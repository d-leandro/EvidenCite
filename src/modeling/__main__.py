"""python -m src.modeling: una corrida de entrenamiento configurada con variables de entorno.

La CLI de Colab pasa los parámetros como variables (--env CLAVE=valor), igual que lora.zip. Este programa arma el
ExperimentConfig de src/config/schema.py con, en orden de prioridad creciente:

1. Los valores por defecto de ExperimentConfig, o un YAML de configs/ si se pasa CONFIG_YAML.
2. Una variable por campo, con su nombre en mayúsculas: SEED, CONFIGURATION, LEARNING_RATE, LORA_VARIANT,
   LR_SCHEDULER, GROUP_BY_LENGTH, TRAIN_FILE, MLFLOW_TRACKING_URI... (top-level, input_representation, paths y la
   familia). Las listas van separadas por comas: TARGET_MODULES=q_proj,v_proj.
3. Los nombres de lora.zip (ALIAS): MODELO, LORA_TIPO, LR, EPOCAS, BATCH, SCHEDULER, PRECISION, DATOS_DIR...

FAMILIA elige la familia: lora (llm_lora, por defecto), scibert (encoder) o tfidf. MODELO=humo usa el modelo
minúsculo de prueba de la familia, y MODO=mini toma 96/64/64 instancias y una época: juntos verifican el pipeline en
segundos.

Se asume un servidor de MLflow: MLFLOW_IP (o MLFLOW_TRACKING_URI, o la IP escrita en mlflow_ip.txt). Todo queda en
la corrida de MLflow, también la salida de la consola (salida.log), y nada en local. Si el servidor no responde,
advierte y la carpeta de la corrida (sin checkpoints/) queda en <SALIDAS_DIR>/<nombre>.zip. La última línea dice
dónde quedaron los resultados: «RESULTADOS EN MLFLOW» o «RESULTADOS LOCALES: <zip>». Con SALTAR_EXISTENTES=1, una
corrida que ya está terminada en MLflow no se repite («YA EN MLFLOW»).
"""

import logging
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import types
import typing
from datetime import datetime

from pydantic import BaseModel

from src.config.schema import ExperimentConfig, load_experiment_config
from src.modeling import registro_mlflow
from src.modeling.dataset import load_jsonl_instances
from src.modeling.train import run_training_experiment

FAMILIAS = {"lora": "llm_lora", "llm_lora": "llm_lora", "scibert": "encoder", "encoder": "encoder", "tfidf": "tfidf"}

# Nombre de lora.zip → (bloque de ExperimentConfig, campo). "familia" es el bloque de la familia elegida.
ALIAS = {
    "NOMBRE_CORRIDA": (None, "experiment_name"),
    "SEMILLA": (None, "seed"),
    "CONFIGURACION": ("input_representation", "configuration"),
    "INCLUIR_SECCION": ("input_representation", "include_citing_section"),
    "MAX_LONGITUD": ("input_representation", "max_seq_length"),
    "MODELO": ("familia", "pretrained_model_name_or_path"),
    "LORA_TIPO": ("familia", "lora_variant"),
    "LORA_R": ("familia", "r"),
    "LORA_MODULOS": ("familia", "target_modules"),
    "LR": ("familia", "learning_rate"),
    "EPOCAS": ("familia", "num_train_epochs"),
    "BATCH": ("familia", "per_device_train_batch_size"),
    "BATCH_EVAL": ("familia", "per_device_eval_batch_size"),
    "ACUMULACION": ("familia", "gradient_accumulation_steps"),
    "WARMUP": ("familia", "warmup_ratio"),
    "SCHEDULER": ("familia", "lr_scheduler"),
    "PESOS_CLASE": ("familia", "class_weight_scheme"),
    "PACIENCIA": ("familia", "early_stopping_patience"),
    "DELTA_MINIMO": ("familia", "early_stopping_threshold"),
    "EVALS_POR_EPOCA": ("familia", "evals_per_epoch"),
    "MAX_PASOS": ("familia", "max_steps"),
    "LOG_CADA": ("familia", "logging_steps"),
    "GRAD_CHECKPOINTING": ("familia", "gradient_checkpointing"),
    "COMPILAR": ("familia", "torch_compile"),
    "MODO_COMPILACION": ("familia", "torch_compile_mode"),
    "ATENCION": ("familia", "attn_implementation"),
    "NUM_WORKERS": ("familia", "dataloader_num_workers"),
    "AGRUPAR_LONGITUD": ("familia", "group_by_length"),
}
# Valores de lora.zip que cambian de nombre en el esquema
VALORES = {
    "PESOS_CLASE": {"ninguno": "none", "inverso": "balanced", "raiz_inversa": "sqrt_balanced"},
    "LORA_MODULOS": {"auto": "all-linear"},
    "COMPILAR": {"auto": "1", "si": "1", "sí": "1", "no": "0"},
}
MINI = {"train": 96, "dev": 64, "test": 64}


def _convertir(valor: str, anotacion):
    """Texto de una variable de entorno al tipo del campo (bool, int, float, lista o texto); None si es un bloque."""
    if typing.get_origin(anotacion) in (typing.Union, types.UnionType):
        anotacion = next(t for t in typing.get_args(anotacion) if t is not type(None))
    if isinstance(anotacion, type) and issubclass(anotacion, BaseModel):
        return None
    if typing.get_origin(anotacion) is list:
        return [v.strip() for v in valor.split(",") if v.strip()]
    if anotacion is bool:
        return valor.strip().lower() in ("1", "true", "si", "sí", "yes", "y")
    if anotacion is int:
        return int(float(valor))
    if anotacion is float:
        return float(valor)
    return valor


def _poner(cambios: dict, bloque, campo: str, valor) -> None:
    """Escribe cambios[bloque][campo] (o cambios[campo] si el bloque es None)."""
    (cambios if bloque is None else cambios.setdefault(bloque, {}))[campo] = valor


def _fusionar(base: dict, cambios: dict) -> dict:
    """Combina dos diccionarios anidados; ganan los valores de `cambios`."""
    salida = dict(base)
    for k, v in cambios.items():
        salida[k] = _fusionar(salida.get(k) or {}, v) if isinstance(v, dict) else v
    return salida


def construir_config(env: typing.Mapping[str, str] = os.environ) -> tuple[ExperimentConfig, str]:
    """Arma el ExperimentConfig desde las variables de entorno. Devuelve (config, modo)."""
    familia = FAMILIAS.get(env.get("FAMILIA") or env.get("MODEL_FAMILY") or "lora")
    if familia is None:
        raise ValueError("FAMILIA debe ser lora, scibert o tfidf")
    datos: dict = {"experiment_name": "corrida", "model_family": familia}
    if env.get("CONFIG_YAML"):
        datos = _fusionar(load_experiment_config(Path(env["CONFIG_YAML"])).model_dump(), datos)
    base = ExperimentConfig.model_validate(datos)
    bloques = {None: ExperimentConfig, "input_representation": type(base.input_representation),
               "paths": type(base.paths), familia: type(getattr(base, familia))}

    cambios: dict = {}
    for bloque, modelo in bloques.items():  # 2. cada campo con su nombre en mayúsculas
        for campo, info in modelo.model_fields.items():
            valor = env.get(campo.upper())
            if campo != "model_family" and valor not in (None, ""):
                convertido = _convertir(valor, info.annotation)
                if convertido is not None:
                    _poner(cambios, bloque, campo, convertido)
    for alias, (bloque, campo) in ALIAS.items():  # 3. los nombres de lora.zip
        bloque = familia if bloque == "familia" else bloque
        if env.get(alias) not in (None, "") and campo in bloques[bloque].model_fields:
            valor = VALORES.get(alias, {}).get(env[alias].strip().lower(), env[alias])
            _poner(cambios, bloque, campo, _convertir(valor, bloques[bloque].model_fields[campo].annotation))
    precision = env.get("PRECISION", "").lower()
    if precision and familia != "tfidf":  # auto = bf16; train.py baja a fp16 sin bf16 nativo y a fp32 sin CUDA
        _poner(cambios, familia, "bf16", precision in ("auto", "bf16"))
        _poner(cambios, familia, "fp16", precision == "fp16")
    if env.get("DATOS_DIR"):
        for split in ("train", "dev", "test"):
            _poner(cambios, "paths", f"{split}_file", str(Path(env["DATOS_DIR"]) / f"{split}.jsonl"))
    if env.get("SALIDAS_DIR"):
        _poner(cambios, "paths", "output_dir", env["SALIDAS_DIR"])
    if env.get("MLFLOW_IP"):
        ip = env["MLFLOW_IP"]
        _poner(cambios, "paths", "mlflow_tracking_uri", ip if "://" in ip else f"http://{ip}")
    if env.get("MLFLOW_EXPERIMENTO"):
        _poner(cambios, "paths", "mlflow_experiment_name", env["MLFLOW_EXPERIMENTO"])

    config = ExperimentConfig.model_validate(_fusionar(base.model_dump(), cambios))
    if "experiment_name" not in cambios:
        sub = getattr(config, familia)
        modelo = getattr(sub, "pretrained_model_name_or_path", "").rstrip("/").split("/")[-1]
        partes = [familia, modelo, getattr(sub, "lora_variant", ""), config.input_representation.configuration]
        config.experiment_name = "_".join(p for p in partes if p) + f"_{datetime.now():%Y%m%d-%H%M%S}"
    modo = env.get("MODO", "completo").lower()
    if modo == "mini":  # pocas instancias (ver main), una época y secuencias cortas
        config.input_representation.max_seq_length = min(config.input_representation.max_seq_length, 256)
        sub = getattr(config, familia)
        if familia != "tfidf":
            sub.num_train_epochs = 1
            sub.per_device_train_batch_size = min(sub.per_device_train_batch_size, 8)
            sub.per_device_eval_batch_size = min(sub.per_device_eval_batch_size, 16)
            sub.logging_steps = min(sub.logging_steps, 2)
    return config, modo


class _Duplicar:
    """Escribe en la consola y además en el archivo de la salida, que se sube a MLflow como salida.log."""

    def __init__(self, consola, archivo) -> None:
        self.consola, self.archivo = consola, archivo

    def write(self, texto: str) -> int:
        self.consola.write(texto)
        self.archivo.write(texto)
        return len(texto)

    def flush(self) -> None:
        self.consola.flush()
        self.archivo.flush()

    def __getattr__(self, nombre: str):
        return getattr(self.consola, nombre)


def _capturar_salida(ruta: Path):
    """Duplica stdout, stderr y los handlers de logging que escriben en ellos hacia `ruta`. Devuelve cómo restaurar."""
    archivo = open(ruta, "w", encoding="utf-8")
    originales = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Duplicar(sys.stdout, archivo), _Duplicar(sys.stderr, archivo)
    cambiados = []
    for logger in [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]:
        for handler in getattr(logger, "handlers", []):
            if isinstance(handler, logging.StreamHandler) and handler.stream in originales:
                cambiados.append((handler, handler.stream))
                handler.setStream(sys.stdout if handler.stream is originales[0] else sys.stderr)

    def restaurar() -> None:
        sys.stdout, sys.stderr = originales
        for handler, stream in cambiados:
            handler.setStream(stream)
        archivo.close()

    return restaurar


def main() -> Path | None:
    """Entrena y evalúa con run_training_experiment. Con servidor de MLflow no deja nada en local; sin él, empaqueta
    la corrida en <SALIDAS_DIR>/<nombre>.zip y devuelve su ruta."""
    config, modo = construir_config()
    salida = Path(tempfile.mkdtemp(prefix="salida_")) / "salida.log"
    restaurar = _capturar_salida(salida)
    registro_mlflow.SALIDA_CONSOLA = salida
    try:
        return _correr(config, modo, salida)
    finally:
        registro_mlflow.SALIDA_CONSOLA = None
        restaurar()
        shutil.rmtree(salida.parent, ignore_errors=True)


def _correr(config: ExperimentConfig, modo: str, salida: Path) -> Path | None:
    """El cuerpo de main(), con la salida de la consola ya capturada en `salida`."""
    sub = getattr(config, config.model_family)
    print(f"== {config.experiment_name} · {config.model_family} · configuración "
          f"{config.input_representation.configuration} · modo {modo}")
    for campo in ("pretrained_model_name_or_path", "lora_variant", "learning_rate", "num_train_epochs",
                  "per_device_train_batch_size", "gradient_accumulation_steps", "lr_scheduler", "evals_per_epoch",
                  "loss_type", "class_weight_scheme", "bf16", "fp16", "group_by_length", "torch_compile"):
        if hasattr(sub, campo):
            print(f"  {campo:<30}{getattr(sub, campo)}")
    if os.environ.get("SALTAR_EXISTENTES") == "1" and registro_mlflow.existe_en_mlflow(config, registro_mlflow.en_colab()):
        print(f"YA EN MLFLOW: experimento {config.paths.mlflow_experiment_name} · corrida {config.experiment_name}")
        return None

    instancias: dict = {}
    if modo == "mini":
        rng = random.Random(config.seed)
        for split, n in MINI.items():
            todas = load_jsonl_instances(getattr(config.paths, f"{split}_file"))
            instancias[f"{split}_instances"] = rng.sample(todas, min(n, len(todas)))
    artefacto = run_training_experiment(config, enable_mlflow=True, **instancias)

    print(f"  dev  macro-F1 (9) {artefacto.dev_metrics_a6.macro_f1_all_9:.4f}")
    if artefacto.test_metrics_a6 is not None:
        print(f"  test macro-F1 (9) {artefacto.test_metrics_a6.macro_f1_all_9:.4f}")
    carpeta = Path(config.paths.output_dir) / config.experiment_name
    if not carpeta.exists():  # train.py la borró porque todo llegó a MLflow
        print(f"RESULTADOS EN MLFLOW: experimento {config.paths.mlflow_experiment_name} · corrida {config.experiment_name}")
        return None
    shutil.rmtree(carpeta / "checkpoints", ignore_errors=True)  # el mejor modelo ya está en best_model/
    sys.stdout.flush()
    shutil.copyfile(salida, carpeta / "salida.log")
    zip_corrida = Path(shutil.make_archive(str(carpeta), "zip", root_dir=carpeta))
    print(f"RESULTADOS LOCALES: {zip_corrida}")
    return zip_corrida


if __name__ == "__main__":
    main()
