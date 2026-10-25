"""Configuración del experimento: Config, su lectura desde variables de entorno y CONFIG_JSON, y su
validación (celda 2).
"""

import json
import os
from dataclasses import dataclass, fields
from datetime import datetime


SCHEDULERS = ("linear", "cosine", "cosine_min", "constant", "wsd", "plateau")  # planificadores de build_scheduler


@dataclass
class Config:
    """Parámetros del experimento. Cada campo acepta una variable de entorno con su nombre en mayúsculas."""

    # Ejecución
    modo: str = "auto"                  # auto | mini | completo
    nombre_corrida: str = ""            # vacío: se arma con modelo, variante, configuración y hora
    semilla: int = 42
    # Datos
    datos_dir: str = ""                 # carpeta con train/dev/test.jsonl o un .zip; vacío: se busca
    configuracion: str = "A"            # A: título + abstract del citado · B: top-3 fragmentos del citado
    incluir_seccion: bool = True        # antepone la sección retórica del citante, como el baseline SciBERT
    max_longitud: int = 1024            # tokens; la configuración B llega a ~800 con el tokenizador de SmolLM2
    evaluar_test: bool = True          # test se usa una sola vez, con el modelo finalista
    # Modelo
    modelo: str = "HuggingFaceTB/SmolLM2-135M"   # id de Hugging Face; "humo": modelo diminuto aleatorio, sin descargas
    # Adaptador
    lora_tipo: str = "lora"             # ver listar_variantes()
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_modulos: str = "auto"          # auto: todas las capas lineales salvo la cabeza; o "q_proj,v_proj"
    # Entrenamiento
    epocas: int = 1
    lr: float = 2e-4
    batch: int = 16
    batch_eval: int = 32
    acumulacion: int = 1
    weight_decay: float = 0.0
    warmup: float = 0.05                # fracción de pasos con calentamiento lineal; después manda `scheduler`
    scheduler: str = "linear"           # linear | cosine | cosine_min | constant | wsd | plateau (ver build_scheduler)
    max_grad_norm: float = 1.0
    max_pasos: int = 0                  # >0 corta el entrenamiento tras N pasos de optimizador
    pesos_clase: str = "ninguno"        # ninguno | inverso | raiz_inversa
    log_cada: int = 10                  # pasos de optimizador entre líneas de registro
    barra_progreso: bool = True         # barra de progreso del entrenamiento
    # Early stopping por macro-F1 de dev
    evals_por_epoca: int = 2            # evaluaciones de dev por época, repartidas en partes iguales
    paciencia: int = 3                  # evaluaciones seguidas sin mejora antes de parar; 0 = nunca para
    delta_minimo: float = 0.0           # mejora mínima de macro-F1 (escala 0-1) que cuenta como mejora
    # Optimizaciones (auto: según el hardware)
    precision: str = "auto"             # auto | bf16 | fp16 | fp32
    compilar: str = "no"                # no | si | auto (si = auto: compila si el hardware lo soporta).
                                        # Compilar cuesta minutos al inicio: solo compensa con modelos grandes y corridas largas.
    modo_compilacion: str = "default"   # default | max-autotune-no-cudagraphs (compila más lento, kernels más rápidos)
    atencion: str = "sdpa"              # sdpa | eager | flash_attention_2 (requiere el paquete flash-attn)
    grad_checkpointing: bool = False    # ahorra memoria a cambio de ~30 % más de tiempo
    num_workers: int = 0                # los datos ya van tokenizados; 0 basta
    # Salidas
    salidas_dir: str = ""               # vacío: /content/salidas en Colab, ./salidas en local
    guardar: bool = True
    # MLflow (vacío: no se usa). La corrida se llama como nombre_corrida; ver RegistroMLflow en registro_mlflow.py
    mlflow_ip: str = ""                 # ip:puerto del servidor, por ejemplo 54.225.26.128:8080
    mlflow_experimento: str = "baseline_lora"
    mlflow_espera_min: float = 20.0     # si el servidor cae, al terminar se lo espera hasta N min desde la caída
    mlflow_adaptador: bool = False      # sube también los pesos del adaptador (unos 175 MB con 8B y r=16)
    # Ajustes del modo mini
    mini_train: int = 96
    mini_dev: int = 64
    mini_max_longitud: int = 256
    mini_max_pasos: int = 12


def _convertir(valor, tipo):
    """Convierte el texto de una variable de entorno o de CONFIG_JSON al tipo del campo de Config."""
    if tipo is bool:
        return str(valor).strip().lower() in ("1", "true", "si", "sí", "yes", "y")
    if tipo is int:
        return int(float(valor))
    if tipo is float:
        return float(valor)
    return str(valor)


def construir_config(colab: bool, cuda: bool, **cambios) -> Config:
    """Arma la Config con esta prioridad: valores por defecto < variables de entorno < CONFIG_JSON < argumentos."""
    cfg = Config()
    tipos = {f.name: f.type for f in fields(Config)}
    for nombre, tipo in tipos.items():
        valor = os.environ.get(nombre.upper())
        if valor not in (None, ""):
            setattr(cfg, nombre, _convertir(valor, tipo))
    if os.environ.get("CONFIG_JSON"):
        for nombre, valor in json.loads(os.environ["CONFIG_JSON"]).items():
            if nombre not in tipos:
                raise KeyError(f"CONFIG_JSON: campo desconocido '{nombre}'. Campos: {', '.join(tipos)}")
            setattr(cfg, nombre, _convertir(valor, tipos[nombre]))
    for nombre, valor in cambios.items():
        if nombre not in tipos:
            raise KeyError(f"Campo desconocido '{nombre}'. Campos: {', '.join(tipos)}")
        setattr(cfg, nombre, _convertir(valor, tipos[nombre]))
    return resolver_config(cfg, colab, cuda)


def resolver_config(cfg: Config, colab: bool, cuda: bool) -> Config:
    """Valida los valores, resuelve los automáticos (modo, carpeta de salida, nombre) y aplica los límites del modo mini."""
    cfg.configuracion = cfg.configuracion.upper()
    if cfg.configuracion not in ("A", "B"):
        raise ValueError(f"CONFIGURACION debe ser A o B, no '{cfg.configuracion}'")
    if cfg.modo == "auto":
        cfg.modo = "completo" if (colab and cuda) else "mini"
    if cfg.modo not in ("mini", "completo"):
        raise ValueError(f"MODO debe ser auto, mini o completo, no '{cfg.modo}'")
    if cfg.modo_compilacion not in ("default", "max-autotune-no-cudagraphs", "max-autotune", "reduce-overhead"):
        raise ValueError(f"MODO_COMPILACION no válido: '{cfg.modo_compilacion}'")
    if cfg.pesos_clase not in ("ninguno", "inverso", "raiz_inversa"):
        raise ValueError(f"PESOS_CLASE debe ser ninguno, inverso o raiz_inversa, no '{cfg.pesos_clase}'")
    if cfg.scheduler not in SCHEDULERS:
        raise ValueError(f"SCHEDULER debe ser uno de {', '.join(SCHEDULERS)}, no '{cfg.scheduler}'")
    if cfg.evals_por_epoca < 1:
        raise ValueError(f"EVALS_POR_EPOCA debe ser 1 o más, no {cfg.evals_por_epoca}")
    if cfg.paciencia < 0 or cfg.delta_minimo < 0:
        raise ValueError("PACIENCIA y DELTA_MINIMO no pueden ser negativos")
    if cfg.mlflow_espera_min < 0:
        raise ValueError("MLFLOW_ESPERA_MIN no puede ser negativo")
    if cfg.modo == "mini":
        cfg.max_longitud = min(cfg.max_longitud, cfg.mini_max_longitud)
        cfg.max_pasos = cfg.max_pasos or cfg.mini_max_pasos
        cfg.batch = min(cfg.batch, 8)
        cfg.batch_eval = min(cfg.batch_eval, 16)
        cfg.log_cada = min(cfg.log_cada, 2)
    if not cfg.salidas_dir:
        cfg.salidas_dir = "/content/salidas" if colab else "salidas"
    if not cfg.nombre_corrida:
        corto = cfg.modelo.rstrip("/").split("/")[-1]
        cfg.nombre_corrida = f"{corto}-{cfg.lora_tipo}-cfg{cfg.configuracion}-{cfg.modo}-{datetime.now():%Y%m%d-%H%M%S}"
    return cfg


def imprimir_config(cfg: Config) -> None:
    """Imprime los campos de la Config que más cambian entre corridas."""
    claves = ("modo", "modelo", "lora_tipo", "configuracion", "max_longitud", "epocas", "lr", "batch",
              "acumulacion", "scheduler", "lora_r", "lora_alpha", "evals_por_epoca", "paciencia", "precision", "compilar",
              "max_pasos", "salidas_dir", "nombre_corrida")
    for k in claves:
        print(f"  {k:<16}{getattr(cfg, k)}")
