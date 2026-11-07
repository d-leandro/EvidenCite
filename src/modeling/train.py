"""Entrenamiento unificado para TF-IDF, Encoders Científicos (SciBERT/SPECTER2) y LLMs con PEFT/LoRA."""

import contextlib
import itertools
import json
import os
from pathlib import Path
import shutil
import time
import sys
from tqdm.std import tqdm as tqdm_texto
from transformers import ProgressCallback
from transformers.trainer_utils import has_length
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import DataLoader, SequentialSampler, Sampler
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config.schema import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    ExperimentConfig,
    LoraModelConfig,
    resolve_canonical_label,
    resolve_device,
)
from src.data.schemas import DatasetInstance
from src.modeling.dataset import (
    CitationIntentDataset,
    compute_split_sha256,
    format_instance_as_single_text,
    load_jsonl_instances,
)
from src.modeling.evaluate import (
    OperationalMetricsA7,
    TechnicalMetricsA6,
    compute_a6_metrics,
    compute_hf_metrics,
)
from src.modeling.heads import (
    SMOKE_MODEL_NAME,
    build_sequence_classifier,
    count_model_parameters,
    create_smoke_model,
    ensure_tokenizer_padding,
)
from src.modeling import registro_mlflow
from src.modeling.lora import create_lora_optimizer, initialize_eva, save_converted_adapter
from src.modeling.loss import build_loss_fn, compute_class_weights
from src.modeling.registro_mlflow import (
    RealTimeMLflowCallback,
    RegistroMLflow,
    abrir_mlflow,
    en_colab,
    parametros_experimento,
    versiones,
)


class ExperimentRunArtifact(BaseModel):
    """Artefacto consolidado devuelto al finalizar una corrida de entrenamiento y evaluación."""

    experiment_name: str
    model_family: str
    configuration: str
    output_dir: Path
    dev_metrics_a6: TechnicalMetricsA6
    test_metrics_a6: Optional[TechnicalMetricsA6] = None
    operational_metrics_a7: OperationalMetricsA7
    dev_predictions: List[str]
    test_predictions: Optional[List[str]] = None
    training_history: List[Dict[str, Any]] = []

class _OrdenFijo(Sampler):
    """Recorre el dataset en un orden dado."""

    def __init__(self, orden: List[int]) -> None:
        self.orden = orden

    def __iter__(self):
        return iter(self.orden)

    def __len__(self) -> int:
        return len(self.orden)


def _orden_por_longitud(dataset: Any) -> List[int]:
    """Índices de la instancia más larga a la más corta. Es determinista: el sampler y _al_orden_original
    obtienen el mismo orden."""
    longitudes = dataset.longitudes()
    return sorted(range(len(longitudes)), key=lambda i: -longitudes[i])


def _al_orden_original(valores: np.ndarray, dataset: Any) -> np.ndarray:
    """predict() entrega las filas en el orden del sampler; esto las devuelve al orden de las instancias."""
    salida = np.empty_like(valores)
    salida[_orden_por_longitud(dataset)] = valores
    return salida

class WeightedSequenceTrainer(Trainer):
    """Subclase de HuggingFace Trainer que inyecta Weighted Cross-Entropy o Focal Loss en compute_loss.

    Complementada con el entrenamiento de lora.zip: el forward va en autocast y la pérdida se calcula sobre los
    logits en fp32; LoRA+ y LoRA-FA usan su optimizador propio; y como compute_loss devuelve la media del micro-lote,
    el Trainer la divide por la acumulación de gradiente (loss_is_scaled_for_ga = False).
    """

    loss_is_scaled_for_ga = False

    def __init__(
        self,
        *args: Any,
        class_weights: Optional[torch.Tensor] = None,
        loss_type: str = "weighted_ce",
        focal_gamma: float = 2.0,
        autocast_dtype: Optional[torch.dtype] = None,
        lora_config: Optional[LoraModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.autocast_dtype = autocast_dtype
        self.lora_config = lora_config
        self._custom_loss_fn: Optional[torch.nn.Module] = None

    def create_optimizer(self, model: Optional[torch.nn.Module] = None) -> torch.optim.Optimizer:
        """AdamW de TrainingArguments, o el optimizador propio de LoRA+ y LoRA-FA (src/modeling/lora)."""
        if self.optimizer is None and self.lora_config is not None:
            self.optimizer = create_lora_optimizer(
                self.model,
                self.lora_config,
                learning_rate=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                fused="fused" in str(self.args.optim),
            )
        return super().create_optimizer(model)

    def _get_eval_sampler(self, eval_dataset: Any) -> Optional[torch.utils.data.Sampler]:
        """Evaluación y predicción de la instancia más larga a la más corta, para rellenar lo mínimo. Las métricas de
        evaluate() no dependen del orden; train.py devuelve las predicciones de predict() al orden de las instancias."""
        if eval_dataset is None:
            return None
        if not hasattr(eval_dataset, "longitudes"):
            return SequentialSampler(eval_dataset)
        return _OrdenFijo(_orden_por_longitud(eval_dataset))

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        labels = inputs.get("labels")
        autocast = (
            torch.autocast(device_type=self.args.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with autocast:
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        loss = None
        if labels is not None and logits is not None:
            if self._custom_loss_fn is None:
                self._custom_loss_fn = build_loss_fn(
                    loss_type=self.loss_type,  # type: ignore[arg-type]
                    class_weights=self.class_weights,
                    focal_gamma=self.focal_gamma,
                    device=logits.device,
                )
            else:
                self._custom_loss_fn = self._custom_loss_fn.to(logits.device)
            num_labels = getattr(self.model.config, "num_labels", logits.shape[-1])
            loss = self._custom_loss_fn(
                logits.float().view(-1, num_labels),
                labels.view(-1),
            )

        return (loss, outputs) if return_outputs else loss

class BarraTexto(ProgressCallback):
    """Barras de texto de tqdm, como en lora.zip. Dentro de un kernel de Jupyter (y por eso con `colab exec`) el
    Trainer usa NotebookProgressCallback, una tabla HTML que se redibuja en cada paso: por la CLI cada redibujo
    llega como texto nuevo."""

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm_texto(total=state.max_steps, desc="entrenando", unit="paso",
                                           ncols=100, mininterval=2, file=sys.stdout)
        self.current_step = 0

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm_texto(total=len(eval_dataloader), desc="evaluando", unit="lote",
                                                 ncols=100, mininterval=2, leave=False, file=sys.stdout)
            self.prediction_bar.update(1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or self.training_bar is None or not logs:
            return
        if "loss" in logs:  # pérdida y tasa dentro de la barra, sin una línea nueva cada LOG_CADA pasos
            self.training_bar.set_postfix(loss=f"{logs['loss']:.4f}", lr=f"{logs.get('learning_rate', 0):.1e}",
                                          refresh=False)
        if "eval_f1_macro" in logs:  # una línea por evaluación de dev
            self.training_bar.write(f"época {logs.get('epoch', 0):.2f} · dev: pérdida {logs['eval_loss']:.4f}"
                                    f" · macro-F1 {logs['eval_f1_macro']:.4f}", file=sys.stdout)

def _extract_label_ids(instances: Sequence[DatasetInstance]) -> List[int]:
    label_ids: List[int] = []
    for inst in instances:
        canonical = resolve_canonical_label(inst.label)
        if canonical is None or canonical not in LABEL_TO_ID:
            raise ValueError(
                f"Instancia {inst.instance_id} tiene etiqueta inválida o nula: {inst.label!r}"
            )
        label_ids.append(LABEL_TO_ID[canonical])
    return label_ids


def _get_peak_memory_mb(device_str: str) -> float:
    if device_str == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    if (
        device_str == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "current_allocated_memory")
    ):
        try:
            return float(torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
        except Exception:
            return 0.0
    return 0.0


def _scheduler_args(name: str, total_steps: int, threshold: float) -> Tuple[str, Dict[str, Any]]:
    """Planificador de lora.zip como (lr_scheduler_type, lr_scheduler_kwargs) de TrainingArguments."""
    if name == "cosine_min":
        return "cosine_with_min_lr", {"min_lr_rate": 0.1}
    if name == "constant":
        return "constant_with_warmup", {}
    if name == "wsd":
        return "warmup_stable_decay", {"num_decay_steps": max(1, int(0.2 * total_steps)), "decay_type": "1-sqrt"}
    if name == "plateau":
        # Baja la tasa a la mitad en cada evaluación en que eval_f1_macro no mejora (sin warmup)
        return "reduce_lr_on_plateau", {
            "mode": "max",
            "factor": 0.5,
            "patience": 0,
            "threshold": threshold,
            "threshold_mode": "abs",
        }
    return name, {}  # linear, cosine

def _compilar_por_bloques(model, modo=None) -> int:
    # En multimodales (Qwen3.5, Gemma 4) la torre de visión aparece antes en model.modules(): se elige la lista
    # homogénea con más parámetros, que son los bloques del modelo de texto.
    candidatas = [m for m in model.modules()
                  if isinstance(m, torch.nn.ModuleList) and len(m) > 1 and len({type(b) for b in m}) == 1]
    if not candidatas:
        return 0
    bloques = max(candidatas, key=lambda m: sum(p.numel() for p in m.parameters()))
    for bloque in bloques:
        bloque.compile(dynamic=True, mode=modo)
    return len(bloques)

def _convertir_a_fp8(model: torch.nn.Module) -> int:
    """FP8 en las capas lineales congeladas del modelo base (torchao, escala por tensor). LoRA y la cabeza se
    entrenan y quedan como estaban. Devuelve cuántas capas convirtió."""
    from torchao.float8 import convert_to_float8_training  # solo se importa si se usa

    convertidas: List[str] = []

    def filtro(modulo: torch.nn.Module, nombre: str) -> bool:
        ok = (
            type(modulo) is torch.nn.Linear  # ni las capas de 4 bits de QLoRA ni otras subclases
            and not modulo.weight.requires_grad  # solo pesos congelados
            and modulo.in_features % 16 == 0 and modulo.out_features % 16 == 0  # requisito de FP8
            and not (modulo.in_features <= 4096 and modulo.out_features <= 1024)  # muy chicas: no compensa
        )
        if ok:
            convertidas.append(nombre)
        return ok

    convert_to_float8_training(model, module_filter_fn=filtro)
    return len(convertidas)

def _log_to_mlflow_safe(
    config: ExperimentConfig,
    artifact: ExperimentRunArtifact,
    split_hashes: Dict[str, str],
    registro: Optional[RegistroMLflow] = None,
) -> bool:
    """Registra hiperparámetros, hashes SHA-256, métricas A6/A7, archivos y el mejor modelo en MLflow de forma
    tolerante a fallos. Devuelve True si todo llegó al servidor: entonces no hace falta guardar nada en local.

    Usa la corrida que run_training_experiment abrió al empezar (`registro`, en RUNNING mientras entrena) o, si no
    hay, abre una con abrir_mlflow. Al final la cierra en FINISHED. El modelo queda como modelo registrado de la
    corrida (columna Models): best_model/ (los .safetensors con config y tokenizador; con el humo de LoRA, también su
    base en modelo_base/) o tfidf_pipeline.joblib.
    """
    exp_dir = artifact.output_dir
    staging = exp_dir / "modelo"  # TF-IDF: el modelo es un archivo; se sube desde esta carpeta
    try:
        registro = registro or abrir_mlflow(config, en_colab(), versiones())
        if registro is None:
            return False
        registro.parametros({f"sha256_{k}": v for k, v in split_hashes.items()})
        registro.metricas(
            {
                "dev_macro_f1_all_9": artifact.dev_metrics_a6.macro_f1_all_9,
                "dev_macro_f1_present": artifact.dev_metrics_a6.macro_f1_present,
                "dev_micro_f1": artifact.dev_metrics_a6.micro_f1,
                "dev_weighted_f1": artifact.dev_metrics_a6.weighted_f1,
                "a7_latency_ms": artifact.operational_metrics_a7.inference_latency_ms_per_instance,
                "a7_trainable_ratio_pct": artifact.operational_metrics_a7.trainable_ratio_pct,
                "a7_train_wall_time_sec": artifact.operational_metrics_a7.train_wall_time_sec,
            }
        )
        if artifact.test_metrics_a6 is not None:
            registro.metricas(
                {
                    "test_macro_f1_all_9": artifact.test_metrics_a6.macro_f1_all_9,
                    "test_macro_f1_present": artifact.test_metrics_a6.macro_f1_present,
                    "test_micro_f1": artifact.test_metrics_a6.micro_f1,
                    "test_weighted_f1": artifact.test_metrics_a6.weighted_f1,
                }
            )
        for archivo in ("config.json", "metrics_summary.json"):
            registro.artefacto(exp_dir / archivo)
        if config.model_family == "tfidf":
            staging.mkdir(exist_ok=True)
            enlace = staging / "tfidf_pipeline.joblib"
            if not enlace.exists():
                try:
                    os.link(exp_dir / "tfidf_pipeline.joblib", enlace)
                except OSError:  # sistemas de archivos sin enlaces duros
                    shutil.copy2(exp_dir / "tfidf_pipeline.joblib", enlace)
            carpetas, tipo = {None: staging}, "sklearn"
        else:
            carpetas, tipo = {None: exp_dir / "best_model"}, "transformers"
            if config.model_family == "llm_lora":
                tipo = "peft"
                if (exp_dir / "modelo_humo").is_dir():
                    carpetas["modelo_base"] = exp_dir / "modelo_humo"
        registro.modelo(config.experiment_name, carpetas, tipo, parametros_experimento(config))
        if registro_mlflow.SALIDA_CONSOLA is not None:
            registro.artefacto(registro_mlflow.SALIDA_CONSOLA)
        completo = registro.cerrar("FINISHED")
        print(f"MLflow {'listo' if completo else 'incompleto (los resultados quedan también en local)'}: "
              f"{registro.url_corrida() or 'sin corrida'}")
        return completo
    except Exception:
        # Tolerancia a fallos si MLflow no está disponible o el backend está bloqueado; la corrida no queda en RUNNING
        if registro is not None:
            registro.cerrar("FAILED", esperar=False)
        return False
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def train_tfidf_baseline(
    config: ExperimentConfig,
    train_instances: Sequence[DatasetInstance],
    dev_instances: Sequence[DatasetInstance],
    test_instances: Optional[Sequence[DatasetInstance]] = None,
    enable_mlflow: bool = False,
    split_hashes: Optional[Dict[str, str]] = None,
    registro: Optional[RegistroMLflow] = None,
) -> ExperimentRunArtifact:
    """Entrena y evalúa el baseline clásico TF-IDF + LogisticRegression."""
    assert config.tfidf is not None
    tfidf_cfg = config.tfidf
    rep_cfg = config.input_representation

    x_train = [format_instance_as_single_text(inst, rep_cfg) for inst in train_instances]
    y_train = _extract_label_ids(train_instances)

    x_dev = [format_instance_as_single_text(inst, rep_cfg) for inst in dev_instances]
    y_dev = _extract_label_ids(dev_instances)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(tfidf_cfg.ngram_min, tfidf_cfg.ngram_max),
                    min_df=tfidf_cfg.min_df,
                    max_df=tfidf_cfg.max_df,
                    max_features=tfidf_cfg.max_features,
                    sublinear_tf=tfidf_cfg.sublinear_tf,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=tfidf_cfg.c_penalty,
                    class_weight=tfidf_cfg.class_weight,
                    max_iter=tfidf_cfg.max_iter,
                    random_state=config.seed,
                ),
            ),
        ]
    )

    t0 = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_wall_time = time.perf_counter() - t0

    t_inf0 = time.perf_counter()
    dev_pred_ids = pipeline.predict(x_dev).tolist()
    inf_elapsed = max(1e-9, time.perf_counter() - t_inf0)

    dev_metrics = compute_a6_metrics(y_dev, dev_pred_ids)
    dev_pred_labels = [ID_TO_LABEL[int(i)] for i in dev_pred_ids]

    test_metrics: Optional[TechnicalMetricsA6] = None
    test_pred_labels: Optional[List[str]] = None
    eval_count = len(x_dev)

    if test_instances is not None and len(test_instances) > 0:
        x_test = [format_instance_as_single_text(inst, rep_cfg) for inst in test_instances]
        y_test = _extract_label_ids(test_instances)
        t_test0 = time.perf_counter()
        test_pred_ids = pipeline.predict(x_test).tolist()
        inf_elapsed = max(1e-9, time.perf_counter() - t_test0)
        eval_count = len(x_test)
        test_metrics = compute_a6_metrics(y_test, test_pred_ids)
        test_pred_labels = [ID_TO_LABEL[int(i)] for i in test_pred_ids]

    clf_step: LogisticRegression = pipeline.named_steps["clf"]
    num_params = int(clf_step.coef_.size + clf_step.intercept_.size)
    latency_ms = (inf_elapsed * 1000.0) / max(1, eval_count)
    throughput = float(eval_count) / inf_elapsed

    op_metrics = OperationalMetricsA7(
        model_family="tfidf",
        configuration=rep_cfg.configuration,
        total_params=num_params,
        trainable_params=num_params,
        trainable_ratio_pct=100.0,
        train_wall_time_sec=float(train_wall_time),
        inference_latency_ms_per_instance=float(latency_ms),
        throughput_instances_per_sec=float(throughput),
        peak_memory_mb=0.0,
        invalid_format_rate=0.0,
        estimated_api_cost_usd=0.0,
    )

    exp_dir = config.paths.output_dir / config.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, exp_dir / "tfidf_pipeline.joblib")

    artifact = ExperimentRunArtifact(
        experiment_name=config.experiment_name,
        model_family=config.model_family,
        configuration=rep_cfg.configuration,
        output_dir=exp_dir,
        dev_metrics_a6=dev_metrics,
        test_metrics_a6=test_metrics,
        operational_metrics_a7=op_metrics,
        dev_predictions=dev_pred_labels,
        test_predictions=test_pred_labels,
        training_history=[],
    )
    (exp_dir / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    (exp_dir / "metrics_summary.json").write_text(
        artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if enable_mlflow and _log_to_mlflow_safe(config, artifact, split_hashes or {}, registro):
        shutil.rmtree(exp_dir, ignore_errors=True)  # todo quedó en MLflow: nada en local
        with contextlib.suppress(OSError):
            exp_dir.parent.rmdir()  # la carpeta de salidas, si quedó vacía
    return artifact


def train_transformer_experiment(
    config: ExperimentConfig,
    train_instances: Sequence[DatasetInstance],
    dev_instances: Sequence[DatasetInstance],
    test_instances: Optional[Sequence[DatasetInstance]] = None,
    tokenizer: Optional[Any] = None,
    model: Optional[torch.nn.Module] = None,
    enable_mlflow: bool = False,
    split_hashes: Optional[Dict[str, str]] = None,
    registro: Optional[RegistroMLflow] = None,
) -> ExperimentRunArtifact:
    """Entrena y evalúa un clasificador basado en Transformer (SciBERT/SPECTER2 o LLM con LoRA SEQ_CLS)."""
    set_seed(config.seed)
    rep_cfg = config.input_representation

    if config.model_family == "encoder":
        assert config.encoder is not None
        sub_cfg: Any = config.encoder
    elif config.model_family == "llm_lora":
        assert config.llm_lora is not None
        sub_cfg = config.llm_lora
    else:
        raise ValueError(f"Familia no soportada en train_transformer_experiment: {config.model_family}")

    if sub_cfg.pretrained_model_name_or_path == SMOKE_MODEL_NAME:
        # Modelo de humo: BERT o Qwen3 minúsculo creado aquí, con tokenizador entrenado sobre train (sin descargas)
        texts = [format_instance_as_single_text(inst, rep_cfg) for inst in train_instances]
        smoke_dir = create_smoke_model(config, texts, config.paths.output_dir / config.experiment_name / "modelo_humo")
        sub_cfg = sub_cfg.model_copy(update={"pretrained_model_name_or_path": smoke_dir})
        config = config.model_copy(update={config.model_family: sub_cfg})

    # Precisión mixta (lora.zip): bf16 en GPUs con cc >= 8.0, fp16 en las anteriores (T4) y fp32 sin CUDA
    device_str = resolve_device(config.device)
    cuda = device_str == "cuda"
    ampere = cuda and torch.cuda.get_device_capability() >= (8, 0)
    bf16 = bool(sub_cfg.bf16 and ampere)
    fp16 = bool(cuda and not bf16 and (sub_cfg.fp16 or sub_cfg.bf16))
    autocast_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(sub_cfg.pretrained_model_name_or_path)
    if model is None:
        model = build_sequence_classifier(config, dtype=autocast_dtype, device=device_str)

    ensure_tokenizer_padding(tokenizer, model)  # type: ignore[arg-type]

    train_ds = CitationIntentDataset(
        tokenizer=tokenizer,
        instances=list(train_instances),
        rep_config=rep_cfg,
        dynamic_padding=sub_cfg.dynamic_padding,
    )
    dev_ds = CitationIntentDataset(
        tokenizer=tokenizer,
        instances=list(dev_instances),
        rep_config=rep_cfg,
        dynamic_padding=sub_cfg.dynamic_padding,
    )
    test_ds = (
        CitationIntentDataset(
            tokenizer=tokenizer,
            instances=list(test_instances),
            rep_config=rep_cfg,
            dynamic_padding=sub_cfg.dynamic_padding,
        )
        if (test_instances is not None and len(test_instances) > 0)
        else None
    )

    train_label_ids = _extract_label_ids(train_instances)
    class_weights = compute_class_weights(
        train_label_ids,
        scheme=sub_cfg.class_weight_scheme,
    )

    exp_dir = config.paths.output_dir / config.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    use_cpu = device_str == "cpu"

    steps_per_epoch = max(
        1,
        len(train_ds)
        // max(1, sub_cfg.per_device_train_batch_size * sub_cfg.gradient_accumulation_steps),
    )
    total_train_steps = max(1, int(steps_per_epoch * sub_cfg.num_train_epochs))
    if sub_cfg.max_steps > 0:
        total_train_steps = sub_cfg.max_steps
    warmup_steps = int(total_train_steps * sub_cfg.warmup_ratio)
    scheduler_type, scheduler_kwargs = _scheduler_args(
        sub_cfg.lr_scheduler, total_train_steps, sub_cfg.early_stopping_threshold
    )
    # Dev (y el checkpoint del mejor punto) una vez por época o evals_per_epoch veces, repartidas en partes iguales
    eval_kwargs: Dict[str, Any] = {"eval_strategy": "epoch", "save_strategy": "epoch"}
    if sub_cfg.evals_per_epoch > 1:
        eval_steps = max(1, round(steps_per_epoch / sub_cfg.evals_per_epoch))
        eval_kwargs = {
            "eval_strategy": "steps",
            "eval_steps": eval_steps,
            "save_strategy": "steps",
            "save_steps": eval_steps,
        }
    compile_model = bool(sub_cfg.torch_compile and cuda)
    fp8 = bool(getattr(sub_cfg, "fp8_base", False))
    if fp8 and not (cuda and torch.cuda.get_device_capability() >= (8, 9)):
        print("AVISO: FP8 desactivado: necesita una GPU con cc >= 8.9 (L4, G4, H100); la A100 y la T4 no tienen FP8")
        fp8 = False
    if fp8 and not compile_model:
        print("AVISO: FP8 desactivado: necesita COMPILAR=si (sin compilar, torchao calcula también el gradiente de "
              "los pesos congelados y sale más lento)")
        fp8 = False

    training_args = TrainingArguments(
        output_dir=str(exp_dir / "checkpoints"),
        learning_rate=sub_cfg.learning_rate,
        per_device_train_batch_size=sub_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=sub_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=sub_cfg.gradient_accumulation_steps,
        num_train_epochs=sub_cfg.num_train_epochs,
        max_steps=sub_cfg.max_steps,
        weight_decay=sub_cfg.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=scheduler_type,
        lr_scheduler_kwargs=scheduler_kwargs,
        max_grad_norm=sub_cfg.max_grad_norm,
        **eval_kwargs,
        save_total_limit=1,
        save_only_model=True,  # sin estado del optimizador: no se reanudan corridas
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=sub_cfg.logging_steps,
        seed=config.seed,
        fp16=fp16,
        bf16=bf16,
        tf32=True if ampere else None,  # multiplicaciones fp32 en tensor cores
        use_cpu=use_cpu,
        report_to="none",
        dataloader_pin_memory=cuda,
        dataloader_num_workers=sub_cfg.dataloader_num_workers,
        dataloader_persistent_workers=sub_cfg.dataloader_num_workers > 0,
        accelerator_config={"non_blocking": cuda},  # copias asíncronas CPU → GPU desde memoria fijada
        train_sampling_strategy="group_by_length" if sub_cfg.group_by_length else "random",
        gradient_checkpointing=sub_cfg.gradient_checkpointing,
        label_names=["labels"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if sub_cfg.gradient_checkpointing else None,
    )

    # Relleno dinámico hasta un múltiplo de 8 en CUDA (tensor cores); de 16 con FP8, que lo exige
    pad_multiple = (16 if fp8 else 8) if cuda else None
    collator = (
        DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_multiple)
        if sub_cfg.dynamic_padding
        else None
    )

    if config.model_family == "llm_lora" and sub_cfg.lora_variant == "eva":
        # EVA inicia el adaptador con las activaciones de los primeros lotes de train
        eva_loader = DataLoader(train_ds, batch_size=sub_cfg.per_device_train_batch_size, collate_fn=collator)
        initialize_eva(model, itertools.islice(eva_loader, 8), autocast_dtype)  # type: ignore[arg-type]

    if fp8:
        capas = _convertir_a_fp8(model)
        print(f"FP8: {capas} capas lineales del modelo base multiplican en FP8")

    if compile_model:
        bloques = _compilar_por_bloques(model, sub_cfg.torch_compile_mode)
        print(f"torch.compile: {bloques} bloques compilados en su lugar, con formas dinámicas")

    callbacks: List[Any] = [
        EarlyStoppingCallback(
            early_stopping_patience=sub_cfg.early_stopping_patience,
            early_stopping_threshold=sub_cfg.early_stopping_threshold,
        )
    ]
    if registro is not None:
        callbacks.append(RealTimeMLflowCallback(registro))  # curvas en MLflow mientras entrena

    trainer = WeightedSequenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_hf_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
        loss_type=sub_cfg.loss_type,
        focal_gamma=sub_cfg.focal_gamma,
        autocast_dtype=autocast_dtype,
        lora_config=sub_cfg if config.model_family == "llm_lora" else None,
    )

    # La barra de texto en lugar de la tabla de Jupyter (o de la barra de tqdm.auto fuera de Jupyter)
    for callback in list(trainer.callback_handler.callbacks):
        if type(callback).__name__ in ("NotebookProgressCallback", "ProgressCallback"):
            trainer.remove_callback(callback)
            trainer.add_callback(BarraTexto())

    t0 = time.perf_counter()
    trainer.train()
    if compile_model:
        from torch._dynamo.utils import counters
        print(f"torch.compile: {counters['stats']['unique_graphs']} grafos compilados")
    train_wall_time = time.perf_counter() - t0
    if registro is not None:  # el mejor punto que eligió el early stopping y los pasos entrenados
        registro.metricas(
            {
                "best_step": trainer.state.best_global_step,
                "best_eval_f1_macro": trainer.state.best_metric,
                "trained_steps": trainer.state.global_step,
                "planned_steps": trainer.state.max_steps,
            }
        )

    # Evaluación en Dev
    t_inf0 = time.perf_counter()
    dev_output = trainer.predict(dev_ds)
    inf_elapsed = max(1e-9, time.perf_counter() - t_inf0)
    dev_logits = dev_output.predictions[0] if isinstance(dev_output.predictions, tuple) else dev_output.predictions
    dev_logits = _al_orden_original(dev_logits, dev_ds)
    dev_pred_ids = np.argmax(dev_logits, axis=-1).tolist()
    dev_true_ids = _extract_label_ids(dev_instances)
    # Comprobación: las etiquetas que devolvió predict, en el orden original, son las de las instancias
    assert _al_orden_original(dev_output.label_ids, dev_ds).tolist() == dev_true_ids
    dev_metrics = compute_a6_metrics(dev_true_ids, dev_pred_ids)
    dev_pred_labels = [ID_TO_LABEL[int(i)] for i in dev_pred_ids]
    eval_count = len(dev_instances)

    # Evaluación en Test (si se proporciona)
    test_metrics: Optional[TechnicalMetricsA6] = None
    test_pred_labels: Optional[List[str]] = None
    if test_ds is not None and test_instances is not None:
        t_test0 = time.perf_counter()
        test_output = trainer.predict(test_ds)
        inf_elapsed = max(1e-9, time.perf_counter() - t_test0)
        eval_count = len(test_instances)
        test_logits = (
            test_output.predictions[0]
            if isinstance(test_output.predictions, tuple)
            else test_output.predictions
        )
        test_logits = _al_orden_original(test_logits, test_ds)
        test_pred_ids = np.argmax(test_logits, axis=-1).tolist()
        test_true_ids = _extract_label_ids(test_instances)
        assert _al_orden_original(test_output.label_ids, test_ds).tolist() == test_true_ids
        test_metrics = compute_a6_metrics(test_true_ids, test_pred_ids)
        test_pred_labels = [ID_TO_LABEL[int(i)] for i in test_pred_ids]

    total_params, trainable_params, trainable_ratio_pct = count_model_parameters(model)
    latency_ms = (inf_elapsed * 1000.0) / max(1, eval_count)
    throughput = float(eval_count) / inf_elapsed

    op_metrics = OperationalMetricsA7(
        model_family=config.model_family,  # type: ignore[arg-type]
        configuration=rep_cfg.configuration,
        total_params=total_params,
        trainable_params=trainable_params,
        trainable_ratio_pct=trainable_ratio_pct,
        train_wall_time_sec=float(train_wall_time),
        inference_latency_ms_per_instance=float(latency_ms),
        throughput_instances_per_sec=float(throughput),
        peak_memory_mb=_get_peak_memory_mb(device_str),
        invalid_format_rate=0.0,
        estimated_api_cost_usd=0.0,
    )

    best_model_dir = exp_dir / "best_model"
    if getattr(model, "initial_adapter_dir", None):
        # PiSSA / OLoRA modifican W: el adaptador se guarda convertido a LoRA estándar sobre la base original
        save_converted_adapter(model, best_model_dir)  # type: ignore[arg-type]
        tokenizer.save_pretrained(best_model_dir)
    else:
        trainer.save_model(str(best_model_dir))

    artifact = ExperimentRunArtifact(
        experiment_name=config.experiment_name,
        model_family=config.model_family,
        configuration=rep_cfg.configuration,
        output_dir=exp_dir,
        dev_metrics_a6=dev_metrics,
        test_metrics_a6=test_metrics,
        operational_metrics_a7=op_metrics,
        dev_predictions=dev_pred_labels,
        test_predictions=test_pred_labels,
        training_history=list(trainer.state.log_history),
    )
    (exp_dir / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    (exp_dir / "metrics_summary.json").write_text(
        artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if enable_mlflow and _log_to_mlflow_safe(config, artifact, split_hashes or {}, registro):
        shutil.rmtree(exp_dir, ignore_errors=True)  # todo quedó en MLflow: nada en local
        with contextlib.suppress(OSError):
            exp_dir.parent.rmdir()  # la carpeta de salidas, si quedó vacía
    return artifact


def run_training_experiment(
    config: ExperimentConfig,
    train_instances: Optional[Sequence[DatasetInstance]] = None,
    dev_instances: Optional[Sequence[DatasetInstance]] = None,
    test_instances: Optional[Sequence[DatasetInstance]] = None,
    tokenizer: Optional[Any] = None,
    model: Optional[torch.nn.Module] = None,
    enable_mlflow: bool = True,
) -> ExperimentRunArtifact:
    """Punto de entrada unificado para ejecutar experimentos de entrenamiento (TF-IDF, Encoder o LoRA).

    Se asume un servidor de MLflow (registro_mlflow.direcciones_mlflow): la corrida se abre antes de entrenar
    (RUNNING, con curvas en tiempo real) y se cierra en FINISHED al final, o en FAILED si algo falla. Si todo llega
    al servidor, la carpeta local de la corrida se borra. Sin servidor, advierte y los resultados quedan en local.
    """
    split_hashes: Dict[str, str] = {}
    if train_instances is None:
        train_instances = load_jsonl_instances(config.paths.train_file)
        if config.paths.train_file.exists():
            split_hashes["train"] = compute_split_sha256(config.paths.train_file)
    if dev_instances is None:
        dev_instances = load_jsonl_instances(config.paths.dev_file)
        if config.paths.dev_file.exists():
            split_hashes["dev"] = compute_split_sha256(config.paths.dev_file)
    if test_instances is None and config.paths.test_file.exists():
        test_instances = load_jsonl_instances(config.paths.test_file)
        split_hashes["test"] = compute_split_sha256(config.paths.test_file)

    if config.model_family not in ("tfidf", "encoder", "llm_lora"):
        raise ValueError(
            f"Para model_family='{config.model_family}' utilice las funciones de inferencia en src.modeling.predict."
        )
    registro: Optional[RegistroMLflow] = None
    if enable_mlflow:
        try:
            registro = abrir_mlflow(config, en_colab(), versiones())
        except Exception as e:
            print(f"AVISO: no se pudo abrir la corrida en MLflow ({type(e).__name__}: {e}); se entrena sin MLflow.")
    try:
        if config.model_family == "tfidf":
            return train_tfidf_baseline(
                config=config,
                train_instances=train_instances,
                dev_instances=dev_instances,
                test_instances=test_instances,
                enable_mlflow=registro is not None,
                split_hashes=split_hashes,
                registro=registro,
            )
        return train_transformer_experiment(
            config=config,
            train_instances=train_instances,
            dev_instances=dev_instances,
            test_instances=test_instances,
            tokenizer=tokenizer,
            model=model,
            enable_mlflow=registro is not None,
            split_hashes=split_hashes,
            registro=registro,
        )
    except BaseException as e:
        if registro is not None:
            if registro_mlflow.SALIDA_CONSOLA is not None:
                registro.artefacto(registro_mlflow.SALIDA_CONSOLA)  # la salida hasta el error, para revisarlo
            registro.cerrar("KILLED" if isinstance(e, KeyboardInterrupt) else "FAILED", esperar=False)
        raise