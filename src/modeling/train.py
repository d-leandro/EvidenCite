"""Entrenamiento unificado para TF-IDF, Encoders Científicos (SciBERT/SPECTER2) y LLMs con PEFT/LoRA."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import torch
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
    build_sequence_classifier,
    count_model_parameters,
    ensure_tokenizer_padding,
)
from src.modeling.loss import build_loss_fn, compute_class_weights


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


class WeightedSequenceTrainer(Trainer):
    """Subclase de HuggingFace Trainer que inyecta Weighted Cross-Entropy o Focal Loss en compute_loss."""

    def __init__(
        self,
        *args: Any,
        class_weights: Optional[torch.Tensor] = None,
        loss_type: str = "weighted_ce",
        focal_gamma: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self._custom_loss_fn: Optional[torch.nn.Module] = None

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is not None and logits is not None and (
            self.loss_type != "cross_entropy" or self.class_weights is not None
        ):
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
                logits.view(-1, num_labels),
                labels.view(-1),
            )
        else:
            loss = outputs.get("loss")
            if loss is None and logits is not None and labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


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


def _log_to_mlflow_safe(
    config: ExperimentConfig,
    artifact: ExperimentRunArtifact,
    split_hashes: Dict[str, str],
) -> None:
    """Registra hiperparámetros, hashes SHA-256 y métricas A6/A7 en MLflow de forma tolerante a fallos."""
    try:
        import mlflow

        mlflow.set_tracking_uri(config.paths.mlflow_tracking_uri)
        mlflow.set_experiment(config.paths.mlflow_experiment_name)
        with mlflow.start_run(run_name=config.experiment_name):
            mlflow.log_params(
                {
                    "experiment_name": config.experiment_name,
                    "model_family": config.model_family,
                    "configuration": config.input_representation.configuration,
                    "seed": config.seed,
                    "max_seq_length": config.input_representation.max_seq_length,
                    "enforce_per_chunk_budget": config.input_representation.enforce_per_chunk_budget,
                    **{f"sha256_{k}": v for k, v in split_hashes.items()},
                }
            )
            mlflow.log_metrics(
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
                mlflow.log_metrics(
                    {
                        "test_macro_f1_all_9": artifact.test_metrics_a6.macro_f1_all_9,
                        "test_macro_f1_present": artifact.test_metrics_a6.macro_f1_present,
                        "test_micro_f1": artifact.test_metrics_a6.micro_f1,
                        "test_weighted_f1": artifact.test_metrics_a6.weighted_f1,
                    }
                )
    except Exception:
        # Tolerancia a fallos si MLflow no está disponible o el backend está bloqueado
        pass


def train_tfidf_baseline(
    config: ExperimentConfig,
    train_instances: Sequence[DatasetInstance],
    dev_instances: Sequence[DatasetInstance],
    test_instances: Optional[Sequence[DatasetInstance]] = None,
    enable_mlflow: bool = False,
    split_hashes: Optional[Dict[str, str]] = None,
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
    (exp_dir / "metrics_summary.json").write_text(
        artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if enable_mlflow:
        _log_to_mlflow_safe(config, artifact, split_hashes or {})
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

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(sub_cfg.pretrained_model_name_or_path)
    if model is None:
        model = build_sequence_classifier(config)

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
    device_str = resolve_device(config.device)
    use_cpu = device_str == "cpu"

    steps_per_epoch = max(
        1,
        len(train_ds)
        // max(1, sub_cfg.per_device_train_batch_size * sub_cfg.gradient_accumulation_steps),
    )
    total_train_steps = max(1, int(steps_per_epoch * sub_cfg.num_train_epochs))
    warmup_steps = int(total_train_steps * sub_cfg.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(exp_dir / "checkpoints"),
        learning_rate=sub_cfg.learning_rate,
        per_device_train_batch_size=sub_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=sub_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=sub_cfg.gradient_accumulation_steps,
        num_train_epochs=sub_cfg.num_train_epochs,
        weight_decay=sub_cfg.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=sub_cfg.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_strategy="epoch",
        seed=config.seed,
        fp16=bool(sub_cfg.fp16 and device_str == "cuda"),
        bf16=bool(sub_cfg.bf16 and device_str == "cuda"),
        use_cpu=use_cpu,
        report_to="none",
        dataloader_pin_memory=False,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer) if sub_cfg.dynamic_padding else None

    trainer = WeightedSequenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=sub_cfg.early_stopping_patience)],
        class_weights=class_weights,
        loss_type=sub_cfg.loss_type,
        focal_gamma=sub_cfg.focal_gamma,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_wall_time = time.perf_counter() - t0

    # Evaluación en Dev
    t_inf0 = time.perf_counter()
    dev_output = trainer.predict(dev_ds)
    inf_elapsed = max(1e-9, time.perf_counter() - t_inf0)
    dev_logits = dev_output.predictions[0] if isinstance(dev_output.predictions, tuple) else dev_output.predictions
    dev_pred_ids = np.argmax(dev_logits, axis=-1).tolist()
    dev_true_ids = _extract_label_ids(dev_instances)
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
        test_pred_ids = np.argmax(test_logits, axis=-1).tolist()
        test_true_ids = _extract_label_ids(test_instances)
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
    (exp_dir / "metrics_summary.json").write_text(
        artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if enable_mlflow:
        _log_to_mlflow_safe(config, artifact, split_hashes or {})
    return artifact


def run_training_experiment(
    config: ExperimentConfig,
    train_instances: Optional[Sequence[DatasetInstance]] = None,
    dev_instances: Optional[Sequence[DatasetInstance]] = None,
    test_instances: Optional[Sequence[DatasetInstance]] = None,
    tokenizer: Optional[Any] = None,
    model: Optional[torch.nn.Module] = None,
    enable_mlflow: bool = False,
) -> ExperimentRunArtifact:
    """Punto de entrada unificado para ejecutar experimentos de entrenamiento (TF-IDF, Encoder o LoRA)."""
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

    if config.model_family == "tfidf":
        return train_tfidf_baseline(
            config=config,
            train_instances=train_instances,
            dev_instances=dev_instances,
            test_instances=test_instances,
            enable_mlflow=enable_mlflow,
            split_hashes=split_hashes,
        )
    if config.model_family in ("encoder", "llm_lora"):
        return train_transformer_experiment(
            config=config,
            train_instances=train_instances,
            dev_instances=dev_instances,
            test_instances=test_instances,
            tokenizer=tokenizer,
            model=model,
            enable_mlflow=enable_mlflow,
            split_hashes=split_hashes,
        )
    raise ValueError(
        f"Para model_family='{config.model_family}' utilice las funciones de inferencia en src.modeling.predict."
    )