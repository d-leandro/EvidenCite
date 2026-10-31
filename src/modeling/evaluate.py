"""Evaluación técnica (Actividad A6), operativa (Actividad A7) y pruebas de significancia estadística pareadas."""

import math
from typing import Dict, List, Literal, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config.schema import (
    CANONICAL_9_CLASSES,
    LABEL_TO_ID,
    resolve_canonical_label,
)


def _to_class_indices(labels: Sequence[Union[int, str]]) -> np.ndarray:
    """Convierte una secuencia de etiquetas (enteros 0..8 o strings) a un arreglo NumPy de índices."""
    indices: list[int] = []
    for item in labels:
        if isinstance(item, (int, np.integer)):
            indices.append(int(item))
        else:
            canonical = resolve_canonical_label(str(item))
            if canonical is None or canonical not in LABEL_TO_ID:
                raise ValueError(f"Etiqueta desconocida o fuera de las 9 clases canónicas: {item!r}")
            indices.append(LABEL_TO_ID[canonical])
    return np.asarray(indices, dtype=np.int64)


class TechnicalMetricsA6(BaseModel):
    """Contrato estandarizado de métricas técnicas (Model-Centric) para la Actividad A6."""

    macro_f1_all_9: float = Field(
        ...,
        description="Macro-F1 promediado sobre las 9 clases canónicas (0.0 en clases sin soporte).",
    )
    macro_f1_present: float = Field(
        ...,
        description="Macro-F1 promediado únicamente sobre las clases con soporte > 0 en la partición.",
    )
    micro_f1: float = Field(..., description="Micro-F1 global (equivalente a Accuracy en clasificación multiclase).")
    weighted_f1: float = Field(..., description="F1 ponderado por el soporte de cada clase.")
    macro_precision: float = Field(..., description="Precisión macro sobre las 9 clases canónicas.")
    macro_recall: float = Field(..., description="Recall macro sobre las 9 clases canónicas.")
    per_class_f1: Dict[str, float] = Field(..., description="F1-score desglosado para cada una de las 9 clases.")
    per_class_precision: Dict[str, float] = Field(..., description="Precisión por clase.")
    per_class_recall: Dict[str, float] = Field(..., description="Recall por clase.")
    per_class_support: Dict[str, int] = Field(..., description="Soporte (número de instancias reales) por clase.")
    confusion_matrix: List[List[int]] = Field(
        ...,
        description="Matriz de confusión 9x9 ordenada según CANONICAL_9_CLASSES.",
    )


class OperationalMetricsA7(BaseModel):
    """Contrato estandarizado de métricas operativas (System-Centric) para la Actividad A7."""

    model_family: Literal["tfidf", "encoder", "llm_lora", "llm_api"]
    configuration: Literal["context_only", "A", "B"]
    total_params: int = Field(..., ge=0)
    trainable_params: int = Field(..., ge=0)
    trainable_ratio_pct: float = Field(..., ge=0.0, le=100.0)
    train_wall_time_sec: float = Field(default=0.0, ge=0.0)
    inference_latency_ms_per_instance: float = Field(..., ge=0.0)
    throughput_instances_per_sec: float = Field(..., ge=0.0)
    peak_memory_mb: float = Field(default=0.0, ge=0.0)
    invalid_format_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_api_cost_usd: float = Field(default=0.0, ge=0.0)


class PairedSignificanceResult(BaseModel):
    """Resultado de las pruebas de significancia estadística pareadas (Paired Bootstrap + McNemar)."""

    score_a: float = Field(..., description="Macro-F1 observado en Configuración A.")
    score_b: float = Field(..., description="Macro-F1 observado en Configuración B.")
    delta_b_minus_a: float = Field(..., description="Diferencia observada: Score(B) - Score(A).")
    ci_95_low: float = Field(..., description="Límite inferior del intervalo de confianza del 95% para Delta.")
    ci_95_high: float = Field(..., description="Límite superior del intervalo de confianza del 95% para Delta.")
    p_value_bootstrap: float = Field(..., description="Valor-p bilateral del test Paired Bootstrap.")
    mcnemar_b_only_correct: int = Field(
        ...,
        description="Instancias donde B acierta y A falla (n_01).",
    )
    mcnemar_a_only_correct: int = Field(
        ...,
        description="Instancias donde A acierta y B falla (n_10).",
    )
    p_value_mcnemar: float = Field(..., description="Valor-p exacto/chi2 del test de McNemar.")
    n_bootstrap: int = Field(..., ge=1)
    is_significant_95: bool = Field(
        ...,
        description="True si p_value_bootstrap < 0.05.",
    )


def compute_a6_metrics(
    y_true: Sequence[Union[int, str]],
    y_pred: Sequence[Union[int, str]],
) -> TechnicalMetricsA6:
    """Calcula el conjunto completo de métricas técnicas A6 sobre las 9 clases canónicas."""
    true_idx = _to_class_indices(y_true)
    pred_idx = _to_class_indices(y_pred)
    if len(true_idx) != len(pred_idx):
        raise ValueError(
            f"Longitudes incompatibles: y_true ({len(true_idx)}) != y_pred ({len(pred_idx)})"
        )

    num_classes = len(CANONICAL_9_CLASSES)
    all_label_ids = list(range(num_classes))

    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        true_idx,
        pred_idx,
        labels=all_label_ids,
        zero_division=0,
    )

    macro_f1_all_9 = float(np.mean(f1_arr))
    present_mask = sup_arr > 0
    macro_f1_present = (
        float(np.mean(f1_arr[present_mask])) if np.any(present_mask) else 0.0
    )

    micro_f1 = float(
        f1_score(true_idx, pred_idx, labels=all_label_ids, average="micro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(true_idx, pred_idx, labels=all_label_ids, average="weighted", zero_division=0)
    )
    macro_precision = float(np.mean(prec_arr))
    macro_recall = float(np.mean(rec_arr))

    cm = confusion_matrix(true_idx, pred_idx, labels=all_label_ids)

    per_class_f1 = {
        CANONICAL_9_CLASSES[i]: float(f1_arr[i]) for i in all_label_ids
    }
    per_class_precision = {
        CANONICAL_9_CLASSES[i]: float(prec_arr[i]) for i in all_label_ids
    }
    per_class_recall = {
        CANONICAL_9_CLASSES[i]: float(rec_arr[i]) for i in all_label_ids
    }
    per_class_support = {
        CANONICAL_9_CLASSES[i]: int(sup_arr[i]) for i in all_label_ids
    }

    return TechnicalMetricsA6(
        macro_f1_all_9=macro_f1_all_9,
        macro_f1_present=macro_f1_present,
        micro_f1=micro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_support=per_class_support,
        confusion_matrix=cm.tolist(),
    )


def compute_hf_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Función compatible con `transformers.Trainer(compute_metrics=...)` para selección por `eval_f1_macro`."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    a6 = compute_a6_metrics(labels.tolist(), preds.tolist())
    return {
        "f1_macro": a6.macro_f1_all_9,
        "f1_macro_present": a6.macro_f1_present,
        "f1_micro": a6.micro_f1,
        "f1_weighted": a6.weighted_f1,
        "precision_macro": a6.macro_precision,
        "recall_macro": a6.macro_recall,
    }


def _mcnemar_exact_or_chi2_pvalue(n_01: int, n_10: int) -> float:
    """Calcula el valor-p bilateral de McNemar a partir de los pares discordantes (n_01, n_10)."""
    n_disc = n_01 + n_10
    if n_disc == 0:
        return 1.0
    if n_disc <= 50:
        # Cálculo binomial exacto bilateral con p=0.5
        k = min(n_01, n_10)
        tail_prob = sum(math.comb(n_disc, i) for i in range(k + 1)) * (0.5 ** n_disc)
        return float(min(1.0, 2.0 * tail_prob))
    # Aproximación normal / chi-cuadrado con corrección de continuidad de Edwards
    chi2_stat = ((abs(n_01 - n_10) - 1.0) ** 2) / float(n_disc)
    return float(math.erfc(math.sqrt(chi2_stat / 2.0)))


def _macro_f1_from_flat_cm(
    flat_cm: np.ndarray,
    num_classes: int,
    class_mask: np.ndarray,
) -> np.ndarray:
    """Calcula de forma vectorizada el Macro-F1 para un lote de matrices de confusión [B, K*K]."""
    cm = flat_cm.reshape(-1, num_classes, num_classes).astype(np.float64)
    tp = np.diagonal(cm, axis1=1, axis2=2)  # [B, K]
    fp = cm.sum(axis=1) - tp                # [B, K]
    fn = cm.sum(axis=2) - tp                # [B, K]
    denom = 2.0 * tp + fp + fn
    f1_per_class = np.where(denom > 0.0, (2.0 * tp) / np.maximum(denom, 1e-12), 0.0)
    active_f1 = f1_per_class[:, class_mask]
    return active_f1.mean(axis=1)


def paired_significance_test(
    y_true: Sequence[Union[int, str]],
    y_pred_a: Sequence[Union[int, str]],
    y_pred_b: Sequence[Union[int, str]],
    n_bootstrap: int = 10000,
    seed: int = 42,
    use_present_classes_only: bool = False,
) -> PairedSignificanceResult:
    """Ejecuta Paired Bootstrap (B=10.000 por defecto) sobre Macro-F1 y test de McNemar pareado."""
    true_idx = _to_class_indices(y_true)
    pred_a = _to_class_indices(y_pred_a)
    pred_b = _to_class_indices(y_pred_b)

    n_samples = len(true_idx)
    if len(pred_a) != n_samples or len(pred_b) != n_samples:
        raise ValueError("y_true, y_pred_a y y_pred_b deben tener exactamente la misma longitud.")
    if n_samples == 0:
        raise ValueError("No se puede ejecutar prueba de significancia sobre listas vacías.")

    num_classes = len(CANONICAL_9_CLASSES)
    metrics_a = compute_a6_metrics(true_idx.tolist(), pred_a.tolist())
    metrics_b = compute_a6_metrics(true_idx.tolist(), pred_b.tolist())

    if use_present_classes_only:
        class_mask = np.array(
            [metrics_a.per_class_support[c] > 0 for c in CANONICAL_9_CLASSES],
            dtype=bool,
        )
        if not np.any(class_mask):
            class_mask = np.ones(num_classes, dtype=bool)
        score_a = metrics_a.macro_f1_present
        score_b = metrics_b.macro_f1_present
    else:
        class_mask = np.ones(num_classes, dtype=bool)
        score_a = metrics_a.macro_f1_all_9
        score_b = metrics_b.macro_f1_all_9

    observed_delta = float(score_b - score_a)

    # Codificamos cada instancia i en su celda conjunta del cubo (y_true, y_pred_a, y_pred_b)
    # y usamos muestreo multinomial para calcular B=10.000 matrices de confusión en ~20 ms
    cell_a = true_idx * num_classes + pred_a
    cell_b = true_idx * num_classes + pred_b

    rng = np.random.default_rng(seed)
    batch_size = min(n_bootstrap, 2000)
    deltas_list: list[np.ndarray] = []

    eye_k2 = np.eye(num_classes * num_classes, dtype=np.int32)
    one_hot_a = eye_k2[cell_a]  # [N, K*K]
    one_hot_b = eye_k2[cell_b]  # [N, K*K]

    remaining = n_bootstrap
    while remaining > 0:
        cur_b = min(batch_size, remaining)
        # Pesos de remuestreo con reemplazo para cada réplica bootstrap: Multinomial(N, [1/N..1/N])
        counts = rng.multinomial(n_samples, np.full(n_samples, 1.0 / n_samples), size=cur_b)
        cm_boot_a = counts @ one_hot_a  # [cur_b, K*K]
        cm_boot_b = counts @ one_hot_b  # [cur_b, K*K]

        f1_boot_a = _macro_f1_from_flat_cm(cm_boot_a, num_classes, class_mask)
        f1_boot_b = _macro_f1_from_flat_cm(cm_boot_b, num_classes, class_mask)
        deltas_list.append(f1_boot_b - f1_boot_a)
        remaining -= cur_b

    boot_deltas = np.concatenate(deltas_list, axis=0)
    ci_95_low = float(np.percentile(boot_deltas, 2.5))
    ci_95_high = float(np.percentile(boot_deltas, 97.5))

    # Valor-p bilateral por desplazamiento de la distribución bootstrap centrada en la hipótesis nula (delta = 0)
    centered_deltas = boot_deltas - np.mean(boot_deltas)
    p_val_boot = float(np.mean(np.abs(centered_deltas) >= abs(observed_delta)))

    # McNemar sobre aciertos a nivel de instancia
    correct_a = pred_a == true_idx
    correct_b = pred_b == true_idx
    n_01 = int(np.sum((~correct_a) & correct_b))
    n_10 = int(np.sum(correct_a & (~correct_b)))
    p_val_mcnemar = _mcnemar_exact_or_chi2_pvalue(n_01, n_10)

    return PairedSignificanceResult(
        score_a=float(score_a),
        score_b=float(score_b),
        delta_b_minus_a=observed_delta,
        ci_95_low=ci_95_low,
        ci_95_high=ci_95_high,
        p_value_bootstrap=p_val_boot,
        mcnemar_b_only_correct=n_01,
        mcnemar_a_only_correct=n_10,
        p_value_mcnemar=p_val_mcnemar,
        n_bootstrap=n_bootstrap,
        is_significant_95=bool(p_val_boot < 0.05),
    )

