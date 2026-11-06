"""Cálculo de métricas de acuerdo inter-jueces, concordancia y distribución de etiquetas."""

from collections import Counter, defaultdict
import itertools
import math
from typing import Any, Dict, List, Optional

from sklearn.metrics import cohen_kappa_score

from src.data.schemas import DatasetInstance
from src.labeling.schemas import PanelVote

# Clases cubiertas por el benchmark Human Gold de MultiCite
MULTICITE_GOLD_CLASSES = {"Application", "Background", "Comparison", "Gap", "Improvement"}


def _pairwise_kappas(
    instances: List[DatasetInstance],
    votes_by_instance: Dict[str, List[PanelVote]],
    judge_ids: List[str],
) -> Dict[str, Optional[float]]:
    """Cohen's Kappa por pares de jueces; None cuando no está definido (sin variación de etiquetas)."""
    kappas: Dict[str, Optional[float]] = {}
    for j1, j2 in itertools.combinations(judge_ids, 2):
        y1, y2 = [], []
        for inst in instances:
            v_map = {v.judge_id: v.label for v in votes_by_instance.get(inst.instance_id, [])}
            if v_map.get(j1) and v_map.get(j2):
                y1.append(v_map[j1])
                y2.append(v_map[j2])
        if y1:
            score = float(cohen_kappa_score(y1, y2))
            kappas[f"{j1}__vs__{j2}"] = None if math.isnan(score) else score
    return kappas


def _judge_usage(votes_by_instance: Dict[str, List[PanelVote]]) -> Dict[str, Dict[str, Any]]:
    usage: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"votes": 0, "abstentions": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms_total": 0.0}
    )
    for votes in votes_by_instance.values():
        for v in votes:
            u = usage[v.judge_id]
            u["votes"] += 1
            u["abstentions"] += int(v.label is None)
            u["prompt_tokens"] += v.prompt_tokens
            u["completion_tokens"] += v.completion_tokens
            u["latency_ms_total"] += v.latency_ms
    for u in usage.values():
        u["mean_latency_ms"] = u["latency_ms_total"] / u["votes"] if u["votes"] else 0.0
        u["abstention_rate"] = u["abstentions"] / u["votes"] if u["votes"] else 0.0
    return {k: dict(v) for k, v in usage.items()}


def compute_labeling_metrics(
    instances_before: List[DatasetInstance],
    instances_after: List[DatasetInstance],
    votes_by_instance: Dict[str, List[PanelVote]],
    quota_per_class: int,
    case_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Calcula la matriz de transición, acuerdos Kappa, exactitud contra gold y balance de cuotas."""
    total = len(instances_after)
    splits_count = Counter(inst.annotation_metadata.split for inst in instances_after)
    sources_after = Counter(inst.annotation_metadata.label_source for inst in instances_after)

    sources_by_split: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for inst in instances_after:
        sources_by_split[str(inst.annotation_metadata.split)][inst.annotation_metadata.label_source] += 1

    # 1. Matriz de transición de procedencia y de etiqueta (antes -> después)
    source_transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    label_transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for b, a in zip(instances_before, instances_after):
        source_transitions[b.annotation_metadata.label_source][a.annotation_metadata.label_source] += 1
        if not b.annotation_metadata.is_gold:
            panel = a.annotation_metadata.panel_label or "sin_mayoria"
            label_transitions[str(b.label)][panel] += 1

    # 2. Acuerdo inter-jueces por pares con Cohen's Kappa
    judge_ids = sorted({v.judge_id for votes in votes_by_instance.values() for v in votes})
    pairwise_kappas = _pairwise_kappas(instances_after, votes_by_instance, judge_ids)
    defined_kappas = [k for k in pairwise_kappas.values() if k is not None]
    mean_kappa = sum(defined_kappas) / len(defined_kappas) if defined_kappas else None

    # 3. Exactitud y matriz de confusión frente a Human Gold (5 clases de MultiCite)
    gold_instances = [
        inst for inst in instances_after
        if inst.annotation_metadata.is_gold and inst.label in MULTICITE_GOLD_CLASSES
    ]
    panel_correct = sum(1 for inst in gold_instances if inst.annotation_metadata.panel_label == inst.label)
    gold_confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for inst in gold_instances:
        gold_confusion[str(inst.label)][inst.annotation_metadata.panel_label or "sin_mayoria"] += 1

    per_judge_gold_acc: Dict[str, float] = {}
    for j_id in judge_ids:
        correct = total_eval = 0
        for inst in gold_instances:
            label = next((v.label for v in votes_by_instance.get(inst.instance_id, []) if v.judge_id == j_id), None)
            if label:
                total_eval += 1
                correct += int(label == inst.label)
        per_judge_gold_acc[j_id] = correct / total_eval if total_eval else 0.0

    # 4. Estado de cuota por clase en el dataset completo
    class_distribution = Counter(inst.label for inst in instances_after if inst.label)
    quota_status = {
        cls_name: {"total_count": count, "quota_target": quota_per_class, "deficit": max(0, quota_per_class - count)}
        for cls_name, count in sorted(class_distribution.items())
    }

    return {
        "total_instances": total,
        "splits": dict(splits_count),
        "label_source_distribution": dict(sources_after),
        "label_source_by_split": {k: dict(v) for k, v in sources_by_split.items()},
        "decision_cases": dict(sorted((case_counts or {}).items())),
        "source_transitions": {k: dict(v) for k, v in source_transitions.items()},
        "label_transitions_non_gold": {k: dict(v) for k, v in label_transitions.items()},
        "pairwise_cohen_kappa": pairwise_kappas,
        "mean_pairwise_kappa": mean_kappa,
        "gold_calibration": {
            "evaluated_instances": len(gold_instances),
            "evaluated_classes": sorted(MULTICITE_GOLD_CLASSES),
            "panel_accuracy": panel_correct / len(gold_instances) if gold_instances else 0.0,
            "per_judge_accuracy": per_judge_gold_acc,
            "confusion_matrix": {k: dict(v) for k, v in gold_confusion.items()},
        },
        "judge_usage": _judge_usage(votes_by_instance),
        "quota_status": quota_status,
    }
