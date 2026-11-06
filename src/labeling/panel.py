"""Lógica de decisión, agregación y aplicación de consensos del panel de jueces.

Matriz de decisión (D7):
    A  Unanimidad del panel                         -> llm_panel (puede corregir la etiqueta débil)
    B  Mayoría calificada que coincide con la previa -> llm_panel
    C  Mayoría calificada que difiere de la previa   -> needs_review (conserva la etiqueta previa)
    D  Empate o dispersión sin mayoría calificada    -> needs_review (conserva la etiqueta previa)
    E  Ningún voto válido (abstención total)         -> needs_review (conserva la etiqueta previa)
    F  Instancia gold                                -> human_gold (solo calibración, etiqueta intacta)
"""

from collections import Counter
from typing import List, Optional

from src.data.schemas import DatasetInstance
from src.labeling.schemas import AggregationConfig, PanelDecision, PanelVote

CASE_UNANIMOUS = "A"
CASE_MAJORITY_CONFIRMS_PRIOR = "B"
CASE_MAJORITY_DIFFERS_PRIOR = "C"
CASE_NO_MAJORITY = "D"
CASE_ALL_ABSTAINED = "E"
CASE_GOLD = "F"


def aggregate_votes(
    votes: List[PanelVote],
    prior_label: Optional[str],
    is_gold: bool,
    config: AggregationConfig,
) -> PanelDecision:
    """Agrega los votos de los jueces de forma pura determinando consenso y etiqueta resultante."""
    n_votes = len(votes)
    counts = Counter(v.label for v in votes if v.label is not None)

    if is_gold:
        majority_label, top_count = counts.most_common(1)[0] if counts else (None, 0)
        return PanelDecision(
            final_label=prior_label,
            final_label_source="human_gold",
            panel_agreement=top_count / n_votes if n_votes else 0.0,
            panel_label=majority_label,
            case=CASE_GOLD,
            votes=votes,
        )

    if not counts:
        return PanelDecision(
            final_label=prior_label,
            final_label_source="needs_review",
            panel_agreement=0.0,
            panel_label=None,
            case=CASE_ALL_ABSTAINED,
            votes=votes,
        )

    ranked = counts.most_common()
    majority_label, top_count = ranked[0]
    is_tie = len(ranked) > 1 and ranked[1][1] == top_count
    panel_agreement = top_count / n_votes

    if panel_agreement == 1.0:
        return PanelDecision(
            final_label=majority_label,
            final_label_source="llm_panel",
            panel_agreement=1.0,
            panel_label=majority_label,
            case=CASE_UNANIMOUS,
            votes=votes,
        )

    if not is_tie and panel_agreement >= config.min_agreement:
        if majority_label == prior_label:
            return PanelDecision(
                final_label=majority_label,
                final_label_source="llm_panel",
                panel_agreement=panel_agreement,
                panel_label=majority_label,
                case=CASE_MAJORITY_CONFIRMS_PRIOR,
                votes=votes,
            )
        return PanelDecision(
            final_label=prior_label,
            final_label_source="needs_review",
            panel_agreement=panel_agreement,
            panel_label=majority_label,
            case=CASE_MAJORITY_DIFFERS_PRIOR,
            votes=votes,
        )

    return PanelDecision(
        final_label=prior_label,
        final_label_source="needs_review",
        panel_agreement=panel_agreement,
        panel_label=None if is_tie else majority_label,
        case=CASE_NO_MAJORITY,
        votes=votes,
    )


def apply_decision(
    instance: DatasetInstance,
    decision: PanelDecision,
) -> DatasetInstance:
    """Retorna una copia profunda de la instancia con AnnotationMetadata actualizado de forma inmutable."""
    inst_copy = instance.model_copy(deep=True)
    meta = inst_copy.annotation_metadata

    if not meta.is_gold:
        meta.prior_label_source = meta.label_source
        meta.label_source = decision.final_label_source
        inst_copy.label = decision.final_label

    meta.panel_agreement = decision.panel_agreement
    meta.panel_label = decision.panel_label
    meta.panel_votes = [v.model_dump() for v in decision.votes]

    return inst_copy
