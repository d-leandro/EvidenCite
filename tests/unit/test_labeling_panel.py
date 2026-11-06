"""Pruebas unitarias de la agregación del panel de jueces y la aplicación de decisiones (src.labeling.panel)."""

from typing import List, Optional

import pytest

from src.data.schemas import (
    AnnotationMetadata,
    CitedPaperInfo,
    CitingPaperInfo,
    DatasetInstance,
    RetrievedChunk,
)
from src.labeling.panel import aggregate_votes, apply_decision
from src.labeling.schemas import AggregationConfig, PanelVote

CONFIG = AggregationConfig(min_agreement=0.66)


def make_votes(*labels: Optional[str]) -> List[PanelVote]:
    return [
        PanelVote(judge_id=f"j{i}", model_name=f"m{i}", provider="test", label=label, rationale="r" if label else None)
        for i, label in enumerate(labels)
    ]


def make_instance(label: str = "Background", is_gold: bool = False, source: str = "weak_nli") -> DatasetInstance:
    return DatasetInstance(
        instance_id="inst_00001",
        label=label,
        annotation_metadata=AnnotationMetadata(
            source_dataset="multicite" if is_gold else "unarxive_mined",
            label_source=source,
            is_gold=is_gold,
            split="test",
        ),
        citing_paper=CitingPaperInfo(id="citing_1", rhetorical_section="Introduction"),
        citation_context="We follow <cite>Smith et al. (2019)</cite> to train the model.",
        cited_paper=CitedPaperInfo(
            id="cited_1",
            title="Cited Title",
            abstract="Cited abstract.",
            retrieved_chunks=[RetrievedChunk(chunk_id="c1", section_heading="Method", text="Chunk text.", similarity_score=0.9, rank=1)],
        ),
    )


def test_case_a_unanimity_overrides_prior():
    decision = aggregate_votes(make_votes("Application", "Application", "Application"), "Background", False, CONFIG)
    assert decision.case == "A"
    assert decision.final_label == "Application"
    assert decision.final_label_source == "llm_panel"
    assert decision.panel_agreement == 1.0
    assert decision.panel_label == "Application"


def test_case_b_majority_confirms_prior():
    decision = aggregate_votes(make_votes("Gap", "Gap", "Background"), "Gap", False, CONFIG)
    assert decision.case == "B"
    assert decision.final_label == "Gap"
    assert decision.final_label_source == "llm_panel"
    assert decision.panel_agreement == pytest.approx(2 / 3)


def test_case_c_majority_differs_from_prior_keeps_prior_label():
    decision = aggregate_votes(make_votes("Comparison", "Comparison", "Gap"), "Background", False, CONFIG)
    assert decision.case == "C"
    assert decision.final_label == "Background"
    assert decision.final_label_source == "needs_review"
    assert decision.panel_label == "Comparison"


def test_case_d_total_dispersion():
    decision = aggregate_votes(make_votes("Gap", "Basis", "Evidence"), "Gap", False, CONFIG)
    assert decision.case == "D"
    assert decision.final_label == "Gap"
    assert decision.final_label_source == "needs_review"
    assert decision.panel_label is None
    assert decision.panel_agreement == pytest.approx(1 / 3)


def test_case_d_tie_with_abstention():
    decision = aggregate_votes(make_votes("Gap", "Basis", None), "Gap", False, CONFIG)
    assert decision.case == "D"
    assert decision.final_label_source == "needs_review"
    assert decision.panel_label is None


def test_case_e_all_abstained():
    decision = aggregate_votes(make_votes(None, None, None), "Evidence", False, CONFIG)
    assert decision.case == "E"
    assert decision.final_label == "Evidence"
    assert decision.final_label_source == "needs_review"
    assert decision.panel_agreement == 0.0
    assert decision.panel_label is None


def test_case_f_gold_is_never_overwritten():
    decision = aggregate_votes(make_votes("Comparison", "Comparison", "Comparison"), "Application", True, CONFIG)
    assert decision.case == "F"
    assert decision.final_label == "Application"
    assert decision.final_label_source == "human_gold"
    assert decision.panel_label == "Comparison"
    assert decision.panel_agreement == 1.0


def test_gold_without_valid_votes():
    decision = aggregate_votes(make_votes(None, None, None), "Background", True, CONFIG)
    assert decision.case == "F"
    assert decision.panel_label is None
    assert decision.panel_agreement == 0.0


def test_majority_with_one_abstention_counts_over_all_judges():
    decision = aggregate_votes(make_votes("Gap", "Gap", None), "Gap", False, CONFIG)
    assert decision.case == "B"
    assert decision.panel_agreement == pytest.approx(2 / 3)


def test_single_vote_after_two_abstentions_is_not_enough():
    decision = aggregate_votes(make_votes("Gap", None, None), "Gap", False, CONFIG)
    assert decision.case == "D"
    assert decision.final_label_source == "needs_review"


def test_apply_decision_updates_non_gold_without_mutating_input():
    instance = make_instance(label="Background", source="weak_nli")
    decision = aggregate_votes(make_votes("Application", "Application", "Application"), instance.label, False, CONFIG)
    updated = apply_decision(instance, decision)

    assert updated.label == "Application"
    assert updated.annotation_metadata.label_source == "llm_panel"
    assert updated.annotation_metadata.prior_label_source == "weak_nli"
    assert updated.annotation_metadata.panel_label == "Application"
    assert len(updated.annotation_metadata.panel_votes) == 3
    assert updated.annotation_metadata.split == "test"

    assert instance.label == "Background"
    assert instance.annotation_metadata.label_source == "weak_nli"
    assert instance.annotation_metadata.panel_votes is None


def test_apply_decision_needs_review_keeps_prior_label():
    instance = make_instance(label="Background", source="heuristic_rule")
    decision = aggregate_votes(make_votes("Gap", "Basis", "Evidence"), instance.label, False, CONFIG)
    updated = apply_decision(instance, decision)
    assert updated.label == "Background"
    assert updated.annotation_metadata.label_source == "needs_review"
    assert updated.annotation_metadata.prior_label_source == "heuristic_rule"


def test_apply_decision_gold_label_and_source_untouched():
    instance = make_instance(label="Comparison", is_gold=True, source="human_gold")
    decision = aggregate_votes(make_votes("Background", "Background", "Background"), instance.label, True, CONFIG)
    updated = apply_decision(instance, decision)
    assert updated.label == "Comparison"
    assert updated.annotation_metadata.label_source == "human_gold"
    assert updated.annotation_metadata.is_gold is True
    assert updated.annotation_metadata.prior_label_source is None
    assert updated.annotation_metadata.panel_label == "Background"
