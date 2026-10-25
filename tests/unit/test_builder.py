"""Pruebas unitarias para el módulo de ensamblado y particionamiento del dataset (src.data.builder)."""

import json
from pathlib import Path
import tempfile
import pytest

from src.data.builder import (
    BuilderConfig,
    DatasetBuilder,
    group_stratified_split,
    map_to_canonical_label,
)
from src.data.schemas import (
    AnnotationMetadata,
    CitedPaperInfo,
    CitingPaperInfo,
    DatasetInstance,
    RetrievedChunk,
)


def test_map_to_canonical_label_exact():
    """Valida el mapeo de etiquetas individuales a la taxonomía canónica de 9 clases (RF-04 / ADR-10)."""
    assert map_to_canonical_label("Background") == ("Background", "exact")
    assert map_to_canonical_label("Uses") == ("Application", "exact")
    assert map_to_canonical_label("Application") == ("Application", "exact")
    assert map_to_canonical_label("Differences") == ("Comparison", "exact")
    assert map_to_canonical_label("Similarities") == ("Comparison", "exact")
    assert map_to_canonical_label("Comparison") == ("Comparison", "exact")
    assert map_to_canonical_label("Extends") == ("Improvement", "exact")
    assert map_to_canonical_label("Improvement") == ("Improvement", "exact")
    assert map_to_canonical_label("Improvement / Modification") == ("Improvement", "exact")
    assert map_to_canonical_label("Basis") == ("Basis", "exact")
    assert map_to_canonical_label("Motivation") == ("Gap", "exact")
    assert map_to_canonical_label("Future Work") == ("Gap", "exact")
    assert map_to_canonical_label("Gap") == ("Gap", "exact")
    assert map_to_canonical_label("Evidence") == ("Evidence", "exact")
    assert map_to_canonical_label("Originator") == ("Originator", "exact")
    assert map_to_canonical_label("Identification of the Originator") == ("Originator", "exact")
    assert map_to_canonical_label("Further Reading") == ("Further Reading", "exact")
    assert map_to_canonical_label(None) == (None, None)
    assert map_to_canonical_label("UnknownLabel") == (None, None)


def test_map_to_canonical_label_compound():
    """Valida el mapeo de etiquetas combinadas de MultiCite a su función primaria con tipo 'partial'."""
    assert map_to_canonical_label("Similarities; Uses") == ("Comparison", "partial")
    assert map_to_canonical_label("Background; Motivation") == ("Background", "partial")
    assert map_to_canonical_label("Extends; Differences") == ("Improvement", "partial")


def _create_synthetic_instance(idx: int, citing_paper_id: str, label: str) -> DatasetInstance:
    """Helper para crear una instancia sintética válida."""
    return DatasetInstance(
        instance_id=f"inst_{idx:05d}",
        label=label,
        annotation_metadata=AnnotationMetadata(
            source_dataset="multicite",
            original_label=label,
            mapping_type="exact",
            label_source="human_gold",
            is_gold=True,
            split=None,
        ),
        citing_paper=CitingPaperInfo(
            id=citing_paper_id,
            id_source="arxiv",
            title="Synthetic Paper Title",
            rhetorical_section="Methods",
        ),
        citation_context="We use the model proposed by <cite>Author et al.</cite> for evaluation.",
        cited_paper=CitedPaperInfo(
            id="cited_paper_1",
            title="Cited Title",
            abstract="Cited abstract text.",
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="chunk_1",
                    section_heading="Methods",
                    text="Chunk text.",
                    similarity_score=0.85,
                    rank=1,
                )
            ],
        ),
    )


def test_group_stratified_split_zero_leakage():
    """Valida que ninguna cita del mismo paper citante aparezca en múltiples particiones (RF-06)."""
    instances: list[DatasetInstance] = []
    # 20 papers citantes distintos, cada uno con entre 2 y 5 citas
    inst_idx = 1
    for p_idx in range(1, 21):
        citing_pid = f"paper_citante_{p_idx}"
        for _ in range(4):
            lbl = "Background" if p_idx % 2 == 0 else "Uses"
            instances.append(_create_synthetic_instance(inst_idx, citing_pid, lbl))
            inst_idx += 1

    splits = group_stratified_split(
        instances,
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_papers = set(inst.citing_paper.id for inst in splits["train"])
    dev_papers = set(inst.citing_paper.id for inst in splits["dev"])
    test_papers = set(inst.citing_paper.id for inst in splits["test"])

    # Cero intersección garantizada matemáticamente
    assert train_papers.isdisjoint(dev_papers), "Fuga de datos detectada entre Train y Dev"
    assert train_papers.isdisjoint(test_papers), "Fuga de datos detectada entre Train y Test"
    assert dev_papers.isdisjoint(test_papers), "Fuga de datos detectada entre Dev y Test"

    # Todas las instancias fueron asignadas
    assert len(splits["train"]) + len(splits["dev"]) + len(splits["test"]) == len(instances)

    # Cada instancia tiene su split actualizado
    for inst in splits["train"]:
        assert inst.annotation_metadata.split == "train"
    for inst in splits["dev"]:
        assert inst.annotation_metadata.split == "dev"
    for inst in splits["test"]:
        assert inst.annotation_metadata.split == "test"


def test_dataset_instance_pydantic_serialization():
    """Valida la serialización y deserialización Pydantic de DatasetInstance (RF-07)."""
    inst = _create_synthetic_instance(1, "2105.0001", "Comparison")
    inst.annotation_metadata.split = "train"

    json_str = inst.model_dump_json()
    assert '"instance_id":"inst_00001"' in json_str
    assert '"label":"Comparison"' in json_str
    assert '"split":"train"' in json_str

    deserialized = DatasetInstance.model_validate_json(json_str)
    assert deserialized.instance_id == "inst_00001"
    assert deserialized.label == "Comparison"
    assert deserialized.cited_paper.retrieved_chunks[0].similarity_score == 0.85


def test_builder_metrics_computation():
    """Valida el cálculo de métricas de auditoría y verificación de paridad (RF-08)."""
    instances = [
        _create_synthetic_instance(1, "paper_1", "Background"),
        _create_synthetic_instance(2, "paper_2", "Uses"),
    ]
    splits = {"train": [instances[0]], "dev": [instances[1]], "test": []}

    builder = DatasetBuilder.__new__(DatasetBuilder)
    metrics = builder._compute_metrics(instances, splits, elapsed_sec=1.5)

    assert metrics["total_instances"] == 2
    assert metrics["gold_instances"] == 2
    assert metrics["parity_verification"]["strict_parity_1_to_1"] is True
    assert metrics["leakage_verification"]["zero_leakage_guaranteed"] is True
    assert metrics["splits"]["train"]["count"] == 1
    assert metrics["splits"]["dev"]["count"] == 1


def test_builder_nli_tagger_integration():
    """Valida la integración en memoria de NLITagger en DatasetBuilder para cubrir clases deficitarias."""
    from unittest.mock import MagicMock
    from src.data.schemas import NLITaggingResult
    from src.data.weak_supervision.nli_tagger import NLITagger, NLITaggerConfig

    mock_tagger = MagicMock(spec=NLITagger)
    mock_tagger.tag_batch.return_value = [
        NLITaggingResult(
            context_id="unlabeled_01",
            top1_class="Gap",
            is_accepted=True,
            top1_confidence=0.92,
            margin=0.40,
            class_probabilities={"Gap": 0.92, "Background": 0.08},
        ),
        NLITaggingResult(
            context_id="unlabeled_02",
            top1_class=None,
            is_accepted=False,
            top1_confidence=0.60,
            margin=0.10,
            reason_rejected="low_confidence",
        ),
    ]
    mock_tagger.get_telemetry.return_value = {
        "total_evaluated": 2,
        "total_accepted": 1,
        "total_rejected": 1,
        "accepted_per_class": {"Gap": 1},
    }

    config = BuilderConfig(
        quota_per_class=5,
        nli_config=NLITaggerConfig(quota_buffer=2),
    )

    builder = DatasetBuilder.__new__(DatasetBuilder)
    builder.config = config
    builder.nli_tagger = mock_tagger

    candidates = [
        # 1 instancia Gold de Background
        {
            "context_id": "c_gold_bg",
            "citation_context": "Background context",
            "normalized_label": "Background",
            "is_gold": True,
            "source_dataset": "multicite",
        },
        # 2 candidatos sin etiqueta
        {
            "context_id": "unlabeled_01",
            "citation_context": "Gap context: problem remains unsolved <cite>X</cite>",
            "normalized_label": None,
            "original_label": None,
            "is_gold": False,
            "source_dataset": "unarxive_mined",
        },
        {
            "context_id": "unlabeled_02",
            "citation_context": "Ambiguous context <cite>Y</cite>",
            "normalized_label": None,
            "original_label": None,
            "is_gold": False,
            "source_dataset": "unarxive_mined",
        },
    ]

    balanced = builder.apply_quota_and_balance(candidates, quota_per_class=5)

    # Debe contener 1 Background (Gold) y 1 Gap (enriquecido por NLI)
    gap_items = [x for x in balanced if x.get("normalized_label") == "Gap"]
    assert len(gap_items) == 1
    assert gap_items[0]["label_source"] == "weak_nli"
    assert gap_items[0]["confidence_score"] == 0.92
    assert gap_items[0]["margin"] == 0.40
    assert gap_items[0]["is_gold"] is False

    # Validar telemetría en métricas
    metrics = builder._compute_metrics([], {"train": [], "dev": [], "test": []}, elapsed_sec=0.5)
    assert "nli_weak_supervision" in metrics
    assert metrics["nli_weak_supervision"]["accepted_per_class"]["Gap"] == 1

