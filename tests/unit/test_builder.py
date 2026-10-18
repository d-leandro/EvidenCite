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
    """Valida el mapeo de etiquetas individuales a la taxonomía canónica de 9 clases (RF-04)."""
    assert map_to_canonical_label("Background") == ("Background", "exact")
    assert map_to_canonical_label("Uses") == ("Uses", "exact")
    assert map_to_canonical_label("Differences") == ("Comparison", "exact")
    assert map_to_canonical_label("Similarities") == ("Comparison", "exact")
    assert map_to_canonical_label("Comparison") == ("Comparison", "exact")
    assert map_to_canonical_label("Extends") == ("Extends", "exact")
    assert map_to_canonical_label("Basis") == ("Extends", "exact")
    assert map_to_canonical_label("Motivation") == ("Motivation", "exact")
    assert map_to_canonical_label("Future Work") == ("Future Work", "exact")
    assert map_to_canonical_label("Evidence") == ("Evidence", "exact")
    assert map_to_canonical_label("Originator") == ("Originator", "exact")
    assert map_to_canonical_label("Further Reading") == ("Further Reading", "exact")
    assert map_to_canonical_label(None) == (None, None)
    assert map_to_canonical_label("UnknownLabel") == (None, None)


def test_map_to_canonical_label_compound():
    """Valida el mapeo de etiquetas combinadas de MultiCite a su función primaria con tipo 'partial'."""
    assert map_to_canonical_label("Similarities; Uses") == ("Comparison", "partial")
    assert map_to_canonical_label("Background; Motivation") == ("Background", "partial")
    assert map_to_canonical_label("Extends; Differences") == ("Extends", "partial")


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
