"""Pruebas unitarias para validación declarativa de esquemas y archivos YAML en configs/ (src.config.schema)."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config.schema import (
    CANONICAL_9_CLASSES,
    ID_TO_LABEL,
    LABEL_TO_ID,
    ExperimentConfig,
    InputRepresentationConfig,
    TfidfModelConfig,
    load_base_config,
    load_experiment_config,
    resolve_canonical_label,
    resolve_device,
)


def test_canonical_9_classes_mapping() -> None:
    """Verifica que las 9 clases canónicas de ADR-10 estén completas y mapeadas bívocamente."""
    assert len(CANONICAL_9_CLASSES) == 9
    assert len(LABEL_TO_ID) == 9
    assert len(ID_TO_LABEL) == 9
    for idx, label in enumerate(CANONICAL_9_CLASSES):
        assert LABEL_TO_ID[label] == idx
        assert ID_TO_LABEL[idx] == label

    # Resolución de etiquetas canónicas y heredadas
    assert resolve_canonical_label("Background") == "Background"
    assert resolve_canonical_label("Further Reading") == "Further Reading"
    assert resolve_canonical_label("Uses") == "Application"
    assert resolve_canonical_label("Extends") == "Improvement"
    assert resolve_canonical_label("Motivation") == "Gap"
    assert resolve_canonical_label("Future Work") == "Gap"
    assert resolve_canonical_label(None) is None
    assert resolve_canonical_label("NonExistentLabel") is None


def test_load_all_declarative_yaml_configs() -> None:
    """Verifica que todos los archivos YAML en configs/ carguen y validen sin errores."""
    base_cfg = load_base_config(Path("configs/base.yaml"))
    assert base_cfg.seed == 42

    yaml_files = [
        ("configs/baseline_tfidf.yaml", "tfidf", "A"),
        ("configs/model_config_a.yaml", "encoder", "A"),
        ("configs/model_config_b.yaml", "encoder", "B"),
        ("configs/llm_lora.yaml", "llm_lora", "B"),
        ("configs/llm_api.yaml", "llm_api", "B"),
    ]
    for rel_path, expected_family, expected_rep in yaml_files:
        cfg = load_experiment_config(Path(rel_path), Path("configs/base.yaml"))
        assert cfg.model_family == expected_family
        assert cfg.input_representation.configuration == expected_rep
        assert cfg.input_representation.max_seq_length == 512
        assert cfg.input_representation.enforce_per_chunk_budget is True


def test_invalid_configs_raise_validation_error() -> None:
    """Valida que Pydantic rechace configuraciones inválidas antes de ejecutar cómputo."""
    with pytest.raises(ValidationError):
        InputRepresentationConfig(configuration="C")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        TfidfModelConfig(ngram_min=3, ngram_max=1)

    with pytest.raises(ValidationError):
        ExperimentConfig(
            experiment_name="",
            model_family="encoder",
        )


def test_resolve_device() -> None:
    """Verifica resolución de dispositivo explícito y automático."""
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cuda", "mps", "cpu")

