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
from src.data.builder import BuilderConfig
from src.data.pipeline import PipelineConfig
from src.data.retrieval import RetrievalConfig


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
        # Verifica herencia estricta de base.yaml (Single Source of Truth)
        assert cfg.seed == base_cfg.seed == 42
        assert cfg.device == base_cfg.device == "auto"


def test_base_config_inheritance_and_override(tmp_path: Path) -> None:
    """Verifica que la configuración base se herede correctamente y admita overrides locales."""
    custom_base = tmp_path / "custom_base.yaml"
    custom_base.write_text("seed: 999\ndevice: 'cpu'\n", encoding="utf-8")

    # Sub-config sin seed ni device: debe heredar de custom_base
    cfg = load_experiment_config(Path("configs/model_config_a.yaml"), custom_base)
    assert cfg.seed == 999
    assert cfg.device == "cpu"

    # Sub-config con override explícito: debe prevalecer el valor específico
    exp_override = tmp_path / "exp_override.yaml"
    exp_override.write_text(
        "experiment_name: 'test_exp'\nmodel_family: 'tfidf'\nseed: 777\n",
        encoding="utf-8",
    )
    cfg_override = load_experiment_config(exp_override, custom_base)
    assert cfg_override.seed == 777
    assert cfg_override.device == "cpu"


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


def test_data_pipeline_configs_inherit_base(tmp_path: Path) -> None:
    """Verifica que los configs de datos hereden seed y device de configs/base.yaml (SSOT)."""
    # 1. Carga directa: heredan los defaults de configs/base.yaml (seed=42, device=auto)
    pipe_cfg = PipelineConfig.from_yaml(Path("configs/data_pipeline.yaml"))
    assert pipe_cfg.seed == 42

    builder_cfg = BuilderConfig.from_yaml(Path("configs/builder.yaml"))
    assert builder_cfg.seed == 42
    assert builder_cfg.nli_config is not None
    assert builder_cfg.nli_config.device == "auto"

    retrieval_cfg = RetrievalConfig.from_yaml(Path("configs/retrieval.yaml"))
    assert retrieval_cfg.seed == 42
    assert retrieval_cfg.device == "auto"

    # 2. Carga con base_path personalizado: propaga seed y device a todos los módulos
    custom_base = tmp_path / "custom_base.yaml"
    custom_base.write_text("seed: 1234\ndevice: 'cuda'\n", encoding="utf-8")

    pipe_custom = PipelineConfig.from_yaml(Path("configs/data_pipeline.yaml"), base_path=custom_base)
    assert pipe_custom.seed == 1234

    builder_custom = BuilderConfig.from_yaml(Path("configs/builder.yaml"), base_path=custom_base)
    assert builder_custom.seed == 1234
    assert builder_custom.nli_config is not None
    assert builder_custom.nli_config.device == "cuda"

    retrieval_custom = RetrievalConfig.from_yaml(Path("configs/retrieval.yaml"), base_path=custom_base)
    assert retrieval_custom.seed == 1234
    assert retrieval_custom.device == "cuda"

    # 3. Override local en YAML particular prevalece sobre base.yaml
    custom_pipe_yaml = tmp_path / "custom_pipe.yaml"
    custom_pipe_yaml.write_text("seed: 888\n", encoding="utf-8")
    pipe_override = PipelineConfig.from_yaml(custom_pipe_yaml, base_path=custom_base)
    assert pipe_override.seed == 888

