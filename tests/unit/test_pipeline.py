"""Pruebas unitarias para el módulo del pipeline orquestador de datos (src/data/pipeline.py)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.pipeline import (
    DataPipeline,
    PipelineConfig,
    STAGE_NAMES,
    StageResult,
)


@pytest.fixture
def dummy_pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Crea una configuración con rutas temporales para pruebas aisladas."""
    return PipelineConfig(
        raw_unarxive_dir=tmp_path / "raw" / "unarxive",
        raw_multicite_dir=tmp_path / "raw" / "multicite",
        s2_cache_path=tmp_path / "raw" / "s2_cache.db",
        structured_papers_path=tmp_path / "structured" / "papers.jsonl",
        citing_contexts_path=tmp_path / "processed" / "citing_contexts.jsonl",
        cited_chunks_path=tmp_path / "processed" / "cited_chunks.jsonl",
        dataset_dir=tmp_path / "dataset",
        seed=42,
        force=False,
    )


def test_pipeline_config_defaults():
    """Valida los valores por defecto canónicos de PipelineConfig."""
    cfg = PipelineConfig()
    assert cfg.seed == 42
    assert not cfg.force
    assert not cfg.skip_seminal
    assert cfg.raw_unarxive_dir == Path("data/raw/unarxive")
    assert cfg.dataset_dir == Path("data/dataset")


def test_pipeline_config_from_yaml(tmp_path: Path):
    """Verifica la carga de configuración desde un archivo YAML."""
    yaml_content = """
raw_unarxive_dir: "custom/raw/unarxive"
seed: 999
force: true
skip_seminal: true
"""
    cfg_file = tmp_path / "test_pipeline.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    loaded_cfg = PipelineConfig.from_yaml(cfg_file)
    assert loaded_cfg.raw_unarxive_dir == Path("custom/raw/unarxive")
    assert loaded_cfg.seed == 999
    assert loaded_cfg.force is True
    assert loaded_cfg.skip_seminal is True


def test_should_skip_stage_when_artifact_exists(dummy_pipeline_config: PipelineConfig):
    """Verifica que should_skip_stage detecte artefactos existentes (RF-03)."""
    pipeline = DataPipeline(dummy_pipeline_config)

    # 1. Cuando no existe, no debe saltar
    skip, reason = pipeline.should_skip_stage("structuring")
    assert not skip

    # 2. Cuando existe con contenido, debe saltar
    dummy_pipeline_config.structured_papers_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_pipeline_config.structured_papers_path.write_text('{"id": "test"}\n', encoding="utf-8")
    skip, reason = pipeline.should_skip_stage("structuring")
    assert skip
    assert "Artefacto existente" in reason

    # 3. Cuando force es True, nunca debe saltar
    dummy_pipeline_config.force = True
    skip, reason = pipeline.should_skip_stage("structuring")
    assert not skip
    assert "forzada" in reason


def test_validate_preconditions(dummy_pipeline_config: PipelineConfig):
    """Valida la verificación fail-fast de insumos requeridos (RF-02)."""
    pipeline = DataPipeline(dummy_pipeline_config)

    # Inicialmente los directorios crudos no existen en tmp_path
    errors = pipeline.validate_preconditions(["structuring", "mining"])
    assert len(errors) == 2
    assert "unarXive" in errors[0]
    assert "MultiCite" in errors[1]

    # Crear directorios requeridos
    dummy_pipeline_config.raw_unarxive_dir.mkdir(parents=True, exist_ok=True)
    dummy_pipeline_config.raw_multicite_dir.mkdir(parents=True, exist_ok=True)

    errors = pipeline.validate_preconditions(["structuring", "mining"])
    assert len(errors) == 0


def test_invalid_stage_selection(dummy_pipeline_config: PipelineConfig):
    """Verifica que seleccionar una etapa inválida arroje ValueError."""
    pipeline = DataPipeline(dummy_pipeline_config)
    with pytest.raises(ValueError, match="Etapa inválida"):
        pipeline.execute(single_stage="etapa_inexistente")

    with pytest.raises(ValueError, match="Etapa inválida"):
        pipeline.execute(from_stage="etapa_falsa")


def test_dry_run_execution(dummy_pipeline_config: PipelineConfig):
    """Verifica que el modo dry-run reporte el plan sin ejecutar llamadas reales (RF-05)."""
    pipeline = DataPipeline(dummy_pipeline_config)
    results = pipeline.execute(dry_run=True)

    assert len(results) == 5
    for r in results:
        assert r.status.startswith("DRY-RUN:")
        assert r.duration_seconds == 0.0


@patch.object(DataPipeline, "run_stage_structuring")
@patch.object(DataPipeline, "run_stage_seminal")
@patch.object(DataPipeline, "run_stage_mining")
@patch.object(DataPipeline, "run_stage_chunking")
@patch.object(DataPipeline, "run_stage_builder")
def test_full_pipeline_mocked_execution(
    mock_builder,
    mock_chunking,
    mock_mining,
    mock_seminal,
    mock_structuring,
    dummy_pipeline_config: PipelineConfig,
):
    """Verifica la ejecución secuencial completa de las 5 etapas (RF-01)."""
    mock_structuring.return_value = {"structured_papers": 100}
    mock_seminal.return_value = {"catalog_size": 146, "resolved_s2": 146}
    mock_mining.return_value = {"total_records": 500}
    mock_chunking.return_value = {"total_chunks": 1200}
    mock_builder.return_value = {"total_instances": 3232}

    dummy_pipeline_config.force = True  # Forzar para que ninguna salte
    pipeline = DataPipeline(dummy_pipeline_config)

    # Evitar validación de preconditions para la prueba mockeada
    with patch.object(pipeline, "validate_preconditions", return_value=[]):
        results = pipeline.execute()

    assert len(results) == 5
    assert all(r.status == "COMPLETED" for r in results)
    assert mock_structuring.called
    assert mock_seminal.called
    assert mock_mining.called
    assert mock_chunking.called
    assert mock_builder.called


@patch.object(DataPipeline, "run_stage_mining")
def test_pipeline_failure_stops_execution(mock_mining, dummy_pipeline_config: PipelineConfig):
    """Verifica que un error en una etapa detenga la ejecución y reporte FAILED (RF-07)."""
    mock_mining.side_effect = RuntimeError("Error simulado de red o base de datos")

    dummy_pipeline_config.force = True
    pipeline = DataPipeline(dummy_pipeline_config)

    with patch.object(pipeline, "validate_preconditions", return_value=[]):
        with patch.object(pipeline, "run_stage_structuring", return_value={}):
            with patch.object(pipeline, "run_stage_seminal", return_value={}):
                with patch.object(pipeline, "run_stage_chunking") as mock_chunking:
                    results = pipeline.execute()

    # structuring: COMPLETED, seminal: COMPLETED, mining: FAILED, chunking: NOT CALLED
    assert len(results) == 3
    assert results[0].status == "COMPLETED"
    assert results[1].status == "COMPLETED"
    assert results[2].status == "FAILED"
    assert "Error simulado" in results[2].error
    assert not mock_chunking.called
