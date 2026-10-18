"""Pruebas unitarias para el módulo src.data.ingestion.multicite."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.ingestion.multicite import (
    MultiCiteDownloader,
    MultiCiteIngestionConfig,
)


def test_config_defaults():
    """Valida los valores por defecto de la configuración."""
    config = MultiCiteIngestionConfig()
    assert config.subset == "classification_3_context"
    assert config.output_dir == Path("data/raw/multicite")
    assert config.include_full is False
    assert config.force is False
    assert config.max_retries == 3


def test_download_file_success(tmp_path):
    """Valida la descarga exitosa de un archivo sin reintentos."""
    downloader = MultiCiteDownloader(MultiCiteIngestionConfig(output_dir=tmp_path))
    dest_path = tmp_path / "train.json"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'[{"id": "1", "x": "text", "y": "background"}]'
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response) as mock_get:
        result_path = downloader.download_file("https://example.com/train.json", dest_path)
        assert result_path == dest_path
        assert dest_path.exists()
        assert dest_path.read_text(encoding="utf-8") == '[{"id": "1", "x": "text", "y": "background"}]'
        mock_get.assert_called_once()


def test_download_file_retry_on_network_error(tmp_path):
    """Valida que el downloader reintente ante fallos transitorios de red."""
    config = MultiCiteIngestionConfig(output_dir=tmp_path, max_retries=3, backoff_factor=0.01)
    downloader = MultiCiteDownloader(config)
    dest_path = tmp_path / "dev.json"

    mock_fail = requests.RequestException("Network connection reset")
    mock_success = MagicMock()
    mock_success.status_code = 200
    mock_success.content = b'[{"id": "2", "x": "citation context", "y": "uses"}]'
    mock_success.raise_for_status = MagicMock()

    with patch("requests.get", side_effect=[mock_fail, mock_fail, mock_success]) as mock_get:
        result_path = downloader.download_file("https://example.com/dev.json", dest_path)
        assert result_path.exists()
        assert mock_get.call_count == 3


def test_download_file_404_raises_value_error(tmp_path):
    """Valida que un error HTTP 404 levante ValueError inmediatamente."""
    downloader = MultiCiteDownloader(MultiCiteIngestionConfig(output_dir=tmp_path))
    dest_path = tmp_path / "invalid.json"

    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="Recurso no encontrado en GitHub"):
            downloader.download_file("https://example.com/invalid.json", dest_path)


def test_validate_split_file_valid(tmp_path):
    """Valida que un split válido retorne el conteo exacto de registros."""
    file_path = tmp_path / "train.json"
    data = [
        {"id": "p1_0", "x": "Context 1", "y": "background"},
        {"id": "p1_1", "x": "Context 2", "y": "uses"},
    ]
    file_path.write_text(json.dumps(data), encoding="utf-8")

    count = MultiCiteDownloader.validate_split_file(file_path)
    assert count == 2


def test_validate_split_file_invalid_schema(tmp_path):
    """Valida que un split con esquema incorrecto lance ValueError."""
    file_path = tmp_path / "corrupt.json"
    data = [{"id": "p1_0", "missing_x": "Context 1"}]
    file_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="faltan llaves"):
        MultiCiteDownloader.validate_split_file(file_path)


def test_validate_full_file(tmp_path):
    """Valida la verificación de estructura del archivo completo full-v20210918.json."""
    file_path = tmp_path / "full-v20210918.json"
    data = {
        "paper_1": {"x": [], "y": {}},
        "paper_2": {"x": [], "y": {}},
    }
    file_path.write_text(json.dumps(data), encoding="utf-8")

    count = MultiCiteDownloader.validate_full_file(file_path)
    assert count == 2


def test_run_orchestration(tmp_path):
    """Valida la ejecución del pipeline completo de descarga y conteo."""
    config = MultiCiteIngestionConfig(
        output_dir=tmp_path,
        include_full=False,
    )
    downloader = MultiCiteDownloader(config)

    # Simular descarga escribiendo archivos JSON sintéticos válidos
    def fake_download(url: str, dest_path: Path):
        content = json.dumps([{"id": f"{dest_path.name}_0", "x": "ctx", "y": "background"}])
        dest_path.write_text(content, encoding="utf-8")
        return dest_path

    downloader.download_file = MagicMock(side_effect=fake_download)

    results = downloader.run()

    # Verificar llamadas a los 3 splits estándar
    assert downloader.download_file.call_count == 3
    assert set(results.keys()) == {"train.json", "dev.json", "test.json"}
    assert results["train.json"] == 1
    assert results["dev.json"] == 1
    assert results["test.json"] == 1


def test_run_idempotency_skip_existing(tmp_path):
    """Valida que si los archivos ya existen no se vuelvan a descargar."""
    config = MultiCiteIngestionConfig(output_dir=tmp_path, force=False)
    downloader = MultiCiteDownloader(config)

    # Pre-crear los 3 archivos
    for split in ["train.json", "dev.json", "test.json"]:
        p = tmp_path / split
        p.write_text(json.dumps([{"id": "1", "x": "c", "y": "b"}]), encoding="utf-8")

    downloader.download_file = MagicMock()
    results = downloader.run()

    # download_file no debe ser invocado
    downloader.download_file.assert_not_called()
    assert len(results) == 3
