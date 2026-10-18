"""Pruebas unitarias para el módulo src.data.ingestion.unarxive."""

import io
import json
from pathlib import Path
import tarfile
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.ingestion.unarxive import (
    UnarxiveDownloader,
    UnarxiveIngestionConfig,
)


def test_config_defaults():
    """Valida los valores por defecto de la configuración."""
    config = UnarxiveIngestionConfig()
    assert config.mode == "sample"
    assert config.years == ["2022", "2021"]
    assert config.output_dir == Path("data/raw/unarxive")
    assert config.keep_archive is False
    assert config.force is False
    assert config.max_retries == 3


def test_get_download_url():
    """Valida la resolución de URLs según el modo configurado."""
    cfg_sample = UnarxiveIngestionConfig(mode="sample")
    assert "unarXive_data_sample" in UnarxiveDownloader(cfg_sample).get_download_url()

    cfg_zenodo = UnarxiveIngestionConfig(mode="zenodo")
    assert "7752615" in UnarxiveDownloader(cfg_zenodo).get_download_url()

    cfg_custom = UnarxiveIngestionConfig(mode="custom", custom_url="https://example.com/shard.tar.gz")
    assert UnarxiveDownloader(cfg_custom).get_download_url() == "https://example.com/shard.tar.gz"

    with pytest.raises(ValueError, match="custom_url es None"):
        UnarxiveDownloader(UnarxiveIngestionConfig(mode="custom")).get_download_url()


def test_download_file_success(tmp_path):
    """Valida la descarga streaming exitosa."""
    downloader = UnarxiveDownloader(UnarxiveIngestionConfig(output_dir=tmp_path))
    dest_path = tmp_path / "test.tar.gz"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b"chunk1_", b"chunk2"]
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response) as mock_get:
        result_path = downloader.download_file("https://example.com/test.tar.gz", dest_path)
        assert result_path == dest_path
        assert dest_path.exists()
        assert dest_path.read_bytes() == b"chunk1_chunk2"
        mock_get.assert_called_once()


def test_download_file_retry_on_network_error(tmp_path):
    """Valida reintentos con backoff ante fallas transitorias de red."""
    config = UnarxiveIngestionConfig(output_dir=tmp_path, max_retries=3, backoff_factor=0.01)
    downloader = UnarxiveDownloader(config)
    dest_path = tmp_path / "retry.tar.gz"

    mock_fail = requests.RequestException("Connection error")
    mock_success = MagicMock()
    mock_success.status_code = 200
    mock_success.iter_content.return_value = [b"data"]
    mock_success.raise_for_status = MagicMock()

    with patch("requests.get", side_effect=[mock_fail, mock_fail, mock_success]) as mock_get:
        result_path = downloader.download_file("https://example.com/retry.tar.gz", dest_path)
        assert result_path.exists()
        assert mock_get.call_count == 3


def test_download_file_404_raises_value_error(tmp_path):
    """Valida que un error HTTP 404 levante ValueError de inmediato."""
    downloader = UnarxiveDownloader(UnarxiveIngestionConfig(output_dir=tmp_path))
    dest_path = tmp_path / "missing.tar.gz"

    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="Recurso no encontrado"):
            downloader.download_file("https://example.com/missing.tar.gz", dest_path)


def test_extract_archive_selective_years(tmp_path):
    """Valida que la descompresión extraiga únicamente los shards de los años solicitados."""
    downloader = UnarxiveDownloader(UnarxiveIngestionConfig(output_dir=tmp_path))
    archive_path = tmp_path / "mock_unarxive.tar.gz"

    # Crear un .tar.gz sintético en memoria con 3 archivos
    with tarfile.open(archive_path, "w:gz") as tar:
        # Shard de 2022
        content_22 = b'{"paper_id": "2212.0001", "body_text": []}\n'
        ti_22 = tarfile.TarInfo(name="22/arXiv_src_2212_001.jsonl")
        ti_22.size = len(content_22)
        tar.addfile(ti_22, io.BytesIO(content_22))

        # Shard de 2019
        content_19 = b'{"paper_id": "1905.0001", "body_text": []}\n'
        ti_19 = tarfile.TarInfo(name="19/arXiv_src_1905_001.jsonl")
        ti_19.size = len(content_19)
        tar.addfile(ti_19, io.BytesIO(content_19))

        # Archivo que no es JSONL
        content_readme = b"README content"
        ti_readme = tarfile.TarInfo(name="README.txt")
        ti_readme.size = len(content_readme)
        tar.addfile(ti_readme, io.BytesIO(content_readme))

    dest_dir = tmp_path / "extracted"

    # Extraer únicamente el año 2022
    extracted = downloader.extract_archive(archive_path, dest_dir, years=["2022"])

    assert len(extracted) == 1
    assert extracted[0].name == "arXiv_src_2212_001.jsonl"
    assert (dest_dir / "arXiv_src_2212_001.jsonl").exists()
    assert not (dest_dir / "arXiv_src_1905_001.jsonl").exists()
    assert not (dest_dir / "README.txt").exists()


def test_validate_shard_valid(tmp_path):
    """Valida la verificación exitosa de un shard JSONL sintético."""
    shard_path = tmp_path / "test_shard.jsonl"
    lines = [
        json.dumps({"paper_id": "2212.0001", "body_text": [{"section": "Intro"}]}),
        json.dumps({"paper_id": "2212.0002", "body_text": [{"section": "Methods"}]}),
    ]
    shard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    count = UnarxiveDownloader.validate_shard(shard_path)
    assert count == 2


def test_validate_shard_invalid_missing_key(tmp_path):
    """Valida que falle si faltan llaves estructurales en el JSONL."""
    shard_path = tmp_path / "bad_shard.jsonl"
    lines = [
        json.dumps({"title_only": "Test Title"}),
    ]
    shard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="faltan llaves"):
        UnarxiveDownloader.validate_shard(shard_path)


def test_run_idempotency_skip_existing(tmp_path):
    """Valida que si los shards correspondientes ya existen, no se vuelva a descargar."""
    config = UnarxiveIngestionConfig(
        output_dir=tmp_path,
        mode="sample",
        force=False,
    )
    downloader = UnarxiveDownloader(config)

    # Pre-crear shard de muestra en disco
    sample_shard = tmp_path / "arXiv_src_2212_086.jsonl"
    sample_shard.write_text(
        json.dumps({"paper_id": "2212.0001", "body_text": []}) + "\n",
        encoding="utf-8",
    )

    downloader.download_file = MagicMock()
    results = downloader.run()

    downloader.download_file.assert_not_called()
    assert "arXiv_src_2212_086.jsonl" in results
    assert results["arXiv_src_2212_086.jsonl"] == 1
