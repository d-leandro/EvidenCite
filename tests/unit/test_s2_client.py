"""Pruebas unitarias para el cliente de Semantic Scholar (src.data.ingestion.s2_client)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.ingestion.s2_client import (
    S2ClientConfig,
    S2PaperMetadata,
    SemanticScholarClient,
)


def test_config_api_key_from_env(monkeypatch, tmp_path):
    """Valida que la clave de API se tome del entorno si no se provee explícitamente."""
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-api-key-123")
    config = S2ClientConfig(cache_path=tmp_path / "cache.db")
    client = SemanticScholarClient(config)
    assert client.api_key == "test-api-key-123"
    assert client.rate_limit_delay == 10.0


def test_cache_hit_avoids_http_call(tmp_path):
    """Valida que si el artículo ya está en la caché SQLite, no se ejecute ninguna llamada de red."""
    cache_path = tmp_path / "cache.db"
    config = S2ClientConfig(cache_path=cache_path)
    client = SemanticScholarClient(config)

    # Pre-guardar en la caché
    sample_paper = S2PaperMetadata(
        paper_id="s2_123",
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models...",
        arxiv_id="1706.03762",
    )
    client._save_to_cache("search:attention is all you need", sample_paper)

    with patch("requests.get") as mock_get:
        result = client.search_paper_by_title("Attention Is All You Need")
        assert result is not None
        assert result.arxiv_id == "1706.03762"
        mock_get.assert_not_called()


def test_search_paper_by_title_success(tmp_path):
    """Valida búsqueda textual exitosa con parseo de abstract y arXiv ID."""
    config = S2ClientConfig(cache_path=tmp_path / "cache.db", rate_limit_delay=0.0)
    client = SemanticScholarClient(config)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "total": 1,
        "data": [
            {
                "paperId": "abc1234567890",
                "title": "vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations",
                "abstract": "We propose vq-wav2vec to learn discrete representations of audio...",
                "year": 2019,
                "externalIds": {
                    "ArXiv": "1910.05453",
                    "DOI": "10.48550/arXiv.1910.05453",
                },
                "authors": [{"name": "Alexei Baevski"}, {"name": "Michael Auli"}],
            }
        ],
    }

    with patch("requests.get", return_value=mock_response) as mock_get:
        paper = client.search_paper_by_title("vq-wav2vec")
        assert paper is not None
        assert paper.paper_id == "abc1234567890"
        assert paper.arxiv_id == "1910.05453"
        assert paper.abstract.startswith("We propose vq-wav2vec")
        assert paper.authors == ["Alexei Baevski", "Michael Auli"]
        mock_get.assert_called_once()

    # Segunda llamada debe ser cache hit
    with patch("requests.get") as mock_get_2:
        cached_paper = client.search_paper_by_title("vq-wav2vec")
        assert cached_paper.arxiv_id == "1910.05453"
        mock_get_2.assert_not_called()


def test_search_paper_not_found(tmp_path):
    """Valida el manejo de búsquedas que no arrojan resultados."""
    config = S2ClientConfig(cache_path=tmp_path / "cache.db", rate_limit_delay=0.0)
    client = SemanticScholarClient(config)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"total": 0, "data": []}

    with patch("requests.get", return_value=mock_response):
        paper = client.search_paper_by_title("Non Existent Paper 99999999")
        assert paper is None

    # Debe haber cacheado la ausencia (tombstone)
    with patch("requests.get") as mock_get:
        paper_cached = client.search_paper_by_title("Non Existent Paper 99999999")
        assert paper_cached is None
        mock_get.assert_not_called()


def test_get_paper_by_id_success(tmp_path):
    """Valida consulta directa por identificador canónico."""
    config = S2ClientConfig(cache_path=tmp_path / "cache.db", rate_limit_delay=0.0)
    client = SemanticScholarClient(config)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "paperId": "sha_xyz",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce a new language representation model called BERT...",
        "year": 2018,
        "externalIds": {"ArXiv": "1810.04805"},
    }

    with patch("requests.get", return_value=mock_response):
        paper = client.get_paper_by_id("ARXIV:1810.04805")
        assert paper is not None
        assert paper.arxiv_id == "1810.04805"
        assert paper.title.startswith("BERT:")


def test_rate_limit_retry_after(tmp_path):
    """Valida que ante un error HTTP 429 con Retry-After, el cliente reintente y tenga éxito."""
    config = S2ClientConfig(cache_path=tmp_path / "cache.db", rate_limit_delay=0.0, max_retries=2)
    client = SemanticScholarClient(config)

    mock_429 = MagicMock()
    mock_429.status_code = 429
    mock_429.headers = {"Retry-After": "0.01"}

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "total": 1,
        "data": [{"paperId": "p1", "title": "Test Title", "externalIds": {}}],
    }

    with patch("requests.get", side_effect=[mock_429, mock_200]) as mock_get:
        paper = client.search_paper_by_title("Test Title")
        assert paper is not None
        assert paper.title == "Test Title"
        assert mock_get.call_count == 2


def test_unauthorized_raises_value_error(tmp_path):
    """Valida que si la API retorna HTTP 403 o 401 levante ValueError."""
    config = S2ClientConfig(cache_path=tmp_path / "cache.db", rate_limit_delay=0.0)
    client = SemanticScholarClient(config)

    mock_403 = MagicMock()
    mock_403.status_code = 403

    with patch("requests.get", return_value=mock_403):
        with pytest.raises(ValueError, match="Autenticación fallida"):
            client.search_paper_by_title("Some Paper")


def test_resolve_arxiv_id_helper(tmp_path):
    """Valida la función abreviada resolve_arxiv_id."""
    config = S2ClientConfig(cache_path=tmp_path / "cache.db", rate_limit_delay=0.0)
    client = SemanticScholarClient(config)

    with patch.object(
        client,
        "search_paper_by_title",
        return_value=S2PaperMetadata(paper_id="1", title="T", arxiv_id="2105.05862"),
    ):
        arxiv_id = client.resolve_arxiv_id("Some Title")
        assert arxiv_id == "2105.05862"

    with patch.object(
        client,
        "search_paper_by_title",
        return_value=S2PaperMetadata(paper_id="2", title="T2", arxiv_id=None),
    ):
        arxiv_id_none = client.resolve_arxiv_id("Title Without Arxiv")
        assert arxiv_id_none is None
