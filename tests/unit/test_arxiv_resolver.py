"""Pruebas unitarias para el resolutor y extractor de arXiv (src.data.ingestion.arxiv_resolver)."""

from pathlib import Path
import time
from unittest.mock import MagicMock, patch

import pytest

from src.data.ingestion.arxiv_resolver import (
    ArxivClient,
    ArxivMetadata,
    ArxivPaperResolver,
    ArxivResolverConfig,
    DoclingPDFParser,
    ExtractedRawSection,
    is_excluded_heading,
    normalize_arxiv_id,
)
from src.data.schemas import StructuredPaper


SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.</summary>
    <published>2017-06-12T17:57:34Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" type="application/pdf"/>
  </entry>
</feed>
"""


def test_normalize_arxiv_id():
    """Valida la normalización de identificadores con versiones y prefijos."""
    assert normalize_arxiv_id("2101.00098") == "2101.00098"
    assert normalize_arxiv_id("arxiv:2101.00098v2") == "2101.00098"
    assert normalize_arxiv_id("  arXiv:1706.03762v7  ") == "1706.03762"
    assert normalize_arxiv_id("cs/0101001v1") == "cs/0101001"


def test_is_excluded_heading():
    """Valida el filtrado de encabezados no discursivos."""
    assert is_excluded_heading("References") is True
    assert is_excluded_heading("10. Bibliography") is True
    assert is_excluded_heading("Acknowledgements and Funding") is False  # Funding solito sí
    assert is_excluded_heading("Funding") is True
    assert is_excluded_heading("Acknowledgments") is True
    assert is_excluded_heading("Author Contributions") is True

    # Encabezados válidos
    assert is_excluded_heading("1. Introduction") is False
    assert is_excluded_heading("Related Work") is False
    assert is_excluded_heading("3. Methodology") is False
    assert is_excluded_heading("Experiments and Results") is False


def test_arxiv_client_rate_limiting(tmp_path):
    """Valida que ArxivClient aplique una pausa mínima entre peticiones."""
    cfg = ArxivResolverConfig(cache_dir=tmp_path, rate_limit_delay=0.2)
    client = ArxivClient(cfg)

    client._last_request_time = time.time()
    t0 = time.time()
    client._enforce_rate_limit()
    elapsed = time.time() - t0

    # Debe haber esperado aproximadamente ~0.2 segundos
    assert elapsed >= 0.15


def test_arxiv_client_fetch_metadata_success(tmp_path):
    """Valida la obtención y parseo de metadatos oficiales desde Atom XML."""
    cfg = ArxivResolverConfig(cache_dir=tmp_path, rate_limit_delay=0.0)
    client = ArxivClient(cfg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_ATOM_XML

    with patch("httpx.Client.get", return_value=mock_resp):
        meta = client.fetch_metadata("1706.03762")

    assert meta is not None
    assert meta.arxiv_id == "1706.03762"
    assert meta.title == "Attention Is All You Need"
    assert "dominant sequence transduction" in meta.abstract
    assert meta.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert "cs.CL" in meta.categories
    assert meta.pdf_url == "https://arxiv.org/pdf/1706.03762v7"


def test_arxiv_client_download_pdf_cache_hit(tmp_path):
    """Valida que si el PDF ya existe en la caché, no se realice llamada de red."""
    cfg = ArxivResolverConfig(cache_dir=tmp_path, rate_limit_delay=0.0)
    client = ArxivClient(cfg)

    pdf_file = tmp_path / "1706.03762.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")

    with patch("httpx.Client.get") as mock_get:
        path = client.download_pdf("1706.03762")
        assert path == pdf_file
        mock_get.assert_not_called()


def test_arxiv_client_download_pdf_network(tmp_path):
    """Valida la descarga de red cuando el archivo no existe en caché."""
    cfg = ArxivResolverConfig(cache_dir=tmp_path, rate_limit_delay=0.0)
    client = ArxivClient(cfg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"%PDF-1.5 valid scientific pdf content"

    with patch("httpx.Client.get", return_value=mock_resp):
        path = client.download_pdf("1706.03762")
        assert path is not None
        assert path.exists()
        assert path.read_bytes().startswith(b"%PDF-1.5")


def test_docling_parser_rejects_pages_exceeding_limit(tmp_path):
    """Valida el descarte total si el PDF excede el número máximo de páginas (sin truncamiento)."""
    parser = DoclingPDFParser(max_pages=20)
    fake_pdf = tmp_path / "thesis.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content")

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 45  # 45 páginas > 20

    with patch("pypdfium2.PdfDocument", return_value=mock_doc):
        is_valid = parser.validate_pdf(fake_pdf)
        assert is_valid is False  # Descartado completamente


def test_docling_parser_accepts_valid_length(tmp_path):
    """Valida que un PDF dentro del límite de páginas sea aceptado."""
    parser = DoclingPDFParser(max_pages=20)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content")

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 12  # 12 páginas <= 20

    with patch("pypdfium2.PdfDocument", return_value=mock_doc):
        is_valid = parser.validate_pdf(fake_pdf)
        assert is_valid is True


def test_resolver_tier1_ar5iv_success_avoids_pdf_download(tmp_path):
    """Valida que si ar5iv (Tier 1) entrega secciones HTML, no se descargue el PDF ni se use Docling."""
    cfg = ArxivResolverConfig(cache_dir=tmp_path, rate_limit_delay=0.0)
    resolver = ArxivPaperResolver(cfg)

    sample_meta = ArxivMetadata(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        abstract="Transformers abstract.",
        authors=["Ashish Vaswani"],
        categories=["cs.CL"],
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
    )

    ar5iv_sections = [
        ExtractedRawSection(heading="1. Introduction", raw_text="Recurrent models typically factor computation. This is sentence two."),
        ExtractedRawSection(heading="2. Background", raw_text="The goal of reducing sequential computation also forms the foundation."),
    ]

    with patch.object(resolver.client, "fetch_metadata", return_value=sample_meta), \
         patch.object(resolver, "_try_extract_ar5iv", return_value=ar5iv_sections), \
         patch.object(resolver.client, "download_pdf") as mock_dl:

        paper = resolver.resolve_paper("1706.03762")

        assert paper is not None
        assert isinstance(paper, StructuredPaper)
        assert paper.paper_id == "1706.03762"
        assert len(paper.sections) == 2
        assert paper.sections[0].heading == "1. Introduction"
        # Comprobar que no descargó PDF
        mock_dl.assert_not_called()


def test_resolver_tier2_docling_fallback_when_ar5iv_fails(tmp_path):
    """Valida que si ar5iv falla (Tier 1 vacío), se active la descarga de PDF y Docling (Tier 2)."""
    cfg = ArxivResolverConfig(cache_dir=tmp_path, rate_limit_delay=0.0)
    resolver = ArxivPaperResolver(cfg)

    sample_meta = ArxivMetadata(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        abstract="Transformers abstract.",
        authors=["Ashish Vaswani"],
        categories=["cs.CL"],
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
    )

    fake_pdf = tmp_path / "1706.03762.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy")

    docling_sections = [
        ExtractedRawSection(heading="1 Introduction", raw_text="Extracted via Docling parser. Another long sentence for test."),
        ExtractedRawSection(heading="References", raw_text="Vaswani et al. 2017."),  # Debe ser descartada
    ]

    with patch.object(resolver.client, "fetch_metadata", return_value=sample_meta), \
         patch.object(resolver, "_try_extract_ar5iv", return_value=[]), \
         patch.object(resolver.client, "download_pdf", return_value=fake_pdf) as mock_dl, \
         patch.object(resolver.docling_parser, "extract_sections", return_value=docling_sections):

        paper = resolver.resolve_paper("1706.03762")

        assert paper is not None
        assert mock_dl.called
        # Debe haber filtrado 'References'
        assert len(paper.sections) == 1
        assert paper.sections[0].heading == "1 Introduction"
        assert len(paper.sections[0].sentences) >= 1


def test_extract_missing_cited_ids(tmp_path):
    """Valida la auto-detección de arXiv IDs citados que tienen has_cited_chunks=False."""
    import json
    from src.data.ingestion.arxiv_resolver import extract_missing_cited_ids

    contexts_file = tmp_path / "citing_contexts.jsonl"
    papers_file = tmp_path / "papers.jsonl"

    # 1. Simular papers.jsonl con 1 paper ya estructurado (1706.03762)
    with open(papers_file, "w", encoding="utf-8") as f_p:
        f_p.write(json.dumps({"paper_id": "1706.03762", "title": "Attention"}) + "\n")

    # 2. Simular citing_contexts.jsonl con 3 citas:
    # - Cita A: target_cited_id="1706.03762" pero ya existe en papers.jsonl -> se excluye
    # - Cita B: target_cited_id="2101.00098", has_cited_chunks=False -> SE DEBE DETECTAR
    # - Cita C: target_cited_id="1810.04805v2", has_cited_chunks=False -> SE DEBE DETECTAR NORMALIZADO
    # - Cita D: target_cited_id=None -> se ignora
    # - Cita E: target_cited_id="2101.00098", duplicada en otro contexto -> no duplicar en lista
    records = [
        {"context_id": "c1", "target_cited_id": "1706.03762", "has_cited_chunks": True},
        {"context_id": "c2", "target_cited_id": "2101.00098", "has_cited_chunks": False},
        {"context_id": "c3", "target_cited_id": "1810.04805v2", "has_cited_chunks": False},
        {"context_id": "c4", "target_cited_id": None, "has_cited_chunks": False},
        {"context_id": "c5", "target_cited_id": "2101.00098", "has_cited_chunks": False},
    ]
    with open(contexts_file, "w", encoding="utf-8") as f_c:
        for r in records:
            f_c.write(json.dumps(r) + "\n")

    missing = extract_missing_cited_ids(contexts_file, papers_file)

    # Debe retornar exactamente los 2 IDs faltantes normalizados y ordenados
    assert missing == ["1810.04805", "2101.00098"]


def test_try_extract_ar5iv_nested_sections(tmp_path):
    """Valida que ar5iv extraiga subsecciones respetando el principio de Nearest Enclosing Scope."""
    sample_html = """
    <html>
      <body>
        <section class="ltx_section">
          <h2 class="ltx_title"><span class="ltx_tag">1</span> Introduction</h2>
          <p class="ltx_p">Introductory text before subsections.</p>
          <section class="ltx_subsection">
            <h3 class="ltx_title"><span class="ltx_tag">1.1</span> Deep Learning</h3>
            <p class="ltx_p">Paragraph under deep learning subsection.</p>
          </section>
          <section class="ltx_subsection">
            <h3 class="ltx_title"><span class="ltx_tag">1.2</span> Autoencoders</h3>
            <p class="ltx_p">Paragraph under autoencoders subsection.</p>
          </section>
        </section>
      </body>
    </html>
    """
    cfg = ArxivResolverConfig(cache_dir=tmp_path, rate_limit_delay=0.0)
    resolver = ArxivPaperResolver(cfg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = sample_html

    with patch("httpx.Client.get", return_value=mock_resp):
        sections = resolver._try_extract_ar5iv("1206.5533")

    assert len(sections) == 3
    assert sections[0].heading == "1 Introduction"
    assert "Introductory text" in sections[0].raw_text
    assert "Paragraph under deep learning" not in sections[0].raw_text

    assert sections[1].heading == "1.1 Deep Learning"
    assert "Paragraph under deep learning" in sections[1].raw_text

    assert sections[2].heading == "1.2 Autoencoders"
    assert "Paragraph under autoencoders" in sections[2].raw_text

