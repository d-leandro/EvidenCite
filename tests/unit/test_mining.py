"""Pruebas unitarias para el módulo de minado y unificación de contextos citantes (src.data.mining)."""

import json
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from src.data.ingestion.s2_client import S2ClientConfig, S2PaperMetadata, SemanticScholarClient
from src.data.mining import (
    MiningConfig,
    MiningPipeline,
    MultiCiteAdapter,
    UnarxiveMiner,
    extract_rhetorical_section_from_text,
    load_known_arxiv_ids,
    mask_secondary_citations,
    normalize_multicite_label,
    parse_multicite_surface,
)
from src.data.schemas import (
    BibliographyEntry,
    CitationMarker,
    CitingContext,
    StructuredPaper,
    StructuredSection,
    StructuredSentence,
)


def test_parse_multicite_surface():
    """Valida la extracción de autor y año a partir de marcadores textuales de citas."""
    assert parse_multicite_surface("(Baevski et al., 2019b)") == ("Baevski", 2019)
    assert parse_multicite_surface("Salehi et al. (2014b)") == ("Salehi", 2014)
    assert parse_multicite_surface("Rashkin et al., 2017") == ("Rashkin", 2017)
    assert parse_multicite_surface("[12]") == (None, None)
    assert parse_multicite_surface("it captures") == (None, None)


def test_extract_rhetorical_section_from_text():
    """Valida la detección de encabezados discursivos en mayúsculas estilo markdown."""
    text_with_header = "**CONCLUSIONS** We presented the first approach to using word embeddings."
    heading, clean_text = extract_rhetorical_section_from_text(text_with_header)
    assert heading == "Conclusions"
    assert clean_text == "We presented the first approach to using word embeddings."

    text_no_header = "Automatic detection of fake from legitimate news has been approached."
    heading, clean_text = extract_rhetorical_section_from_text(text_no_header)
    assert heading == "Unknown Section"
    assert clean_text == text_no_header


def test_mask_secondary_citations():
    """Valida el enmascaramiento con [OTHERCIT] preservando la cita objetivo."""
    text = (
        "Previous work <cite>Rashkin et al., 2017</cite> addressed fake news. "
        "Also related is <cite>Wang, 2017</cite> and another mention <cite>Rashkin et al., 2017</cite>."
    )
    masked = mask_secondary_citations(text, target_mention="Rashkin et al., 2017")
    assert "<cite>Rashkin et al., 2017</cite>" in masked
    assert "[OTHERCIT]" in masked


def test_normalize_multicite_label():
    """Valida la normalización de etiquetas de MultiCite al esquema de 9 clases."""
    assert normalize_multicite_label("differences") == "Differences"
    assert normalize_multicite_label("background") == "Background"
    assert normalize_multicite_label("similarities future_work") == "Similarities; Future Work"
    assert normalize_multicite_label("extends background") == "Extends; Background"


def test_multicite_adapter_s2_cache_hit(tmp_path):
    """Valida que MultiCiteAdapter resuelva metadatos desde s2_cache.db sin realizar peticiones de red."""
    cache_path = tmp_path / "s2_cache.db"
    s2_config = S2ClientConfig(cache_path=cache_path, rate_limit_delay=0.0)
    s2_client = SemanticScholarClient(s2_config)

    # Pre-cargar artículo en la caché SQLite
    s2_paper = S2PaperMetadata(
        paper_id="sha_baevski",
        title="vq-wav2vec: Self-Supervised Learning",
        abstract="We learn discrete speech representations...",
        year=2019,
        arxiv_id="1910.05453",
        authors=["Alexei Baevski"],
    )
    s2_client._save_to_cache("search:baevski 2019", s2_paper)

    # Crear partición de prueba en tmp_path
    mc_dir = tmp_path / "multicite"
    mc_dir.mkdir(parents=True)
    sample_data = [
        {
            "id": "02521fd9721c_0",
            "x": "Prior work uses tandem models <cite>(Baevski et al., 2019b)</cite> for speech.",
            "y": "differences",
        }
    ]
    (mc_dir / "train.json").write_text(json.dumps(sample_data), encoding="utf-8")

    known_arxiv_ids = {"1910.05453"}  # Simulamos que el paper citado existe en papers.jsonl
    adapter = MultiCiteAdapter(
        multicite_dir=mc_dir,
        s2_client=s2_client,
        known_arxiv_ids=known_arxiv_ids,
        skip_network=True,
    )

    with patch("requests.get") as mock_get:
        contexts = list(adapter.process_split("train"))
        assert len(contexts) == 1
        ctx = contexts[0]
        assert ctx.context_id == "mc_train_02521fd9721c_0"
        assert ctx.target_cited_id == "1910.05453"
        assert ctx.target_cited_title == "vq-wav2vec: Self-Supervised Learning"
        assert ctx.target_cited_authors == "Alexei Baevski"
        assert ctx.has_cited_chunks is True
        assert ctx.normalized_label == "Differences"
        # Comprobar que no hubo peticiones HTTP
        mock_get.assert_not_called()


def test_unarxive_miner_window_m3():
    """Valida la extracción de ventana estricta m=3 oraciones dentro de los límites de una sección."""
    miner = UnarxiveMiner(known_arxiv_ids={"1706.03762"})

    paper = StructuredPaper(
        paper_id="2101.00098",
        id_source="arxiv",
        title="Sample Paper",
        abstract="Abstract sample.",
        sections=[
            StructuredSection(
                section_idx=0,
                heading="Related Work",
                sentences=[
                    StructuredSentence(sent_idx=0, text="First sentence introduces context.", citations=[]),
                    StructuredSentence(
                        sent_idx=1,
                        text="We build upon the framework of {{cite:ref1}} for our model.",
                        citations=[
                            CitationMarker(
                                marker_text="{{cite:ref1}}",
                                char_start=33,
                                char_end=48,
                                ref_key="ref1",
                            )
                        ],
                    ),
                    StructuredSentence(sent_idx=2, text="Third sentence follows up.", citations=[]),
                    StructuredSentence(sent_idx=3, text="Fourth sentence is out of window.", citations=[]),
                ],
            )
        ],
        bibliography={
            "ref1": BibliographyEntry(
                title="Attention Is All You Need",
                authors="Vaswani et al.",
                year=2017,
                raw_text="Vaswani et al. 2017. Attention...",
                arxiv_id="1706.03762",
            )
        },
    )

    contexts = list(miner.mine_paper(paper))
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx.source_dataset == "unarxive_mined"
    assert ctx.normalized_label == "Basis"
    assert ctx.target_cited_id == "1706.03762"
    assert ctx.has_cited_chunks is True
    assert ctx.rhetorical_section == "Related Work"
    assert "First sentence introduces context." in ctx.citation_context
    assert "<cite>Vaswani et al., 2017</cite>" in ctx.citation_context
    assert "Third sentence follows up." in ctx.citation_context
    assert "Fourth sentence is out of window." not in ctx.citation_context
    assert ctx.label_source == "heuristic_rule"
    assert ctx.is_gold is False


def test_unarxive_miner_window_boundary_start():
    """RF-01: Valida que si la cita ocurre al inicio de la sección (i=0), la ventana se desplace hacia adelante [s0, s1, s2]."""
    miner = UnarxiveMiner(known_arxiv_ids=set())

    paper = StructuredPaper(
        paper_id="2101.00001",
        id_source="arxiv",
        title="Sample",
        abstract="Abstract",
        sections=[
            StructuredSection(
                section_idx=0,
                heading="Introduction",
                sentences=[
                    StructuredSentence(
                        sent_idx=0,
                        text="We build upon the framework of {{cite:ref1}}.",
                        citations=[CitationMarker(marker_text="{{cite:ref1}}", char_start=0, char_end=35, ref_key="ref1")],
                    ),
                    StructuredSentence(sent_idx=1, text="Second sentence provides detail.", citations=[]),
                    StructuredSentence(sent_idx=2, text="Third sentence concludes setup.", citations=[]),
                    StructuredSentence(sent_idx=3, text="Fourth sentence is extra.", citations=[]),
                ],
            )
        ],
        bibliography={"ref1": BibliographyEntry(title="Ref Title", raw_text="Ref Title")},
    )

    contexts = list(miner.mine_paper(paper))
    assert len(contexts) == 1
    ctx = contexts[0]
    # Debe contener s0, s1, s2 y no s3
    assert "<cite>Ref Title</cite>" in ctx.citation_context
    assert "Second sentence provides detail." in ctx.citation_context
    assert "Third sentence concludes setup." in ctx.citation_context
    assert "Fourth sentence is extra." not in ctx.citation_context


def test_unarxive_miner_window_boundary_end():
    """RF-01: Valida que si la cita ocurre al final de la sección (i=n-1), la ventana se desplace hacia atrás [s_{n-3}, s_{n-2}, s_{n-1}]."""
    miner = UnarxiveMiner(known_arxiv_ids=set())

    paper = StructuredPaper(
        paper_id="2101.00002",
        id_source="arxiv",
        title="Sample",
        abstract="Abstract",
        sections=[
            StructuredSection(
                section_idx=0,
                heading="Discussion",
                sentences=[
                    StructuredSentence(sent_idx=0, text="Introductory sentence.", citations=[]),
                    StructuredSentence(sent_idx=1, text="Second background sentence.", citations=[]),
                    StructuredSentence(sent_idx=2, text="Penultimate discussion point.", citations=[]),
                    StructuredSentence(
                        sent_idx=3,
                        text="Our findings are consistent with {{cite:ref2}}.",
                        citations=[CitationMarker(marker_text="{{cite:ref2}}", char_start=0, char_end=40, ref_key="ref2")],
                    ),
                ],
            )
        ],
        bibliography={"ref2": BibliographyEntry(title="Prior Work", raw_text="Prior Work")},
    )

    contexts = list(miner.mine_paper(paper))
    assert len(contexts) == 1
    ctx = contexts[0]
    # Debe contener s1, s2, s3 y excluir s0
    assert "Introductory sentence." not in ctx.citation_context
    assert "Second background sentence." in ctx.citation_context
    assert "Penultimate discussion point." in ctx.citation_context
    assert "<cite>Prior Work</cite>" in ctx.citation_context



def test_unarxive_orphan_class_patterns():
    """Valida la detección léxica para las 4 clases huérfanas."""
    miner = UnarxiveMiner(known_arxiv_ids=set())

    assert miner.match_orphan_class("We build upon the framework of {{cite:1}}.") == "Basis"
    assert miner.match_orphan_class("Our findings are consistent with {{cite:2}}.") == "Evidence"
    assert miner.match_orphan_class("The concept was first introduced by {{cite:3}}.") == "Originator"
    assert miner.match_orphan_class("For a comprehensive review, see {{cite:4}}.") == "Further Reading"
    assert miner.match_orphan_class("We simply use {{cite:5}}.") is None


def test_citing_context_pydantic_validation():
    """Valida la serialización y deserialización conforme al esquema CitingContext."""
    ctx = CitingContext(
        context_id="mc_train_001",
        source_dataset="multicite",
        split="train",
        citing_paper_id="paper_1",
        rhetorical_section="Methodology",
        window_rule="m_3_sentences",
        citation_context="Sentence 1. <cite>Author, 2020</cite> is used. Sentence 3.",
        target_ref_key="Author, 2020",
        target_cited_id="2001.00001",
        target_cited_title="Paper Title",
        target_cited_authors="Author",
        target_cited_year=2020,
        target_cited_abstract="Abstract text",
        has_cited_chunks=True,
        original_label="uses",
        normalized_label="Uses",
    )
    json_str = ctx.model_dump_json()
    reconstructed = CitingContext.model_validate_json(json_str)
    assert reconstructed.context_id == "mc_train_001"
    assert reconstructed.has_cited_chunks is True
    assert reconstructed.normalized_label == "Uses"


def test_mining_pipeline_force_and_idempotency(tmp_path):
    """Valida que el pipeline respete la bandera --force ante archivos existentes."""
    out_file = tmp_path / "citing_contexts.jsonl"
    out_file.write_text("existing content\n", encoding="utf-8")

    config = MiningConfig(
        multicite_dir=tmp_path / "multicite",
        papers_file=tmp_path / "papers.jsonl",
        output_file=out_file,
        s2_cache_path=tmp_path / "s2_cache.db",
        force=False,
    )
    pipeline = MiningPipeline(config)

    with pytest.raises(FileExistsError, match="ya existe"):
        pipeline.run()

    # Con force=True debe sobreescribir sin error
    config.force = True
    config.skip_network = True
    pipeline = MiningPipeline(config)
    stats = pipeline.run()
    assert stats["total_records"] == 0
    assert out_file.read_text(encoding="utf-8") == ""
