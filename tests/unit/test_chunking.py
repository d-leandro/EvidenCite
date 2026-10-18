"""Pruebas unitarias para el módulo de chunking granular de artículos citados (src.data.chunking)."""

import json
from pathlib import Path
import tempfile
import pytest

from src.data.chunking import (
    ChunkingConfig,
    ChunkingPipeline,
    chunk_paper,
    chunk_sentences_greedy,
    extract_target_cited_ids,
)
from src.data.schemas import CitedChunk


def test_chunk_sentences_greedy_basic():
    """Valida la acumulación greedy hasta el límite de palabras."""
    sentences = [
        "This is sentence one with exactly eight words.",  # 8 words
        "This is sentence two with eight words too.",      # 8 words
        "Sentence three has six words.",                   # 5 words
    ]
    # Total: 21 words. If max_words = 17:
    # 8 + 8 = 16 <= 17 -> Chunk 0 has 16 words.
    # 16 + 5 = 21 > 17 -> Chunk 1 has 5 words.
    chunks = chunk_sentences_greedy(sentences, max_words=17)
    assert len(chunks) == 2
    assert chunks[0][1] == 16
    assert chunks[1][1] == 5
    assert "sentence one" in chunks[0][0] and "sentence two" in chunks[0][0]
    assert "Sentence three" in chunks[1][0]


def test_chunk_sentences_greedy_oversized_sentence():
    """Valida que una oración que excede max_words se preserve íntegra sin fracturas."""
    long_sentence = "word " * 150  # 150 words
    sentences = [
        "Normal sentence of five words.",  # 5 words
        long_sentence.strip(),              # 150 words
        "Another normal sentence of six words.",  # 6 words
    ]
    chunks = chunk_sentences_greedy(sentences, max_words=110)
    assert len(chunks) == 3
    assert chunks[0][1] == 5
    assert chunks[1][1] == 150
    assert chunks[2][1] == 6


def test_chunk_sentences_greedy_empty_and_whitespace():
    """Valida que oraciones vacías o de puros espacios sean descartadas limpiamente."""
    sentences = ["", "   ", "Valid sentence with five words.", "   ", ""]
    chunks = chunk_sentences_greedy(sentences, max_words=110)
    assert len(chunks) == 1
    assert chunks[0][0] == "Valid sentence with five words."
    assert chunks[0][1] == 5


def test_chunk_paper_respects_section_boundaries():
    """Valida que los chunks nunca crucen límites entre secciones (RF-02) y su chunk_id canónico (RF-04)."""
    paper_data = {
        "paper_id": "1706.03762",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models...",
        "sections": [
            {
                "section_idx": 1,
                "heading": "Introduction",
                "sentences": [
                    {"text": "Recurrent neural networks have been firmly established."},
                    {"text": "Attention mechanisms have become an integral part."},
                ],
            },
            {
                "section_idx": 2,
                "heading": "Model Architecture",
                "sentences": [
                    {"text": "Most competitive neural sequence models have an encoder-decoder structure."},
                    {"text": "The Transformer follows this overall architecture."},
                ],
            },
        ],
    }

    chunks = chunk_paper(paper_data, max_words=110)
    assert len(chunks) == 2
    # Chunk 0 pertenece a section 1
    assert chunks[0].chunk_id == "1706.03762_s1_c0"
    assert chunks[0].section_idx == 1
    assert chunks[0].section_heading == "Introduction"
    assert "Recurrent neural networks" in chunks[0].text

    # Chunk 1 pertenece a section 2
    assert chunks[1].chunk_id == "1706.03762_s2_c0"
    assert chunks[1].section_idx == 2
    assert chunks[1].section_heading == "Model Architecture"
    assert "encoder-decoder structure" in chunks[1].text


def test_chunk_paper_fallback_to_abstract():
    """Valida la salvaguarda cuando un paper estructurado no posee secciones de cuerpo."""
    paper_data = {
        "paper_id": "2103.10431",
        "title": "Convexification inversion method",
        "abstract": "First sentence of abstract. Second sentence of abstract.",
        "sections": [],
    }

    chunks = chunk_paper(paper_data, max_words=110)
    assert len(chunks) >= 1
    assert chunks[0].chunk_id == "2103.10431_s0_c0"
    assert chunks[0].section_heading == "Abstract"
    assert "First sentence of abstract" in chunks[0].text


def test_extract_target_cited_ids(tmp_path: Path):
    """Valida que extract_target_cited_ids filtre estrictamente has_cited_chunks=True."""
    contexts_file = tmp_path / "citing_contexts.jsonl"
    records = [
        {"context_id": "c1", "target_cited_id": "1706.03762", "has_cited_chunks": True},
        {"context_id": "c2", "target_cited_id": "1810.04805", "has_cited_chunks": False},
        {"context_id": "c3", "target_cited_id": "1706.03762", "has_cited_chunks": True},  # Duplicado
        {"context_id": "c4", "target_cited_id": None, "has_cited_chunks": False},
    ]
    with open(contexts_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    target_ids = extract_target_cited_ids(contexts_file)
    assert target_ids == {"1706.03762"}
