"""Pruebas unitarias para el motor de recuperación semántica con SciBERT (src.data.retrieval)."""

import json
from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn.functional as F

from src.data.retrieval import (
    RetrievalConfig,
    SciBERTRetriever,
    mean_pooling,
    resolve_device,
)
from src.data.schemas import CitedChunk, RetrievedChunk


def test_resolve_device():
    """Valida la resolución transparente de dispositivos (RNF-01)."""
    # Fallback forzado a CPU
    cpu_dev = resolve_device("cpu")
    assert cpu_dev.type == "cpu"

    # Resolución automática
    auto_dev = resolve_device("auto")
    assert auto_dev.type in ("mps", "cuda", "cpu")


def test_mean_pooling_ignores_padding():
    """Valida que el mean-pooling excluya estrictamente los tokens con attention_mask=0 (RF-07 / ADR-05)."""
    # Batch de 1 secuencia de longitud 4, dimensión 2
    # Token 0: [2.0, 4.0] (real)
    # Token 1: [4.0, 6.0] (real)
    # Token 2: [99.0, 99.0] (padding!)
    # Token 3: [-50.0, -50.0] (padding!)
    last_hidden_state = torch.tensor(
        [[[2.0, 4.0], [4.0, 6.0], [99.0, 99.0], [-50.0, -50.0]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    pooled = mean_pooling(last_hidden_state, attention_mask)

    # El promedio de los tokens reales (Token 0 y Token 1) debe ser exactamente:
    # [(2.0 + 4.0)/2, (4.0 + 6.0)/2] = [3.0, 5.0]
    expected = torch.tensor([[3.0, 5.0]], dtype=torch.float32)
    assert torch.allclose(pooled, expected, atol=1e-5)


def test_mean_pooling_handles_empty_sequence_gracefully():
    """Valida que una máscara enteramente de ceros no produzca NaN o división por cero."""
    last_hidden_state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    attention_mask = torch.tensor([[0, 0]], dtype=torch.long)

    pooled = mean_pooling(last_hidden_state, attention_mask)
    assert not torch.isnan(pooled).any()
    assert not torch.isinf(pooled).any()


def test_l2_normalization_and_unit_norm():
    """Valida que la normalización L2 proyecte los vectores a la esfera unitaria (RF-08)."""
    vectors = torch.tensor([[3.0, 4.0], [1.0, 1.0], [10.0, 0.0]], dtype=torch.float32)
    normalized = F.normalize(vectors, p=2, dim=-1)

    norms = torch.norm(normalized, p=2, dim=-1)
    expected_norms = torch.ones_like(norms)
    assert torch.allclose(norms, expected_norms, atol=1e-6)


def test_retrieval_handles_fewer_than_k_chunks():
    """Valida que la selección Top-k maneje artículos con menos de k chunks sin errores (RF-09)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chunks_file = Path(tmp_dir) / "test_chunks.jsonl"
        # Paper con solo 1 chunk disponible
        single_chunk = {
            "chunk_id": "test_paper_s1_c0",
            "paper_id": "test_paper",
            "section_idx": 1,
            "section_heading": "Methods",
            "chunk_idx": 0,
            "text": "Deep learning models require extensive computational resources and data.",
            "token_count": 12,
            "word_count": 9,
        }
        with open(chunks_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(single_chunk) + "\n")

        config = RetrievalConfig(
            chunks_path=chunks_file,
            device="cpu",  # CPU para pruebas unitarias rápidas
            k=3,
        )
        retriever = SciBERTRetriever(config=config)

        # Pedimos k=3 sobre un paper con solo 1 chunk
        results = retriever.retrieve_top_k(
            citation_context="Machine learning architectures require large scale data.",
            target_cited_id="test_paper",
            k=3,
        )

        assert len(results) == 1
        assert results[0].chunk_id == "test_paper_s1_c0"
        assert results[0].rank == 1
        assert -1.0 <= results[0].similarity_score <= 1.0


def test_retrieval_ranking_is_descending():
    """Valida que los chunks recuperados estén estrictamente ordenados de mayor a menor similitud (RF-09)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chunks_file = Path(tmp_dir) / "test_chunks.jsonl"
        # Paper con 4 chunks con diferente afinidad temática
        chunks_data = [
            {
                "chunk_id": "paper_nlp_s1_c0",
                "paper_id": "paper_nlp",
                "section_idx": 1,
                "section_heading": "Introduction",
                "chunk_idx": 0,
                "text": "Transformers and self-attention mechanisms in natural language processing.",
                "token_count": 12,
                "word_count": 9,
            },
            {
                "chunk_id": "paper_nlp_s1_c1",
                "paper_id": "paper_nlp",
                "section_idx": 1,
                "section_heading": "Methods",
                "chunk_idx": 1,
                "text": "We implement multi-head attention with scaled dot-product attention layers.",
                "token_count": 12,
                "word_count": 10,
            },
            {
                "chunk_id": "paper_nlp_s2_c0",
                "paper_id": "paper_nlp",
                "section_idx": 2,
                "section_heading": "Experiments",
                "chunk_idx": 0,
                "text": "Agricultural crop yields depend on seasonal rainfall and soil nitrogen levels.",
                "token_count": 12,
                "word_count": 10,
            },
            {
                "chunk_id": "paper_nlp_s3_c0",
                "paper_id": "paper_nlp",
                "section_idx": 3,
                "section_heading": "Conclusion",
                "chunk_idx": 0,
                "text": "Protein folding predictions using crystallography and electron microscopy.",
                "token_count": 12,
                "word_count": 9,
            },
        ]
        with open(chunks_file, "w", encoding="utf-8") as f:
            for c in chunks_data:
                f.write(json.dumps(c) + "\n")

        config = RetrievalConfig(
            chunks_path=chunks_file,
            device="cpu",
            k=3,
        )
        retriever = SciBERTRetriever(config=config)

        context = "We build upon the self-attention mechanism <cite>Vaswani 2017</cite> for language modeling."
        results = retriever.retrieve_top_k(
            citation_context=context,
            target_cited_id="paper_nlp",
            k=3,
        )

        assert len(results) == 3
        # Validar orden estrictamente descendente
        assert results[0].similarity_score >= results[1].similarity_score
        assert results[1].similarity_score >= results[2].similarity_score
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3

        # Los chunks sobre NLP/attention deben tener mayor similitud que el de agricultura
        assert "attention" in results[0].text or "Transformers" in results[0].text


def test_in_memory_cache_efficiency():
    """Valida que la caché en memoria evite codificar repetidamente el mismo artículo seminal (RF-10)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chunks_file = Path(tmp_dir) / "test_chunks.jsonl"
        chunk_data = {
            "chunk_id": "seminal_s1_c0",
            "paper_id": "seminal_paper",
            "section_idx": 1,
            "section_heading": "Introduction",
            "chunk_idx": 0,
            "text": "Deep residual learning for image recognition with skip connections.",
            "token_count": 12,
            "word_count": 9,
        }
        with open(chunks_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(chunk_data) + "\n")

        config = RetrievalConfig(chunks_path=chunks_file, device="cpu", k=1)
        retriever = SciBERTRetriever(config=config)

        assert "seminal_paper" not in retriever._paper_embeddings_cache

        # Primera consulta: codifica y guarda en caché
        _ = retriever.retrieve_top_k("ResNet models utilize skip connections.", "seminal_paper")
        assert "seminal_paper" in retriever._paper_embeddings_cache

        # Guardar referencia del tensor en caché
        cached_chunks, cached_tensor = retriever._paper_embeddings_cache["seminal_paper"]

        # Segunda consulta con otro contexto diferente pero el mismo paper
        _ = retriever.retrieve_top_k("Residual networks optimize gradient flow.", "seminal_paper")

        # La tupla en caché debe ser idénticamente el mismo objeto
        assert retriever._paper_embeddings_cache["seminal_paper"][1] is cached_tensor
