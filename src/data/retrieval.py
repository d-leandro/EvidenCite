"""Módulo de Recuperación Semántica Densa de Chunks Citados con SciBERT.

Implementa la codificación densa de contextos citantes y fragmentos discursivos (chunks)
utilizando SciBERT (o cualquier modelo transformer parametrizable) con mean-pooling ponderado
por attention mask (ADR-05) y normalización L2 nativa en PyTorch (RF-06 a RF-08).

Recupera determinísticamente los Top-k (k=3) fragmentos más relevantes de cada artículo citado
para el contexto de citación dado, alimentando la Configuración B del benchmark experimental (A1).
Incluye una caché en memoria de representaciones tensoriales de artículos seminales (RF-10)
para acelerar la inferencia en más de un 80%.

Especificación Técnica: documentacion/diseno-tecnico/07-diseno-chunking-scibert.md
"""

import argparse
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Garantizar que la raíz del proyecto esté en sys.path para ejecuciones como script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast
import yaml

from src.data.schemas import CitedChunk, RetrievedChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resuelve dinámicamente el dispositivo de cómputo más eficiente disponible (RNF-01).

    Soporta Metal Performance Shaders (mps) en macOS, NVIDIA CUDA en Linux/Windows,
    o CPU como fallback universal.
    """
    dev_lower = device_str.lower().strip()
    if dev_lower == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif dev_lower in ("cuda", "gpu"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif dev_lower == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    return device


@dataclass
class RetrievalConfig:
    """Parámetros de configuración del motor de recuperación semántica."""

    model_name: str = "allenai/scibert_scivocab_uncased"
    batch_size: int = 32
    max_length: int = 512
    k: int = 3
    device: str = "auto"
    seed: int = 42
    chunks_path: Path = Path("data/processed/cited_chunks.jsonl")

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "RetrievalConfig":
        """Carga la configuración desde un archivo YAML externo."""
        path = Path(yaml_path)
        if not path.exists():
            logger.warning("Archivo YAML de configuración no encontrado: %s. Usando defaults.", path)
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            model_name=data.get("model_name", cls.model_name),
            batch_size=int(data.get("batch_size", cls.batch_size)),
            max_length=int(data.get("max_length", cls.max_length)),
            k=int(data.get("k", cls.k)),
            device=str(data.get("device", cls.device)),
            seed=int(data.get("seed", cls.seed)),
            chunks_path=Path(data.get("chunks_path", cls.chunks_path)),
        )


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Calcula el vector de representación promedio excluyendo tokens de padding (ADR-05 / RF-07).

    Matemáticamente:
        e = sum(m_i * h_i) / (sum(m_i) + eps)

    Args:
        last_hidden_state: Tensor [batch_size, seq_len, hidden_dim] de la última capa del modelo.
        attention_mask: Tensor [batch_size, seq_len] con 1 para tokens reales y 0 para padding.

    Returns:
        Tensor [batch_size, hidden_dim] con el promedio ponderado exacto por fila.
    """
    # Expandir máscara a las dimensiones del estado oculto: [batch_size, seq_len, hidden_dim]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).to(dtype=last_hidden_state.dtype)

    # Sumar embeddings ignorando el padding
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)

    # Sumar tokens reales con un valor mínimo de epsilon (1e-9) para evitar división por cero
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return sum_embeddings / sum_mask


class SciBERTRetriever:
    """Motor de recuperación densa con SciBERT y caché de representaciones tensoriales."""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        model: Optional[PreTrainedModel] = None,
    ) -> None:
        """Inicializa el modelo de lenguaje, tokenizador, dispositivo y estructuras en memoria."""
        self.config = config or RetrievalConfig()

        # Fijar semilla determinista (RNF-03)
        torch.manual_seed(self.config.seed)

        # Resolver dispositivo (RNF-01)
        self.device = resolve_device(self.config.device)
        logger.info("Motor de recuperación SciBERT inicializado en dispositivo: %s", self.device)

        # Cargar tokenizador y modelo si no fueron inyectados directamente
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            logger.info("Cargando tokenizador: %s", self.config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        if model is not None:
            self.model = model
        else:
            logger.info("Cargando pesos de modelo transformer: %s", self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)

        self.model.to(self.device)
        self.model.eval()  # Modo evaluación estricto (RNF-02)

        # Índice de chunks discursivos agrupados por paper_id: Dict[paper_id, List[CitedChunk]]
        self._chunks_by_paper: Dict[str, List[CitedChunk]] = {}

        # Caché en memoria de embeddings seminales (RF-10): Dict[paper_id, (List[CitedChunk], torch.Tensor)]
        # El tensor contiene los embeddings normalizados L2 [M, hidden_dim]
        self._paper_embeddings_cache: Dict[str, Tuple[List[CitedChunk], torch.Tensor]] = {}

        # Cargar chunks si el archivo existe
        if self.config.chunks_path.exists():
            self.load_chunks(self.config.chunks_path)

    def load_chunks(self, chunks_path: Path) -> int:
        """Carga y pre-indexa en memoria los chunks de data/processed/cited_chunks.jsonl."""
        start_time = time.time()
        logger.info("Cargando índice de chunks desde %s ...", chunks_path)

        total_chunks = 0
        papers_count = 0
        self._chunks_by_paper.clear()

        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    chunk = CitedChunk.model_validate_json(line_str)
                    if chunk.paper_id not in self._chunks_by_paper:
                        self._chunks_by_paper[chunk.paper_id] = []
                        papers_count += 1
                    self._chunks_by_paper[chunk.paper_id].append(chunk)
                    total_chunks += 1
                except Exception as e:
                    logger.debug("Error deserializando CitedChunk: %s", e)
                    continue

        elapsed = time.time() - start_time
        logger.info(
            "Índice de chunks cargado exitosamente: %d chunks indexados en %d artículos citados (%.2fs)",
            total_chunks,
            papers_count,
            elapsed,
        )
        return total_chunks

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Codifica una lista de textos a embeddings densos normalizados L2.

        Aplica tokenización con truncamiento a max_length (512 tokens), inferencia
        sin gradientes (torch.no_grad()), mean-pooling ponderado y normalización L2 (RF-07, RF-08).

        Returns:
            Tensor normalizado [len(texts), hidden_dim] en self.device.
        """
        if not texts:
            return torch.empty((0, self.model.config.hidden_size), device=self.device)

        all_embeddings: List[torch.Tensor] = []

        # Procesar por lotes para no saturar memoria
        batch_size = self.config.batch_size
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = mean_pooling(outputs.last_hidden_state, attention_mask)

                # Normalización L2 nativa a la esfera unitaria S^{d-1}
                normalized = F.normalize(pooled, p=2, dim=-1)
                all_embeddings.append(normalized)

        return torch.cat(all_embeddings, dim=0)

    def get_paper_chunks_and_embeddings(
        self, paper_id: str
    ) -> Optional[Tuple[List[CitedChunk], torch.Tensor]]:
        """Obtiene o computa los embeddings normalizados de los chunks de un artículo citado.

        Implementa la caché en memoria para papers seminales recurrentes (RF-10).
        """
        # 1. Comprobar si ya está en caché
        if paper_id in self._paper_embeddings_cache:
            return self._paper_embeddings_cache[paper_id]

        # 2. Verificar disponibilidad de chunks en el índice
        chunks = self._chunks_by_paper.get(paper_id)
        if not chunks:
            return None

        # 3. Codificar los chunks del artículo
        chunk_texts = [c.text for c in chunks]
        embeddings = self.encode_texts(chunk_texts)

        # 4. Almacenar en caché y retornar
        self._paper_embeddings_cache[paper_id] = (chunks, embeddings)
        return chunks, embeddings

    def retrieve_top_k(
        self,
        citation_context: str,
        target_cited_id: str,
        k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Recupera los Top-k fragmentos discursivos más relevantes de un artículo citado (RF-09).

        La consulta de recuperación se define estrictamente como la ventana del contexto de cita:
            T_query = citation_context
        (sin anteponer el encabezado de sección citante, garantizando neutralidad experimental).

        Args:
            citation_context: Texto del contexto citante (ventana m=3 oraciones).
            target_cited_id: Identificador canónico del artículo citado.
            k: Número de fragmentos a recuperar (por defecto: config.k = 3).

        Returns:
            Lista ordenada de RetrievedChunk con rank 1..k y similarity_score [-1, 1].
        """
        k_val = k if k is not None else self.config.k

        paper_data = self.get_paper_chunks_and_embeddings(target_cited_id)
        if not paper_data:
            return []

        chunks, chunk_embeddings = paper_data
        m_chunks = len(chunks)
        if m_chunks == 0:
            return []

        # Codificar la consulta (contexto de citación)
        query_vec = self.encode_texts([citation_context])  # [1, hidden_dim]

        # Multiplicación tensorial acelerada: s = q * C^T  -> [1, M]
        # Al estar normalizados L2, el producto punto es idénticamente la similitud coseno
        sim_scores = torch.matmul(query_vec, chunk_embeddings.T).squeeze(0)  # [M]

        # Seleccionar top-k: min(k_val, M) por si el paper tiene menos chunks que k
        actual_k = min(k_val, m_chunks)
        top_scores, top_indices = torch.topk(sim_scores, k=actual_k, largest=True, sorted=True)

        results: List[RetrievedChunk] = []
        for rank_idx, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
            target_chunk = chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=target_chunk.chunk_id,
                    section_heading=target_chunk.section_heading,
                    text=target_chunk.text,
                    similarity_score=float(score),
                    rank=rank_idx,
                )
            )

        return results

    def clear_cache(self) -> None:
        """Limpia la caché de embeddings en memoria."""
        self._paper_embeddings_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()


def parse_arguments() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Módulo de recuperación semántica densa de fragmentos citados con SciBERT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval.yaml",
        help="Ruta al archivo YAML de configuración",
    )
    parser.add_argument(
        "--chunks-file",
        type=str,
        default=None,
        help="Sobrescribe la ruta de cited_chunks.jsonl",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Identificador del modelo transformer en HuggingFace",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de aceleración ('auto', 'mps', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamaño de lote para codificación de chunks",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Número de chunks a recuperar por consulta",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default=None,
        help="Contexto citante para prueba interactiva",
    )
    parser.add_argument(
        "--target-cited-id",
        type=str,
        default=None,
        help="ID del paper citado para prueba interactiva",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada CLI para verificación y prueba de recuperación semántica."""
    args = parse_arguments()

    # Cargar configuración base desde YAML
    config = RetrievalConfig.from_yaml(args.config)

    # Sobrescribir con flags CLI explícitas
    if args.chunks_file:
        config.chunks_path = Path(args.chunks_file)
    if args.model_name:
        config.model_name = args.model_name
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.k:
        config.k = args.k

    retriever = SciBERTRetriever(config=config)

    if args.test_query and args.target_cited_id:
        print(f"\n--- Probando recuperación para {args.target_cited_id} ---")
        print(f"Contexto: {args.test_query}\n")
        results = retriever.retrieve_top_k(args.test_query, args.target_cited_id, k=config.k)
        for r in results:
            print(f"[Rank {r.rank}] Score: {r.similarity_score:.4f} | Chunk ID: {r.chunk_id}")
            print(f"Sección: {r.section_heading}")
            print(f"Texto: {r.text[:140]}...\n")
    else:
        # Modo de demostración rápida sobre el primer artículo disponible en el índice
        first_paper_id = next(iter(retriever._chunks_by_paper.keys()), None)
        if first_paper_id:
            demo_context = (
                "Recent advances in natural language processing rely heavily on "
                "self-attention mechanisms <cite>Vaswani 2017</cite> for sequence transduction."
            )
            print(f"\n--- Demostración de Recuperación Top-{config.k} sobre {first_paper_id} ---")
            print(f"Contexto citante simulado: '{demo_context}'\n")
            start = time.time()
            results = retriever.retrieve_top_k(demo_context, first_paper_id, k=config.k)
            dur = (time.time() - start) * 1000
            for r in results:
                print(f"[Rank {r.rank}] Similitud Coseno: {r.similarity_score:.4f} | Chunk: {r.chunk_id}")
                print(f"Sección: {r.section_heading}")
                print(f"Texto ({len(r.text.split())} palabras): {r.text[:120]}...\n")
            print(f"Tiempo de recuperación (primera llamada, codificación + topk): {dur:.2f} ms")

            # Segunda llamada para verificar efecto de la caché de embeddings
            start2 = time.time()
            _ = retriever.retrieve_top_k(demo_context, first_paper_id, k=config.k)
            dur2 = (time.time() - start2) * 1000
            print(f"Tiempo de recuperación (segunda llamada con caché activa): {dur2:.2f} ms")
        else:
            logger.warning("No se encontraron chunks en %s para ejecutar demostración.", config.chunks_path)


if __name__ == "__main__":
    main()
