"""Módulo de segmentación discursiva granular (Chunking) para artículos citados.

Implementa la partición en fragmentos discursivos (chunks) acotados a un umbral
de palabras parametrizable (por defecto: 110 palabras, conforme a ADR-04 Alternativa a),
respetando estrictamente las fronteras de sección (RF-02) y oracionales (RF-03).

Filtra quirúrgicamente el universo de artículos citados (RF-01) a partir de los
identificadores con has_cited_chunks=True en data/processed/citing_contexts.jsonl,
generando data/processed/cited_chunks.jsonl como insumo desacoplado para la recuperación
semántica densa con SciBERT (A1 / Configuración B).
"""

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

# Garantizar que la raíz del proyecto esté en sys.path para ejecuciones como script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import CitedChunk, StructuredPaper, StructuredSection, StructuredSentence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuración del proceso de chunking de artículos citados."""

    citing_contexts_path: Path = Path("data/processed/citing_contexts.jsonl")
    papers_path: Path = Path("data/structured/papers.jsonl")
    output_path: Path = Path("data/processed/cited_chunks.jsonl")
    max_words: int = 110
    model_name: str = "allenai/scibert_scivocab_uncased"
    force: bool = False


def extract_target_cited_ids(citing_contexts_path: Path) -> Set[str]:
    """Extrae el conjunto único de target_cited_id que tienen texto completo disponible.

    Filtra estrictamente las instancias con has_cited_chunks = True (RF-01).
    """
    if not citing_contexts_path.exists():
        raise FileNotFoundError(f"Archivo de contextos citantes no encontrado: {citing_contexts_path}")

    target_ids: Set[str] = set()
    total_records = 0
    with open(citing_contexts_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            total_records += 1
            try:
                data = json.loads(line_str)
                if data.get("has_cited_chunks") and data.get("target_cited_id"):
                    target_ids.add(str(data["target_cited_id"]))
            except json.JSONDecodeError:
                continue

    logger.info(
        "Contextos analizados: %d | Artículos citados únicos con texto completo (has_cited_chunks=True): %d",
        total_records,
        len(target_ids),
    )
    return target_ids


def chunk_sentences_greedy(
    sentences: List[str],
    max_words: int = 110,
) -> List[Tuple[str, int]]:
    """Particiona una secuencia de oraciones en bloques textuales de forma greedy.

    Reglas (RF-03 y ADR-04):
    1. Las oraciones se acumulan en el chunk actual mientras la suma de palabras <= max_words.
    2. Si agregar la siguiente oración excede max_words:
       - Se emite el chunk acumulado.
       - Si la oración entrante sola ya supera max_words, se emite inmediatamente como
         un chunk individual completo para evitar fracturas léxicas arbitrarias.
       - De lo contrario, la oración entrante inicia el siguiente chunk.
    3. Al finalizar la lista, cualquier oración acumulada se emite como el último chunk.
    4. Oraciones vacías o compuestas exclusivamente de espacios se omiten.

    Retorna una lista de tuplas: [(chunk_text, word_count), ...]
    """
    chunks: List[Tuple[str, int]] = []
    current_sentences: List[str] = []
    current_words: int = 0

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        words = len(s_clean.split())
        if words == 0:
            continue

        if current_words + words > max_words:
            if current_sentences:
                chunks.append((" ".join(current_sentences), current_words))
                current_sentences = []
                current_words = 0

            if words > max_words:
                chunks.append((s_clean, words))
            else:
                current_sentences = [s_clean]
                current_words = words
        else:
            current_sentences.append(s_clean)
            current_words += words

    if current_sentences:
        chunks.append((" ".join(current_sentences), current_words))

    return chunks


def load_optional_tokenizer(model_name: str) -> Optional[Any]:
    """Carga el tokenizador Hugging Face si transformers está disponible."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        logger.info("Tokenizador '%s' cargado exitosamente para conteo exacto de tokens.", model_name)
        return tokenizer
    except Exception as exc:
        logger.warning(
            "No se pudo cargar el tokenizador '%s' (%s). Se usará estimación heurística (palabras * 1.3).",
            model_name,
            exc,
        )
        return None


def chunk_paper(
    paper_data: Dict[str, Any],
    max_words: int = 110,
    tokenizer: Optional[Any] = None,
) -> List[CitedChunk]:
    """Genera los chunks de un artículo respetando estrictamente las fronteras de sección (RF-02).

    Si el artículo no contiene secciones de cuerpo pero sí un abstract válido,
    genera un chunk de respaldo a partir del abstract para garantizar completitud.
    """
    paper_id = str(paper_data.get("paper_id", "")).strip()
    sections = paper_data.get("sections", [])

    # Salvaguarda: si el artículo estructurado no posee secciones de cuerpo, utilizar abstract
    if not sections and paper_data.get("abstract"):
        abstract_text = str(paper_data["abstract"]).strip()
        if abstract_text:
            # Segmentar abstract en oraciones simples por punto y espacio
            raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract_text) if s.strip()]
            sections = [
                {
                    "section_idx": 0,
                    "heading": "Abstract",
                    "sentences": [{"text": s} for s in raw_sents],
                }
            ]

    raw_chunks: List[CitedChunk] = []
    seen_sec_indices: Set[int] = set()

    for s_idx, sec in enumerate(sections):
        orig_sec_idx = sec.get("section_idx")
        # Asegurar un índice de sección numérico y no colisionante
        if orig_sec_idx is None or orig_sec_idx in seen_sec_indices:
            sec_idx = s_idx
        else:
            sec_idx = int(orig_sec_idx)
        seen_sec_indices.add(sec_idx)

        heading = str(sec.get("heading", "")).strip() or "Untitled Section"
        sents_data = sec.get("sentences", [])

        # Extraer textos limpios de oraciones
        sentence_texts: List[str] = []
        for sent in sents_data:
            if isinstance(sent, dict):
                t = str(sent.get("text", "")).strip()
            elif hasattr(sent, "text"):
                t = str(sent.text).strip()
            else:
                t = str(sent).strip()
            if t:
                sentence_texts.append(t)

        if not sentence_texts:
            continue

        section_chunks = chunk_sentences_greedy(sentence_texts, max_words=max_words)

        for c_idx, (chunk_text, word_count) in enumerate(section_chunks):
            chunk_id = f"{paper_id}_s{sec_idx}_c{c_idx}"
            # Estimación inicial de tokens
            estimated_tokens = int(round(word_count * 1.3))
            raw_chunks.append(
                CitedChunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    section_idx=sec_idx,
                    section_heading=heading,
                    chunk_idx=c_idx,
                    text=chunk_text,
                    token_count=estimated_tokens,
                    word_count=word_count,
                )
            )

    # Conteo exacto de tokens en lote con el tokenizador si está disponible
    if tokenizer is not None and raw_chunks:
        try:
            texts = [c.text for c in raw_chunks]
            batch_enc = tokenizer(texts, truncation=False, add_special_tokens=False)
            for chunk, token_ids in zip(raw_chunks, batch_enc["input_ids"]):
                chunk.token_count = len(token_ids)
        except Exception as exc:
            logger.debug("Fallo en tokenización por lotes para %s (%s). Se conserva estimación.", paper_id, exc)

    return raw_chunks


class ChunkingPipeline:
    """Orquestador del pipeline de chunking de artículos citados."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def run(self) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de segmentación y persistencia."""
        start_time = time.time()
        logger.info("=== Iniciando Pipeline de Chunking de Artículos Citados ===")
        logger.info("Umbral max_words: %d palabras", self.config.max_words)
        logger.info("Modelo de tokens: %s", self.config.model_name)
        logger.info("Destino de salida: %s", self.config.output_path)

        if self.config.output_path.exists() and not self.config.force:
            logger.warning(
                "El archivo de salida %s ya existe. Use --force para regenerarlo.",
                self.config.output_path,
            )
            return {"status": "skipped", "reason": "output_already_exists"}

        if not self.config.papers_path.exists():
            raise FileNotFoundError(f"Archivo de papers no encontrado: {self.config.papers_path}")

        # 1. Extraer identificadores objetivo
        target_ids = extract_target_cited_ids(self.config.citing_contexts_path)
        if not target_ids:
            logger.error("No se identificaron artículos citados objetivo con has_cited_chunks=True.")
            return {"status": "empty_targets"}

        # 2. Cargar tokenizador opcional
        tokenizer = load_optional_tokenizer(self.config.model_name)

        # 3. Escanear papers.jsonl y emitir chunks
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = self.config.output_path.with_suffix(".tmp")

        papers_found = 0
        total_chunks = 0
        total_words = 0
        total_tokens = 0
        chunks_per_paper: List[int] = []
        words_per_chunk: List[int] = []

        logger.info("Escaneando %s en busca de los %d artículos objetivo...", self.config.papers_path, len(target_ids))

        with open(self.config.papers_path, "r", encoding="utf-8") as in_f, open(
            temp_output_path, "w", encoding="utf-8"
        ) as out_f:
            for line_idx, line in enumerate(in_f):
                # Filtro rápido por subcadena para no parsear JSONs innecesarios de 66k líneas
                idx = line.find('"paper_id":')
                if idx == -1:
                    continue
                start = line.find('"', idx + 11) + 1
                end = line.find('"', start)
                pid = line[start:end]

                if pid not in target_ids:
                    continue

                try:
                    paper_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                papers_found += 1
                paper_chunks = chunk_paper(paper_data, max_words=self.config.max_words, tokenizer=tokenizer)

                p_chunk_count = len(paper_chunks)
                chunks_per_paper.append(p_chunk_count)

                for chunk in paper_chunks:
                    total_chunks += 1
                    total_words += chunk.word_count
                    total_tokens += chunk.token_count
                    words_per_chunk.append(chunk.word_count)
                    out_f.write(chunk.model_dump_json() + "\n")

                if papers_found % 50 == 0 or papers_found == len(target_ids):
                    logger.info(
                        "Progreso: %d/%d papers procesados | %d chunks generados...",
                        papers_found,
                        len(target_ids),
                        total_chunks,
                    )

        # Reemplazo atómico del archivo final
        temp_output_path.replace(self.config.output_path)
        elapsed = time.time() - start_time

        avg_words = (total_words / total_chunks) if total_chunks > 0 else 0.0
        avg_tokens = (total_tokens / total_chunks) if total_chunks > 0 else 0.0
        avg_chunks = (total_chunks / papers_found) if papers_found > 0 else 0.0

        stats = {
            "target_papers_required": len(target_ids),
            "target_papers_found": papers_found,
            "missing_papers": len(target_ids) - papers_found,
            "total_chunks_generated": total_chunks,
            "avg_chunks_per_paper": round(avg_chunks, 2),
            "avg_words_per_chunk": round(avg_words, 2),
            "avg_tokens_per_chunk": round(avg_tokens, 2),
            "min_words_chunk": min(words_per_chunk) if words_per_chunk else 0,
            "max_words_chunk": max(words_per_chunk) if words_per_chunk else 0,
            "elapsed_seconds": round(elapsed, 2),
            "output_file": str(self.config.output_path),
        }

        logger.info("=== Chunking Finalizado Exitosamente ===")
        logger.info("Artículos objetivo procesados: %d / %d", papers_found, len(target_ids))
        logger.info("Total de chunks persistidos: %d", total_chunks)
        logger.info("Promedio de chunks por artículo: %.1f", avg_chunks)
        logger.info("Promedio de palabras por chunk: %.1f (min: %d, max: %d)", avg_words, stats["min_words_chunk"], stats["max_words_chunk"])
        logger.info("Promedio de tokens por chunk: %.1f", avg_tokens)
        logger.info("Tiempo total transcurrido: %.2f segundos", elapsed)

        return stats


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Chunking discursivo de artículos citados hacia data/processed/cited_chunks.jsonl",
    )
    parser.add_argument(
        "--citing-contexts",
        type=Path,
        default=Path("data/processed/citing_contexts.jsonl"),
        help="Ruta al archivo de contextos citantes procesados.",
    )
    parser.add_argument(
        "--papers-file",
        type=Path,
        default=Path("data/structured/papers.jsonl"),
        help="Ruta al corpus canónico estructurado de artículos.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/cited_chunks.jsonl"),
        help="Ruta de destino para los chunks estructurados.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=110,
        help="Límite máximo de palabras por chunk (default: 110 según ADR-04 Alternativa a).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="Identificador del modelo para tokenización exacta de SciBERT.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe el archivo de salida si ya existe.",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada de ejecución por consola."""
    args = parse_args()
    config = ChunkingConfig(
        citing_contexts_path=args.citing_contexts,
        papers_path=args.papers_file,
        output_path=args.output_file,
        max_words=args.max_words,
        model_name=args.model_name,
        force=args.force,
    )
    pipeline = ChunkingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
