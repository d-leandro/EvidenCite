"""Módulo de minado y unificación de contextos citantes hacia data/processed/citing_contexts.jsonl.

Integra el dataset heredado MultiCite (con resolución semántica y caché SQLite vía s2_client)
y el minado nativo sobre data/structured/papers.jsonl (ventana m=3 y reglas léxicas para
las 4 clases huérfanas: Basis, Evidence, Originator, Further Reading).
"""

import argparse
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

# Garantizar que la raíz del proyecto esté en sys.path para ejecuciones como script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field

from src.data.ingestion.s2_client import S2ClientConfig, S2PaperMetadata, SemanticScholarClient
from src.data.schemas import BibliographyEntry, CitationMarker, CitingContext, StructuredPaper, StructuredSection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Mapeo oficial de etiquetas de MultiCite al esquema de 9 clases (Lyu et al.)
MULTICITE_LABEL_MAP: Dict[str, str] = {
    "background": "Background",
    "uses": "Uses",
    "motivation": "Motivation",
    "extends": "Extends",
    "similarities": "Similarities",
    "differences": "Differences",
    "future_work": "Future Work",
}

# Reglas léxico-sintácticas de alta precisión para las 4 clases huérfanas en unarXive
ORPHAN_CLASS_PATTERNS: Dict[str, re.Pattern] = {
    "Basis": re.compile(
        r"\b(build(?:s|ing|t)? upon|adapt(?:s|ed|ing)? the (?:framework|method|approach|model) of|"
        r"based on the work of|following the approach of|derived from the work of|"
        r"extend(?:s|ed|ing)? the framework of)\b",
        re.IGNORECASE,
    ),
    "Evidence": re.compile(
        r"\b(consistent with|confirm(?:s|ed|ing)? (?:the )?(?:findings|results)|"
        r"in line with|support(?:s|ed|ing)? by|corroborat(?:es|ed|ing)|"
        r"aligns? with the findings of|in agreement with)\b",
        re.IGNORECASE,
    ),
    "Originator": re.compile(
        r"\b(first introduced by|originally proposed by|originated in|pioneered by|"
        r"coined by|first formulated by|seminal work of)\b",
        re.IGNORECASE,
    ),
    "Further Reading": re.compile(
        r"\b(for a (?:comprehensive )?review(?:,?\s*see)?|see (?:also )?.*?for (?:a )?(?:more )?detailed|"
        r"refer (?:to )?.*?for|we refer the reader to|for more details(?:,?\s*see)?|"
        r"for an overview(?:,?\s*see)?)\b",
        re.IGNORECASE,
    ),
}


class MiningConfig(BaseModel):
    """Configuración para el proceso de minado y unificación de contextos citantes."""

    multicite_dir: Path = Field(
        default=Path("data/raw/multicite"),
        description="Directorio con las particiones de MultiCite (classification_3_context)",
    )
    papers_file: Path = Field(
        default=Path("data/structured/papers.jsonl"),
        description="Ruta al archivo canonical estructurado de unarXive",
    )
    output_file: Path = Field(
        default=Path("data/processed/citing_contexts.jsonl"),
        description="Ruta destino para los contextos citantes procesados",
    )
    s2_cache_path: Path = Field(
        default=Path("data/raw/s2_cache.db"),
        description="Ruta a la base de datos SQLite para caché persistente de Semantic Scholar",
    )
    max_multicite: Optional[int] = Field(
        default=None,
        description="Límite opcional de instancias de MultiCite a procesar (para pruebas)",
    )
    max_unarxive: Optional[int] = Field(
        default=None,
        description="Límite opcional de instancias de unarXive a procesar",
    )
    skip_network: bool = Field(
        default=False,
        description="Si es True, sólo consulta s2_cache.db sin realizar peticiones HTTP a Semantic Scholar",
    )
    force: bool = Field(
        default=False,
        description="Si es True, sobreescribe el archivo de salida si ya existe",
    )


def load_known_arxiv_metadata(
    papers_file: Path,
) -> Tuple[Set[str], Dict[str, str], Dict[str, str]]:
    """Carga los arXiv IDs y sus metadatos (título y abstract) presentes en papers.jsonl.

    Permite evaluar 'has_cited_chunks' en tiempo constante O(1) y poblar 'target_cited_abstract'
    y 'target_cited_title' de forma fidedigna para las citas minadas de unarXive.
    """
    arxiv_ids: Set[str] = set()
    abstracts: Dict[str, str] = {}
    titles: Dict[str, str] = {}
    if not papers_file.exists():
        logger.warning("Archivo de papers %s no existe; metadatos estarán vacíos.", papers_file)
        return arxiv_ids, abstracts, titles

    logger.info("Indexando metadatos de arXiv IDs disponibles desde %s...", papers_file)
    t0 = time.time()
    with open(papers_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                pid = data.get("paper_id")
                if pid:
                    arxiv_ids.add(pid)
                    if data.get("abstract"):
                        abstracts[pid] = data["abstract"]
                    if data.get("title"):
                        titles[pid] = data["title"]
            except Exception:
                continue

    logger.info(
        "Indexados %d arXiv IDs (%d abstracts, %d títulos) en %.2f segundos.",
        len(arxiv_ids),
        len(abstracts),
        len(titles),
        time.time() - t0,
    )
    return arxiv_ids, abstracts, titles


def load_known_arxiv_ids(papers_file: Path) -> Set[str]:
    """Wrapper de retrocompatibilidad para retornar exclusivamente el conjunto de IDs."""
    ids, _, _ = load_known_arxiv_metadata(papers_file)
    return ids


def parse_multicite_surface(raw_mention: str) -> Tuple[Optional[str], Optional[int]]:
    """Extrae el primer autor y año de una mención de cita de MultiCite.

    Ejemplos:
        '(Baevski et al., 2019b)' -> ('Baevski', 2019)
        'Salehi et al. (2014b)' -> ('Salehi', 2014)
        'Rashkin et al., 2017' -> ('Rashkin', 2017)
    """
    clean = re.sub(r"^[(\[\"\'\s]+|[)\]\"\'\s]+$", "", raw_mention).strip()
    author_m = re.search(r"^([A-Z][a-zA-Z\u00C0-\u017F\-\']+)", clean)
    author = author_m.group(1) if author_m else None

    year_m = re.search(r"\b(19\d{2}|20\d{2})[a-z]?\b", clean)
    year = int(year_m.group(1)) if year_m else None

    return author, year


def extract_rhetorical_section_from_text(text: str) -> Tuple[str, str]:
    """Detecta encabezados de sección en mayúsculas estilo markdown al inicio del texto.

    Retorna tupla (encabezado_detectado, texto_sin_prefijo).
    """
    header_m = re.match(r"^\s*\*\*([A-Za-z\s]+)\*\*\s*(.*)$", text, re.DOTALL)
    if header_m:
        raw_heading = header_m.group(1).strip()
        cleaned_text = header_m.group(2).strip()
        heading_formatted = raw_heading.capitalize()
        return heading_formatted, cleaned_text
    return "Unknown Section", text


def mask_secondary_citations(text: str, target_mention: str) -> str:
    """Enmascara citas secundarias con [OTHERCIT] preservando la cita objetivo delimitada con <cite>.

    En MultiCite, múltiples <cite> pueden referirse al mismo paper o a coreferencias.
    Esta función normaliza para que solo el target quede en <cite>...</cite> y menciones secundarias se enmascaren.
    """
    # 1. Identificar todas las menciones <cite>...</cite>
    cite_spans = list(re.finditer(r"<cite>(.*?)</cite>", text, flags=re.DOTALL))
    if not cite_spans:
        return text

    target_norm = re.sub(r"[^a-zA-Z0-9]", "", target_mention.lower())

    parts = []
    last_end = 0
    first_target_handled = False

    for span in cite_spans:
        parts.append(text[last_end : span.start()])
        inner = span.group(1)
        inner_norm = re.sub(r"[^a-zA-Z0-9]", "", inner.lower())

        # Si coincide con el target y aún no hemos fijado la cita principal
        if inner_norm == target_norm or not first_target_handled:
            parts.append(f"<cite>{inner.strip()}</cite>")
            first_target_handled = True
        else:
            # Mención secundaria o coreferencia secundaria
            parts.append("[OTHERCIT]")
        last_end = span.end()

    parts.append(text[last_end:])
    return "".join(parts)


def normalize_multicite_label(raw_label: str) -> str:
    """Mapea la etiqueta de MultiCite (simple o múltiple) al vocabulario unificado."""
    tokens = raw_label.strip().split()
    mapped_tokens = [MULTICITE_LABEL_MAP.get(tok.lower(), tok.capitalize()) for tok in tokens]
    return "; ".join(mapped_tokens) if mapped_tokens else "Background"


class MultiCiteAdapter:
    """Adaptador de ingestión y enriquecimiento para el dataset MultiCite."""

    def __init__(
        self,
        multicite_dir: Path,
        s2_client: SemanticScholarClient,
        known_arxiv_ids: Set[str],
        skip_network: bool = False,
    ) -> None:
        self.multicite_dir = multicite_dir
        self.s2_client = s2_client
        self.known_arxiv_ids = known_arxiv_ids
        self.skip_network = skip_network

    def _resolve_via_s2(
        self, raw_mention: str, author: Optional[str], year: Optional[int]
    ) -> Optional[S2PaperMetadata]:
        """Consulta Semantic Scholar con caché SQLite para resolver metadatos bibliográficos."""
        clean_m = re.sub(r"^[(\[\s]+", "", raw_mention)
        clean_m = re.sub(r"[)\]\s,;:]+$", "", clean_m).strip().lower()

        # Prioridad 1: Marcador de citación exacto pre-calentado en s2_cache.db
        candidate_keys = [
            f"search:{clean_m}",
            f"search:{clean_m.replace(',', '')}",
        ]
        if author and year:
            candidate_keys.append(f"search:{author.lower()} {year}")

        for k in candidate_keys:
            is_cached, cached = self.s2_client._get_from_cache(k)
            if is_cached:
                return cached

        # Si skip_network es True, no realizar llamadas HTTP
        if self.skip_network:
            return None

        # Filtro estricto de ruido: descartar menciones sin autor o sin año canónico (ej. '5,', '[16,', '9]', 'same approach')
        if not (author and year and len(author) >= 2 and 1950 <= year <= 2026):
            # Guardar tombstone en caché para no volver a evaluar esta mención con ruido
            for k in candidate_keys:
                self.s2_client._save_to_cache(k, None)
            return None

        query = f"{author} {year}"
        metadata = self.s2_client.search_paper_by_title(query)
        # Asociar resultado (o tombstone) a todas las claves candidatas para cache hits directos O(1)
        for k in candidate_keys:
            self.s2_client._save_to_cache(k, metadata)

        return metadata

    def process_split(self, split_name: str, limit: Optional[int] = None) -> Generator[CitingContext, None, None]:
        """Procesa una partición (train.json, dev.json o test.json) de MultiCite."""
        file_path = self.multicite_dir / f"{split_name}.json"
        if not file_path.exists():
            logger.warning("Partición de MultiCite %s no encontrada en %s.", split_name, file_path)
            return

        logger.info("Procesando partición MultiCite '%s' desde %s...", split_name, file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for item in data:
            if limit and count >= limit:
                break

            item_id = item.get("id", f"mc_{split_name}_{count}")
            raw_x = item.get("x", "")
            raw_y = item.get("y", "background")

            # Normalizar x a string
            if isinstance(raw_x, list):
                text = " ".join(str(s) for s in raw_x)
            else:
                text = str(raw_x)

            # Extraer sección si existe en formato **HEADING**
            rhetorical_section, text_clean = extract_rhetorical_section_from_text(text)

            # Extraer mención citante entre <cite>...</cite>
            cite_matches = re.findall(r"<cite>(.*?)</cite>", text_clean, flags=re.DOTALL)
            target_mention = cite_matches[0].strip() if cite_matches else "citation"
            author, year = parse_multicite_surface(target_mention)

            # Enmascarar citas secundarias
            masked_context = mask_secondary_citations(text_clean, target_mention)

            # Resolver mediante Semantic Scholar / s2_cache.db
            s2_meta = self._resolve_via_s2(target_mention, author, year)

            target_cited_id: Optional[str] = None
            target_cited_title: Optional[str] = None
            target_cited_authors: Optional[str] = None
            target_cited_year: Optional[int] = year
            target_cited_abstract: Optional[str] = None
            has_cited_chunks: bool = False

            if s2_meta:
                target_cited_id = s2_meta.arxiv_id
                target_cited_title = s2_meta.title
                target_cited_authors = ", ".join(s2_meta.authors) if s2_meta.authors else None
                target_cited_year = s2_meta.year or year
                target_cited_abstract = s2_meta.abstract
                if target_cited_id and target_cited_id in self.known_arxiv_ids:
                    has_cited_chunks = True
            else:
                target_cited_authors = author

            citing_paper_id = item_id.split("_")[0]

            context_obj = CitingContext(
                context_id=f"mc_{split_name}_{item_id}",
                source_dataset="multicite",
                split=split_name,
                citing_paper_id=citing_paper_id,
                rhetorical_section=rhetorical_section,
                window_rule="m_3_sentences",
                citation_context=masked_context,
                target_ref_key=target_mention,
                target_cited_id=target_cited_id,
                target_cited_title=target_cited_title,
                target_cited_authors=target_cited_authors,
                target_cited_year=target_cited_year,
                target_cited_abstract=target_cited_abstract,
                has_cited_chunks=has_cited_chunks,
                original_label=raw_y,
                normalized_label=normalize_multicite_label(raw_y),
                label_source="human_gold",
                is_gold=True,
            )

            count += 1
            yield context_obj


class UnarxiveMiner:
    """Minador nativo de oraciones citantes sobre data/structured/papers.jsonl."""

    def __init__(
        self,
        known_arxiv_ids: Set[str],
        known_abstracts: Optional[Dict[str, str]] = None,
        known_titles: Optional[Dict[str, str]] = None,
    ) -> None:
        self.known_arxiv_ids = known_arxiv_ids
        self.known_abstracts = known_abstracts or {}
        self.known_titles = known_titles or {}

    @staticmethod
    def match_orphan_class(text: str) -> Optional[str]:
        """Aplica las expresiones regulares sobre el texto para detectar las 4 clases huérfanas."""
        for label, pattern in ORPHAN_CLASS_PATTERNS.items():
            if pattern.search(text):
                return label
        return None

    def mine_section(
        self,
        paper: StructuredPaper,
        section: StructuredSection,
    ) -> Generator[CitingContext, None, None]:
        """Extrae contextos citantes con ventana m=3 oraciones dentro de una sección estructurada."""
        sentences = section.sentences
        n_sents = len(sentences)

        for sent_idx_in_sec, sentence in enumerate(sentences):
            if not sentence.citations:
                continue

            # Evaluar si la oración contiene patrones de clases huérfanas
            matched_label = self.match_orphan_class(sentence.text)
            if not matched_label:
                continue

            for cite_marker in sentence.citations:
                target_ref = cite_marker.ref_key
                bib_entry: Optional[BibliographyEntry] = paper.bibliography.get(target_ref)

                # Definir ventana estricta m=3 oraciones respetando límites de la sección:
                # - Si la cita está al inicio (i=0): ventana hacia adelante [s_i, s_{i+1}, s_{i+2}]
                # - Si la cita está al final (i=n-1): ventana hacia atrás [s_{i-2}, s_{i-1}, s_i]
                # - Interior: ventana centrada [s_{i-1}, s_i, s_{i+1}]
                # - Si la sección tiene menos de 3 oraciones: toma todas las oraciones disponibles
                if n_sents >= 3:
                    if sent_idx_in_sec == 0:
                        w_start, w_end = 0, 3
                    elif sent_idx_in_sec == n_sents - 1:
                        w_start, w_end = n_sents - 3, n_sents
                    else:
                        w_start, w_end = sent_idx_in_sec - 1, sent_idx_in_sec + 2
                else:
                    w_start, w_end = 0, n_sents

                window_sents = sentences[w_start:w_end]

                # Construir texto de la ventana reemplazando citas
                window_parts: List[str] = []
                for s_curr in window_sents:
                    s_text = s_curr.text
                    if s_curr.sent_idx == sentence.sent_idx:
                        # Oración objetivo: sustituir la cita objetivo por <cite>...</cite>
                        surface_title = bib_entry.title if bib_entry else "citation"
                        surface_repr = f"{bib_entry.authors or 'Author'}, {bib_entry.year or 'Year'}" if bib_entry and bib_entry.authors else surface_title
                        s_text = s_text.replace(cite_marker.marker_text, f"<cite>{surface_repr}</cite>")

                    # Reemplazar cualquier otra cita secundaria en la ventana por [OTHERCIT]
                    s_text = re.sub(r"\{\{cite:[a-f0-9]+\}\}", "[OTHERCIT]", s_text)
                    window_parts.append(s_text.strip())

                citation_context = " ".join(window_parts)

                target_cited_id = bib_entry.arxiv_id if bib_entry else None
                has_cited_chunks = bool(target_cited_id and target_cited_id in self.known_arxiv_ids)

                target_cited_title = (bib_entry.title if bib_entry and bib_entry.title else None) or self.known_titles.get(target_cited_id or "")
                target_cited_abstract = self.known_abstracts.get(target_cited_id or "") if has_cited_chunks else None

                context_id = f"unx_{paper.paper_id}_s{section.section_idx}_sent{sentence.sent_idx}_{target_ref[:8]}"

                yield CitingContext(
                    context_id=context_id,
                    source_dataset="unarxive_mined",
                    split=None,
                    citing_paper_id=paper.paper_id,
                    rhetorical_section=section.heading,
                    window_rule="m_3_sentences",
                    citation_context=citation_context,
                    target_ref_key=target_ref,
                    target_cited_id=target_cited_id,
                    target_cited_title=target_cited_title,
                    target_cited_authors=bib_entry.authors if bib_entry else None,
                    target_cited_year=bib_entry.year if bib_entry else None,
                    target_cited_abstract=target_cited_abstract,
                    has_cited_chunks=has_cited_chunks,
                    original_label=matched_label,
                    normalized_label=matched_label,
                    label_source="heuristic_rule",
                    is_gold=False,
                )

    def mine_paper(self, paper: StructuredPaper) -> Generator[CitingContext, None, None]:
        """Itera por todas las secciones del artículo estructurado."""
        for section in paper.sections:
            yield from self.mine_section(paper, section)


class MiningPipeline:
    """Orquestador principal de extracción y consolidación hacia data/processed/citing_contexts.jsonl."""

    def __init__(self, config: MiningConfig) -> None:
        self.config = config
        self.known_arxiv_ids: Set[str] = set()
        self.known_abstracts: Dict[str, str] = {}
        self.known_titles: Dict[str, str] = {}

    def run(self) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de minado con streaming y validación Pydantic."""
        out_path = Path(self.config.output_file)
        if out_path.exists() and not self.config.force:
            raise FileExistsError(
                f"El archivo de salida {out_path} ya existe. Usa --force para sobreescribir."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Cargar índice de arXiv IDs conocidos y metadatos
        self.known_arxiv_ids, self.known_abstracts, self.known_titles = load_known_arxiv_metadata(self.config.papers_file)

        # 2. Inicializar cliente de Semantic Scholar
        s2_config = S2ClientConfig(
            cache_path=self.config.s2_cache_path,
        )
        s2_client = SemanticScholarClient(s2_config)

        # 3. Inicializar adaptadores
        mc_adapter = MultiCiteAdapter(
            multicite_dir=self.config.multicite_dir,
            s2_client=s2_client,
            known_arxiv_ids=self.known_arxiv_ids,
            skip_network=self.config.skip_network,
        )
        unarxive_miner = UnarxiveMiner(
            known_arxiv_ids=self.known_arxiv_ids,
            known_abstracts=self.known_abstracts,
            known_titles=self.known_titles,
        )

        stats: Dict[str, Any] = {
            "total_records": 0,
            "multicite_records": 0,
            "unarxive_records": 0,
            "has_cited_chunks_true": 0,
            "label_distribution": {},
        }

        t_start = time.time()
        logger.info("Iniciando generación de contextos citantes hacia %s...", out_path)

        with open(out_path, "w", encoding="utf-8") as out_f:
            # --- Vertiente A: MultiCite ---
            splits = ["train", "dev", "test"]
            mc_total_processed = 0

            for split in splits:
                if self.config.max_multicite and mc_total_processed >= self.config.max_multicite:
                    break

                split_limit = None
                if self.config.max_multicite:
                    split_limit = self.config.max_multicite - mc_total_processed

                for context in mc_adapter.process_split(split, limit=split_limit):
                    out_f.write(context.model_dump_json() + "\n")
                    stats["total_records"] += 1
                    stats["multicite_records"] += 1
                    mc_total_processed += 1
                    if context.has_cited_chunks:
                        stats["has_cited_chunks_true"] += 1

                    lbl = context.normalized_label or "Unknown"
                    stats["label_distribution"][lbl] = stats["label_distribution"].get(lbl, 0) + 1

                    if stats["total_records"] % 1000 == 0:
                        logger.info("Progreso minado: %d registros procesados...", stats["total_records"])

            # --- Vertiente B: unarXive ---
            unx_processed = 0
            if self.config.papers_file.exists():
                logger.info("Minando oraciones para clases huérfanas desde %s...", self.config.papers_file)
                with open(self.config.papers_file, "r", encoding="utf-8") as p_file:
                    for line in p_file:
                        if self.config.max_unarxive and unx_processed >= self.config.max_unarxive:
                            break

                        try:
                            paper_dict = json.loads(line)
                            paper = StructuredPaper(**paper_dict)
                        except Exception as exc:
                            logger.debug("Error deserializando StructuredPaper: %s", exc)
                            continue

                        for context in unarxive_miner.mine_paper(paper):
                            if self.config.max_unarxive and unx_processed >= self.config.max_unarxive:
                                break
                            out_f.write(context.model_dump_json() + "\n")
                            stats["total_records"] += 1
                            stats["unarxive_records"] += 1
                            unx_processed += 1
                            if context.has_cited_chunks:
                                stats["has_cited_chunks_true"] += 1

                            lbl = context.normalized_label or "Unknown"
                            stats["label_distribution"][lbl] = stats["label_distribution"].get(lbl, 0) + 1

                            if stats["total_records"] % 1000 == 0:
                                logger.info("Progreso minado: %d registros procesados...", stats["total_records"])

        elapsed = time.time() - t_start
        stats["elapsed_seconds"] = round(elapsed, 2)
        logger.info(
            "Minado completado exitosamente: %d registros generados en %.2fs (MultiCite: %d, unarXive: %d).",
            stats["total_records"],
            elapsed,
            stats["multicite_records"],
            stats["unarxive_records"],
        )
        return stats


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline de minado y unificación de contextos citantes hacia data/processed/citing_contexts.jsonl"
    )
    parser.add_argument(
        "--multicite-dir",
        type=Path,
        default=Path("data/raw/multicite"),
        help="Directorio de las particiones MultiCite",
    )
    parser.add_argument(
        "--papers-file",
        type=Path,
        default=Path("data/structured/papers.jsonl"),
        help="Ruta al archivo canónico de unarXive",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/citing_contexts.jsonl"),
        help="Ruta destino para citing_contexts.jsonl",
    )
    parser.add_argument(
        "--s2-cache",
        type=Path,
        default=Path("data/raw/s2_cache.db"),
        help="Ruta a la base de datos de caché SQLite de Semantic Scholar",
    )
    parser.add_argument(
        "--max-multicite",
        type=int,
        default=None,
        help="Límite máximo de instancias MultiCite a procesar (para pruebas)",
    )
    parser.add_argument(
        "--max-unarxive",
        type=int,
        default=None,
        help="Límite máximo de instancias unarXive a procesar",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Si se activa, no realiza peticiones HTTP a Semantic Scholar y sólo consulta la caché SQLite local",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza la sobreescritura del archivo de salida si ya existe",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal para CLI."""
    args = parse_args()
    config = MiningConfig(
        multicite_dir=args.multicite_dir,
        papers_file=args.papers_file,
        output_file=args.output_file,
        s2_cache_path=args.s2_cache,
        max_multicite=args.max_multicite,
        max_unarxive=args.max_unarxive,
        skip_network=args.skip_network,
        force=args.force,
    )
    pipeline = MiningPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
