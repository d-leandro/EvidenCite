"""Módulo de estructuración de artículos científicos desde unarXive hacia data/structured/papers.jsonl.

Transforma shards crudos JSONL de unarXive en el corpus canónico unificado:
- Filtra exclusivamente artículos de Computación (Computer Science / cs.*).
- Preserva fielmente los encabezados de sección originales.
- Resuelve marcadores de fórmulas a su representación LaTeX y remueve flotantes de tablas/figuras.
- Pre-segmenta párrafos en oraciones atómicas respetando abreviaturas científicas.
- Ancla menciones explícitas de citas (cite_spans) con offsets exactos dentro de su oración.
- Normaliza la bibliografía extrayendo títulos limpios y metadatos (arXiv ID, DOI, OpenAlex).
- Opera en streaming O(1) en RAM con soporte de anexión incremental (--append).
"""

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from src.data.schemas import (
    BibliographyEntry,
    CitationMarker,
    StructuredPaper,
    StructuredSection,
    StructuredSentence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Expresiones regulares para filtrado y extracción
CS_CATEGORY_PATTERN = re.compile(r"(?:^|\s)cs\.[A-Za-z\-]+", re.IGNORECASE)
CITE_REGEX = re.compile(r"\{\{cite:([^\}]+)\}\}")
FORMULA_REGEX = re.compile(r"\{\{formula:([^\}]+)\}\}")
FLOAT_REGEX = re.compile(r"\{\{(table|figure):([^\}]+)\}\}")

# Abreviaturas científicas comunes para no fracturar oraciones
SCIENTIFIC_ABBREVIATIONS = {
    "al",
    "e.g",
    "i.e",
    "eg",
    "ie",
    "fig",
    "figs",
    "tab",
    "tabs",
    "eq",
    "eqs",
    "sec",
    "secs",
    "ref",
    "refs",
    "approx",
    "cf",
    "vs",
    "dept",
    "dr",
    "prof",
    "mr",
    "mrs",
    "ms",
    "no",
    "vol",
    "pp",
    "ed",
    "eds",
    "thm",
    "lem",
    "prop",
    "cor",
    "defn",
}


class StructuringConfig(BaseModel):
    """Configuración para el proceso de estructuración de unarXive."""

    input_dir: Path = Field(
        default=Path("data/raw/unarxive"),
        description="Directorio que contiene los shards crudos JSONL de unarXive",
    )
    output_file: Path = Field(
        default=Path("data/structured/papers.jsonl"),
        description="Ruta de destino del corpus canónico estructurado",
    )
    force: bool = Field(
        default=False,
        description="Sobreescribe el archivo destino si ya existe",
    )
    append: bool = Field(
        default=False,
        description="Anexa los artículos al archivo destino si ya existe (útil para fallbacks)",
    )
    limit: Optional[int] = Field(
        default=None,
        description="Número máximo opcional de artículos a estructurar",
    )
    years: Optional[List[str]] = Field(
        default=None,
        description="Años específicos a estructurar (ej. ['2015', '2016']). Si es None, procesa todos los shards.",
    )


def is_computer_science_paper(record: Dict[str, Any]) -> bool:
    """Evalúa si un artículo pertenece a Ciencias de la Computación (Computer Science).

    Criterios:
    1. discipline == 'Computer Science', O
    2. Alguna categoría en metadata.categories contiene el prefijo 'cs.'.
    """
    if record.get("discipline") == "Computer Science":
        return True

    categories = record.get("metadata", {}).get("categories", "")
    if isinstance(categories, str) and CS_CATEGORY_PATTERN.search(categories):
        return True

    return False


def clean_whitespace(text: str) -> str:
    """Normaliza espacios en blanco múltiples, tabulaciones y saltos de línea."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_bib_entry_raw(raw_text: str) -> Dict[str, Any]:
    """Extrae heurísticamente autores, año y título desde una referencia bib_entry_raw.

    Aplica una estrategia múltiple según el estilo de citación:
    1. IEEE/LNCS: Título entre comillas (\"Título\" o 'Título' o ``Título'').
    2. ACM/APA/Harvard: Delimitador de año explícito en posición 2 (ej. 'Autores. 2012. Título...').
    3. Año en paréntesis tras autores (ej. 'Autores (2012). Título...').
    4. Estilo estándar con año al final o en revista (ej. 'Autores. Título. Revista, 2016.').
    """
    cleaned = clean_whitespace(raw_text)
    if not cleaned:
        return {"title": "", "authors": None, "year": None, "raw_text": ""}

    # Remover corchetes de numeración inicial tipo [1] o [AB20] si existieran
    cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned)

    # Estrategia 1: Título entre comillas (común en IEEE, ACM, LNCS)
    q_match = re.search(r'(?:[\"“\']|``)([^\"“”\']{6,})(?:[\"”\']|\'\')', cleaned)
    if q_match:
        authors = cleaned[: q_match.start()].strip().rstrip(",.")
        title = q_match.group(1).strip().rstrip(",.")
        year_m = re.search(r"\b(19\d{2}|20\d{2})\b", cleaned[q_match.end() :]) or re.search(
            r"\b(19\d{2}|20\d{2})\b", cleaned
        )
        year = int(year_m.group(0)) if year_m else None
        return {"title": title, "authors": authors or None, "year": year, "raw_text": cleaned}

    # División por puntos respetando iniciales de autores (ej. "A. B. Smith")
    parts = [p.strip() for p in re.split(r"(?<!\b[A-Z])\.\s+", cleaned) if p.strip()]

    # Estrategia 2: Año explícito en parts[1] (estilo ACM/APA: 'Autores. 2012. Título...')
    if len(parts) >= 3 and re.match(r"^\(?(?:19|20)\d{2}[a-z]?\)?$", parts[1]):
        authors = parts[0].strip().rstrip(",.")
        year_m = re.search(r"(19\d{2}|20\d{2})", parts[1])
        year = int(year_m.group(0)) if year_m else None
        title = parts[2].rstrip(".")
        return {"title": title, "authors": authors or None, "year": year, "raw_text": cleaned}

    # Estrategia 3: Año en paréntesis dentro de los autores (ej. 'Autores (2019). Título...')
    if parts and re.search(r"\((19\d{2}|20\d{2})[a-z]?\)", parts[0]):
        year_m = re.search(r"\((19\d{2}|20\d{2})[a-z]?\)", parts[0])
        year = int(year_m.group(1))
        authors = re.sub(r"\s*\((19\d{2}|20\d{2})[a-z]?\)", "", parts[0]).strip().rstrip(",.")
        title = parts[1].rstrip(".") if len(parts) >= 2 else cleaned.rstrip(".")
        return {"title": title, "authors": authors or None, "year": year, "raw_text": cleaned}

    # Estrategia 4: Formato estándar con año al final (ej. 'Autores. Título. Conferencia, 2016.')
    year_m = re.search(r"\b(19\d{2}|20\d{2})\b", cleaned)
    year = int(year_m.group(0)) if year_m else None

    if len(parts) >= 2:
        return {
            "title": parts[1].rstrip("."),
            "authors": parts[0].rstrip(",.") or None,
            "year": year,
            "raw_text": cleaned,
        }

    return {"title": cleaned.rstrip("."), "authors": None, "year": year, "raw_text": cleaned}


def extract_title_from_raw(raw_text: str) -> str:
    """Extrae heurísticamente el título del artículo desde la cadena bib_entry_raw."""
    parsed = parse_bib_entry_raw(raw_text)
    return parsed.get("title") or clean_whitespace(raw_text).rstrip(".")


def resolve_ref_entries(text: str, ref_entries: Dict[str, Any]) -> str:
    """Reemplaza marcadores de fórmulas por su código LaTeX y remueve tags flotantes de tablas/figuras.

    - {{formula:<uuid>}} -> '$latex$' si ref_entries[uuid] contiene código LaTeX, o '' si está vacío.
    - {{table:<uuid>}} / {{figure:<uuid>}} -> '' (eliminación de tag flotante sin contenido).
    """
    if not text:
        return ""

    def replace_formula(m: re.Match) -> str:
        uuid = m.group(1)
        entry = ref_entries.get(uuid, {})
        latex = entry.get("latex", "").strip() if isinstance(entry, dict) else ""
        if latex:
            return f"${latex}$"
        return ""

    text = FORMULA_REGEX.sub(replace_formula, text)
    text = FLOAT_REGEX.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    """Segmenta un párrafo en oraciones sin fracturar ante abreviaturas científicas o expresiones matemáticas."""
    cleaned = clean_whitespace(text)
    if not cleaned:
        return []

    pattern = re.compile(r"([.!?])\s+")
    sentences: List[str] = []
    start = 0

    for m in pattern.finditer(cleaned):
        punct_pos = m.start()
        prefix = cleaned[start:punct_pos].strip()
        words = prefix.split()
        if not words:
            continue

        last_word = words[-1].lower().rstrip(".!?")
        if last_word in SCIENTIFIC_ABBREVIATIONS:
            continue

        next_char_pos = m.end()
        if (
            next_char_pos < len(cleaned)
            and not cleaned[next_char_pos].isupper()
            and cleaned[next_char_pos] not in "\"'(["
        ):
            continue

        sentence = cleaned[start : punct_pos + 1].strip()
        if sentence:
            sentences.append(sentence)
        start = m.end()

    last_sentence = cleaned[start:].strip()
    if last_sentence:
        sentences.append(last_sentence)

    return sentences


def structure_single_paper(record: Dict[str, Any]) -> Optional[StructuredPaper]:
    """Estructura un único registro crudo de unarXive en el modelo StructuredPaper.

    Retorna None si el artículo no pertenece a Ciencias de la Computación o si carece de texto.
    """
    if not is_computer_science_paper(record):
        return None

    paper_id = record.get("paper_id") or record.get("metadata", {}).get("id") or ""
    if not paper_id:
        return None

    # Título y abstract
    title = clean_whitespace(record.get("metadata", {}).get("title", ""))
    abstract_obj = record.get("abstract") or {}
    if isinstance(abstract_obj, dict):
        abstract_text = abstract_obj.get("text", "")
    else:
        abstract_text = str(abstract_obj)
    abstract = clean_whitespace(abstract_text or record.get("metadata", {}).get("abstract", ""))

    # Categorías
    cats_raw = record.get("metadata", {}).get("categories", "")
    categories = cats_raw.split() if isinstance(cats_raw, str) else []

    # Bibliografía
    ref_entries = record.get("ref_entries") or {}
    bib_entries = record.get("bib_entries") or {}
    structured_bib: Dict[str, BibliographyEntry] = {}

    for ref_key, entry in bib_entries.items():
        if not isinstance(entry, dict):
            continue

        raw_text = clean_whitespace(entry.get("bib_entry_raw", ""))
        parsed_bib = parse_bib_entry_raw(raw_text)
        ids = entry.get("ids") or {}

        arxiv_id = ids.get("arxiv_id")
        if arxiv_id:
            arxiv_id = arxiv_id.strip()
            if arxiv_id.lower().startswith("arxiv:"):
                arxiv_id = arxiv_id[6:].strip()
            if not arxiv_id:
                arxiv_id = None
        else:
            arxiv_id = None

        doi = ids.get("doi")
        if doi and isinstance(doi, str):
            doi = doi.strip() or None
        else:
            doi = None

        open_alex_id = ids.get("open_alex_id")
        if open_alex_id and isinstance(open_alex_id, str):
            open_alex_id = open_alex_id.strip() or None
        else:
            open_alex_id = None

        structured_bib[ref_key] = BibliographyEntry(
            title=parsed_bib["title"] or raw_text,
            authors=parsed_bib["authors"],
            year=parsed_bib["year"],
            raw_text=raw_text,
            arxiv_id=arxiv_id,
            doi=doi,
            open_alex_id=open_alex_id,
        )

    # Secciones y oraciones
    body_text = record.get("body_text") or []
    structured_sections: List[StructuredSection] = []

    current_heading: Optional[str] = None
    current_sentences: List[StructuredSentence] = []
    section_counter = 0
    sentence_counter = 0

    for paragraph in body_text:
        if not isinstance(paragraph, dict):
            continue

        raw_heading_val = paragraph.get("section")
        # Nearest Enclosing Scope: Si el párrafo no trae encabezado explícito, hereda el encabezado activo previo
        if raw_heading_val and clean_whitespace(str(raw_heading_val)):
            effective_heading = clean_whitespace(str(raw_heading_val))
        else:
            effective_heading = current_heading or "Preamble"

        p_text = paragraph.get("text", "")
        cleaned_p_text = resolve_ref_entries(p_text, ref_entries)

        if not cleaned_p_text:
            continue

        # Cambio de sección lógica (agrupar párrafos consecutivos del mismo encabezado efectivo)
        if current_heading is not None and effective_heading != current_heading:
            if current_sentences:
                structured_sections.append(
                    StructuredSection(
                        section_idx=section_counter,
                        heading=current_heading,
                        sentences=current_sentences,
                    )
                )
                section_counter += 1
                current_sentences = []

        current_heading = effective_heading
        p_sentences = split_sentences(cleaned_p_text)

        for sent_text in p_sentences:
            # Anclaje de citas presentes en esta oración
            citations: List[CitationMarker] = []
            for m in CITE_REGEX.finditer(sent_text):
                citations.append(
                    CitationMarker(
                        marker_text=m.group(0),
                        char_start=m.start(),
                        char_end=m.end(),
                        ref_key=m.group(1),
                    )
                )

            current_sentences.append(
                StructuredSentence(
                    sent_idx=sentence_counter,
                    text=sent_text,
                    citations=citations,
                )
            )
            sentence_counter += 1

    # Guardar la última sección
    if current_heading and current_sentences:
        structured_sections.append(
            StructuredSection(
                section_idx=section_counter,
                heading=current_heading,
                sentences=current_sentences,
            )
        )

    return StructuredPaper(
        paper_id=paper_id,
        id_source="arxiv",
        title=title,
        abstract=abstract,
        discipline="Computer Science",
        categories=categories,
        sections=structured_sections,
        bibliography=structured_bib,
    )


class UnarxiveStructurer:
    """Orquestador para estructurar shards de unarXive hacia el corpus canónico."""

    def __init__(self, config: Optional[StructuringConfig] = None) -> None:
        self.config = config or StructuringConfig()

    def process_shard(self, shard_path: Path) -> Iterator[StructuredPaper]:
        """Lee un archivo shard JSONL línea por línea en streaming y emite objetos StructuredPaper."""
        logger.info("Procesando shard crudo: %s", shard_path.name)
        with open(shard_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                    structured = structure_single_paper(record)
                    if structured:
                        yield structured
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning("Error parseando línea %d de %s: %s", line_idx, shard_path.name, exc)

    def run(self) -> Dict[str, int]:
        """Ejecuta el pipeline de estructuración procesando los shards del directorio de entrada."""
        output_file = Path(self.config.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists() and not self.config.force and not self.config.append:
            logger.info("El archivo %s ya existe. Usa --force o --append. Omitiendo ejecución.", output_file)
            return {"structured_papers": 0, "status": "skipped"}

        mode = "a" if self.config.append else "w"
        input_dir = Path(self.config.input_dir)
        shard_files = sorted(list(input_dir.glob("*.jsonl")))

        if self.config.years:
            year_suffixes = [y[-2:] for y in self.config.years]
            shard_files = [
                s for s in shard_files
                if any(f"_{y}" in s.name or f"arXiv_src_{y}" in s.name for y in year_suffixes)
            ]

        if not shard_files:
            logger.warning("No se encontraron archivos .jsonl en %s coincidentes con los filtros", input_dir)
            return {"structured_papers": 0, "status": "no_shards_found"}

        count = 0
        logger.info(
            "Iniciando estructuración de %d shard(s) hacia %s (modo: '%s')",
            len(shard_files),
            output_file,
            mode,
        )

        with open(output_file, mode, encoding="utf-8") as out_f:
            for shard_path in shard_files:
                for paper in self.process_shard(shard_path):
                    line_data = paper.model_dump_json()
                    out_f.write(line_data + "\n")
                    count += 1
                    if count % 100 == 0:
                        logger.info("Artículos estructurados hasta el momento: %d", count)
                    if self.config.limit and count >= self.config.limit:
                        logger.info("Alcanzado límite de %d artículos. Finalizando.", self.config.limit)
                        return {"structured_papers": count, "status": "limit_reached"}

        logger.info("Estructuración completada exitosamente. Total artículos estructurados: %d", count)
        return {"structured_papers": count, "status": "success"}


def main() -> None:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Estructura shards crudos de unarXive a data/structured/papers.jsonl"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/unarxive"),
        help="Directorio con shards JSONL de unarXive",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/structured/papers.jsonl"),
        help="Ruta al archivo canónico resultante",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobreescribe el archivo destino si ya existe",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Anexa registros al archivo existente",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Límite máximo de artículos a estructurar",
    )

    parser.add_argument(
        "--years",
        nargs="+",
        default=None,
        help="Años a estructurar (ej. 2015 2016 2017). Si se omite, estructura todos los shards.",
    )

    args = parser.parse_args()
    config = StructuringConfig(
        input_dir=args.input_dir,
        output_file=args.output_file,
        force=args.force,
        append=args.append,
        limit=args.limit,
        years=args.years,
    )
    structurer = UnarxiveStructurer(config)
    structurer.run()


if __name__ == "__main__":
    main()
