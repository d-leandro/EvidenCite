"""Resolutor y extractor de artículos científicos desde arXiv hacia el esquema canónico StructuredPaper.

Implementa un flujo de dos niveles (Tier 1: ar5iv HTML, Tier 2: Docling PDF) con cortesía
estricta de red (rate limit de 3s para arXiv API), validación de complejidad con pypdfium2
y segmentación atómica de oraciones compatible con el contrato canónico del proyecto.
"""

from enum import Enum
import json
import logging
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import httpx
from pydantic import BaseModel, Field

from src.data.schemas import (
    StructuredPaper,
    StructuredSection,
    StructuredSentence,
)
from src.data.structuring import split_sentences

logger = logging.getLogger(__name__)

# Namespaces para parseo de respuestas Atom XML de arXiv API
ARXIV_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Secciones no discursivas a excluir del corpus de texto
EXCLUDED_HEADINGS = {
    "references",
    "bibliography",
    "works cited",
    "literature cited",
    "acknowledgments",
    "acknowledgements",
    "author contributions",
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "funding",
    "data availability",
}


class ExtractionSource(str, Enum):
    """Fuente técnica utilizada para extraer el texto estructurado."""

    AR5IV_HTML = "ar5iv_html"
    DOCLING_PDF = "docling_pdf"


class ArxivMetadata(BaseModel):
    """Metadatos bibliográficos obtenidos de la API oficial de arXiv."""

    arxiv_id: str = Field(..., description="Identificador normalizado (ej. '2105.05862')")
    title: str = Field(..., description="Título del artículo")
    authors: List[str] = Field(default_factory=list, description="Lista de autores")
    abstract: str = Field(..., description="Texto completo del abstract")
    categories: List[str] = Field(default_factory=list, description="Categorías temáticas (ej. ['cs.CL', 'cs.AI'])")
    published_date: Optional[str] = Field(default=None, description="Fecha de publicación en formato ISO")
    pdf_url: str = Field(..., description="URL oficial de descarga del PDF")


class ExtractedRawSection(BaseModel):
    """Representación intermedia de una sección extraída antes de la segmentación."""

    heading: str = Field(..., description="Encabezado original del autor")
    raw_text: str = Field(..., description="Texto de los párrafos de la sección")


class ArxivResolverConfig(BaseModel):
    """Configuración operativa para la resolución de preprints de arXiv."""

    cache_dir: Path = Field(default=Path("data/raw/arxiv_cache"), description="Directorio para caché local de PDFs")
    rate_limit_delay: float = Field(default=3.5, description="Retraso mínimo en segundos entre peticiones a arXiv API")
    max_pages: int = Field(default=20, description="Límite máximo de páginas para PDF (descarte si es mayor)")
    max_file_size_mb: int = Field(default=20, description="Tamaño máximo de archivo PDF en MB")
    timeout_seconds: int = Field(default=30, description="Timeout en segundos para peticiones de red")
    max_retries: int = Field(default=5, description="Número máximo de reintentos ante errores transitorios")
    user_agent: str = Field(
        default="EvidenCite-Research/1.0 (academic research; mailto:contact@example.org)",
        description="Encabezado User-Agent para peticiones web",
    )


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normaliza un identificador de arXiv removiendo prefijos y versiones."""
    clean = arxiv_id.strip()
    if clean.lower().startswith("arxiv:"):
        clean = clean[6:].strip()
    # Remover sufijos de versión (ej. '2101.00098v2' -> '2101.00098')
    return clean.split("v")[0]


def is_excluded_heading(heading: str) -> bool:
    """Evalúa si un encabezado corresponde a metadatos, bibliografía o agradecimientos."""
    norm = re.sub(r"^\d+(\.\d+)*\s*", "", heading.strip().lower())
    norm = re.sub(r"[^\w\s]", "", norm).strip()
    return norm in EXCLUDED_HEADINGS


class ArxivClient:
    """Cliente para la API REST/Atom de arXiv y descarga gestionada de PDFs."""

    def __init__(self, config: Optional[ArxivResolverConfig] = None) -> None:
        self.config = config or ArxivResolverConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time: Optional[float] = None

    def _enforce_rate_limit(self) -> None:
        """Aplica cortesía estricta de red respetando el delay mínimo entre peticiones con jitter."""
        if self._last_request_time is not None:
            jitter = random.uniform(0.2, 0.8)
            required_delay = self.config.rate_limit_delay + jitter
            elapsed = time.time() - self._last_request_time
            if elapsed < required_delay:
                time.sleep(required_delay - elapsed)
        self._last_request_time = time.time()

    def fetch_metadata(self, arxiv_id: str) -> Optional[ArxivMetadata]:
        """Consulta metadatos bibliográficos oficiales desde la API de arXiv.

        Args:
            arxiv_id: Identificador canónico del artículo (ej. '1706.03762').

        Returns:
            ArxivMetadata o None si no fue encontrado o hubo error insuperable.
        """
        clean_id = normalize_arxiv_id(arxiv_id)
        url = f"https://export.arxiv.org/api/query?id_list={clean_id}&max_results=1"

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._enforce_rate_limit()
                headers = {"User-Agent": self.config.user_agent}
                with httpx.Client(timeout=self.config.timeout_seconds, follow_redirects=True) as client:
                    resp = client.get(url, headers=headers)
                    resp.raise_for_status()
                    xml_data = resp.text

                return self._parse_atom_xml(clean_id, xml_data)

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (404, 400):
                    logger.warning("Paper no encontrado en arXiv API (HTTP %s): %s", status, clean_id)
                    return None
                if status == 429:
                    wait_time = max(20.0, 20.0 * attempt)
                    logger.warning(
                        "Rate limit (HTTP 429) en arXiv API para %s en intento %d/%d. Enfriando por %.1fs antes de reintentar...",
                        clean_id,
                        attempt,
                        self.config.max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    self._last_request_time = time.time()
                    continue

                logger.warning(
                    "Intento %d/%d: Error HTTP %s al consultar arXiv para %s",
                    attempt,
                    self.config.max_retries,
                    status,
                    clean_id,
                )
            except Exception as exc:
                logger.warning(
                    "Intento %d/%d: Error al consultar arXiv API para %s: %s",
                    attempt,
                    self.config.max_retries,
                    clean_id,
                    exc,
                )

            time.sleep(2.0 * attempt)

        return None

    def _parse_atom_xml(self, clean_id: str, xml_data: str) -> Optional[ArxivMetadata]:
        """Parsea la respuesta XML Atom de la API de arXiv."""
        try:
            root = ET.fromstring(xml_data)
            entry = root.find("atom:entry", ARXIV_NAMESPACES)
            if entry is None:
                return None

            title_elem = entry.find("atom:title", ARXIV_NAMESPACES)
            if title_elem is None or not title_elem.text:
                return None

            title = title_elem.text.strip().replace("\n", " ")
            if title.lower() == "error":
                return None

            summary_elem = entry.find("atom:summary", ARXIV_NAMESPACES)
            abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""

            published_elem = entry.find("atom:published", ARXIV_NAMESPACES)
            published = published_elem.text.strip() if published_elem is not None and published_elem.text else None

            authors = []
            for author in entry.findall("atom:author", ARXIV_NAMESPACES):
                name_elem = author.find("atom:name", ARXIV_NAMESPACES)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            categories = []
            for cat in entry.findall("atom:category", ARXIV_NAMESPACES):
                term = cat.attrib.get("term")
                if term:
                    categories.append(term)

            pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
            for link in entry.findall("atom:link", ARXIV_NAMESPACES):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", pdf_url)
                    break

            if pdf_url.startswith("http://"):
                pdf_url = "https://" + pdf_url[7:]

            return ArxivMetadata(
                arxiv_id=clean_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published_date=published,
                pdf_url=pdf_url,
            )
        except Exception as exc:
            logger.error("Fallo al parsear respuesta Atom XML para %s: %s", clean_id, exc)
            return None

    def download_pdf(
        self,
        arxiv_id: str,
        pdf_url: Optional[str] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """Descarga el PDF oficial de arXiv y lo almacena en la caché local persistente.

        Args:
            arxiv_id: Identificador canónico del artículo.
            pdf_url: URL directa de descarga (opcional).
            force: Si True, fuerza la descarga ignorando la caché.

        Returns:
            Ruta Path al archivo PDF en caché o None si la descarga falló.
        """
        clean_id = normalize_arxiv_id(arxiv_id)
        safe_filename = clean_id.replace("/", "_") + ".pdf"
        target_path = self.config.cache_dir / safe_filename

        if target_path.exists() and target_path.stat().st_size > 0 and not force:
            logger.debug("Usando PDF en caché: %s", target_path)
            return target_path

        download_url = pdf_url or f"https://arxiv.org/pdf/{clean_id}.pdf"
        temp_path = self.config.cache_dir / f"{safe_filename}.tmp"

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._enforce_rate_limit()
                headers = {"User-Agent": self.config.user_agent}

                with httpx.Client(timeout=self.config.timeout_seconds, follow_redirects=True) as client:
                    resp = client.get(download_url, headers=headers)
                    resp.raise_for_status()

                    content = resp.content
                    if not content.startswith(b"%PDF-"):
                        logger.warning("El archivo descargado para %s no contiene cabecera %%PDF-.", clean_id)
                        return None

                    with open(temp_path, "wb") as f_out:
                        f_out.write(content)

                    temp_path.replace(target_path)
                    logger.info("PDF descargado exitosamente en caché: %s", target_path)
                    return target_path

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    wait_time = max(20.0, 20.0 * attempt)
                    logger.warning(
                        "Rate limit (HTTP 429) al descargar PDF para %s en intento %d/%d. Enfriando por %.1fs...",
                        clean_id,
                        attempt,
                        self.config.max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    self._last_request_time = time.time()
                    continue
                logger.warning(
                    "Intento %d/%d: Error HTTP %s al descargar PDF para %s",
                    attempt,
                    self.config.max_retries,
                    exc.response.status_code,
                    clean_id,
                )
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(
                    "Intento %d/%d: Error al descargar PDF para %s: %s",
                    attempt,
                    self.config.max_retries,
                    clean_id,
                    exc,
                )
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

            time.sleep(2.0 * attempt)

        return None


class DoclingPDFParser:
    """Parser de PDFs científicos basado en IBM Docling con inspección preventiva pypdfium2."""

    def __init__(self, max_pages: int = 20, max_file_size_mb: int = 20) -> None:
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._converter = None

    def validate_pdf(self, pdf_path: Path) -> bool:
        """Valida que el PDF exista, sea íntegro y no supere los límites de tamaño o páginas.

        Regla estricta: Si supera max_pages, se descarta totalmente (no se trunca).
        """
        try:
            if not pdf_path.exists():
                logger.error("El archivo PDF no existe: %s", pdf_path)
                return False

            file_size = pdf_path.stat().st_size
            if file_size == 0:
                logger.error("El archivo PDF está vacío: %s", pdf_path)
                return False

            if file_size > self.max_file_size_bytes:
                logger.warning(
                    "PDF %s excede límite de tamaño (%.1f MB > %.1f MB). Omitiendo.",
                    pdf_path.name,
                    file_size / (1024 * 1024),
                    self.max_file_size_bytes / (1024 * 1024),
                )
                return False

            with open(pdf_path, "rb") as f_in:
                header = f_in.read(8)
                if not header.startswith(b"%PDF-"):
                    logger.error("El archivo %s no es un PDF válido (falta firma %%PDF-)", pdf_path.name)
                    return False

            import pypdfium2 as pdfium

            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            total_pages = len(pdf_doc)
            pdf_doc.close()

            if total_pages > self.max_pages:
                logger.warning(
                    "PDF %s excede límite de complejidad (%d págs > %d págs). Omitido totalmente por ser monografía/tesis.",
                    pdf_path.name,
                    total_pages,
                    self.max_pages,
                )
                return False

            return True

        except Exception as exc:
            logger.error("Error al validar PDF %s: %s", pdf_path, exc)
            return False

    def _get_converter(self):
        """Inicialización perezosa de DocumentConverter con OCR desactivado para máxima velocidad."""
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions(
                do_ocr=False,  # arXiv ya tiene texto incrustado digital; OCR ralentizaría 50x
                do_table_structure=True,
            )
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        return self._converter

    def extract_sections(self, pdf_path: Path) -> List[ExtractedRawSection]:
        """Procesa el PDF mediante Docling y extrae secciones estructuradas."""
        if not self.validate_pdf(pdf_path):
            return []

        try:
            converter = self._get_converter()
            result = converter.convert(
                str(pdf_path),
                max_num_pages=self.max_pages,
                max_file_size=self.max_file_size_bytes,
            )
            doc = result.document

            sections: List[ExtractedRawSection] = []
            current_heading = "Content"
            current_text = ""

            for element in doc.texts:
                label = getattr(element, "label", None)
                text = getattr(element, "text", "")

                if label in ["title", "section_header"]:
                    if current_text.strip():
                        sections.append(ExtractedRawSection(heading=current_heading, raw_text=current_text.strip()))
                    current_heading = text.strip()
                    current_text = ""
                else:
                    if text:
                        current_text += text + "\n"

            if current_text.strip():
                sections.append(ExtractedRawSection(heading=current_heading, raw_text=current_text.strip()))

            return sections

        except Exception as exc:
            logger.error("Fallo durante la extracción de secciones con Docling en %s: %s", pdf_path.name, exc)
            return []


class ArxivPaperResolver:
    """Orquestador de resolución y extracción de artículos de arXiv hacia StructuredPaper."""

    def __init__(self, config: Optional[ArxivResolverConfig] = None) -> None:
        self.config = config or ArxivResolverConfig()
        self.client = ArxivClient(self.config)
        self.docling_parser = DoclingPDFParser(
            max_pages=self.config.max_pages,
            max_file_size_mb=self.config.max_file_size_mb,
        )

    def _try_extract_ar5iv(self, arxiv_id: str) -> List[ExtractedRawSection]:
        """Tier 1: Intento de extracción rápida desde ar5iv.labs.arxiv.org (HTML)."""
        url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        headers = {"User-Agent": self.config.user_agent}

        try:
            with httpx.Client(timeout=self.config.timeout_seconds, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code != 200:
                    return []
                html = resp.text

            soup = BeautifulSoup(html, "html.parser")
            # ar5iv estructura las secciones jerárquicamente: ltx_section -> ltx_subsection -> ltx_subsubsection.
            # Capturamos todas las unidades estructurales para respetar el principio de Nearest Enclosing Scope
            section_tags = soup.find_all(
                ["section", "div", "article"],
                class_=lambda c: c and any(k in c for k in ["ltx_section", "ltx_subsection", "ltx_subsubsection", "ltx_appendix"])
            )
            if not section_tags:
                return []

            extracted: List[ExtractedRawSection] = []
            for s in section_tags:
                heading_elem = s.find(["h1", "h2", "h3", "h4", "h5", "h6"], class_=lambda c: c and "ltx_title" in c)
                heading = heading_elem.get_text(separator=" ", strip=True) if heading_elem else "Section"

                # Párrafos directos: pertenencia estricta al scope más inmediato (evitar duplicar en ancestros)
                direct_paras: List[str] = []
                for p in s.find_all("p", class_=lambda c: c and "ltx_p" in c):
                    parent_sec = p.find_parent(
                        ["section", "div", "article"],
                        class_=lambda c: c and any(k in c for k in ["ltx_section", "ltx_subsection", "ltx_subsubsection", "ltx_appendix"])
                    )
                    if parent_sec == s:
                        p_text = p.get_text(separator=" ", strip=True)
                        if p_text:
                            direct_paras.append(p_text)

                if direct_paras:
                    extracted.append(ExtractedRawSection(heading=heading, raw_text="\n".join(direct_paras)))

            return extracted

        except Exception as exc:
            logger.debug("ar5iv no disponible para %s: %s", arxiv_id, exc)
            return []

    def resolve_paper(self, arxiv_id: str) -> Optional[StructuredPaper]:
        """Resuelve un artículo de arXiv y retorna su representación canónica StructuredPaper.

        Aplica la cascada:
        1. Consulta metadatos oficiales (título, autores, abstract, categorías).
        2. Tier 1: Intento ultrarrápido con ar5iv HTML.
        3. Tier 2: Si ar5iv no tiene HTML, descarga PDF y procesa con Docling.
        4. Segmenta oraciones y normaliza secciones retóricas.
        """
        clean_id = normalize_arxiv_id(arxiv_id)

        # 1. Obtener Metadatos Oficiales (con tolerancia ante fallas de API)
        meta = self.client.fetch_metadata(clean_id)
        if not meta:
            logger.warning("Metadatos de arXiv API no disponibles para %s. Intentando extraer texto directamente...", clean_id)

        # 2. Tier 1: Intento con ar5iv HTML
        raw_sections = self._try_extract_ar5iv(clean_id)
        source = ExtractionSource.AR5IV_HTML

        # 3. Tier 2: Fallback a PDF con Docling
        if not raw_sections:
            logger.info("ar5iv no disponible para %s. Conmutando a Tier 2 (Docling PDF)...", clean_id)
            pdf_url = meta.pdf_url if meta else f"https://arxiv.org/pdf/{clean_id}.pdf"
            pdf_path = self.client.download_pdf(clean_id, pdf_url)
            if not pdf_path:
                logger.warning("No fue posible descargar PDF para %s", clean_id)
                return None

            raw_sections = self.docling_parser.extract_sections(pdf_path)
            source = ExtractionSource.DOCLING_PDF

        if not raw_sections:
            logger.warning("No fue posible extraer texto estructurado para %s", clean_id)
            return None

        # 4. Normalización de Secciones y Segmentación Atómica de Oraciones
        structured_sections: List[StructuredSection] = []
        sent_idx = 0

        for raw_sec in raw_sections:
            if is_excluded_heading(raw_sec.heading):
                continue

            raw_sents = split_sentences(raw_sec.raw_text)
            sec_sentences: List[StructuredSentence] = []

            for s in raw_sents:
                clean_s = s.strip()
                if len(clean_s) > 15:
                    sec_sentences.append(StructuredSentence(sent_idx=sent_idx, text=clean_s, citations=[]))
                    sent_idx += 1

            if sec_sentences:
                structured_sections.append(
                    StructuredSection(
                        section_idx=len(structured_sections),
                        heading=raw_sec.heading,
                        sentences=sec_sentences,
                    )
                )

        if not structured_sections:
            logger.warning("El artículo %s no produjo secciones discursivas útiles.", clean_id)
            return None

        logger.info(
            "Artículo %s exitosamente estructurado (%s): %d secciones, %d oraciones.",
            clean_id,
            source.value,
            len(structured_sections),
            sent_idx,
        )

        return StructuredPaper(
            paper_id=clean_id,
            id_source="arxiv",
            title=meta.title if meta else f"arXiv:{clean_id}",
            abstract=meta.abstract if meta else "",
            discipline="Computer Science",
            categories=meta.categories if (meta and meta.categories) else ["cs.AI"],
            sections=structured_sections,
            bibliography={},
        )

    def resolve_batch(
        self,
        arxiv_ids: List[str],
        output_file: Optional[Path] = None,
    ) -> Tuple[List[StructuredPaper], List[str]]:
        """Resuelve un lote de arXiv IDs y opcionalmente anexa los resultados a disco.

        Args:
            arxiv_ids: Lista de identificadores de arXiv.
            output_file: Si se provee, anexa incrementalmente cada StructuredPaper como línea JSON.

        Returns:
            Tupla con (lista de StructuredPaper resueltos, lista de IDs fallidos).
        """
        successful: List[StructuredPaper] = []
        failed: List[str] = []

        out_fh = None
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            out_fh = open(output_file, "a", encoding="utf-8")

        try:
            total = len(arxiv_ids)
            for idx, raw_id in enumerate(arxiv_ids, 1):
                clean_id = normalize_arxiv_id(raw_id)
                logger.info("[%d/%d] Resolviendo paper: %s...", idx, total, clean_id)

                paper = self.resolve_paper(clean_id)
                if paper:
                    successful.append(paper)
                    if out_fh:
                        out_fh.write(json.dumps(paper.model_dump()) + "\n")
                        out_fh.flush()
                else:
                    failed.append(clean_id)

        finally:
            if out_fh:
                out_fh.close()

        logger.info(
            "Lote completado: %d exitosos, %d fallidos de un total de %d.",
            len(successful),
            len(failed),
            len(arxiv_ids),
        )
        return successful, failed


def extract_missing_cited_ids(
    contexts_path: Path,
    papers_path: Optional[Path] = None,
) -> List[str]:
    """Identifica automáticamente los arXiv IDs citados que faltan por estructurar.

    Examina citing_contexts.jsonl para encontrar todos los target_cited_id donde
    has_cited_chunks es False, y excluye aquellos que ya existan en papers.jsonl.

    Args:
        contexts_path: Ruta al archivo de contextos citantes (data/processed/citing_contexts.jsonl).
        papers_path: Ruta al corpus estructurado canónico (data/structured/papers.jsonl).

    Returns:
        Lista ordenada y deduplicada de arXiv IDs pendientes por resolver.
    """
    if not contexts_path.exists():
        logger.warning("El archivo de contextos %s no existe.", contexts_path)
        return []

    existing_ids: Set[str] = set()
    if papers_path and papers_path.exists():
        with open(papers_path, "r", encoding="utf-8") as f_p:
            for line in f_p:
                try:
                    pdata = json.loads(line)
                    pid = pdata.get("paper_id")
                    if pid:
                        existing_ids.add(normalize_arxiv_id(pid))
                except Exception:
                    continue

    missing_ids: Set[str] = set()
    with open(contexts_path, "r", encoding="utf-8") as f_c:
        for line in f_c:
            try:
                cdata = json.loads(line)
                target_id = cdata.get("target_cited_id")
                has_chunks = cdata.get("has_cited_chunks", False)
                if target_id and not has_chunks:
                    clean_id = normalize_arxiv_id(target_id)
                    if clean_id not in existing_ids:
                        missing_ids.add(clean_id)
            except Exception:
                continue

    sorted_ids = sorted(missing_ids)
    logger.info(
        "Identificados %d arXiv IDs citados faltantes en %s (excluidos %d ya estructurados en %s).",
        len(sorted_ids),
        contexts_path,
        len(existing_ids),
        papers_path or "N/A",
    )
    return sorted_ids


def main() -> None:
    """Punto de entrada de línea de comandos para ejecución directa y autosuficiente."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Resuelve preprints de arXiv y extrae su texto estructurado hacia StructuredPaper de forma autosuficiente."
    )
    parser.add_argument("--id", type=str, help="arXiv ID individual a resolver (ej. 1706.03762)")
    parser.add_argument("--input-file", type=Path, help="Archivo de texto opcional con un arXiv ID por línea")
    parser.add_argument(
        "--contexts-path",
        type=Path,
        default=Path("data/processed/citing_contexts.jsonl"),
        help="Ruta a citing_contexts.jsonl para auto-detección de citados faltantes",
    )
    parser.add_argument(
        "--papers-path",
        type=Path,
        default=Path("data/structured/papers.jsonl"),
        help="Ruta a papers.jsonl donde consultar existentes y anexar resueltos",
    )
    parser.add_argument("--limit", type=int, default=None, help="Límite máximo de papers a resolver en esta corrida (útil para pruebas)")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/arxiv_cache"), help="Directorio de caché PDF")
    parser.add_argument("--max-pages", type=int, default=20, help="Límite máximo de páginas por PDF")
    parser.add_argument("--rate-limit", type=float, default=3.0, help="Retraso en segundos para arXiv API")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    cfg = ArxivResolverConfig(
        cache_dir=args.cache_dir,
        max_pages=args.max_pages,
        rate_limit_delay=args.rate_limit,
    )
    resolver = ArxivPaperResolver(cfg)

    # Modo 1: ID individual para pruebas rápidas
    if args.id:
        paper = resolver.resolve_paper(args.id)
        if paper:
            print(f"Paper estructurado exitosamente: {paper.title} ({paper.paper_id})")
            print(f"Total secciones: {len(paper.sections)}")
            if args.papers_path:
                with open(args.papers_path, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(paper.model_dump()) + "\n")
                print(f"Anexado a {args.papers_path}")
        else:
            print(f"Fallo al resolver paper {args.id}")

    # Modo 2: Archivo de texto explícito (compatibilidad)
    elif args.input_file:
        if not args.input_file.exists():
            print(f"Error: {args.input_file} no existe.")
            return

        with open(args.input_file, "r", encoding="utf-8") as f_in:
            ids = [line.strip() for line in f_in if line.strip() and not line.startswith("#")]

        if args.limit:
            ids = ids[: args.limit]

        succ, fail = resolver.resolve_batch(ids, output_file=args.papers_path)
        print(f"Procesamiento finalizado. Exitosos: {len(succ)}, Fallidos: {len(fail)}")

    # Modo 3: Modo por defecto AUTOSUFICIENTE (Auto-detección desde citing_contexts.jsonl)
    else:
        logger.info("Modo autosuficiente: detectando papers citados faltantes desde %s...", args.contexts_path)
        missing_ids = extract_missing_cited_ids(args.contexts_path, args.papers_path)

        if not missing_ids:
            logger.info("No se encontraron arXiv IDs citados pendientes por resolver.")
            return

        if args.limit:
            logger.info("Aplicando límite de corrida: procesando los primeros %d de %d IDs...", args.limit, len(missing_ids))
            missing_ids = missing_ids[: args.limit]

        succ, fail = resolver.resolve_batch(missing_ids, output_file=args.papers_path)
        print(f"Resolución completada. Exitosos anexados a papers.jsonl: {len(succ)}, Fallidos: {len(fail)}")


if __name__ == "__main__":
    main()

