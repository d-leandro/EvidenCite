"""Esquemas Pydantic para el contrato de datos del proyecto.

Define los modelos estructurados para la capa canonical (data/structured/papers.jsonl),
así como los componentes atómicos de oraciones, citas, secciones y bibliografía.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CitationMarker(BaseModel):
    """Mención explícita de cita dentro de una oración."""

    marker_text: str = Field(..., description="Texto del marcador, ej. '{{cite:ref_id}}'")
    char_start: int = Field(..., description="Índice de carácter inicial dentro de la oración")
    char_end: int = Field(..., description="Índice de carácter final dentro de la oración")
    ref_key: str = Field(..., description="Clave única que apunta a la bibliografía del artículo")


class StructuredSentence(BaseModel):
    """Oración atómica pre-segmentada."""

    sent_idx: int = Field(..., description="Índice secuencial único de la oración en el artículo")
    text: str = Field(..., description="Texto limpio de la oración")
    citations: List[CitationMarker] = Field(default_factory=list, description="Citas presentes en la oración")


class StructuredSection(BaseModel):
    """Sección retórica del artículo con encabezado original."""

    section_idx: int = Field(..., description="Índice de la sección en el documento")
    heading: str = Field(..., description="Encabezado original exacto del autor")
    sentences: List[StructuredSentence] = Field(default_factory=list, description="Oraciones de la sección")


class BibliographyEntry(BaseModel):
    """Entrada bibliográfica resuelta del artículo."""

    title: str = Field(..., description="Título extraído de la publicación citada")
    authors: Optional[str] = Field(default=None, description="Cadena de autores extraídos de la referencia")
    year: Optional[int] = Field(default=None, description="Año de publicación extraído")
    raw_text: str = Field(..., description="Cadena de texto completa original de la referencia")
    arxiv_id: Optional[str] = Field(default=None, description="Identificador arXiv canónico si fue resuelto")
    doi: Optional[str] = Field(default=None, description="Identificador DOI si existe")
    open_alex_id: Optional[str] = Field(default=None, description="URI de OpenAlex si existe")


class StructuredPaper(BaseModel):
    """Contrato del artículo científico estructurado canónico en papers.jsonl."""

    paper_id: str = Field(..., description="Identificador único arXiv (ej. '2212.11808')")
    id_source: str = Field(default="arxiv", description="Fuente de procedencia del identificador")
    title: str = Field(..., description="Título normalizado del artículo")
    abstract: str = Field(..., description="Texto completo del abstract")
    discipline: str = Field(default="Computer Science", description="Disciplina normalizada")
    categories: List[str] = Field(default_factory=list, description="Lista de categorías arXiv (ej. ['cs.CL', 'cs.LG'])")
    sections: List[StructuredSection] = Field(default_factory=list, description="Secciones estructuradas")
    bibliography: Dict[str, BibliographyEntry] = Field(default_factory=dict, description="Bibliografía indexada")


