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


class CitingContext(BaseModel):
    """Contexto citante atómico procesado en data/processed/citing_contexts.jsonl."""

    context_id: str = Field(..., description="Identificador único del contexto (ej. 'mc_train_021c423c_0' o 'unx_2101.00098_s0')")
    source_dataset: str = Field(..., description="Origen de la instancia: 'multicite' o 'unarxive_mined'")
    split: Optional[str] = Field(default=None, description="Partición del dataset ('train', 'dev', 'test') si aplica")
    citing_paper_id: str = Field(..., description="Identificador del artículo citante (arXiv ID o hash MultiCite)")
    rhetorical_section: str = Field(..., description="Encabezado de sección fiel donde ocurre la cita")
    window_rule: str = Field(default="m_3_sentences", description="Regla de ventana aplicada (m=3 oraciones)")
    citation_context: str = Field(..., description="Texto de la ventana con cita delimitada por <cite>...</cite>")

    # Metadatos del artículo citado resuelto
    target_ref_key: Optional[str] = Field(default=None, description="Clave de referencia bibliográfica original")
    target_cited_id: Optional[str] = Field(default=None, description="Identificador canónico arXiv del citado si fue resuelto")
    target_cited_title: Optional[str] = Field(default=None, description="Título del artículo citado")
    target_cited_authors: Optional[str] = Field(default=None, description="Autores del artículo citado")
    target_cited_year: Optional[int] = Field(default=None, description="Año del artículo citado")
    target_cited_abstract: Optional[str] = Field(default=None, description="Abstract del citado (clave para Configuración A)")
    has_cited_chunks: bool = Field(default=False, description="Indica si el artículo citado cuenta con texto completo en papers.jsonl")

    # Etiquetas e intención de cita
    original_label: Optional[str] = Field(default=None, description="Etiqueta original en el dataset fuente")
    normalized_label: Optional[str] = Field(default=None, description="Etiqueta normalizada en el esquema de 9 clases de Lyu et al.")
    label_source: str = Field(
        default="human_gold",
        description="Origen de la etiqueta: 'human_gold' (anotación humana experta) o 'heuristic_rule' (regla léxica minada)",
    )
    is_gold: bool = Field(
        default=True,
        description="Booleano que certifica si la etiqueta es un benchmark de oro humano (True) o una heurística débil (False)",
    )


class CitedChunk(BaseModel):
    """Representación granular de un fragmento discursivo de un artículo citado."""

    chunk_id: str = Field(..., description="Identificador único canónico: {paper_id}_s{sec_idx}_c{chunk_idx}")
    paper_id: str = Field(..., description="Identificador del artículo citado en papers.jsonl")
    section_idx: int = Field(..., description="Índice ordinal de la sección dentro del artículo")
    section_heading: str = Field(..., description="Título o encabezado retórico de la sección")
    chunk_idx: int = Field(..., description="Índice ordinal del chunk dentro de la sección")
    text: str = Field(..., description="Texto concatenado de las oraciones que componen el chunk")
    token_count: int = Field(..., description="Número de tokens aproximados (WordPiece)")
    word_count: int = Field(..., description="Número de palabras en el fragmento")


