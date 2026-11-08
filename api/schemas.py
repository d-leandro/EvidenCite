from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    model_id: str
    citing_context: str = Field(min_length=20)
    target_citation: str | None = None
    rhetorical_section: str | None = None

    # Configuración A / búsqueda del artículo
    cited_arxiv_id: str | None = None
    cited_title: str | None = None
    cited_abstract: str | None = None

    # Configuración B
    cited_chunks: list[str] | None = None
    cited_chunk_sections: list[str] | None = None


class PrepareContextRequest(BaseModel):
    citing_context: str = Field(min_length=20)
    target_citation: str = Field(min_length=1)



class EvidenceRequest(BaseModel):
    citing_context: str = Field(min_length=20)
    target_citation: str | None = None
    cited_arxiv_id: str = Field(min_length=4)
    cited_title: str | None = None
    cited_abstract: str | None = None
