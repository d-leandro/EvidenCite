"""Esquemas Pydantic y contratos de datos para el módulo de etiquetado por jueces (Actividad A2)."""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data.taxonomy import CanonicalLabel


class JudgeVote(BaseModel):
    """Contrato estricto de respuesta individual emitido por cada modelo juez."""

    model_config = ConfigDict(extra="ignore")

    rationale: str = Field(
        ...,
        max_length=300,
        description="Breve justificación textual previa de la clasificación (máx. 40 palabras).",
    )
    label: CanonicalLabel = Field(
        ...,
        description="Etiqueta canónica asignada estrictamente dentro de las 9 clases.",
    )

    @field_validator("rationale", mode="before")
    @classmethod
    def _truncate_rationale(cls, value: object) -> str:
        """Trunca justificaciones extensas antes de la validación para evitar fallos en structured outputs."""
        return str(value or "").strip()[:300]


class PanelVote(BaseModel):
    """Voto procesado y trazable registrado por un juez en el panel."""

    judge_id: str = Field(..., description="Identificador único del juez (ej. 'llama31_8b')")
    model_name: str = Field(..., description="Nombre canónico del modelo en el proveedor")
    provider: str = Field(..., description="Proveedor del endpoint ('openai', 'openrouter', 'ollama', etc.)")
    label: Optional[CanonicalLabel] = Field(default=None, description="Etiqueta votada o None si hubo abstención/error")
    rationale: Optional[str] = Field(default=None, description="Explicación emitida por el modelo")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Latencia en milisegundos")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    error: Optional[str] = Field(default=None, description="Mensaje de error si la llamada falló")


class PanelDecision(BaseModel):
    """Resultado de la agregación y ruteo sobre el conjunto de votos de una instancia."""

    final_label: Optional[str] = Field(..., description="Etiqueta final resultante para la instancia")
    final_label_source: Literal["human_gold", "llm_panel", "needs_review"] = Field(
        ..., description="Nueva procedencia metodológica de la etiqueta"
    )
    panel_agreement: float = Field(..., ge=0.0, le=1.0, description="Fracción de concordancia del panel")
    panel_label: Optional[str] = Field(default=None, description="Etiqueta mayoritaria emitida por el panel")
    case: str = Field(..., description="Caso de la matriz de decisión (A–F)")
    votes: List[PanelVote] = Field(default_factory=list, description="Lista de votos recolectados")


class JudgeConfig(BaseModel):
    """Configuración individual de un juez del panel."""

    id: str
    provider: str = "openrouter"
    model_name: str
    base_url: str
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 150
    max_retries: int = 3
    timeout_seconds: float = 60.0
    max_concurrency: int = Field(default=1, ge=1)


class AggregationConfig(BaseModel):
    """Parámetros para la función de decisión y agregación de votos."""

    min_agreement: float = Field(default=0.66, ge=0.0, le=1.0)
    quota_per_class: int = Field(default=2000, ge=1)


class LabelingConfig(BaseModel):
    """Configuración maestra del módulo de pre-etiquetado A2."""

    input_path: Path = Path("data/dataset/instances.jsonl")
    output_dir: Path = Path("data/labeled")
    log_path: Path = Path("data/logs/labeling.log")
    seed: int = 42
    prompt_version: Literal["v1_zeroshot_definitions_only", "v2_zeroshot_context_only"] = "v1_zeroshot_definitions_only"
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    judges: List[JudgeConfig]
