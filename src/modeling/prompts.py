"""Plantillas de prompts y contratos Pydantic estructurados (Forma 2) para la Actividad A5 (LLMs vía API)."""

from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.config.schema import (
    CANONICAL_9_CLASSES,
    InputRepresentationConfig,
    resolve_canonical_label,
)
from src.data.schemas import DatasetInstance
from src.data.taxonomy import CANONICAL_CLASS_DEFINITIONS, CanonicalLabel
from src.modeling.dataset import format_instance_parts

CanonicalLabelType = CanonicalLabel


class NineClassScores(BaseModel):
    """Vector estructurado con los 9 puntajes de pertenencia en [0.0, 1.0] (Forma 2 aprobada)."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    Background: float = Field(..., ge=0.0, le=1.0)
    Gap: float = Field(..., ge=0.0, le=1.0)
    Basis: float = Field(..., ge=0.0, le=1.0)
    Comparison: float = Field(..., ge=0.0, le=1.0)
    Application: float = Field(..., ge=0.0, le=1.0)
    Improvement: float = Field(..., ge=0.0, le=1.0)
    Evidence: float = Field(..., ge=0.0, le=1.0)
    Originator: float = Field(..., ge=0.0, le=1.0)
    Further_Reading: float = Field(..., ge=0.0, le=1.0, alias="Further Reading")

    def to_dict_canonical(self) -> Dict[str, float]:
        """Retorna el diccionario indexado exactamente por CANONICAL_9_CLASSES."""
        return {
            "Background": float(self.Background),
            "Gap": float(self.Gap),
            "Basis": float(self.Basis),
            "Comparison": float(self.Comparison),
            "Application": float(self.Application),
            "Improvement": float(self.Improvement),
            "Evidence": float(self.Evidence),
            "Originator": float(self.Originator),
            "Further Reading": float(self.Further_Reading),
        }

    def to_prob_vector(self) -> List[float]:
        """Retorna la lista de 9 puntajes en el orden canónico de CANONICAL_9_CLASSES."""
        canonical_map = self.to_dict_canonical()
        return [canonical_map[cls_name] for cls_name in CANONICAL_9_CLASSES]


class LLMResponse(BaseModel):
    """Contrato estricto de respuesta JSON para GPT-4o / Gemini en la Actividad A5."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    predicted_label: CanonicalLabelType = Field(
        ...,
        description="Etiqueta canónica predicha (debe coincidir con la clase de mayor puntaje).",
    )
    scores: NineClassScores = Field(
        ...,
        description="Puntajes de confianza en [0.0, 1.0] para cada una de las 9 clases canónicas.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Breve justificación textual de la intención de cita predicha.",
    )


def _build_system_prompt(prompt_mode: Literal["generic", "section_conditioned"]) -> str:
    taxonomy_lines = "\n".join(
        f"- {cls_name}: {CANONICAL_CLASS_DEFINITIONS[cls_name]}"
        for cls_name in CANONICAL_9_CLASSES
    )
    section_guidance = ""
    if prompt_mode == "section_conditioned":
        section_guidance = (
            "\nPay special attention to the rhetorical section headings provided at the start of "
            "the citing context and cited passages (formatted as 'Section: Text'), as discourse "
            "location strongly constrains the rhetorical function of a citation.\n"
        )

    return (
        "You are an expert computational linguistics annotator specialized in scientific citation "
        "intent classification.\n"
        "Classify the rhetorical function of the target citation (marked with <cite>...</cite> in "
        "the citing context) into one of the following 9 canonical classes (Lyu et al.):\n\n"
        f"{taxonomy_lines}\n"
        f"{section_guidance}\n"
        "Respond strictly with a valid JSON object matching the required schema:\n"
        "{\n"
        '  "predicted_label": "<one of the 9 canonical labels>",\n'
        '  "scores": {\n'
        '    "Background": 0.0,\n'
        '    "Gap": 0.0,\n'
        '    "Basis": 0.0,\n'
        '    "Comparison": 0.0,\n'
        '    "Application": 0.0,\n'
        '    "Improvement": 0.0,\n'
        '    "Evidence": 0.0,\n'
        '    "Originator": 0.0,\n'
        '    "Further Reading": 0.0\n'
        "  }\n"
        "}"
    )


def format_user_prompt(
    instance: DatasetInstance,
    rep_config: InputRepresentationConfig,
    prompt_mode: Literal["generic", "section_conditioned"] = "section_conditioned",
) -> str:
    """Construye el prompt de usuario para una instancia según configuración (context_only, A o B) y prompt_mode."""
    effective_rep = rep_config.model_copy(
        update={
            "include_citing_section": (
                rep_config.include_citing_section if prompt_mode == "section_conditioned" else False
            ),
            "include_cited_section": (
                rep_config.include_cited_section if prompt_mode == "section_conditioned" else False
            ),
        }
    )
    text_citing, text_cited = format_instance_parts(instance, effective_rep, tokenizer=None)
    if text_cited is None:
        return f"Citing Context:\n{text_citing}"
    return f"Citing Context:\n{text_citing}\n\nCited Paper Evidence:\n{text_cited}"


def build_llm_messages(
    instance: DatasetInstance,
    rep_config: InputRepresentationConfig,
    prompt_mode: Literal["generic", "section_conditioned"] = "section_conditioned",
    few_shot_examples: Optional[Sequence[DatasetInstance]] = None,
) -> List[Dict[str, str]]:
    """Genera la lista de mensajes [{'role': ..., 'content': ...}] para inferencia Zero-Shot o Few-Shot en A5."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _build_system_prompt(prompt_mode)}
    ]

    if few_shot_examples:
        for ex in few_shot_examples:
            lbl = resolve_canonical_label(ex.label) or "Background"
            scores_dict = {
                c: (0.92 if c == lbl else 0.01) for c in CANONICAL_9_CLASSES
            }
            exemplar_resp = LLMResponse.model_validate(
                {"predicted_label": lbl, "scores": scores_dict}
            )
            messages.append(
                {
                    "role": "user",
                    "content": format_user_prompt(ex, rep_config, prompt_mode=prompt_mode),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": exemplar_resp.model_dump_json(by_alias=True),
                }
            )

    messages.append(
        {
            "role": "user",
            "content": format_user_prompt(instance, rep_config, prompt_mode=prompt_mode),
        }
    )
    return messages

