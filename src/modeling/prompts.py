"""Plantillas de prompts y contratos Pydantic estructurados (Forma 2) para la Actividad A5 (LLMs vía API)."""

import hashlib
import random
from typing import Dict, List, Literal, Optional, Sequence, Tuple

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

# Vecindad retórica derivada de la meta-síntesis de Lyu et al. (Scientometrics, 2021)
# y la triangulación empírica en el símplex Delta^8 (ARQ-08 / ARQ-09).
_RHETORICAL_NEIGHBORS: Dict[str, Dict[str, object]] = {
    "Background": {
        "gold_range_cents": (62, 78),
        "primary": ("Further Reading", "Gap"),
        "secondary": ("Originator", "Evidence"),
    },
    "Gap": {
        "gold_range_cents": (60, 76),
        "primary": ("Background", "Comparison"),
        "secondary": ("Evidence", "Improvement"),
    },
    "Basis": {
        "gold_range_cents": (56, 72),
        "primary": ("Improvement", "Application"),
        "secondary": ("Background", "Originator"),
    },
    "Comparison": {
        "gold_range_cents": (58, 75),
        "primary": ("Improvement", "Evidence"),
        "secondary": ("Gap", "Background"),
    },
    "Application": {
        "gold_range_cents": (60, 76),
        "primary": ("Basis", "Improvement"),
        "secondary": ("Evidence", "Comparison"),
    },
    "Improvement": {
        "gold_range_cents": (56, 72),
        "primary": ("Basis", "Comparison"),
        "secondary": ("Application", "Evidence"),
    },
    "Evidence": {
        "gold_range_cents": (64, 80),
        "primary": ("Comparison", "Background"),
        "secondary": ("Basis", "Gap"),
    },
    "Originator": {
        "gold_range_cents": (64, 80),
        "primary": ("Background", "Basis"),
        "secondary": ("Application", "Evidence"),
    },
    "Further Reading": {
        "gold_range_cents": (68, 84),
        "primary": ("Background", "Evidence"),
        "secondary": ("Application", "Basis"),
    },
}


def _build_few_shot_scores(gold_label: str, seed_key: str) -> Dict[str, float]:
    """Genera deterministamente una distribución realista en centésimas (suma exacta 1.00) con adyacencia retórica."""
    resolved_gold = resolve_canonical_label(gold_label) or "Background"
    spec = _RHETORICAL_NEIGHBORS[resolved_gold]
    low_c, high_c = spec["gold_range_cents"]  # type: ignore[misc]
    primary: Tuple[str, str] = spec["primary"]  # type: ignore[assignment]
    secondary: Tuple[str, str] = spec["secondary"]  # type: ignore[assignment]

    seed_bytes = hashlib.sha256(f"{seed_key}::{resolved_gold}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))

    gold_cents = rng.randint(low_c, high_c)
    rem = 100 - gold_cents

    # Reparto estratificado del remanente: ~64% vecinos primarios, ~24% secundarios, ~12% cola ortogonal
    prim_budget = max(2, round(rem * rng.uniform(0.58, 0.70)))
    sec_budget = min(rem - prim_budget, max(1, round(rem * rng.uniform(0.20, 0.28))))
    tail_budget = rem - prim_budget - sec_budget

    # División asimétrica en vecinos primarios (garantizando p_i < gold_cents)
    p1_share = rng.uniform(0.55, 0.72)
    p1 = min(gold_cents - 1, max(1, round(prim_budget * p1_share)))
    p2 = max(0, prim_budget - p1)
    if p2 >= gold_cents:
        overflow = p2 - (gold_cents - 1)
        p2 -= overflow
        tail_budget += overflow

    # División asimétrica en vecinos secundarios
    s1 = round(sec_budget * rng.uniform(0.52, 0.75))
    s2 = max(0, sec_budget - s1)

    cents_map: Dict[str, int] = {c: 0 for c in CANONICAL_9_CLASSES}
    cents_map[resolved_gold] = gold_cents
    cents_map[primary[0]] = p1
    cents_map[primary[1]] = p2
    cents_map[secondary[0]] = s1
    cents_map[secondary[1]] = s2

    # Cola ortogonal (4 clases restantes): asignación discreta residual
    tail_classes = [
        c for c in CANONICAL_9_CLASSES if c not in (resolved_gold, *primary, *secondary)
    ]
    while tail_budget > 0:
        pick = rng.choice(tail_classes)
        cents_map[pick] += 1
        tail_budget -= 1

    return {c: round(cents_map[c] / 100.0, 2) for c in CANONICAL_9_CLASSES}


class NineClassScores(BaseModel):
    """Structured vector with confidence scores in [0.0, 1.0] for each of the 9 canonical classes."""

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
        """Retorna el diccionario normalizado (suma 1.0) indexado por CANONICAL_9_CLASSES."""
        raw_scores: Dict[str, float] = {
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
        total = sum(raw_scores.values())
        if total <= 0.0:
            uniform = 1.0 / len(raw_scores)
            return {cls_name: uniform for cls_name in raw_scores}
        return {cls_name: val / total for cls_name, val in raw_scores.items()}

    def to_prob_vector(self) -> List[float]:
        """Retorna la lista de 9 puntajes en el orden canónico de CANONICAL_9_CLASSES."""
        canonical_map = self.to_dict_canonical()
        return [canonical_map[cls_name] for cls_name in CANONICAL_9_CLASSES]


class LLMResponse(BaseModel):
    """Strict JSON response schema for commercial LLM citation intent classification."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    predicted_label: CanonicalLabelType = Field(
        ...,
        description="Predicted canonical citation intent label (must match the highest-scoring class).",
    )
    scores: NineClassScores = Field(
        ...,
        description="Confidence scores in [0.0, 1.0] for each of the 9 canonical classes (summing to 1.0).",
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
        "the citing context) into exactly ONE of the following 9 canonical classes (Lyu et al.):\n\n"
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
    """Construye el prompt de usuario para una instancia según configuración (context_only, A o B) y prompt_mode.

    En modo 'generic', se omite la sección del contexto citante (`include_citing_section=False`),
    pero se preserva `rep_config.include_cited_section` para mantener paridad 1:1 (ADR-07) con
    los encabezados de sección de los Top-3 chunks del citado en Configuración B.
    """
    effective_rep = rep_config.model_copy(
        update={
            "include_citing_section": (
                rep_config.include_citing_section if prompt_mode == "section_conditioned" else False
            ),
            "include_cited_section": rep_config.include_cited_section,
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
        for idx, ex in enumerate(few_shot_examples):
            lbl = resolve_canonical_label(ex.label) or "Background"
            seed_key = ex.instance_id or f"shot_{idx}_{lbl}"
            scores_dict = _build_few_shot_scores(lbl, seed_key=seed_key)
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

