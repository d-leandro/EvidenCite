"""Construcción de prompts zero-shot y contratos textuales para los jueces LLM."""

from typing import Dict, List

from src.data.schemas import DatasetInstance
from src.data.taxonomy import CANONICAL_9_CLASSES, CANONICAL_CLASS_DEFINITIONS

PROMPT_VERSION: str = "v1_zeroshot_definitions_only"


def build_system_prompt() -> str:
    """Retorna el prompt de sistema inmutable con las definiciones de las 9 clases."""
    definitions_text = "\n".join(
        f"- {label}: {CANONICAL_CLASS_DEFINITIONS[label]}" for label in CANONICAL_9_CLASSES
    )
    return (
        "You are an expert computational linguistics annotator specialized in scientific citation intent classification.\n\n"
        "Your task is to classify the rhetorical function of the target citation (enclosed in <cite>...</cite> tags) "
        "within the citing context into exactly ONE of the following 9 canonical classes:\n\n"
        f"{definitions_text}\n\n"
        "Guidelines:\n"
        "1. Pay careful attention to the rhetorical section where the citation appears, as discourse location strongly constrains citation function.\n"
        "2. Provide a concise rationale (maximum 40 words) justifying your decision BEFORE providing the label.\n"
        "3. Output strictly a valid JSON object matching the requested schema with keys 'rationale' and 'label'."
    )


def format_judge_user_prompt(instance: DatasetInstance) -> str:
    """Construye el prompt de usuario conteniendo de forma aislada el contexto citante y sección (cegado D5)."""
    section = instance.citing_paper.rhetorical_section or "Unknown Section"
    return f"Rhetorical Section: {section}\n\nCiting Context:\n{instance.citation_context}"


def build_judge_messages(instance: DatasetInstance) -> List[Dict[str, str]]:
    """Construye la lista completa de mensajes [{'role': ..., 'content': ...}] para el API OpenAI."""
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": format_judge_user_prompt(instance)},
    ]
