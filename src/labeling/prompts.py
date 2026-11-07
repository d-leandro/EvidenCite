"""Construcción de prompts zero-shot y contratos textuales para los jueces LLM."""

from typing import Dict, List

from src.data.schemas import DatasetInstance
from src.data.taxonomy import CANONICAL_9_CLASSES, CANONICAL_CLASS_DEFINITIONS

PROMPT_VERSION: str = "v1_zeroshot_definitions_only"
PROMPT_VERSION_V2: str = "v2_zeroshot_context_only"
PROMPT_VERSIONS = (PROMPT_VERSION, PROMPT_VERSION_V2)

_GUIDELINE_BY_VERSION = {
    PROMPT_VERSION: (
        "1. Pay careful attention to the rhetorical section where the citation appears, "
        "as discourse location strongly constrains citation function.\n"
    ),
    PROMPT_VERSION_V2: "1. Base your classification strictly on the linguistic and rhetorical cues within the citing context.\n",
}


def _check_version(prompt_version: str) -> None:
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"Versión de prompt desconocida: {prompt_version!r}")


def build_system_prompt(prompt_version: str = PROMPT_VERSION) -> str:
    """Retorna el prompt de sistema inmutable con las definiciones de las 9 clases."""
    _check_version(prompt_version)
    definitions_text = "\n".join(
        f"- {label}: {CANONICAL_CLASS_DEFINITIONS[label]}" for label in CANONICAL_9_CLASSES
    )
    return (
        "You are an expert computational linguistics annotator specialized in scientific citation intent classification.\n\n"
        "Your task is to classify the rhetorical function of the target citation (enclosed in <cite>...</cite> tags) "
        "within the citing context into exactly ONE of the following 9 canonical classes:\n\n"
        f"{definitions_text}\n\n"
        "Guidelines:\n"
        f"{_GUIDELINE_BY_VERSION[prompt_version]}"
        "2. Provide a concise rationale (maximum 40 words) justifying your decision BEFORE providing the label.\n"
        "3. Output strictly a valid JSON object matching the requested schema with keys 'rationale' and 'label'."
    )


def format_judge_user_prompt(instance: DatasetInstance, prompt_version: str = PROMPT_VERSION) -> str:
    """Construye el prompt de usuario con el contexto citante (cegado D5); v1 agrega la sección retórica."""
    _check_version(prompt_version)
    if prompt_version == PROMPT_VERSION_V2:
        return f"Citing Context:\n{instance.citation_context}"
    section = instance.citing_paper.rhetorical_section or "Unknown Section"
    return f"Rhetorical Section: {section}\n\nCiting Context:\n{instance.citation_context}"


def build_judge_messages(instance: DatasetInstance, prompt_version: str = PROMPT_VERSION) -> List[Dict[str, str]]:
    """Construye la lista completa de mensajes [{'role': ..., 'content': ...}] para el API OpenAI."""
    return [
        {"role": "system", "content": build_system_prompt(prompt_version)},
        {"role": "user", "content": format_judge_user_prompt(instance, prompt_version)},
    ]
