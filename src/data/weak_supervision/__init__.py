"""Módulo de supervisión débil mediante Inferencia de Lenguaje Natural (NLI Zero-Shot).

Proporciona abstracciones y servicios reutilizables para el enriquecimiento semántico
de citas científicas (Actividad A2), consumido en memoria por el pipeline de builder.py.
"""

from src.data.weak_supervision.nli_tagger import (
    CANONICAL_NLI_HYPOTHESES,
    DeBERTaInferenceEngine,
    NLIPromptFormatter,
    NLIPurityFilter,
    NLITagger,
    NLITaggerConfig,
)

__all__ = [
    "CANONICAL_NLI_HYPOTHESES",
    "DeBERTaInferenceEngine",
    "NLIPromptFormatter",
    "NLIPurityFilter",
    "NLITagger",
    "NLITaggerConfig",
]
