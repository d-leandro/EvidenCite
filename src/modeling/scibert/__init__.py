"""Definición de modelos de la rama encoder (SciBERT / SPECTER2)."""

from src.modeling.scibert.modelo import build_encoder_classifier, create_smoke_model

__all__ = ["build_encoder_classifier", "create_smoke_model"]
