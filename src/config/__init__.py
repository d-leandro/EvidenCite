"""Paquete de configuración tipada con Pydantic para experimentos (Actividades A4–A7)."""

from src.config.schema import (
    CANONICAL_9_CLASSES,
    ID_TO_LABEL,
    LABEL_TO_ID,
    BaseConfig,
    EncoderModelConfig,
    ExperimentConfig,
    InputRepresentationConfig,
    LLMApiConfig,
    LoraModelConfig,
    PathsConfig,
    TfidfModelConfig,
    load_base_config,
    load_experiment_config,
    resolve_canonical_label,
    resolve_device,
)

__all__ = [
    "CANONICAL_9_CLASSES",
    "ID_TO_LABEL",
    "LABEL_TO_ID",
    "BaseConfig",
    "EncoderModelConfig",
    "ExperimentConfig",
    "InputRepresentationConfig",
    "LLMApiConfig",
    "LoraModelConfig",
    "PathsConfig",
    "TfidfModelConfig",
    "load_base_config",
    "load_experiment_config",
    "resolve_canonical_label",
    "resolve_device",
]

