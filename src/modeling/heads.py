"""Fábrica de modelos de clasificación de secuencias (AutoModelForSequenceClassification + PEFT/LoRA).

La definición de cada familia está en su carpeta: src/modeling/scibert (encoder) y src/modeling/lora (LLM con LoRA).
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config.schema import ExperimentConfig
from src.modeling import lora, scibert
from src.modeling.lora import build_lora_classifier
from src.modeling.scibert import build_encoder_classifier

SMOKE_MODEL_NAME = "humo"  # pretrained_model_name_or_path del modelo minúsculo de prueba (sin descargas)


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int, float]:
    """Calcula (total_params, trainable_params, trainable_ratio_pct) para la Actividad A7."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio_pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
    return total_params, trainable_params, ratio_pct


def ensure_tokenizer_padding(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> None:
    """Garantiza que el tokenizador y el modelo tengan pad_token_id definido (requerido en decoders con batch > 1)."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    # La cabeza agrupa en el último token distinto de pad_token_id y lo lee de la config de texto (en Qwen3.5 o
    # Gemma 4 no es la raíz). Las dos quedan con el id con que rellena el tokenizador, aunque el modelo trajera otro.
    for cfg in (model.config, model.config.get_text_config()):
        cfg.pad_token_id = tokenizer.pad_token_id



def build_sequence_classifier(
    config: ExperimentConfig,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
) -> PreTrainedModel:
    """Despachador unificado que construye el clasificador de secuencias según model_family.

    dtype y device solo aplican a llm_lora: la base se carga en esa precisión y en ese dispositivo.
    """
    if config.model_family == "encoder":
        assert config.encoder is not None
        return build_encoder_classifier(config.encoder)
    if config.model_family == "llm_lora":
        assert config.llm_lora is not None
        return build_lora_classifier(config.llm_lora, dtype=dtype, device=device)
    raise ValueError(
        f"build_sequence_classifier solo aplica a 'encoder' o 'llm_lora', recibido: '{config.model_family}'"
    )


def create_smoke_model(config: ExperimentConfig, texts: Sequence[str], folder: Path) -> str:
    """Crea el modelo de humo de la familia (BERT o Qwen3 minúsculo, con tokenizador entrenado con `texts`) y devuelve
    su carpeta, para usarla como pretrained_model_name_or_path."""
    if config.model_family == "encoder":
        return scibert.create_smoke_model(texts, folder)
    if config.model_family == "llm_lora":
        return lora.create_smoke_model(texts, folder)
    raise ValueError(f"No hay modelo de humo para model_family='{config.model_family}'")

