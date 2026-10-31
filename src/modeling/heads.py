"""Fábrica de modelos de clasificación de secuencias (AutoModelForSequenceClassification + PEFT/LoRA)."""

from typing import Tuple

from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config.schema import (
    CANONICAL_9_CLASSES,
    ID_TO_LABEL,
    LABEL_TO_ID,
    EncoderModelConfig,
    ExperimentConfig,
    LoraModelConfig,
)


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
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def build_encoder_classifier(encoder_cfg: EncoderModelConfig) -> PreTrainedModel:
    """Instancia AutoModelForSequenceClassification nativo de Hugging Face sobre las 9 clases canónicas."""
    hf_config = AutoConfig.from_pretrained(
        encoder_cfg.pretrained_model_name_or_path,
        num_labels=len(CANONICAL_9_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    if encoder_cfg.classifier_dropout is not None:
        if hasattr(hf_config, "classifier_dropout"):
            hf_config.classifier_dropout = encoder_cfg.classifier_dropout
        if hasattr(hf_config, "hidden_dropout_prob"):
            hf_config.hidden_dropout_prob = encoder_cfg.classifier_dropout

    model = AutoModelForSequenceClassification.from_pretrained(
        encoder_cfg.pretrained_model_name_or_path,
        config=hf_config,
        ignore_mismatched_sizes=True,
    )
    return model


def build_lora_classifier(lora_cfg: LoraModelConfig) -> PreTrainedModel:
    """Instancia AutoModelForSequenceClassification envuelto con PEFT/LoRA (TaskType.SEQ_CLS)."""
    hf_config = AutoConfig.from_pretrained(
        lora_cfg.pretrained_model_name_or_path,
        num_labels=len(CANONICAL_9_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        lora_cfg.pretrained_model_name_or_path,
        config=hf_config,
        ignore_mismatched_sizes=True,
    )

    # Filtramos modules_to_save para incluir únicamente atributos presentes en el modelo base
    model_module_names = {name.split(".")[0] for name, _ in base_model.named_modules() if name}
    active_modules_to_save = [
        m for m in lora_cfg.modules_to_save if m in model_module_names
    ] or None

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        modules_to_save=active_modules_to_save,
        bias=lora_cfg.bias,
    )
    peft_model = get_peft_model(base_model, peft_config)
    return peft_model


def build_sequence_classifier(config: ExperimentConfig) -> PreTrainedModel:
    """Despachador unificado que construye el clasificador de secuencias según model_family."""
    if config.model_family == "encoder":
        assert config.encoder is not None
        return build_encoder_classifier(config.encoder)
    if config.model_family == "llm_lora":
        assert config.llm_lora is not None
        return build_lora_classifier(config.llm_lora)
    raise ValueError(
        f"build_sequence_classifier solo aplica a 'encoder' o 'llm_lora', recibido: '{config.model_family}'"
    )

