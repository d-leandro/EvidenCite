"""Definición del clasificador encoder (SciBERT / SPECTER2) y su modelo de humo."""

from pathlib import Path
from typing import Sequence

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BertConfig,
    BertForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from src.config.schema import CANONICAL_9_CLASSES, ID_TO_LABEL, LABEL_TO_ID, EncoderModelConfig


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
        attn_implementation=encoder_cfg.attn_implementation,
    )
    return model


def create_smoke_model(texts: Sequence[str], folder: Path) -> str:
    """Crea en `folder` un BERT minúsculo con pesos aleatorios y un tokenizador WordPiece de 1.000 piezas entrenado
    con `texts`. Devuelve la carpeta, que se carga como cualquier encoder de Hugging Face (sin descargas).

    Sirve para verificar en segundos el pipeline completo de la rama encoder (pretrained_model_name_or_path: humo).
    """
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors, trainers
    from tokenizers.models import WordPiece

    especiales = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    wordpiece = Tokenizer(WordPiece(unk_token="[UNK]"))
    wordpiece.normalizer = normalizers.BertNormalizer(lowercase=True)
    wordpiece.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    wordpiece.decoder = decoders.WordPiece()
    wordpiece.train_from_iterator(
        texts, trainers.WordPieceTrainer(vocab_size=1000, special_tokens=especiales, show_progress=False)
    )
    cls_id, sep_id = wordpiece.token_to_id("[CLS]"), wordpiece.token_to_id("[SEP]")
    wordpiece.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=wordpiece,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    config = BertConfig(
        vocab_size=1000,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
    )
    folder.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):  # pesos fijos sin alterar la semilla del experimento
        torch.manual_seed(0)
        BertForMaskedLM(config).save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return str(folder)
