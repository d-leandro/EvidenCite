"""Modelo: tokenizador, carga del modelo de clasificación y modelo de humo para depurar sin red (celda 7)."""

from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerFast

from configuracion import Config
from preprocesamiento import ETIQ2ID, ETIQUETAS


CABEZAS = ("score", "classifier")  # nombres de la cabeza de clasificación en transformers


def tokenizador_humo(textos: list[str]):
    """Entrena un tokenizador BPE de 4000 piezas sobre `textos` para el modelo de humo, sin descargar nada."""
    bpe = Tokenizer(BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(textos, trainers.BpeTrainer(
        vocab_size=4000, special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    return PreTrainedTokenizerFast(tokenizer_object=bpe, pad_token="<pad>", unk_token="<unk>",
                                   bos_token="<s>", eos_token="</s>")


def cargar_tokenizer(ruta: str, token_hf: str | None):
    """Carga el tokenizador del modelo, asegura un token de relleno y pone el relleno a la derecha.

    Si el modelo no trae pad, usa EOS. Es seguro porque el texto nunca contiene EOS: verificar_pooling comprueba
    que la cabeza siga leyendo el último token real.
    """
    tok = AutoTokenizer.from_pretrained(ruta, token=token_hf)
    if tok.pad_token_id is None:
        if tok.eos_token is None:
            raise ValueError("El tokenizador no tiene pad ni eos: define tokenizer.pad_token a mano.")
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def argumentos_modelo(opt: dict, var, token_hf: str | None) -> dict:
    """Argumentos de from_pretrained: 9 etiquetas, dtype de la precisión elegida, atención y token.

    En QLoRA añade la cuantización NF4 de la base, con la cabeza sin cuantizar y todo el modelo en la GPU 0.
    """
    kwargs = dict(num_labels=len(ETIQUETAS), id2label=dict(enumerate(ETIQUETAS)), label2id=ETIQ2ID,
                  dtype=opt["dtype"], attn_implementation=opt["atencion"], token=token_hf)
    if var.cuantizar_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=opt["dtype"], llm_int8_skip_modules=list(CABEZAS))
        kwargs["device_map"] = {"": 0}
    return kwargs


def crear_modelo(cfg: Config, opt: dict, var, textos_humo: list[str] | None = None, token_hf: str | None = None):
    """Crea el tokenizador y el modelo de clasificación a partir de `cfg.modelo`. Devuelve (modelo, tokenizador, ruta).

    Para cambiar de modelo basta con cambiar `cfg.modelo` (variable MODELO) por otro id de Hugging Face.
    Esta es la única función con imports dentro: las clases propias de una familia de modelos se importan aquí,
    solo cuando se usan (por ejemplo, las de Llama para el modelo de humo).

    - "humo": construye un Llama diminuto con pesos aleatorios y un tokenizador BPE entrenado con `textos_humo`,
      lo guarda en disco y lo carga por el mismo camino que un modelo real, sin descargas.
    - Carga AutoModelForSequenceClassification con 9 etiquetas. Si la atención pedida no está disponible, prueba
      sdpa y después eager.
    - Desactiva la caché de KV (solo sirve para generar), mueve el modelo al dispositivo (en QLoRA ya está en la
      GPU) y activa gradient checkpointing si se pidió.
    """
    ruta = cfg.modelo
    if cfg.modelo == "humo":
        from transformers import LlamaConfig, LlamaForCausalLM

        carpeta = Path(cfg.salidas_dir) / "_modelo_humo"
        carpeta.mkdir(parents=True, exist_ok=True)
        tok_humo = tokenizador_humo(textos_humo or ["texto de prueba"])
        tok_humo.save_pretrained(carpeta)
        config = LlamaConfig(vocab_size=len(tok_humo), hidden_size=64, intermediate_size=172, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                             pad_token_id=tok_humo.pad_token_id, bos_token_id=tok_humo.bos_token_id,
                             eos_token_id=tok_humo.eos_token_id)
        with torch.random.fork_rng(devices=[]):  # pesos fijos sin alterar la semilla del experimento
            torch.manual_seed(0)
            LlamaForCausalLM(config).save_pretrained(carpeta)
        ruta = str(carpeta)

    tok = cargar_tokenizer(ruta, token_hf)
    kwargs = argumentos_modelo(opt, var, token_hf)
    try:
        modelo = AutoModelForSequenceClassification.from_pretrained(ruta, **kwargs)
    except (ValueError, ImportError) as e:
        texto = str(e).lower()
        if not any(k in texto for k in ("attn", "attention", "sdpa", "flash")):
            raise
        respaldo = "sdpa" if opt["atencion"] != "sdpa" else "eager"
        print(f"AVISO: no se pudo usar atención '{opt['atencion']}' ({str(e)[:200]}); uso '{respaldo}'.")
        opt["atencion"] = kwargs["attn_implementation"] = respaldo
        modelo = AutoModelForSequenceClassification.from_pretrained(ruta, **kwargs)
    # En los modelos multimodales (Qwen3.5, Gemma 3 desde 4B) la cabeza y el modelo de texto leen pad_token_id y
    # use_cache de la config de texto, no de la principal; en los de solo texto ambas son la misma.
    for config in (modelo.config, modelo.config.get_text_config()):
        config.pad_token_id = tok.pad_token_id
        if hasattr(config, "use_cache"):
            config.use_cache = False
    if not var.cuantizar_4bit:
        modelo.to(opt["dispositivo"])
    if cfg.grad_checkpointing and not var.cuantizar_4bit:  # en QLoRA lo activa prepare_model_for_kbit_training
        modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return modelo, tok, ruta


def verificar_pooling(lote: dict, pad_id: int) -> None:
    """La cabeza lee el último token distinto de pad. Comprueba que ese sea el último token real de cada fila."""
    ids, mascara = lote["input_ids"], lote["attention_mask"]
    ultimo_real = mascara.sum(1) - 1
    posiciones = torch.arange(ids.shape[1], device=ids.device)
    ultimo_no_pad = (posiciones * (ids != pad_id)).argmax(1)
    if not torch.equal(ultimo_real, ultimo_no_pad):
        raise RuntimeError("El token de relleno aparece dentro del texto y la cabeza leería una posición equivocada. "
                           "Asigna un pad_token que no aparezca en el texto.")
