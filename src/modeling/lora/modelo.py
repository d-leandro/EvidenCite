"""Definición del clasificador LLM con PEFT/LoRA (SEQ_CLS): las variantes de lora.zip y su modelo de humo."""

import contextlib
from pathlib import Path
import tempfile
from typing import Iterable, Optional, Sequence

from peft import (
    EvaConfig,
    LoraConfig,
    TaskType,
    VeraConfig,
    get_peft_model,
    initialize_lora_eva_weights,
    prepare_model_for_kbit_training,
)
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from transformers.modeling_layers import GenericForSequenceClassification
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES, MODEL_MAPPING

from src.config.schema import CANONICAL_9_CLASSES, ID_TO_LABEL, LABEL_TO_ID, LoraModelConfig

CLASSIFIER_HEADS = ("score", "classifier")  # nombres de la cabeza de clasificación en transformers
LORA_PLUS_LR_RATIO = 16.0  # LoRA+: la tasa de B es 16 veces la de A
MUTATES_BASE = ("pissa", "olora")  # modifican W al iniciar: se guardan convertidos a LoRA estándar

def _registrar_cabeza(hf_config) -> None:
    """Gemma 4, Nemotron-H, Granite… no traen ForSequenceClassification en transformers: registra la cabeza
    genérica (lineal sobre el último token que no es relleno), la misma que usan Qwen3 o Llama."""
    if hf_config.model_type in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES:
        return
    base = MODEL_MAPPING[type(hf_config)]  # p. ej. Gemma4Model
    preentrenado = next(c for c in base.__mro__[1:] if c.__name__.endswith("PreTrainedModel"))
    clase = type(base.__name__.removesuffix("Model") + "ForSequenceClassification",
                 (GenericForSequenceClassification, preentrenado), {})
    AutoModelForSequenceClassification.register(type(hf_config), clase, exist_ok=True)
    print(f"AVISO: {hf_config.model_type} no trae cabeza de clasificación; uso {clase.__name__} (genérica).")

def _peft_config(lora_cfg: LoraModelConfig, target_modules, modules_to_save):
    """Configuración de PEFT de la variante. lora_plus, lora_fa y qlora usan LoRA estándar: cambian el optimizador
    (create_lora_optimizer) o la carga de la base (4 bits)."""
    if lora_cfg.lora_variant == "vera":
        # Proyecciones aleatorias compartidas entre capas; solo se entrenan vectores de escala (por eso r alto).
        return VeraConfig(
            task_type=TaskType.SEQ_CLS,
            r=256,
            target_modules=target_modules,
            vera_dropout=lora_cfg.lora_dropout,
            d_initial=0.1,
            save_projection=True,
            modules_to_save=modules_to_save,
            bias=lora_cfg.bias,
        )
    extra = {
        "rslora": {"use_rslora": True},  # escala α/√r en vez de α/r
        "dora": {"use_dora": True},  # separa magnitud y dirección de W
        "pissa": {"init_lora_weights": "pissa_niter_4"},  # componentes principales de W (SVD rápida)
        "olora": {"init_lora_weights": "olora"},  # base ortonormal (QR de W)
        "eva": {"init_lora_weights": "eva", "eva_config": EvaConfig(rho=2.0)},  # varianza de las activaciones
    }.get(lora_cfg.lora_variant, {})
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias=lora_cfg.bias,
        **extra,
    )


def _trainables_to_fp32(model: torch.nn.Module) -> None:
    """Pasa a fp32 los pesos que se optimizan (la cabeza copiada en modules_to_save hereda el dtype de la base)."""
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()


def build_lora_classifier(
    lora_cfg: LoraModelConfig,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
) -> PreTrainedModel:
    """Instancia AutoModelForSequenceClassification envuelto con PEFT (TaskType.SEQ_CLS) en la variante de lora_cfg.

    - dtype: precisión de los pesos de la base (bf16/fp16 con precisión mixta); los entrenables quedan en fp32.
    - device: la base se mueve allí antes de crear el adaptador (PiSSA, OLoRA y EVA lo inician a partir de W).
    - qlora: base cuantizada a 4 bits (NF4); sin CUDA se entrena LoRA estándar sin cuantizar.
    - eva: el adaptador queda sin iniciar hasta llamar a initialize_eva con lotes de train.
    """
    variant = lora_cfg.lora_variant
    quantize = variant == "qlora" and torch.cuda.is_available()
    if variant == "qlora" and not quantize:
        print("AVISO: qlora necesita CUDA y bitsandbytes; se entrena LoRA estándar sin cuantizar.")

    hf_config = AutoConfig.from_pretrained(
        lora_cfg.pretrained_model_name_or_path,
        num_labels=len(CANONICAL_9_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    hf_config.use_cache = False  # la caché de KV solo sirve para generar
    quantization = {}
    if quantize:
        quantization = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype or torch.bfloat16,
                llm_int8_skip_modules=list(CLASSIFIER_HEADS),
            ),
            "device_map": {"": 0},
        }

    hf_config.use_cache = False
    hf_config.get_text_config().use_cache = False  # C2: en los multimodales la config de texto es otra
    _registrar_cabeza(hf_config)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        lora_cfg.pretrained_model_name_or_path,
        config=hf_config,
        ignore_mismatched_sizes=True,
        dtype=dtype or torch.float32,
        attn_implementation=lora_cfg.attn_implementation,
        **quantization,
    )
    if quantize:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=lora_cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif device is not None:
        base_model.to(device)

    # Filtramos modules_to_save para incluir únicamente atributos presentes en el modelo base
    model_module_names = {name.split(".")[0] for name, _ in base_model.named_modules() if name}
    active_modules_to_save = [
        m for m in lora_cfg.modules_to_save if m in model_module_names
    ] or None
    # "all-linear": PEFT toma todas las capas lineales salvo la cabeza de clasificación
    target_modules = "all-linear" if lora_cfg.target_modules == ["all-linear"] else list(lora_cfg.target_modules)

    peft_config = _peft_config(lora_cfg, target_modules, active_modules_to_save)
    peft_model = get_peft_model(base_model, peft_config, low_cpu_mem_usage=variant == "eva")
    if variant in MUTATES_BASE:
        # Adaptador inicial: permite guardar el final convertido a LoRA estándar (save_converted_adapter)
        initial_dir = Path(tempfile.mkdtemp(prefix="lora_")) / "adaptador_inicial"
        peft_model.save_pretrained(initial_dir)
        peft_model.initial_adapter_dir = str(initial_dir)
    _trainables_to_fp32(peft_model)
    return peft_model


def initialize_eva(model: PreTrainedModel, batches: Iterable[dict], autocast_dtype: Optional[torch.dtype]) -> None:
    """EVA: inicia el adaptador con las direcciones de mayor varianza de las activaciones de `batches` (de train)."""
    device = next(p for p in model.parameters() if p.device.type != "meta").device
    lotes = [{k: v.to(device) for k, v in b.items() if k in ("input_ids", "attention_mask")} for b in batches]
    contexto = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
    with contexto:
        initialize_lora_eva_weights(model, dataloader=lotes, show_progress_bar=False)
    _trainables_to_fp32(model)


def create_lora_optimizer(
    model: torch.nn.Module,
    lora_cfg: LoraModelConfig,
    learning_rate: float,
    weight_decay: float,
    fused: bool,
) -> Optional[torch.optim.Optimizer]:
    """Optimizador propio de lora_plus (tasa de B × 16) y lora_fa (A congelada); None en las demás variantes."""
    from peft.optimizers import create_lorafa_optimizer, create_loraplus_optimizer

    if lora_cfg.lora_variant == "lora_plus":
        return create_loraplus_optimizer(
            model=model,
            optimizer_cls=torch.optim.AdamW,
            lr=learning_rate,
            loraplus_lr_ratio=LORA_PLUS_LR_RATIO,
            weight_decay=weight_decay,
            **({"fused": True} if fused else {}),
        )
    if lora_cfg.lora_variant == "lora_fa":
        return create_lorafa_optimizer(
            model=model,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    return None


def save_converted_adapter(model: PreTrainedModel, folder: Path) -> None:
    """Guarda un adaptador PiSSA/OLoRA convertido a LoRA estándar, aplicable sobre la base original."""
    model.save_pretrained(folder, path_initial_model_for_weight_conversion=model.initial_adapter_dir)
    # La conversión de PEFT 0.21 omite los modules_to_save (la cabeza entrenada): sin ellos no se puede recargar.
    archivo = Path(folder) / "adapter_model.safetensors"
    convertido = load_file(archivo)
    faltan = {
        k: v.detach().cpu().contiguous()
        for k, v in get_peft_model_state_dict(model).items()
        if k not in convertido and ".lora_" not in k
    }
    if faltan:
        convertido.update(faltan)
        save_file(convertido, archivo, metadata={"format": "pt"})


def create_smoke_model(texts: Sequence[str], folder: Path) -> str:
    """Crea en `folder` un Qwen3 minúsculo con pesos aleatorios y un tokenizador BPE de 1.000 piezas entrenado con
    `texts` (sin BOS, con <|endoftext|> como eos y pad, como Qwen3-Base). Devuelve la carpeta, que se carga como
    cualquier modelo de Hugging Face (sin descargas) y recibe el mismo LoRA que un modelo real.
    """
    from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
    from tokenizers.models import BPE
    from transformers import Qwen3Config, Qwen3ForCausalLM

    fin_texto = "<|endoftext|>"
    bpe = Tokenizer(BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=1000,
            special_tokens=[fin_texto],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token=fin_texto, pad_token=fin_texto)

    config = Qwen3Config(
        vocab_size=1000,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        bos_token_id=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    folder.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):  # pesos fijos sin alterar la semilla del experimento
        torch.manual_seed(0)
        Qwen3ForCausalLM(config).save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return str(folder)
