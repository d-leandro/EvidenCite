"""Esquemas Pydantic declarativos para validación de configuraciones YAML (Actividades A4–A7)."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
import torch
import yaml

from src.data.builder import CANONICAL_9_CLASSES, map_to_canonical_label

LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(CANONICAL_9_CLASSES)}
ID_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(CANONICAL_9_CLASSES)}


def resolve_canonical_label(label: Optional[str]) -> Optional[str]:
    """Resuelve una etiqueta a su nombre canónico en CANONICAL_9_CLASSES (soporta etiquetas heredadas)."""
    if not label:
        return None
    if label in LABEL_TO_ID:
        return label
    mapped, _ = map_to_canonical_label(label)
    return mapped if mapped in LABEL_TO_ID else None


def resolve_device(device_pref: str = "auto") -> str:
    """Resuelve el dispositivo de ejecución ('cuda', 'mps' o 'cpu') según disponibilidad de hardware."""
    pref = (device_pref or "auto").lower().strip()
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PathsConfig(BaseModel):
    """Rutas base de entrada/salida y tracking de experimentos."""

    train_file: Path = Field(
        default=Path("data/labeled/train.jsonl"),
        description="Ruta al archivo JSONL de entrenamiento.",
    )
    dev_file: Path = Field(
        default=Path("data/labeled/dev.jsonl"),
        description="Ruta al archivo JSONL de validación (dev).",
    )
    test_file: Path = Field(
        default=Path("data/labeled/test.jsonl"),
        description="Ruta al archivo JSONL de prueba (test).",
    )
    output_dir: Path = Field(
        default=Path("models"),
        description="Directorio base donde se almacenan checkpoints, métricas y figuras.",
    )
    mlflow_tracking_uri: str = Field(
        default="",
        description=(
            "Servidor de MLflow (http://ip:puerto). Vacío: MLFLOW_TRACKING_URI, MLFLOW_IP o mlflow_ip.txt. "
            "Sin servidor, los resultados se guardan en output_dir."
        ),
    )
    mlflow_experiment_name: str = Field(
        default="EvidenCite-A4-A7",
        description="Nombre del experimento en MLflow.",
    )


class BaseConfig(BaseModel):
    """Configuración transversal del proyecto (`configs/base.yaml`)."""

    seed: int = Field(default=42, ge=0, description="Semilla global de reproducibilidad.")
    device: Literal["auto", "cuda", "mps", "cpu"] = Field(
        default="auto",
        description="Dispositivo de cómputo preferido.",
    )
    paths: PathsConfig = Field(default_factory=PathsConfig)


class InputRepresentationConfig(BaseModel):
    """Configuración del Eje 1: Representación de Entrada a partir de DatasetInstance."""

    configuration: Literal["context_only", "A", "B"] = Field(
        default="A",
        description=(
            "Representación de entrada: 'context_only' (Ablación solo citante), "
            "'A' (Citante + Título y Abstract del citado) o "
            "'B' (Citante + Top-k Chunks recuperados del citado)."
        ),
    )
    include_citing_section: bool = Field(
        default=False,
        description="Si True, antepone '{rhetorical_section}: ' al citation_context.",
    )
    include_cited_section: bool = Field(
        default=True,
        description="Si True en Config B, antepone '{section_heading}: ' a cada chunk citado.",
    )
    max_seq_length: int = Field(
        default=512,
        ge=32,
        le=8192,
        description="Presupuesto máximo de tokens para la secuencia de entrada (ADR-04: 512).",
    )
    enforce_per_chunk_budget: bool = Field(
        default=True,
        description=(
            "Si True (ADR-04 por defecto), reparte equitativamente los tokens restantes "
            "entre los Top-k chunks garantizando que el chunk 3 entre en la ventana. "
            "Si False (retrocompatible), concatena secuencialmente y trunca al final."
        ),
    )
    top_k_chunks: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Cantidad máxima de chunks recuperados a concatenar en Configuración B.",
    )
    strip_cite_tags: bool = Field(
        default=False,
        description="Si True, elimina las etiquetas '<cite>' y '</cite>' conservando el texto interno.",
    )


class TfidfModelConfig(BaseModel):
    """Hiperparámetros para la línea base clásica TF-IDF + LogisticRegression."""

    ngram_min: int = Field(default=1, ge=1)
    ngram_max: int = Field(default=2, ge=1)
    min_df: int = Field(default=2, ge=1)
    max_df: float = Field(default=0.95, gt=0.0, le=1.0)
    max_features: Optional[int] = Field(default=50000, ge=100)
    sublinear_tf: bool = Field(default=True)
    c_penalty: float = Field(default=1.0, gt=0.0, description="Parámetro de regularización inversa C.")
    class_weight: Optional[Literal["balanced"]] = Field(default="balanced")
    max_iter: int = Field(default=1000, ge=100)

    @model_validator(mode="after")
    def _validate_ngrams(self) -> "TfidfModelConfig":
        if self.ngram_max < self.ngram_min:
            raise ValueError(
                f"ngram_max ({self.ngram_max}) debe ser >= ngram_min ({self.ngram_min})."
            )
        return self


class TrainerOptimizationConfig(BaseModel):
    """Planificador, evaluación y optimizaciones de entrenamiento traídas de lora.zip (comunes a Encoder y LoRA).

    train.py las traduce a parámetros de TrainingArguments. Los valores por defecto conservan el comportamiento
    anterior: planificador lineal, una evaluación por época y ninguna optimización opcional activa.
    """

    lr_scheduler: Literal["linear", "cosine", "cosine_min", "constant", "wsd", "plateau"] = Field(
        default="linear",
        description=(
            "Planificador de la tasa (todos con warmup lineal salvo plateau): 'linear', 'cosine', "
            "'cosine_min' (coseno hasta el 10 %), 'constant', 'wsd' (constante y decae en el 20 % final) o "
            "'plateau' (la tasa baja a la mitad en cada evaluación sin mejora)."
        ),
    )
    evals_per_epoch: int = Field(
        default=1,
        ge=1,
        description="Evaluaciones de dev por época, repartidas en partes iguales (early stopping y mejor punto).",
    )
    early_stopping_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description="Mejora mínima de eval_f1_macro que cuenta como mejora para el early stopping.",
    )
    logging_steps: int = Field(default=10, ge=1, description="Pasos de optimizador entre registros (y MLflow).")
    max_steps: int = Field(default=-1, description="Si es > 0, corta el entrenamiento tras N pasos de optimizador.")
    group_by_length: bool = Field(
        default=False,
        description="Agrupa en cada lote ejemplos de longitud parecida para rellenar menos.",
    )
    gradient_checkpointing: bool = Field(default=False, description="Ahorra memoria a cambio de ~30 % más de tiempo.")
    torch_compile: bool = Field(default=False, description="Compila el modelo con torch.compile (solo CUDA).")
    torch_compile_mode: Optional[
        Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
    ] = Field(default=None)
    attn_implementation: Optional[Literal["sdpa", "eager", "flash_attention_2"]] = Field(
        default=None,
        description="Implementación de la atención; None usa la de transformers (sdpa).",
    )
    dataloader_num_workers: int = Field(default=0, ge=0)


class EncoderModelConfig(TrainerOptimizationConfig):
    """Hiperparámetros para Actividad A4: Encoders Científicos Fine-Tuned (SciBERT / SPECTER2)."""

    pretrained_model_name_or_path: str = Field(
        default="allenai/scibert_scivocab_uncased",
        description="Identificador Hugging Face o ruta local del encoder preentrenado.",
    )
    learning_rate: float = Field(default=2.0e-5, gt=0.0)
    per_device_train_batch_size: int = Field(default=16, ge=1)
    per_device_eval_batch_size: int = Field(default=32, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    num_train_epochs: int = Field(default=5, ge=1)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    classifier_dropout: Optional[float] = Field(default=0.1, ge=0.0, lt=1.0)
    loss_type: Literal["cross_entropy", "weighted_ce", "focal"] = Field(
        default="weighted_ce",
        description="Función de pérdida: 'cross_entropy', 'weighted_ce' o 'focal'.",
    )
    class_weight_scheme: Literal["none", "balanced", "sqrt_balanced"] = Field(
        default="balanced",
        description="Esquema de pesos por clase: 'none', 'balanced' (N/(K*n_c)) o 'sqrt_balanced'.",
    )
    focal_gamma: float = Field(
        default=2.0,
        ge=0.0,
        description="Parámetro de enfoque gamma para Focal Loss.",
    )
    early_stopping_patience: int = Field(default=2, ge=1)
    dynamic_padding: bool = Field(
        default=True,
        description="Si True, delega el padding al batch vía DataCollatorWithPadding.",
    )
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=False)


class LoraModelConfig(TrainerOptimizationConfig):
    """Hiperparámetros para Extensión A4/A5: LLM Liviano de Pesos Abiertos con PEFT/LoRA (SEQ_CLS)."""

    pretrained_model_name_or_path: str = Field(
        default="HuggingFaceTB/SmolLM2-135M",
        description="Modelo base de pesos abiertos para clasificación de secuencias con LoRA.",
    )
    task_type: Literal["SEQ_CLS"] = Field(
        default="SEQ_CLS",
        description="Tipo de tarea PEFT (fijado a SEQ_CLS con cabeza lineal sobre las 9 clases).",
    )
    lora_variant: Literal[
        "lora", "rslora", "dora", "pissa", "olora", "eva", "lora_plus", "lora_fa", "vera", "qlora"
    ] = Field(
        default="lora",
        description="Variante del adaptador (src/modeling/lora/modelo.py, desde lora.zip).",
    )
    r: int = Field(default=16, ge=1, description="Rango r de las matrices de bajo rango LoRA.")
    lora_alpha: int = Field(default=32, ge=1, description="Factor de escala alpha de LoRA.")
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Módulos lineales donde se inyectan adaptadores LoRA; ['all-linear'] = todos salvo la cabeza.",
    )
    modules_to_save: List[str] = Field(
        default_factory=lambda: ["score", "classifier"],
        description="Cabeza de clasificación entrenada junto con los adaptadores.",
    )
    bias: Literal["none", "all", "lora_only"] = Field(default="none")
    learning_rate: float = Field(default=1.0e-4, gt=0.0)
    per_device_train_batch_size: int = Field(default=8, ge=1)
    per_device_eval_batch_size: int = Field(default=16, ge=1)
    gradient_accumulation_steps: int = Field(default=2, ge=1)
    num_train_epochs: int = Field(default=5, ge=1)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    loss_type: Literal["cross_entropy", "weighted_ce", "focal"] = Field(default="weighted_ce")
    class_weight_scheme: Literal["none", "balanced", "sqrt_balanced"] = Field(default="balanced")
    focal_gamma: float = Field(default=2.0, ge=0.0)
    early_stopping_patience: int = Field(default=2, ge=1)
    dynamic_padding: bool = Field(default=True)
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=False)


class LLMApiConfig(BaseModel):
    """Hiperparámetros para Actividad A5: Inferencia Zero/Few-Shot con LLMs vía API (OpenAI / Gemini)."""

    provider: Literal["openai", "gemini", "ollama"] = Field(
        default="openai",
        description="Proveedor del SDK ('openai', 'gemini' nativo con google-genai, u 'ollama' vía base_url).",
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="Nombre del modelo comercial o local (ej. 'gpt-4o', 'gemini-2.5-flash').",
    )
    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Nombre de la variable de entorno que almacena la API key.",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="URL base personalizada (ej. 'http://localhost:11434/v1' para Ollama/vLLM).",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=64)
    thinking_budget: int = Field(
        default=0,
        ge=0,
        description="Presupuesto de thinking tokens en Gemini 2.5 (0 desactiva thinking para paridad en A7).",
    )
    prompt_mode: Literal["generic", "section_conditioned"] = Field(
        default="section_conditioned",
        description="Modo de prompt en A5: 'generic' (sin sección explícita) vs 'section_conditioned'.",
    )
    num_few_shot: int = Field(default=0, ge=0, le=18)
    enforce_structured_output: bool = Field(
        default=True,
        description=(
            "Si True, utiliza validación estricta nativa del SDK (response_format / response_schema). "
            "Si False, solicita JSON en el prompt para evaluar empíricamente invalid_format_rate en A7."
        ),
    )
    cache_path: Optional[Path] = Field(
        default=Path("models/llm_api_cache.jsonl"),
        description="Archivo JSONL de caché por instance_id para evitar re-facturación ante cortes.",
    )
    cost_per_1m_input_tokens_usd: float = Field(default=0.15, ge=0.0)
    cost_per_1m_output_tokens_usd: float = Field(default=0.60, ge=0.0)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)


class ExperimentConfig(BaseModel):
    """Contrato declarativo completo de un experimento de modelamiento y evaluación (A4–A7)."""

    experiment_name: str = Field(..., min_length=1, description="Identificador único del experimento.")
    model_family: Literal["tfidf", "encoder", "llm_lora", "llm_api"] = Field(
        ...,
        description="Familia del modelo evaluado en el Eje 2.",
    )
    seed: int = Field(default=42, ge=0)
    device: Literal["auto", "cuda", "mps", "cpu"] = Field(default="auto")
    paths: PathsConfig = Field(default_factory=PathsConfig)
    input_representation: InputRepresentationConfig = Field(
        default_factory=InputRepresentationConfig
    )
    tfidf: Optional[TfidfModelConfig] = None
    encoder: Optional[EncoderModelConfig] = None
    llm_lora: Optional[LoraModelConfig] = None
    llm_api: Optional[LLMApiConfig] = None

    @model_validator(mode="after")
    def _ensure_family_subconfig(self) -> "ExperimentConfig":
        if self.model_family == "tfidf" and self.tfidf is None:
            self.tfidf = TfidfModelConfig()
        elif self.model_family == "encoder" and self.encoder is None:
            self.encoder = EncoderModelConfig()
        elif self.model_family == "llm_lora" and self.llm_lora is None:
            self.llm_lora = LoraModelConfig()
        elif self.model_family == "llm_api" and self.llm_api is None:
            self.llm_api = LLMApiConfig()
        return self


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Combina recursivamente dos diccionarios dando prioridad a override."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge_dicts(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_base_config(path: Path = Path("configs/base.yaml")) -> BaseConfig:
    """Carga y valida el archivo `configs/base.yaml`."""
    if not path.exists():
        return BaseConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return BaseConfig.model_validate(raw)


def load_experiment_config(
    config_path: Path,
    base_config_path: Optional[Path] = Path("configs/base.yaml"),
) -> ExperimentConfig:
    """Carga un archivo YAML de experimento fusionándolo con `configs/base.yaml` y validándolo con Pydantic."""
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")

    base_dict: Dict[str, Any] = {}
    if base_config_path is not None and base_config_path.exists():
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_dict = yaml.safe_load(f) or {}

    with open(config_path, "r", encoding="utf-8") as f:
        exp_dict = yaml.safe_load(f) or {}

    merged = _deep_merge_dicts(base_dict, exp_dict)
    return ExperimentConfig.model_validate(merged)

