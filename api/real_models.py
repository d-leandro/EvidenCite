"""Runtime for real citation classifiers, using local artifacts."""

import os
from functools import lru_cache
from pathlib import Path

from api.adapters import to_inference_request
from api.schemas import PredictionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = (
    Path.home()
    / "evidencite-mlflow-backup"
    / "test-revisado"
)

SCIBERT_MODELS = {
    "scibert-a": ("scibert_A", "model_config_a.yaml"),
    "scibert-b": ("scibert_B", "model_config_b.yaml"),
}

REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def scibert_path(model_id: str) -> Path:
    folder, _ = SCIBERT_MODELS[model_id]
    env_name = f"EVIDENCITE_{model_id.upper().replace('-', '_')}_PATH"
    default_path = BACKUP_ROOT / folder
    if model_id == "scibert-b":
        default_path = default_path / "artifacts"

    if model_id == "scibert-b":
        env_name = "EVIDENCITE_SCIBERT_B_PATH"

    return Path(os.getenv(env_name, str(default_path))).expanduser()


def scibert_available(model_id: str) -> bool:
    path = scibert_path(model_id)
    return all((path / filename).is_file() for filename in REQUIRED_FILES)


@lru_cache(maxsize=2)
def load_scibert(model_id: str):
    import yaml

    from src.config.schema import InputRepresentationConfig
    from src.modeling.predict import TransformerPredictor

    if not scibert_available(model_id):
        raise FileNotFoundError(
            f"Artifacts unavailable for {model_id}: {scibert_path(model_id)}"
        )

    _, config_name = SCIBERT_MODELS[model_id]
    config_path = PROJECT_ROOT / "configs" / config_name

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rep_config = InputRepresentationConfig(
        **config["input_representation"]
    )

    return TransformerPredictor.from_pretrained(
        model_dir=scibert_path(model_id),
        rep_config=rep_config,
        device="cpu",
        model_name=model_id,
    )


def predict_scibert(model_id: str, request: PredictionRequest) -> dict:
    inference_request = to_inference_request(request)
    prediction = load_scibert(model_id).predict(inference_request)

    return {
        **prediction.model_dump(),
        "is_demo": False,
    }


# Compatibilidad con los nombres utilizados anteriormente.
def scibert_b_available() -> bool:
    return scibert_available("scibert-b")


def load_scibert_b():
    return load_scibert("scibert-b")


def predict_scibert_b(request: PredictionRequest) -> dict:
    return predict_scibert("scibert-b", request)
