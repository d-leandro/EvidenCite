import os
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: str
    configuration: str
    available: bool = False


def demo_enabled() -> bool:
    return os.getenv(
        "ENABLE_DEMO_MODEL", ""
    ).lower() in ("true", "1", "yes")


MODELS = {
    "tfidf-a": ModelSpec(
        id="tfidf-a",
        name="TF-IDF + Logistic Regression — Configuration A",
        family="tfidf",
        configuration="A",
    ),
    "tfidf-b": ModelSpec(
        id="tfidf-b",
        name="TF-IDF + Logistic Regression — Configuration B",
        family="tfidf",
        configuration="B",
    ),
    "scibert-a": ModelSpec(
        id="scibert-a",
        name="SciBERT — Configuration A (Real Model)",
        family="encoder",
        configuration="A",
    ),
    "scibert-b": ModelSpec(
        id="scibert-b",
        name="SciBERT — Configuration B (Real Model)",
        family="encoder",
        configuration="B",
    ),
    "qwen3-8b-lora-b": ModelSpec(
        id="qwen3-8b-lora-b",
        name="Qwen3 8B + LoRA",
        family="transformer",
        configuration="B",
    ),
    "demo-workflow": ModelSpec(
        id="demo-workflow",
        name="Demo (Synthetic Predictions)",
        family="demo",
        configuration="A/B",
    ),
}


def list_models():
    models = [asdict(model) for model in MODELS.values()]

    for model in models:
        if model["id"].startswith("tfidf-"):
            from api.tfidf_runtime import tfidf_available
            model["available"] = tfidf_available()
        if model["id"] == "scibert-a":
            from api.real_models import scibert_available
            model["available"] = scibert_available("scibert-a")

        if model["id"] == "scibert-b":
            from api.real_models import scibert_b_available
            model["available"] = scibert_b_available()

        if model["id"] == "demo-workflow":
            model["available"] = demo_enabled()

    return models
