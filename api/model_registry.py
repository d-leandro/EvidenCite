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
        if model["id"] == "demo-workflow":
            model["available"] = demo_enabled()

    return models
