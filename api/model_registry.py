from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: str
    configuration: str
    available: bool = False


MODELS = {
    "qwen3-8b-lora-b": ModelSpec(
        id="qwen3-8b-lora-b",
        name="Qwen3 8B + LoRA",
        family="transformer",
        configuration="B",
    ),
}


def list_models():
    return [asdict(model) for model in MODELS.values()]
