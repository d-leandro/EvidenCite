"""Inferencia Gemini para la API interactiva de EvidenCite."""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from api.adapters import to_inference_request
from api.schemas import PredictionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GEMINI_CONFIGS = {
    "gemini-a": "llm_api_gemini38flash_config_a.yaml",
    "gemini-b": "llm_api_gemini38flash_config_b.yaml",
}


def gemini_available() -> bool:
    """Indica si hay credenciales configuradas, no disponibilidad remota."""
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


@lru_cache(maxsize=2)
def load_gemini(model_id: str):
    """Construye el predictor usando las configuraciones originales."""
    if model_id not in GEMINI_CONFIGS:
        raise ValueError(f"Modelo Gemini desconocido: {model_id}")

    if not gemini_available():
        raise RuntimeError("GEMINI_API_KEY no está configurada.")

    from google import genai
    from google.genai import types

    from src.config.schema import load_experiment_config
    from src.modeling.predict import LLMApiPredictor

    config = load_experiment_config(
        PROJECT_ROOT / "configs" / GEMINI_CONFIGS[model_id],
        base_config_path=PROJECT_ROOT / "configs" / "base.yaml",
    )

    # Inferencia interactiva exclusivamente Zero-Shot.
    # Deshabilitamos la caché persistente de experimentos.
    llm_config = config.llm_api.model_copy(
        update={
        "model_name": "gemini-3.5-flash-lite",
        "thinking_level": None,
        "thinking_budget": None,
        "seed": None,
            "num_few_shot": 0,
            "shot_variants": None,
            "cache_path": None,
            "max_retries": 2,
        }
    )

    vertex = os.getenv(
        "GOOGLE_GENAI_USE_VERTEXAI", "false"
    ).lower() in ("1", "true", "yes")

    client = genai.Client(
        vertexai=vertex,
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=45000),
    )

    return LLMApiPredictor(
        llm_config=llm_config,
        rep_config=config.input_representation,
        client=client,
    )


def predict_gemini(
    model_id: str,
    request: PredictionRequest,
) -> dict:
    """Ejecuta una predicción conservando el contrato de EvidenCite."""
    inference_request = to_inference_request(request)

    # La caché debe distinguir cambios en título, abstract y chunks.
    payload = request.model_dump(
        exclude={"model_id"},
        exclude_none=True,
    )

    fingerprint = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    inference_request = inference_request.model_copy(
        update={"instance_id": fingerprint}
    )

    prediction = load_gemini(model_id).predict(inference_request)

    return {
        **prediction.model_dump(),
        "is_demo": False,
    }
