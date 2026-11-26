"""Cliente HTTP para el servicio independiente Granite A/B."""

import os

import httpx

from api.schemas import PredictionRequest


GRANITE_URL = os.getenv(
    "GRANITE_URL",
    "http://granite:8001",
).rstrip("/")


def granite_available() -> bool:
    """Verifica si el servicio Granite terminó de inicializarse."""
    try:
        response = httpx.get(
            f"{GRANITE_URL}/health",
            timeout=3.0,
            trust_env=False,
        )
        return (
            response.status_code == 200
            and response.json().get("status") == "ready"
        )
    except (httpx.HTTPError, ValueError):
        return False


def predict_granite(
    model_id: str,
    request: PredictionRequest,
) -> dict:
    """Delega la inferencia al servicio Granite persistente."""
    if model_id not in ("granite-a", "granite-b"):
        raise ValueError(f"Modelo desconocido: {model_id}")

    try:
        response = httpx.post(
            f"{GRANITE_URL}/predict",
            json=request.model_dump(exclude_none=True),
            timeout=httpx.Timeout(600.0, connect=5.0),
            trust_env=False,
        )
    except httpx.RequestError as exc:
        raise RuntimeError(
            "No fue posible comunicarse con Granite."
        ) from exc

    if response.status_code == 422:
        detail = response.json().get("detail", "Solicitud inválida.")
        raise ValueError(str(detail))

    if not response.is_success:
        raise RuntimeError(
            f"Granite respondió HTTP {response.status_code}."
        )

    return response.json()
