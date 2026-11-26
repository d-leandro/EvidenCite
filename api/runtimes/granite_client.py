"""Authenticated, asynchronous Granite A/B client over Cloudflare Tunnel."""
import os
import time
from urllib.parse import urlsplit

import httpx
from api.schemas import PredictionRequest

GRANITE_URL = os.getenv("GRANITE_URL", "").rstrip("/")
GRANITE_SHARED_TOKEN = os.getenv("GRANITE_SHARED_TOKEN", "")


def _configured() -> bool:
    parsed = urlsplit(GRANITE_URL)
    # Only HTTPS remote services; do not leak the bearer token to localhost or plaintext HTTP.
    return parsed.scheme == "https" and bool(parsed.hostname) and len(GRANITE_SHARED_TOKEN) >= 32


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {GRANITE_SHARED_TOKEN}"}


def granite_available() -> bool:
    if not _configured():
        return False
    try:
        r = httpx.get(f"{GRANITE_URL}/health", headers=_headers(), timeout=8.0, trust_env=False)
        return r.status_code == 200 and r.json().get("status") == "ready"
    except (httpx.HTTPError, ValueError):
        return False


def predict_granite(model_id: str, request: PredictionRequest) -> dict:
    if model_id not in ("granite-a", "granite-b"):
        raise ValueError(f"Modelo desconocido: {model_id}")
    if not _configured():
        raise RuntimeError("Granite remoto no está configurado")
    try:
        with httpx.Client(headers=_headers(), timeout=httpx.Timeout(20.0, connect=5.0), trust_env=False) as client:
            response = client.post(f"{GRANITE_URL}/jobs", json=request.model_dump(exclude_none=True))
            if response.status_code == 422:
                raise ValueError(str(response.json().get("detail", "Solicitud inválida")))
            if response.status_code == 429:
                raise RuntimeError("Granite está ocupado; intenta de nuevo")
            response.raise_for_status()
            job_id = response.json()["job_id"]
            if not isinstance(job_id, str) or len(job_id) != 32:
                raise RuntimeError("Granite devolvió un identificador de trabajo inválido")
            deadline = time.monotonic() + 570
            while time.monotonic() < deadline:
                time.sleep(3)
                poll = client.get(f"{GRANITE_URL}/jobs/{job_id}")
                poll.raise_for_status()
                data = poll.json()
                if data.get("status") == "done":
                    return data["result"]
                if data.get("status") == "error":
                    raise RuntimeError(str(data.get("detail", "Inferencia Granite fallida")))
                if data.get("status") not in ("queued", "running"):
                    raise RuntimeError("Estado de trabajo Granite desconocido")
    except httpx.HTTPError as exc:
        raise RuntimeError("No fue posible comunicarse con Granite remoto") from exc
    raise RuntimeError("Granite agotó el tiempo de espera (570 segundos)")
