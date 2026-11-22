"""Persistent TF-IDF worker managed by FastAPI."""

import json
import os
import subprocess
import threading
from pathlib import Path

from api.schemas import PredictionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORKER_PYTHON = Path(os.getenv(
    "EVIDENCITE_TFIDF_PYTHON",
    "/opt/tfidf-venv/bin/python",
))

MODEL_ROOT = Path(os.getenv(
    "EVIDENCITE_TFIDF_ROOT",
    "/opt/evidencite-models",
))

_lock = threading.Lock()
_process = None


def tfidf_available() -> bool:
    return (
        WORKER_PYTHON.is_file()
        and all(
            (MODEL_ROOT / folder / "tfidf_pipeline.joblib").is_file()
            and (MODEL_ROOT / folder / "input_representation.json").is_file()
            for folder in ("tfidf_A", "tfidf_B")
        )
    )


def _get_worker():
    global _process

    if _process is None or _process.poll() is not None:
        _process = subprocess.Popen(
            [str(WORKER_PYTHON), "-m", "api.workers.tfidf_worker"],
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    return _process


def predict_tfidf(model_id: str, request: PredictionRequest) -> dict:
    if model_id not in ("tfidf-a", "tfidf-b"):
        raise ValueError(f"Unknown TF-IDF model: {model_id}")

    if not tfidf_available():
        raise FileNotFoundError("TF-IDF runtime or artifacts unavailable")

    payload = request.model_dump(
        exclude={"model_id"},
        exclude_none=True,
    )
    payload["model_id"] = model_id

    # Serialize stdin/stdout exchanges to avoid mixing responses.
    with _lock:
        process = _get_worker()

        try:
            process.stdin.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
            process.stdin.flush()
            line = process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("TF-IDF worker disconnected") from exc

    if not line:
        raise RuntimeError(
            f"TF-IDF worker exited unexpectedly: {process.poll()}"
        )

    response = json.loads(line)

    if not response.get("ok"):
        message = response.get("error", "Unknown TF-IDF error")
        if response.get("error_type") in (
            "ValueError",
            "ValidationError",
        ):
            raise ValueError(message)
        raise RuntimeError(message)

    return response["result"]
