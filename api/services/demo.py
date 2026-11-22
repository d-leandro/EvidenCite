"""Temporary predictor for end-to-end integration testing."""

import hashlib

from api.schemas import PredictionRequest
from src.data.taxonomy import CANONICAL_9_CLASSES


def predict_demo(request: PredictionRequest) -> dict:
    labels = CANONICAL_9_CLASSES

    # Deterministic synthetic prediction, NOT machine learning.
    digest = hashlib.sha256(
        request.citing_context.encode("utf-8")
    ).hexdigest()

    predicted_label = labels[int(digest, 16) % len(labels)]

    probabilities = {
        label: 0.2 if label == predicted_label else 0.1
        for label in labels
    }

    return {
        "model_name": "demo-workflow",
        "configuration": "B" if request.cited_chunks else "A",
        "predicted_label": predicted_label,
        "confidence": 0.2,
        "probabilities": probabilities,
        "latency_ms": 0.0,
        "is_demo": True,
        "warning": "Synthetic prediction. Not a real ML inference.",
    }
