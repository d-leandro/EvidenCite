from fastapi import APIRouter, HTTPException

from api.schemas import PredictionRequest
from api.model_registry import list_models
from src.data.builder import CANONICAL_9_CLASSES

router = APIRouter(prefix="/api/v1")


@router.get("/labels")
def get_labels():
    return {"labels": CANONICAL_9_CLASSES}


@router.post("/predict")
def predict(request: PredictionRequest):
    raise HTTPException(
        status_code=503,
        detail="Inference models are not registered yet.",
    )


@router.get("/models")
def get_models():
    return {"models": list_models()}
