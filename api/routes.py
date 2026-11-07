from fastapi import APIRouter, HTTPException

from api.schemas import PredictionRequest, PrepareContextRequest
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


@router.post("/prepare-context")
def prepare_context(request: PrepareContextRequest):
    if request.target_citation not in request.citing_context:
        raise HTTPException(
            status_code=422,
            detail="Target citation not found in citing context.",
        )

    from src.modeling.predict import prepare_inference_citing_context

    prepared = prepare_inference_citing_context(
        request.citing_context,
        request.target_citation,
    )

    return {"citing_context": prepared}
