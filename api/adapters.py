from api.schemas import PredictionRequest


def to_inference_request(request: PredictionRequest):
    """Convert API input to the existing inference contract."""
    from src.modeling.predict import InferenceRequest

    data = request.model_dump(
        exclude={"model_id"},
        exclude_none=True,
    )
    return InferenceRequest(**data)
