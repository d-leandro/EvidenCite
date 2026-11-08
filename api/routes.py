import os

from fastapi import APIRouter, HTTPException

from api.schemas import PredictionRequest, PrepareContextRequest
from api.model_registry import list_models
from api.demo import predict_demo
from src.data.builder import CANONICAL_9_CLASSES

router = APIRouter(prefix="/api/v1")


@router.get("/labels")
def get_labels():
    return {"labels": CANONICAL_9_CLASSES}


@router.post("/predict")
def predict(request: PredictionRequest):
    if request.model_id == "demo-workflow":
        enabled = os.getenv("ENABLE_DEMO_MODEL", "").lower() in ("true", "1", "yes")
        if not enabled:
            raise HTTPException(status_code=503, detail="Demo model is disabled.")
        return predict_demo(request)

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


@router.get("/papers/lookup")
def lookup_cited_paper(query: str):
    from api.paper_lookup import lookup_paper

    if len(query.strip()) < 4:
        raise HTTPException(
            status_code=422,
            detail="Please provide a valid title or arXiv ID.",
        )

    return lookup_paper(query)



from api.schemas import EvidenceRequest


@router.post("/evidence")
def retrieve_evidence(request: EvidenceRequest):
    from src.modeling.predict import (
        InferenceRequest,
        resolve_request_evidence,
    )

    inference_request = InferenceRequest(**request.model_dump())
    from api.evidence import get_retriever

    result = resolve_request_evidence(
        inference_request,
        retriever=get_retriever(),
        top_k=3,
    )

    chunks = result.cited_chunks or []
    sections = result.cited_chunk_sections or []

    if not chunks:
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve evidence from the cited paper.",
        )

    return {
        "cited_title": result.cited_title,
        "cited_abstract": result.cited_abstract,
        "cited_arxiv_id": result.cited_arxiv_id,
        "cited_chunks": chunks,
        "cited_chunk_sections": sections,
        "count": len(chunks),
    }
