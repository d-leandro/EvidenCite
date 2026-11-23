import os

from fastapi import APIRouter, HTTPException

from api.schemas import PredictionRequest, PrepareContextRequest
from api.model_registry import list_models
from api.services.demo import predict_demo
from src.data.builder import CANONICAL_9_CLASSES

router = APIRouter(prefix="/api/v1")


@router.get("/labels")
def get_labels():
    return {"labels": CANONICAL_9_CLASSES}


def _predict_impl(request: PredictionRequest):
    if request.model_id in ("tfidf-a", "tfidf-b"):
        from api.runtimes.tfidf import tfidf_available, predict_tfidf

        if not tfidf_available():
            raise HTTPException(
                status_code=503,
                detail="TF-IDF runtime is unavailable.",
            )

        try:
            return predict_tfidf(request.model_id, request)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except (RuntimeError, OSError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"TF-IDF inference failed: {exc}",
            ) from exc

    if request.model_id == "scibert-a":
        from api.runtimes.scibert import scibert_available, predict_scibert
        if not scibert_available("scibert-a"):
            raise HTTPException(
                status_code=503,
                detail="SciBERT A artifacts are unavailable.",
            )
        try:
            return predict_scibert("scibert-a", request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if request.model_id == "scibert-b":
        from api.runtimes.scibert import (
            scibert_b_available,
            predict_scibert_b,
        )
        if not scibert_b_available():
            raise HTTPException(
                status_code=503,
                detail="SciBERT B artifacts are unavailable.",
            )
        try:
            return predict_scibert_b(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if request.model_id in ("gemini-a", "gemini-b"):
        from api.runtimes.gemini import (
            gemini_available,
            predict_gemini,
        )

        if not gemini_available():
            raise HTTPException(
                status_code=503,
                detail="Gemini API key is not configured.",
            )

        try:
            return predict_gemini(request.model_id, request)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gemini inference failed. "
                    "The Google API may be temporarily unavailable."
                ),
            ) from exc

    if request.model_id in ("granite-a", "granite-b"):
        from api.runtimes.granite_client import (
            granite_available,
            predict_granite,
        )

        if not granite_available():
            raise HTTPException(
                status_code=503,
                detail="El servicio Granite no está disponible.",
            )

        try:
            return predict_granite(request.model_id, request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if request.model_id == "demo-workflow":
        enabled = os.getenv("ENABLE_DEMO_MODEL", "").lower() in ("true", "1", "yes")
        if not enabled:
            raise HTTPException(status_code=503, detail="Demo model is disabled.")
        return predict_demo(request)

    raise HTTPException(
        status_code=503,
        detail="Inference models are not registered yet.",
    )


@router.post("/predict")
def predict(request: PredictionRequest):
    """Measure every real model with one common server-side clock."""
    import time
    from api.services.metrics import log_inference

    start = time.perf_counter()
    try:
        result = _predict_impl(request)
    except Exception as exc:
        log_inference(request.model_id, None,
                      (time.perf_counter() - start) * 1000.0, error=exc)
        raise
    duration_ms = (time.perf_counter() - start) * 1000.0
    run_id = log_inference(request.model_id, result, duration_ms)
    payload = (result if isinstance(result, dict) else
               result.model_dump(mode="json") if hasattr(result, "model_dump") else {})
    return {**payload, "run_id": run_id, "server_ms": round(duration_ms, 2)}


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
    from api.services.paper_lookup import lookup_paper

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
    from api.services.evidence import get_retriever

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
