"""Servicio aislado de inferencia Granite A/B en CPU."""

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import monotonic
from uuid import uuid4
import hmac
import os

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from peft import PeftModel
from torchao.quantization import Int8WeightOnlyConfig
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TorchAoConfig,
)

from api.adapters import to_inference_request
from api.schemas import PredictionRequest
from src.config.schema import (
    CANONICAL_9_CLASSES,
    ID_TO_LABEL,
    LABEL_TO_ID,
    InputRepresentationConfig,
)
from src.modeling.lora.modelo import registrar_cabeza
from src.modeling.predict import TransformerPredictor


BASE_PATH = os.getenv("GRANITE_BASE_PATH", "/models/base")
ADAPTER_PATHS = {
    "A": os.getenv("GRANITE_A_PATH", "/models/a"),
    "B": os.getenv("GRANITE_B_PATH", "/models/b"),
}

state = {}
inference_lock = Lock()
job_lock = Lock()
job_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="granite-infer")
jobs = {}
GRANITE_SHARED_TOKEN = os.getenv("GRANITE_SHARED_TOKEN", "")


def require_auth(authorization: str | None = Header(default=None)):
    """Constant-time Bearer validation for all externally reachable routes."""
    if len(GRANITE_SHARED_TOKEN) < 32 or not authorization or not hmac.compare_digest(
        authorization, f"Bearer {GRANITE_SHARED_TOKEN}"
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")



@asynccontextmanager
async def lifespan(app: FastAPI):
    if len(GRANITE_SHARED_TOKEN) < 32:
        raise RuntimeError("GRANITE_SHARED_TOKEN debe tener al menos 32 caracteres")
    print("Cargando Granite 4.2 8B INT8...", flush=True)

    torch.set_num_threads(8)

    config = AutoConfig.from_pretrained(
        BASE_PATH,
        num_labels=len(CANONICAL_9_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        local_files_only=True,
    )

    registrar_cabeza(config)

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_PATH,
        config=config,
        quantization_config=TorchAoConfig(
            quant_type=Int8WeightOnlyConfig()
        ),
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )

    print("Cargando adaptadores A y B...", flush=True)

    model = PeftModel.from_pretrained(
        base,
        ADAPTER_PATHS["A"],
        adapter_name="a",
        is_trainable=False,
    )

    model.load_adapter(
        ADAPTER_PATHS["B"],
        adapter_name="b",
        is_trainable=False,
    )

    model.eval()

    predictors = {}

    for mode in ("A", "B"):
        name = mode.lower()
        tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_PATHS[mode],
            local_files_only=True,
        )

        predictors[mode] = TransformerPredictor(
            model=model,
            tokenizer=tokenizer,
            rep_config=InputRepresentationConfig(
                configuration=mode,
                max_seq_length=512,
                include_citing_section=False,
                include_cited_section=True,
                enforce_per_chunk_budget=True,
                top_k_chunks=3,
                strip_cite_tags=False,
            ),
            device="cpu",
            model_name=f"granite-4.2-8b-lora-{name}-int8",
        )

    state["model"] = model
    state["predictors"] = predictors

    print("✅ GRANITE A/B LISTO", flush=True)

    try:
        yield
    finally:
        state.clear()


app = FastAPI(
    title="EvidenCite Granite Runtime",
    lifespan=lifespan,
)


@app.get("/health", dependencies=[Depends(require_auth)])
def health():
    return {
        "status": "ready",
        "model": "granite-4.2-8b",
        "configurations": ["A", "B"],
        "quantization": "int8",
    }


@app.post("/predict", dependencies=[Depends(require_auth)])
def predict(request: PredictionRequest):
    mode = {
        "granite-a": "A",
        "granite-b": "B",
    }.get(request.model_id)

    if mode is None:
        raise HTTPException(
            status_code=422,
            detail="Modelo Granite desconocido.",
        )

    with inference_lock:
        model = state["model"]
        model.set_adapter(mode.lower())

        predictor = state["predictors"][mode]

        try:
            result = predictor.predict(
                to_inference_request(request)
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        return {
            **result.model_dump(),
            "is_demo": False,
        }


# Cloudflare's read timeout can be shorter than CPU inference.
# Submit quickly and poll for completion; the computation stays in a bounded queue.
@app.post("/jobs", dependencies=[Depends(require_auth)], status_code=202)
def submit_job(request: PredictionRequest):
    if request.model_id not in ("granite-a", "granite-b"):
        raise HTTPException(status_code=422, detail="Modelo Granite desconocido")
    with job_lock:
        now = monotonic()
        for job_id, entry in list(jobs.items()):
            if entry["future"].done() and now - entry["created"] > 900:
                del jobs[job_id]
        if sum(not entry["future"].done() for entry in jobs.values()) >= 3:
            raise HTTPException(status_code=429, detail="Granite está ocupado; intenta nuevamente")
        job_id = uuid4().hex
        jobs[job_id] = {"created": now, "future": job_pool.submit(predict, request)}
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_auth)])
def get_job(job_id: str):
    with job_lock:
        entry = jobs.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado o expirado")
    future = entry["future"]
    if not future.done():
        return {"job_id": job_id, "status": "running" if future.running() else "queued"}
    try:
        result = future.result()
    except Exception:
        # Avoid leaking internal errors or model context to the public Internet.
        return {"job_id": job_id, "status": "error", "detail": "Granite no pudo completar la inferencia"}
    return {"job_id": job_id, "status": "done", "result": result}
