"""Inferencia unificada y servicios de predicción para TF-IDF, Encoders/LoRA y LLMs Comerciales (Actividades A5, A8)."""

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from src.config.schema import (
    CANONICAL_9_CLASSES,
    ID_TO_LABEL,
    LABEL_TO_ID,
    ExperimentConfig,
    InputRepresentationConfig,
    LLMApiConfig,
    resolve_canonical_label,
    resolve_device,
)
from src.data.schemas import DatasetInstance
from src.modeling.dataset import (
    CitationIntentDataset,
    format_instance_as_single_text,
)
from src.modeling.evaluate import (
    OperationalMetricsA7,
    TechnicalMetricsA6,
    compute_a6_metrics,
)
from src.modeling.prompts import (
    LLMResponse,
    build_llm_messages,
)


CITATION_REGEX = re.compile(
    r"\([^)]*?\b(?:19|20)\d{2}[a-z]?[^)]*?\)|"
    r"\[\d+(?:[,\s–-]+\d+)*\]|"
    r"\b[A-Z][a-zA-Z\s.-]+(?:\s+et\s+al\.?)?\s*\((?:19|20)\d{2}[a-z]?\)"
)


def prepare_inference_citing_context(raw_text: str, target_citation: str) -> str:
    """Prepara un texto libre para inferencia 'in the wild': inserta <cite>...</cite> en target_citation
    y sustituye cualquier otra cita co-ocurrente secundaria como [OTHERCIT] (ADR-03 / A8).
    """
    target = target_citation.strip()
    if not target or target not in raw_text:
        return raw_text

    sentinel = "__TARGET_CITATION_SENTINEL__"
    text_with_sentinel = raw_text.replace(target, sentinel, 1)

    def _mask_secondary(match: re.Match) -> str:
        matched_str = match.group(0).strip()
        if sentinel in matched_str:
            return matched_str
        return "[OTHERCIT]"

    masked_text = CITATION_REGEX.sub(_mask_secondary, text_with_sentinel)
    return masked_text.replace(sentinel, f"<cite>{target}</cite>")


class InferenceRequest(BaseModel):
    """Solicitud formal de inferencia para un contexto de cita (ADR-11 / Actividad A8)."""

    citing_context: str = Field(
        ...,
        description="Texto citante crudo (in the wild) o pre-formateado con <cite>...</cite>.",
    )
    target_citation: Optional[str] = Field(
        default=None,
        description="Fragmento textual de la cita objetivo (ej. 'Devlin et al., 2018') para inyectar etiquetas <cite>.",
    )
    rhetorical_section: Optional[str] = Field(
        default=None,
        description="Sección retórica del paper citante (ej. 'Introduction', 'Methodology').",
    )
    cited_arxiv_id: Optional[str] = Field(
        default=None,
        description="ID de arXiv del paper citado para resolución automática por el orquestador en A8.",
    )
    cited_title: Optional[str] = Field(
        default=None,
        description="Título del paper citado (usado en Configuración A).",
    )
    cited_abstract: Optional[str] = Field(
        default=None,
        description="Abstract del paper citado (usado en Configuración A).",
    )
    cited_chunks: Optional[List[str]] = Field(
        default=None,
        description="Lista de chunks de evidencia del paper citado (usado en Configuración B).",
    )
    instance_id: Optional[str] = Field(
        default=None,
        description="Identificador opcional de la cita para trazabilidad y caché.",
    )

    @classmethod
    def from_instance(cls, instance: DatasetInstance) -> "InferenceRequest":
        """Convierte un DatasetInstance del pipeline en un InferenceRequest tipado."""
        cited = instance.cited_paper
        title: Optional[str] = None
        abstract: Optional[str] = None
        paper_id: Optional[str] = None
        chunks_text: List[str] = []

        if cited is not None:
            if isinstance(cited, dict):
                title = cited.get("title")
                abstract = cited.get("abstract")
                paper_id = cited.get("paper_id") or cited.get("id")
                chunks_raw = cited.get("retrieved_chunks", []) or []
                chunks_sorted = sorted(
                    chunks_raw,
                    key=lambda c: c.get("rank", 999) if isinstance(c, dict) else getattr(c, "rank", 999),
                )
                chunks_text = [
                    (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", ""))
                    for c in chunks_sorted
                ]
            else:
                title = getattr(cited, "title", None)
                abstract = getattr(cited, "abstract", None)
                paper_id = getattr(cited, "id", None) or getattr(cited, "paper_id", None)
                chunks_raw = getattr(cited, "retrieved_chunks", []) or []
                chunks_sorted = sorted(
                    chunks_raw,
                    key=lambda c: getattr(c, "rank", 999) if hasattr(c, "rank") else 999,
                )
                chunks_text = [
                    getattr(c, "text", "")
                    for c in chunks_sorted
                    if getattr(c, "text", "")
                ]

        chunks_clean = [t for t in chunks_text if t]

        citing = getattr(instance, "citing_paper", None)
        sec: Optional[str] = None
        if isinstance(citing, dict):
            sec = citing.get("rhetorical_section")
        elif citing is not None:
            sec = getattr(citing, "rhetorical_section", None)
        else:
            sec = getattr(instance, "rhetorical_section", None)

        return cls(
            citing_context=instance.citation_context,
            rhetorical_section=sec,
            cited_arxiv_id=paper_id,
            cited_title=title,
            cited_abstract=abstract,
            cited_chunks=chunks_clean or None,
            instance_id=instance.instance_id,
        )


class PredictionRecord(BaseModel):
    """Registro estandarizado de predicción individual para cualquier familia de modelos (A4–A8)."""

    instance_id: Optional[str] = None
    predicted_label: str
    probabilities: Dict[str, float] = Field(
        ...,
        description="Mapa de probabilidad/puntaje sobre cada una de las 9 clases canónicas.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Puntaje de confianza asignado a la clase predicha.",
    )
    latency_ms: float = Field(default=0.0, ge=0.0)
    rationale: Optional[str] = Field(
        default=None,
        description="Explicación textual opcional generada por LLMs comerciales.",
    )
    model_name: Optional[str] = None
    configuration: Optional[str] = None
    is_valid_format: bool = Field(
        default=True,
        description="True si la salida cumplió el contrato esperado (relevante para LLMs en A7).",
    )
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    thoughts_tokens: int = Field(default=0, ge=0)


class BasePredictor(ABC):
    """Clase base abstracta para todos los predictores de intención de cita (Strategy Pattern)."""

    def __init__(self, rep_config: InputRepresentationConfig) -> None:
        self.rep_config = rep_config

    def _normalize_request(
        self,
        request_or_text: Optional[Union[InferenceRequest, DatasetInstance, str]] = None,
        target_citation: Optional[str] = None,
        **kwargs: Any,
    ) -> InferenceRequest:
        """Normaliza cualquier entrada a un InferenceRequest validado y pre-procesado."""
        kwargs_clean = dict(kwargs)
        if isinstance(request_or_text, DatasetInstance):
            req = InferenceRequest.from_instance(request_or_text)
        elif isinstance(request_or_text, InferenceRequest):
            req = request_or_text
        elif isinstance(request_or_text, str):
            kwargs_clean.pop("citing_context", None)
            req = InferenceRequest(
                citing_context=request_or_text,
                target_citation=target_citation,
                **kwargs_clean,
            )
        elif request_or_text is None and "citing_context" in kwargs_clean:
            req = InferenceRequest(
                target_citation=target_citation,
                **kwargs_clean,
            )
        else:
            raise TypeError(f"Entrada inválida para inferencia: {type(request_or_text)}")

        # Preprocesamiento de contexto in-the-wild si target_citation está presente
        if req.target_citation and "<cite>" not in req.citing_context:
            prepared_ctx = prepare_inference_citing_context(
                req.citing_context, req.target_citation
            )
            req = req.model_copy(update={"citing_context": prepared_ctx})

        # Validación de evidencia según la configuración del modelo
        cfg_mode = self.rep_config.configuration.upper()
        if cfg_mode == "A":
            if not req.cited_title and not req.cited_abstract:
                raise ValueError(
                    "Configuración A requiere 'cited_title' o 'cited_abstract' como evidencia."
                )
        elif cfg_mode == "B":
            if not req.cited_chunks:
                raise ValueError(
                    "Configuración B requiere al menos un chunk en 'cited_chunks' como evidencia."
                )

        return req

    def _format_request_text(self, req: InferenceRequest) -> str:
        """Formatea el request como texto único según la configuración de representación."""
        ctx = req.citing_context
        if self.rep_config.strip_cite_tags:
            ctx = re.sub(r"</?cite>", "", ctx)

        if self.rep_config.include_citing_section and req.rhetorical_section:
            ctx = f"{req.rhetorical_section}: {ctx}"

        evidence_parts: List[str] = []
        cfg_mode = self.rep_config.configuration.upper()
        if cfg_mode == "A":
            if req.cited_title:
                evidence_parts.append(req.cited_title)
            if req.cited_abstract:
                evidence_parts.append(req.cited_abstract)
        elif cfg_mode == "B":
            if req.cited_chunks:
                evidence_parts.extend(req.cited_chunks[: self.rep_config.top_k_chunks])

        if evidence_parts:
            return f"{ctx}\n" + ". ".join(evidence_parts)
        return ctx

    def _to_dataset_instance(self, req: InferenceRequest) -> DatasetInstance:
        """Crea una instancia sintética en memoria compatible con CitationIntentDataset y prompts."""
        cited_dict: Dict[str, Any] = {
            "id": req.cited_arxiv_id or "cited_adhoc",
            "title": req.cited_title,
            "abstract": req.cited_abstract,
            "retrieved_chunks": [
                {
                    "chunk_id": f"chk_{i}",
                    "section_heading": "Section",
                    "text": chk,
                    "similarity_score": 1.0,
                    "rank": i + 1,
                }
                for i, chk in enumerate(req.cited_chunks or [])
            ],
        }
        citing_dict: Dict[str, Any] = {
            "id": "citing_adhoc",
            "id_source": "arxiv",
            "rhetorical_section": req.rhetorical_section or "Unknown Section",
        }
        meta_dict: Dict[str, Any] = {
            "source_dataset": "adhoc_inference",
            "is_gold": False,
        }

        return DatasetInstance(
            instance_id=req.instance_id or "adhoc_inference",
            label="Background",  # Placeholder
            annotation_metadata=meta_dict,  # type: ignore[arg-type]
            citing_paper=citing_dict,  # type: ignore[arg-type]
            citation_context=req.citing_context,
            cited_paper=cited_dict,  # type: ignore[arg-type]
        )

    @abstractmethod
    def predict(
        self,
        request: Optional[Union[InferenceRequest, DatasetInstance, str]] = None,
        target_citation: Optional[str] = None,
        **kwargs: Any,
    ) -> PredictionRecord:
        """Ejecuta inferencia sobre una única solicitud y retorna un PredictionRecord unificado."""
        pass

    @abstractmethod
    def predict_batch(
        self,
        requests: Sequence[Union[InferenceRequest, DatasetInstance, str]],
        batch_size: int = 32,
    ) -> List[PredictionRecord]:
        """Ejecuta inferencia por lotes optimizada para evaluación sobre datasets."""
        pass


class TfidfPredictor(BasePredictor):
    """Predictor para baselines de aprendizaje estadístico con Scikit-Learn TF-IDF."""

    def __init__(
        self,
        pipeline: Pipeline,
        rep_config: InputRepresentationConfig,
        model_name: str = "tfidf_baseline",
    ) -> None:
        super().__init__(rep_config)
        self.pipeline = pipeline
        self.model_name = model_name
        self.clf_classes = list(self.pipeline.named_steps["clf"].classes_)

    @classmethod
    def from_path(
        cls,
        model_path: Union[str, Path],
        rep_config: InputRepresentationConfig,
        model_name: str = "tfidf_baseline",
    ) -> "TfidfPredictor":
        pipe = joblib.load(str(model_path))
        return cls(pipeline=pipe, rep_config=rep_config, model_name=model_name)

    def predict(
        self,
        request: Optional[Union[InferenceRequest, DatasetInstance, str]] = None,
        target_citation: Optional[str] = None,
        **kwargs: Any,
    ) -> PredictionRecord:
        norm_req = self._normalize_request(request, target_citation, **kwargs)
        single_text = self._format_request_text(norm_req)

        t0 = time.perf_counter()
        proba_matrix = self.pipeline.predict_proba([single_text])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs_9 = {cls_name: 0.0 for cls_name in CANONICAL_9_CLASSES}
        for col_idx, cls_id in enumerate(self.clf_classes):
            cls_name = ID_TO_LABEL.get(int(cls_id))
            if cls_name is not None:
                probs_9[cls_name] = float(proba_matrix[col_idx])

        pred_label = max(probs_9.items(), key=lambda kv: kv[1])[0]
        confidence = probs_9[pred_label]

        return PredictionRecord(
            instance_id=norm_req.instance_id,
            predicted_label=pred_label,
            probabilities=probs_9,
            confidence=confidence,
            latency_ms=float(latency_ms),
            model_name=self.model_name,
            configuration=self.rep_config.configuration,
            is_valid_format=True,
        )

    def predict_batch(
        self,
        requests: Sequence[Union[InferenceRequest, DatasetInstance, str]],
        batch_size: int = 32,
    ) -> List[PredictionRecord]:
        if not requests:
            return []
        norm_reqs = [self._normalize_request(r) for r in requests]
        texts = [self._format_request_text(r) for r in norm_reqs]

        t0 = time.perf_counter()
        proba_matrix = self.pipeline.predict_proba(texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_inst_ms = elapsed_ms / len(requests)

        records: List[PredictionRecord] = []
        for row_idx, req in enumerate(norm_reqs):
            probs_9 = {cls_name: 0.0 for cls_name in CANONICAL_9_CLASSES}
            for col_idx, cls_id in enumerate(self.clf_classes):
                cls_name = ID_TO_LABEL.get(int(cls_id))
                if cls_name is not None:
                    probs_9[cls_name] = float(proba_matrix[row_idx, col_idx])

            pred_label = max(probs_9.items(), key=lambda kv: kv[1])[0]
            records.append(
                PredictionRecord(
                    instance_id=req.instance_id,
                    predicted_label=pred_label,
                    probabilities=probs_9,
                    confidence=probs_9[pred_label],
                    latency_ms=float(per_inst_ms),
                    model_name=self.model_name,
                    configuration=self.rep_config.configuration,
                    is_valid_format=True,
                )
            )
        return records


class TransformerPredictor(BasePredictor):
    """Predictor para modelos basados en Transformers (SciBERT, DeBERTa o LLMs con LoRA SEQ_CLS)."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        rep_config: InputRepresentationConfig,
        device: str = "auto",
        model_name: str = "transformer_classifier",
    ) -> None:
        super().__init__(rep_config)
        self.model = model
        self.tokenizer = tokenizer
        self.device_str = resolve_device(device)
        self.target_device = torch.device(self.device_str)
        self.model = self.model.to(self.target_device)
        self.model.eval()
        self.model_name = model_name

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        rep_config: InputRepresentationConfig,
        device: str = "auto",
        model_name: Optional[str] = None,
    ) -> "TransformerPredictor":
        path_str = str(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(path_str)
        model = AutoModelForSequenceClassification.from_pretrained(path_str)
        name = model_name or Path(path_str).name
        return cls(
            model=model,
            tokenizer=tokenizer,
            rep_config=rep_config,
            device=device,
            model_name=name,
        )

    def _to_dataset_instance(self, req: InferenceRequest) -> DatasetInstance:
        """Crea una instancia sintética en memoria compatible con CitationIntentDataset."""
        cited_dict: Dict[str, Any] = {
            "id": req.cited_arxiv_id or "cited_adhoc",
            "title": req.cited_title,
            "abstract": req.cited_abstract,
            "retrieved_chunks": [
                {
                    "chunk_id": f"chk_{i}",
                    "section_heading": "Section",
                    "text": chk,
                    "similarity_score": 1.0,
                    "rank": i + 1,
                }
                for i, chk in enumerate(req.cited_chunks or [])
            ],
        }
        citing_dict: Dict[str, Any] = {
            "id": "citing_adhoc",
            "id_source": "arxiv",
            "rhetorical_section": req.rhetorical_section or "Unknown Section",
        }
        meta_dict: Dict[str, Any] = {
            "source_dataset": "adhoc_inference",
            "is_gold": False,
        }

        return DatasetInstance(
            instance_id=req.instance_id or "adhoc_inference",
            label="Background",  # Placeholder
            annotation_metadata=meta_dict,  # type: ignore[arg-type]
            citing_paper=citing_dict,  # type: ignore[arg-type]
            citation_context=req.citing_context,
            cited_paper=cited_dict,  # type: ignore[arg-type]
        )

    def predict(
        self,
        request: Optional[Union[InferenceRequest, DatasetInstance, str]] = None,
        target_citation: Optional[str] = None,
        **kwargs: Any,
    ) -> PredictionRecord:
        req_item = request if request is not None else kwargs.get("citing_context", "")
        records = self.predict_batch([req_item], batch_size=1, target_citation=target_citation, **kwargs)
        return records[0]

    def predict_batch(
        self,
        requests: Sequence[Union[InferenceRequest, DatasetInstance, str]],
        batch_size: int = 32,
        target_citation: Optional[str] = None,
        **kwargs: Any,
    ) -> List[PredictionRecord]:
        if not requests:
            return []

        norm_reqs = [
            self._normalize_request(r, target_citation=target_citation, **kwargs)
            for r in requests
        ]
        instances = [self._to_dataset_instance(r) for r in norm_reqs]

        use_dyn_pad = hasattr(self.tokenizer, "pad")
        dataset = CitationIntentDataset(
            tokenizer=self.tokenizer,
            instances=instances,
            rep_config=self.rep_config,
            dynamic_padding=use_dyn_pad,
        )
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer) if use_dyn_pad else None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        records: List[PredictionRecord] = []
        offset = 0

        with torch.no_grad():
            for batch in loader:
                cur_bs = batch["input_ids"].shape[0]
                model_inputs = {
                    k: v.to(self.target_device)
                    for k, v in batch.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")
                }
                t0 = time.perf_counter()
                outputs = self.model(**model_inputs)
                batch_ms = (time.perf_counter() - t0) * 1000.0
                per_item_ms = batch_ms / max(1, cur_bs)

                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                probs_tensor = torch.softmax(logits, dim=-1).cpu().numpy()

                for i in range(cur_bs):
                    req = norm_reqs[offset + i]
                    row_probs = probs_tensor[i]
                    probs_9 = {
                        CANONICAL_9_CLASSES[c_idx]: float(row_probs[c_idx])
                        for c_idx in range(len(CANONICAL_9_CLASSES))
                    }
                    pred_id = int(np.argmax(row_probs))
                    pred_label = ID_TO_LABEL[pred_id]
                    records.append(
                        PredictionRecord(
                            instance_id=req.instance_id,
                            predicted_label=pred_label,
                            probabilities=probs_9,
                            confidence=probs_9[pred_label],
                            latency_ms=float(per_item_ms),
                            model_name=self.model_name,
                            configuration=self.rep_config.configuration,
                            is_valid_format=True,
                        )
                    )
                offset += cur_bs

        return records


def _clean_json_text(raw_text: str) -> str:
    """Extrae el bloque JSON limpiando cercas Markdown ```json ... ``` si existen."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def parse_llm_json_response(raw_text: str) -> Tuple[LLMResponse, bool]:
    """Valida una respuesta textual del LLM contra el contrato LLMResponse (Forma 2)."""
    try:
        cleaned = _clean_json_text(raw_text)
        parsed = LLMResponse.model_validate_json(cleaned)
        return parsed, True
    except Exception:
        uniform = {cls_name: round(1.0 / 9.0, 4) for cls_name in CANONICAL_9_CLASSES}
        fallback = LLMResponse.model_validate(
            {"predicted_label": "Background", "scores": uniform, "rationale": "Fallback parsing error"}
        )
        return fallback, False


class LLMApiPredictor(BasePredictor):
    """Cliente unificado de inferencia Zero/Few-Shot para OpenAI, Google Gemini y Ollama/vLLM (Forma 2)."""

    def __init__(
        self,
        llm_config: LLMApiConfig,
        rep_config: InputRepresentationConfig,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(rep_config)
        self.llm_config = llm_config
        self._client = client
        self._cache: Dict[str, PredictionRecord] = {}
        self._load_cache()

    def _cache_key(self, instance_id: str) -> str:
        return (
            f"{instance_id}::{self.rep_config.configuration}::"
            f"{self.llm_config.prompt_mode}::{self.llm_config.model_name}"
        )

    def _load_cache(self) -> None:
        cache_path = self.llm_config.cache_path
        if cache_path is None or not cache_path.exists():
            return
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    key = payload.get("cache_key")
                    rec_data = payload.get("record")
                    if key and rec_data:
                        self._cache[key] = PredictionRecord.model_validate(rec_data)
                except Exception:
                    continue

    def _append_to_cache(self, key: str, record: PredictionRecord) -> None:
        self._cache[key] = record
        cache_path = self.llm_config.cache_path
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"cache_key": key, "record": record.model_dump()}
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _get_or_create_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = os.getenv(self.llm_config.api_key_env, "")
        if self.llm_config.provider in ("openai", "ollama"):
            from openai import OpenAI

            effective_key = api_key or ("ollama" if self.llm_config.provider == "ollama" else "")
            self._client = OpenAI(
                api_key=effective_key,
                base_url=self.llm_config.base_url,
                max_retries=self.llm_config.max_retries,
                timeout=self.llm_config.timeout_seconds,
            )
            return self._client

        if self.llm_config.provider == "gemini":
            from google import genai

            self._client = genai.Client(api_key=api_key)
            return self._client

        raise ValueError(f"Proveedor LLM no soportado: {self.llm_config.provider}")

    def _call_openai_or_ollama(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[LLMResponse, bool, int, int, int]:
        client = self._get_or_create_client()
        if self.llm_config.enforce_structured_output:
            completion = client.beta.chat.completions.parse(
                model=self.llm_config.model_name,
                messages=messages,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_output_tokens,
                response_format=LLMResponse,
            )
            usage = getattr(completion, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            parsed_msg = completion.choices[0].message.parsed
            if isinstance(parsed_msg, LLMResponse):
                return parsed_msg, True, in_tok, out_tok, 0
            raw_content = completion.choices[0].message.content or ""
            resp, valid = parse_llm_json_response(raw_content)
            return resp, valid, in_tok, out_tok, 0

        completion = client.chat.completions.create(
            model=self.llm_config.model_name,
            messages=messages,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_output_tokens,
        )
        usage = getattr(completion, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        raw_content = completion.choices[0].message.content or ""
        resp, valid = parse_llm_json_response(raw_content)
        return resp, valid, in_tok, out_tok, 0

    def _call_gemini_native(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[LLMResponse, bool, int, int, int]:
        from google.genai import types

        client = self._get_or_create_client()
        system_instruction = ""
        user_parts: List[str] = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role_tag = "User" if msg["role"] == "user" else "Assistant"
                user_parts.append(f"[{role_tag}]\n{msg['content']}")

        combined_contents = "\n\n".join(user_parts)
        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_instruction or None,
            "temperature": self.llm_config.temperature,
            "max_output_tokens": self.llm_config.max_output_tokens,
            "thinking_config": types.ThinkingConfig(
                thinking_budget=self.llm_config.thinking_budget
            ),
        }
        if self.llm_config.enforce_structured_output:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = LLMResponse

        gen_config = types.GenerateContentConfig(**config_kwargs)
        response = client.models.generate_content(
            model=self.llm_config.model_name,
            contents=combined_contents,
            config=gen_config,
        )

        usage = getattr(response, "usage_metadata", None)
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
        thought_tok = int(getattr(usage, "thoughts_token_count", 0) or 0)

        parsed_obj = getattr(response, "parsed", None)
        if isinstance(parsed_obj, LLMResponse):
            return parsed_obj, True, in_tok, out_tok, thought_tok

        raw_text = getattr(response, "text", "") or ""
        resp, valid = parse_llm_json_response(raw_text)
        return resp, valid, in_tok, out_tok, thought_tok

    def predict(
        self,
        request: Optional[Union[InferenceRequest, DatasetInstance, str]] = None,
        target_citation: Optional[str] = None,
        few_shot_examples: Optional[Sequence[DatasetInstance]] = None,
        **kwargs: Any,
    ) -> PredictionRecord:
        norm_req = self._normalize_request(request, target_citation=target_citation, **kwargs)
        cache_id = norm_req.instance_id or norm_req.citing_context
        key = self._cache_key(cache_id)
        if key in self._cache:
            return self._cache[key]

        if isinstance(request, DatasetInstance):
            temp_inst = request
        else:
            temp_inst = self._to_dataset_instance(norm_req)

        messages = build_llm_messages(
            instance=temp_inst,
            rep_config=self.rep_config,
            prompt_mode=self.llm_config.prompt_mode,
            few_shot_examples=few_shot_examples,
        )

        t0 = time.perf_counter()
        if self.llm_config.provider in ("openai", "ollama"):
            resp, valid, in_tok, out_tok, thought_tok = self._call_openai_or_ollama(messages)
        elif self.llm_config.provider == "gemini":
            resp, valid, in_tok, out_tok, thought_tok = self._call_gemini_native(messages)
        else:
            raise ValueError(f"Proveedor no soportado: {self.llm_config.provider}")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        scores_dict = resp.scores.to_dict_canonical()
        pred_label = resp.predicted_label
        confidence = scores_dict.get(pred_label, max(scores_dict.values()))

        record = PredictionRecord(
            instance_id=norm_req.instance_id,
            predicted_label=pred_label,
            probabilities=scores_dict,
            confidence=confidence,
            latency_ms=float(latency_ms),
            rationale=resp.rationale,
            model_name=self.llm_config.model_name,
            configuration=self.rep_config.configuration,
            is_valid_format=valid,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thoughts_tokens=thought_tok,
        )
        self._append_to_cache(key, record)
        return record

    def predict_batch(
        self,
        requests: Sequence[Union[InferenceRequest, DatasetInstance, str]],
        batch_size: int = 32,
        few_shot_examples: Optional[Sequence[DatasetInstance]] = None,
        **kwargs: Any,
    ) -> List[PredictionRecord]:
        return [
            self.predict(req, few_shot_examples=few_shot_examples, **kwargs)
            for req in requests
        ]


# ---------------------------------------------------------------------------
# Factory Registry Pattern con Caché en Memoria (Singleton per Model/Config)
# ---------------------------------------------------------------------------
_PREDICTOR_CACHE: Dict[str, BasePredictor] = {}


def create_predictor(
    config: ExperimentConfig,
    model_or_pipe: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    use_cache: bool = True,
) -> BasePredictor:
    """Fábrica de predictores polimórficos con caché en memoria de modelos cargados."""
    cache_key = f"{config.experiment_name}::{config.model_family}::{config.input_representation.configuration}"
    if use_cache and cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]

    predictor: BasePredictor
    if config.model_family == "tfidf":
        if model_or_pipe is not None:
            predictor = TfidfPredictor(
                pipeline=model_or_pipe,
                rep_config=config.input_representation,
                model_name=config.experiment_name,
            )
        else:
            model_file = config.paths.output_dir / config.experiment_name / "tfidf_pipeline.joblib"
            predictor = TfidfPredictor.from_path(
                model_path=model_file,
                rep_config=config.input_representation,
                model_name=config.experiment_name,
            )

    elif config.model_family in ("encoder", "llm_lora"):
        if model_or_pipe is not None and tokenizer is not None:
            predictor = TransformerPredictor(
                model=model_or_pipe,
                tokenizer=tokenizer,
                rep_config=config.input_representation,
                device=config.device,
                model_name=config.experiment_name,
            )
        else:
            best_dir = config.paths.output_dir / config.experiment_name / "best_model"
            predictor = TransformerPredictor.from_pretrained(
                model_dir=best_dir,
                rep_config=config.input_representation,
                device=config.device,
                model_name=config.experiment_name,
            )

    elif config.model_family == "llm_api":
        if config.llm_api is None:
            raise ValueError("config.llm_api es requerido para model_family='llm_api'")
        predictor = LLMApiPredictor(
            llm_config=config.llm_api,
            rep_config=config.input_representation,
            client=model_or_pipe,
        )

    else:
        raise ValueError(f"Familia de modelo no soportada para predictor: {config.model_family}")

    if use_cache:
        _PREDICTOR_CACHE[cache_key] = predictor
    return predictor


# ---------------------------------------------------------------------------
# Funciones Retrocompatibles para Pipelines Existentes y Evaluación
# ---------------------------------------------------------------------------
def predict_tfidf_batch(
    pipeline: Pipeline,
    instances: Sequence[DatasetInstance],
    rep_config: InputRepresentationConfig,
) -> List[PredictionRecord]:
    """Retrocompatibilidad: Inferencia batch sobre TF-IDF."""
    pred = TfidfPredictor(pipeline=pipeline, rep_config=rep_config)
    return pred.predict_batch(instances)


def predict_transformer_batch(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    instances: Sequence[DatasetInstance],
    rep_config: InputRepresentationConfig,
    batch_size: int = 32,
    device: str = "auto",
) -> List[PredictionRecord]:
    """Retrocompatibilidad: Inferencia batch sobre SciBERT / LoRA."""
    pred = TransformerPredictor(
        model=model,
        tokenizer=tokenizer,
        rep_config=rep_config,
        device=device,
    )
    return pred.predict_batch(instances, batch_size=batch_size)


def run_llm_api_experiment(
    config: ExperimentConfig,
    eval_instances: Optional[Sequence[DatasetInstance]] = None,
    few_shot_examples: Optional[Sequence[DatasetInstance]] = None,
    custom_client: Optional[Any] = None,
) -> Tuple[List[PredictionRecord], TechnicalMetricsA6, OperationalMetricsA7]:
    """Ejecuta una corrida experimental completa para LLMs vía API (Actividad A5)."""
    if config.llm_api is None:
        raise ValueError("Se requiere el bloque `llm_api` en ExperimentConfig para run_llm_api_experiment.")

    llm_cfg = config.llm_api
    rep_cfg = config.input_representation

    if eval_instances is None:
        from src.modeling.dataset import load_jsonl_instances

        eval_instances = load_jsonl_instances(config.paths.test_file)

    predictor = LLMApiPredictor(
        llm_config=llm_cfg,
        rep_config=rep_cfg,
        client=custom_client,
    )

    records = predictor.predict_batch(eval_instances, few_shot_examples=few_shot_examples)

    true_ids = [
        LABEL_TO_ID[resolve_canonical_label(inst.label)]  # type: ignore[index]
        for inst in eval_instances
    ]
    pred_ids = [LABEL_TO_ID[r.predicted_label] for r in records]
    metrics_a6 = compute_a6_metrics(true_ids, pred_ids)

    n_total = max(1, len(records))
    mean_latency_ms = float(sum(r.latency_ms for r in records)) / float(n_total)
    throughput = (1000.0 / mean_latency_ms) if mean_latency_ms > 0 else 0.0
    invalid_rate = float(sum(1 for r in records if not r.is_valid_format)) / float(n_total)

    total_in_tokens = sum(r.input_tokens for r in records)
    total_out_tokens = sum(r.output_tokens + r.thoughts_tokens for r in records)
    est_cost_usd = (
        (total_in_tokens / 1_000_000.0) * llm_cfg.cost_per_1m_input_tokens_usd
        + (total_out_tokens / 1_000_000.0) * llm_cfg.cost_per_1m_output_tokens_usd
    )

    metrics_a7 = OperationalMetricsA7(
        model_family="llm_api",
        configuration=rep_cfg.configuration,
        total_params=0,
        trainable_params=0,
        trainable_ratio_pct=0.0,
        train_wall_time_sec=0.0,
        inference_latency_ms_per_instance=mean_latency_ms,
        throughput_instances_per_sec=throughput,
        peak_memory_mb=0.0,
        invalid_format_rate=invalid_rate,
        estimated_api_cost_usd=float(est_cost_usd),
    )

    exp_dir = config.paths.output_dir / config.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "experiment_name": config.experiment_name,
        "metrics_a6": metrics_a6.model_dump(),
        "metrics_a7": metrics_a7.model_dump(),
    }
    (exp_dir / "llm_api_metrics.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records, metrics_a6, metrics_a7
