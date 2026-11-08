"""Servicio desacoplado y reutilizable de etiquetado débil mediante Zero-Shot NLI.

Módulo: src.data.weak_supervision.nli_tagger
Diseño Técnico: documentacion/diseno-tecnico/11-diseno-weak-supervision-nli-tagger.md
Decisiones Relacionadas: ADR-10 (Taxonomía Meta-Síntesis), ADR-08 (Trazabilidad y Calidad de Etiqueta).

Proporciona la inferencia semántica por lotes, encuadre discursivo (Framing Devices),
calibración softmax de verosimilitud y el filtro estricto de pureza de dos umbrales
para ser consumido en memoria por DatasetBuilder en builder.py.
"""

from collections import Counter
from dataclasses import dataclass, field
import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.schemas import NLITaggingResult

logger = logging.getLogger(__name__)

# Catálogo Formal de las 9 Hipótesis Canónicas (ADR-10 / Meta-síntesis)
CANONICAL_NLI_HYPOTHESES: Dict[str, str] = {
    "Background": (
        "The cited paper is mentioned to provide historical background, survey the literature, "
        "or describe the general state of the art in the domain."
    ),
    "Gap": (
        "The cited paper is referenced to identify an unsolved problem, point out a research gap, "
        "or highlight an open area that needs further study."
    ),
    "Basis": (
        "The cited paper serves as the intellectual foundation, core theoretical pillar, "
        "or conceptual starting point for the current study."
    ),
    "Comparison": (
        "The authors compare their work with the cited paper, highlighting similarities, "
        "differences, advantages, or conflicting findings."
    ),
    "Application": (
        "The authors directly use, apply, or run an existing tool, algorithm, dataset, "
        "or method from the cited paper without making changes."
    ),
    "Improvement": (
        "The authors modify, optimize, adapt, or extend a method, technique, "
        "or algorithm proposed in the cited paper to improve it."
    ),
    "Evidence": (
        "The cited paper is cited as empirical evidence or factual proof to substantiate, "
        "justify, or support a claim, finding, or hypothesis."
    ),
    "Originator": (
        "The cited paper is credited as the original pioneer, historical author, "
        "or first publication that introduced a concept, term, or method."
    ),
    "Further Reading": (
        "The citation simply points the reader to external supplementary reading, "
        "related literature, or deeper material for additional details."
    ),
}


@dataclass
class NLITaggerConfig:
    """Configuración del servicio de inferencia NLI."""

    enabled: bool = True
    model_name_or_path: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    fp16: bool = True
    confidence_threshold: float = 0.50
    margin_threshold: float = 0.15
    temperature: float = 0.6
    quota_buffer: int = 300
    candidate_chunk_size: int = 1000
    hypotheses: Dict[str, str] = field(default_factory=lambda: dict(CANONICAL_NLI_HYPOTHESES))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NLITaggerConfig":
        """Instancia la configuración a partir de un diccionario."""
        valid_keys = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path],
        section_key: str = "nli",
        base_path: Optional[Union[str, Path]] = Path("configs/base.yaml"),
    ) -> "NLITaggerConfig":
        """Carga la configuración desde un archivo YAML (ej. configs/builder.yaml), heredando de base_path."""
        path = Path(yaml_path)
        base_dict: Dict[str, Any] = {}
        if base_path is not None:
            bp = Path(base_path)
            if bp.exists():
                with open(bp, "r", encoding="utf-8") as f:
                    base_dict = yaml.safe_load(f) or {}

        if not path.exists():
            logger.warning("Archivo de configuración %s no existe. Usando defaults.", path)
            return cls(device=str(base_dict.get("device", cls.device)))

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        nli_cfg = dict(cfg.get(section_key, {}))
        if "device" not in nli_cfg and "device" in base_dict:
            nli_cfg["device"] = base_dict["device"]
        return cls.from_dict(nli_cfg)


class NLIPromptFormatter:
    """Gestiona el encuadre discursivo (Framing Devices) y el formateo de premisas."""

    @staticmethod
    def clean_section_heading(section: Optional[str]) -> str:
        """Limpia y normaliza el encabezado retórico de la sección."""
        if not section:
            return "Context"
        cleaned = section.strip()
        if not cleaned or cleaned.isdigit() or len(cleaned) < 2:
            return "Context"
        return cleaned

    @classmethod
    def format_premise(cls, section: Optional[str], context: str) -> str:
        """Construye la premisa discursiva: Section: [S] . Context: [C]."""
        sec = cls.clean_section_heading(section)
        clean_ctx = context.strip() if context else ""
        return f"Section: {sec} . Context: {clean_ctx}"


class NLIPurityFilter:
    """Evalúa la pureza y calibración estadística de las predicciones de implicación."""

    @staticmethod
    def compute_softmax(
        logits: Union[List[float], torch.Tensor],
        temperature: float = 1.0,
    ) -> List[float]:
        """Aplica la transformación softmax con temperatura sobre un vector de logits."""
        temp = max(1e-4, float(temperature))
        if isinstance(logits, list):
            tensor_logits = torch.tensor(logits, dtype=torch.float32)
        else:
            tensor_logits = logits.detach().float()

        scaled = tensor_logits / temp
        probs = torch.softmax(scaled, dim=-1).tolist()
        return probs

    @staticmethod
    def evaluate_prediction(
        context_id: str,
        class_probabilities: Dict[str, float],
        confidence_threshold: float = 0.50,
        margin_threshold: float = 0.15,
    ) -> NLITaggingResult:
        """Evalúa una distribución posterior contra los dos umbrales de pureza.

        Regla de Decisión:
        1. Identifica top1 y top2 según su probabilidad posterior.
        2. Certifica si P(top1) >= theta_conf.
        3. Certifica si Delta = P(top1) - P(top2) >= theta_margin.
        4. Si alguna falla, marca rechazo y abstención (is_accepted=False).
        """
        if not class_probabilities:
            return NLITaggingResult(
                context_id=context_id,
                top1_class=None,
                is_accepted=False,
                top1_confidence=0.0,
                margin=0.0,
                class_probabilities={},
                reason_rejected="empty_probabilities",
            )

        sorted_classes = sorted(
            class_probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top1_class, top1_prob = sorted_classes[0]
        top2_prob = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
        margin = top1_prob - top2_prob

        # Validación del primer umbral: Confianza mínima
        if top1_prob < confidence_threshold:
            return NLITaggingResult(
                context_id=context_id,
                top1_class=None,
                is_accepted=False,
                top1_confidence=top1_prob,
                margin=margin,
                class_probabilities=class_probabilities,
                reason_rejected="low_confidence",
            )

        # Validación del segundo umbral: Margen de separación inter-clase
        if margin < margin_threshold:
            return NLITaggingResult(
                context_id=context_id,
                top1_class=None,
                is_accepted=False,
                top1_confidence=top1_prob,
                margin=margin,
                class_probabilities=class_probabilities,
                reason_rejected="low_margin",
            )

        return NLITaggingResult(
            context_id=context_id,
            top1_class=top1_class,
            is_accepted=True,
            top1_confidence=top1_prob,
            margin=margin,
            class_probabilities=class_probabilities,
            reason_rejected=None,
        )


class DeBERTaInferenceEngine:
    """Motor de inferencia Zero-Shot NLI basado en Hugging Face DeBERTa-v3."""

    def __init__(self, config: NLITaggerConfig) -> None:
        """Inicializa el motor resolviendo el hardware disponible."""
        self.config = config
        self.device = self._resolve_device(config.device)
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.entailment_idx: Optional[int] = None

    @staticmethod
    def _resolve_device(device_setting: str) -> torch.device:
        """Resuelve dinámicamente el backend de cómputo más eficiente disponible."""
        dev_str = (device_setting or "auto").lower()
        if dev_str == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(dev_str)

    def _ensure_loaded(self) -> None:
        """Carga perezosa del modelo y tokenizador para optimizar inicio."""
        if self.model is not None and self.tokenizer is not None:
            return

        logger.info(
            "Cargando modelo NLI '%s' en dispositivo '%s'...",
            self.config.model_name_or_path,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path,
        )

        if self.config.fp16 and self.device.type in ("cuda", "mps"):
            self.model = self.model.half()

        self.model.to(self.device)
        self.model.eval()

        # Determinar índice del logit 'entailment' de forma robusta e insensible a mayúsculas
        label2id = getattr(self.model.config, "label2id", {}) or {}
        norm_label2id = {k.lower(): v for k, v in label2id.items()}

        if "entailment" in norm_label2id:
            self.entailment_idx = norm_label2id["entailment"]
        elif "yes" in norm_label2id:
            self.entailment_idx = norm_label2id["yes"]
        else:
            # Fallback estándar MNLI (0 o 2 típicamente; si no se encuentra, default a 0)
            self.entailment_idx = 0
            logger.warning(
                "No se encontró etiqueta explícita 'entailment' en config.label2id (%s). Usando índice 0.",
                label2id,
            )

        logger.info(
            "Modelo DeBERTa-v3 cargado exitosamente. Índice de entailment: %d",
            self.entailment_idx,
        )

    def score_premises_against_hypotheses(
        self,
        premises: List[str],
        hypotheses: List[str],
    ) -> List[List[float]]:
        """Calcula logits de entailment para cada combinación (premisa, hipótesis).

        Retorna: Matriz de logits de forma [len(premises), len(hypotheses)].
        """
        self._ensure_loaded()
        assert self.tokenizer is not None
        assert self.model is not None
        assert self.entailment_idx is not None

        n_premises = len(premises)
        n_hypotheses = len(hypotheses)
        all_logits: List[List[float]] = [[] for _ in range(n_premises)]

        # Preparar pares aplanados para procesamiento por lotes
        # Para cada premisa, evaluamos las n_hypotheses
        flat_premises: List[str] = []
        flat_hypotheses: List[str] = []
        for prem in premises:
            for hyp in hypotheses:
                flat_premises.append(prem)
                flat_hypotheses.append(hyp)

        total_pairs = len(flat_premises)
        batch_size = max(1, self.config.batch_size)
        extracted_entailment_logits: List[float] = []

        total_batches = (total_pairs + batch_size - 1) // batch_size
        with torch.no_grad():
            for batch_idx, start_idx in enumerate(range(0, total_pairs, batch_size), start=1):
                end_idx = min(start_idx + batch_size, total_pairs)
                batch_p = flat_premises[start_idx:end_idx]
                batch_h = flat_hypotheses[start_idx:end_idx]

                inputs = self.tokenizer(
                    batch_p,
                    batch_h,
                    max_length=self.config.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits  # [batch_size, num_classes]

                # Extraer logit de entailment
                entailment_scores = logits[:, self.entailment_idx].detach().float().cpu().tolist()
                extracted_entailment_logits.extend(entailment_scores)

                # Liberar tensores intermedios
                del inputs
                del outputs
                del logits

                # Log periódico cada 50 lotes para visibilidad en tiempo real
                if batch_idx % 50 == 0 or end_idx == total_pairs:
                    logger.info(
                        "Inferencia NLI en progreso: lote [%d/%d] (%.1f%%) | Pares evaluados: %d/%d",
                        batch_idx,
                        total_batches,
                        (end_idx / total_pairs) * 100,
                        end_idx,
                        total_pairs,
                    )

        # Reconstruir la matriz estructurada [n_premises, n_hypotheses]
        idx = 0
        for p_idx in range(n_premises):
            for _ in range(n_hypotheses):
                all_logits[p_idx].append(extracted_entailment_logits[idx])
                idx += 1

        # Limpieza de memoria
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

        return all_logits


class NLITagger:
    """Fachada principal del servicio de etiquetado débil Zero-Shot NLI.

    Consumido en memoria por DatasetBuilder en builder.py.
    Mantiene caché de inferencia y telemetría de auditoría sin escribir archivos parásitos.
    """

    def __init__(
        self,
        config: Optional[NLITaggerConfig] = None,
        inference_engine: Optional[DeBERTaInferenceEngine] = None,
    ) -> None:
        """Inicializa el tagger con configuración y motor de inferencia."""
        self.config = config or NLITaggerConfig()
        self.inference_engine = inference_engine or DeBERTaInferenceEngine(self.config)
        self.formatter = NLIPromptFormatter()
        self.purity_filter = NLIPurityFilter()

        # Telemetría y estadísticas de auditoría en memoria
        self._telemetry: Dict[str, Any] = {
            "total_evaluated": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "rejected_low_confidence": 0,
            "rejected_low_margin": 0,
            "accepted_per_class": Counter(),
        }
        self._cache: Dict[str, NLITaggingResult] = {}

    def tag_batch(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[NLITaggingResult]:
        """Procesa un conjunto de contextos citantes candidatos mediante inferencia NLI por lotes.

        Cada elemento de `candidates` debe ser un diccionario que contenga al menos:
        - 'citation_context': str
        - 'rhetorical_section': Optional[str]
        - 'instance_id' o 'context_id' o 'target_cited_id': str

        Retorna: Lista de objetos NLITaggingResult en el mismo orden de entrada.
        """
        if not candidates or not self.config.enabled:
            return []

        # Preparar lista de hipótesis canónicas y sus claves
        hypotheses_items = list(self.config.hypotheses.items())
        canonical_classes = [c for c, _ in hypotheses_items]
        hypotheses_texts = [h for _, h in hypotheses_items]

        results: List[NLITaggingResult] = []
        pending_candidates: List[Dict[str, Any]] = []
        pending_indices: List[int] = []

        # 1. Resolver desde caché en memoria si ya fue evaluado
        for idx, cand in enumerate(candidates):
            cid = str(cand.get("context_id") or cand.get("instance_id") or cand.get("target_cited_id") or f"idx_{idx}")
            if cid in self._cache:
                results.append(self._cache[cid])
            else:
                results.append(None)  # Placeholder
                pending_candidates.append(cand)
                pending_indices.append(idx)

        if not pending_candidates:
            return results

        # 2. Formatear premisas con encuadre discursivo
        premises = [
            self.formatter.format_premise(
                section=c.get("rhetorical_section"),
                context=str(c.get("citation_context", "")),
            )
            for c in pending_candidates
        ]

        # 3. Inferencia de logits con el motor
        logits_matrix = self.inference_engine.score_premises_against_hypotheses(
            premises=premises,
            hypotheses=hypotheses_texts,
        )

        # 4. Calibración softmax y evaluación con filtro de pureza
        for cand, logits, orig_idx in zip(pending_candidates, logits_matrix, pending_indices):
            cid = str(cand.get("context_id") or cand.get("instance_id") or cand.get("target_cited_id") or f"idx_{orig_idx}")
            probs = self.purity_filter.compute_softmax(
                logits=logits,
                temperature=self.config.temperature,
            )
            class_probs = {c: p for c, p in zip(canonical_classes, probs)}

            tagging_res = self.purity_filter.evaluate_prediction(
                context_id=cid,
                class_probabilities=class_probs,
                confidence_threshold=self.config.confidence_threshold,
                margin_threshold=self.config.margin_threshold,
            )

            # Actualizar telemetría
            self._telemetry["total_evaluated"] += 1
            if tagging_res.is_accepted:
                self._telemetry["total_accepted"] += 1
                self._telemetry["accepted_per_class"][tagging_res.top1_class] += 1
            else:
                self._telemetry["total_rejected"] += 1
                if tagging_res.reason_rejected == "low_confidence":
                    self._telemetry["rejected_low_confidence"] += 1
                elif tagging_res.reason_rejected == "low_margin":
                    self._telemetry["rejected_low_margin"] += 1

            self._cache[cid] = tagging_res
            results[orig_idx] = tagging_res

        return results

    def get_telemetry(self) -> Dict[str, Any]:
        """Retorna las métricas cuantitativas acumuladas de auditoría del servicio NLI."""
        total = self._telemetry["total_evaluated"]
        acc = self._telemetry["total_accepted"]
        rate = (acc / total * 100.0) if total > 0 else 0.0

        return {
            "total_evaluated": total,
            "total_accepted": acc,
            "total_rejected": self._telemetry["total_rejected"],
            "acceptance_rate_pct": round(rate, 2),
            "rejected_low_confidence": self._telemetry["rejected_low_confidence"],
            "rejected_low_margin": self._telemetry["rejected_low_margin"],
            "accepted_per_class": dict(self._telemetry["accepted_per_class"]),
        }

    def clear_cache(self) -> None:
        """Limpia la caché en memoria y resetea contadores."""
        self._cache.clear()
        gc.collect()
