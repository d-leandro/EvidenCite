"""Pruebas unitarias para el servicio de etiquetado débil Zero-Shot NLI (src.data.weak_supervision.nli_tagger)."""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock
import pytest
import torch

from src.data.schemas import NLITaggingResult
from src.data.weak_supervision.nli_tagger import (
    CANONICAL_NLI_HYPOTHESES,
    DeBERTaInferenceEngine,
    NLIPromptFormatter,
    NLIPurityFilter,
    NLITagger,
    NLITaggerConfig,
)


def test_canonical_hypotheses_integrity():
    """Valida que existan las 9 hipótesis canónicas con textos disyuntivos y completos."""
    expected_classes = {
        "Background",
        "Gap",
        "Basis",
        "Comparison",
        "Application",
        "Improvement",
        "Evidence",
        "Originator",
        "Further Reading",
    }
    assert set(CANONICAL_NLI_HYPOTHESES.keys()) == expected_classes
    for cls_name, hyp_text in CANONICAL_NLI_HYPOTHESES.items():
        assert isinstance(hyp_text, str)
        assert len(hyp_text.strip()) > 30, f"Hipótesis para {cls_name} es demasiado corta"


def test_nli_tagger_config_defaults_and_overrides():
    """Valida la carga de configuración por defecto y sobreescrituras desde dict."""
    cfg = NLITaggerConfig()
    assert cfg.enabled is True
    assert cfg.confidence_threshold == 0.85
    assert cfg.margin_threshold == 0.25
    assert cfg.quota_buffer == 300
    assert cfg.temperature == 1.0

    # Sobreescritura desde diccionario con claves adicionales ignoradas
    custom = NLITaggerConfig.from_dict({
        "confidence_threshold": 0.90,
        "batch_size": 32,
        "unknown_extra_key": "ignored",
    })
    assert custom.confidence_threshold == 0.90
    assert custom.batch_size == 32
    assert custom.margin_threshold == 0.25


def test_nli_tagger_config_from_yaml():
    """Valida la deserialización desde archivo YAML (configs/builder.yaml)."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("""
seed: 42
nli:
  enabled: true
  model_name_or_path: "test/model"
  confidence_threshold: 0.88
  margin_threshold: 0.30
  quota_buffer: 250
""")
        temp_path = f.name

    try:
        cfg = NLITaggerConfig.from_yaml(temp_path, section_key="nli")
        assert cfg.model_name_or_path == "test/model"
        assert cfg.confidence_threshold == 0.88
        assert cfg.margin_threshold == 0.30
        assert cfg.quota_buffer == 250
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_prompt_formatter():
    """Valida el encuadre discursivo (Framing Devices) y normalización de premisas."""
    formatter = NLIPromptFormatter()

    # Caso 1: Sección válida y contexto limpio
    p1 = formatter.format_premise("Related Work", "We build upon the work of <cite>Author et al.</cite>")
    assert p1 == "Section: Related Work . Context: We build upon the work of <cite>Author et al.</cite>"

    # Caso 2: Sección ausente o nula
    p2 = formatter.format_premise(None, "A seminal result was shown in <cite>Ref</cite>.")
    assert p2 == "Section: Context . Context: A seminal result was shown in <cite>Ref</cite>."

    # Caso 3: Sección numérica o espuria
    p3 = formatter.format_premise("1", "A seminal result was shown in <cite>Ref</cite>.")
    assert p3 == "Section: Context . Context: A seminal result was shown in <cite>Ref</cite>."


def test_purity_filter_softmax():
    """Valida el cálculo de softmax y la sensibilidad a la temperatura."""
    logits = [1.0, 2.0, 3.0]
    probs = NLIPurityFilter.compute_softmax(logits, temperature=1.0)
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-5
    assert probs[2] > probs[1] > probs[0]

    # Temperatura baja intensifica la probabilidad del valor máximo
    probs_sharp = NLIPurityFilter.compute_softmax(logits, temperature=0.5)
    assert probs_sharp[2] > probs[2]


def test_purity_filter_decision_rules():
    """Valida la compuerta de dos umbrales y motivos de abstención."""
    # Caso 1: Aceptado (alta confianza y margen amplio)
    res_ok = NLIPurityFilter.evaluate_prediction(
        context_id="ctx_01",
        class_probabilities={
            "Gap": 0.89,
            "Improvement": 0.05,
            "Background": 0.06,
        },
        confidence_threshold=0.85,
        margin_threshold=0.25,
    )
    assert res_ok.is_accepted is True
    assert res_ok.top1_class == "Gap"
    assert res_ok.top1_confidence == 0.89
    assert abs(res_ok.margin - (0.89 - 0.06)) < 1e-5
    assert res_ok.reason_rejected is None

    # Caso 2: Rechazado por baja confianza (P_top1 < 0.85)
    res_low_conf = NLIPurityFilter.evaluate_prediction(
        context_id="ctx_02",
        class_probabilities={
            "Gap": 0.70,
            "Improvement": 0.15,
            "Background": 0.15,
        },
        confidence_threshold=0.85,
        margin_threshold=0.25,
    )
    assert res_low_conf.is_accepted is False
    assert res_low_conf.top1_class is None
    assert res_low_conf.reason_rejected == "low_confidence"

    # Caso 3: Rechazado por bajo margen (Delta < 0.25 aunque P_top1 sea relativamente alta)
    res_low_margin = NLIPurityFilter.evaluate_prediction(
        context_id="ctx_03",
        class_probabilities={
            "Improvement": 0.86,
            "Application": 0.65,  # Margen 0.21 < 0.25
        },
        confidence_threshold=0.85,
        margin_threshold=0.25,
    )
    assert res_low_margin.is_accepted is False
    assert res_low_margin.top1_class is None
    assert res_low_margin.reason_rejected == "low_margin"

    # Caso 4: Diccionario vacío
    res_empty = NLIPurityFilter.evaluate_prediction("ctx_04", {})
    assert res_empty.is_accepted is False
    assert res_empty.reason_rejected == "empty_probabilities"


def test_inference_engine_device_resolution():
    """Valida la resolución de backend de cómputo en el motor."""
    cfg = NLITaggerConfig(device="cpu")
    engine = DeBERTaInferenceEngine(cfg)
    assert engine.device == torch.device("cpu")

    cfg_auto = NLITaggerConfig(device="auto")
    engine_auto = DeBERTaInferenceEngine(cfg_auto)
    assert engine_auto.device.type in ("mps", "cuda", "cpu")


def test_nli_tagger_service_batch_and_telemetry():
    """Valida el servicio NLITagger de extremo a extremo usando un motor mockeado."""
    mock_engine = MagicMock(spec=DeBERTaInferenceEngine)
    cfg = NLITaggerConfig(confidence_threshold=0.85, margin_threshold=0.25)

    # 9 clases canónicas en orden
    hypotheses_keys = list(cfg.hypotheses.keys())
    gap_idx = hypotheses_keys.index("Gap")

    # Simular logits donde la clase 'Gap' domina claramente para ctx_1
    # y hay ambigüedad para ctx_2
    logits_cand1 = [0.0] * 9
    logits_cand1[gap_idx] = 10.0  # Dominante -> P(Gap) ~ 0.99

    logits_cand2 = [2.0] * 9  # Todos iguales -> P_top1 = 1/9 ~ 0.11 -> Rechazado

    mock_engine.score_premises_against_hypotheses.return_value = [
        logits_cand1,
        logits_cand2,
    ]

    tagger = NLITagger(config=cfg, inference_engine=mock_engine)

    candidates = [
        {
            "context_id": "c1",
            "rhetorical_section": "Introduction",
            "citation_context": "However, existing methods fail on large graphs <cite>Ref</cite>.",
        },
        {
            "context_id": "c2",
            "rhetorical_section": "Background",
            "citation_context": "Previous work includes <cite>RefA</cite> and <cite>RefB</cite>.",
        },
    ]

    results = tagger.tag_batch(candidates)
    assert len(results) == 2

    # Candidato 1 aceptado como Gap
    assert results[0].is_accepted is True
    assert results[0].top1_class == "Gap"
    assert results[0].top1_confidence > 0.95

    # Candidato 2 rechazado por baja confianza
    assert results[1].is_accepted is False
    assert results[1].reason_rejected == "low_confidence"

    # Validar telemetría
    telemetry = tagger.get_telemetry()
    assert telemetry["total_evaluated"] == 2
    assert telemetry["total_accepted"] == 1
    assert telemetry["total_rejected"] == 1
    assert telemetry["rejected_low_confidence"] == 1
    assert telemetry["accepted_per_class"]["Gap"] == 1

    # Validar que una segunda llamada usa la caché en memoria y no re-invoca el motor
    mock_engine.score_premises_against_hypotheses.reset_mock()
    cached_results = tagger.tag_batch([candidates[0]])
    assert len(cached_results) == 1
    assert cached_results[0].top1_class == "Gap"
    mock_engine.score_premises_against_hypotheses.assert_not_called()

    # Validar limpieza de caché
    tagger.clear_cache()
    assert len(tagger._cache) == 0
