"""Pruebas unitarias para evaluación (A6/A7), significancia estadística, entrenamiento y predicción."""

from pathlib import Path
from types import SimpleNamespace
from typing import List

from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import BertConfig, BertForSequenceClassification

from src.config.schema import (
    CANONICAL_9_CLASSES,
    EncoderModelConfig,
    ExperimentConfig,
    ID_TO_LABEL,
    InputRepresentationConfig,
    LABEL_TO_ID,
    LLMApiConfig,
    PathsConfig,
    TfidfModelConfig,
)
from src.modeling.evaluate import (
    compute_a6_metrics,
    compute_hf_metrics,
    paired_significance_test,
)
from src.modeling.heads import count_model_parameters
from src.modeling.predict import (
    parse_llm_json_response,
    predict_tfidf_batch,
    predict_transformer_batch,
    run_llm_api_experiment,
)
from src.modeling.prompts import (
    LLMResponse,
    build_llm_messages,
)
from src.modeling.train import run_training_experiment
from tests.unit.test_modeling_dataset_and_loss import (
    DummyWordTokenizer,
    make_synthetic_instance,
)


def test_compute_a6_metrics_and_paired_significance() -> None:
    """Valida el cálculo de TechnicalMetricsA6 y la prueba pareada (Paired Bootstrap + McNemar)."""
    y_true = [
        "Background",
        "Background",
        "Comparison",
        "Comparison",
        "Application",
        "Application",
        "Basis",
        "Improvement",
    ] * 5
    # Modelo A acierta la mitad; Modelo B acierta casi todas
    y_pred_a = [
        "Background",
        "Comparison",
        "Comparison",
        "Background",
        "Application",
        "Background",
        "Background",
        "Background",
    ] * 5
    y_pred_b = [
        "Background",
        "Background",
        "Comparison",
        "Comparison",
        "Application",
        "Application",
        "Basis",
        "Background",
    ] * 5

    m_a = compute_a6_metrics(y_true, y_pred_a)
    m_b = compute_a6_metrics(y_true, y_pred_b)

    assert m_b.macro_f1_all_9 > m_a.macro_f1_all_9
    assert m_b.macro_f1_present > m_b.macro_f1_all_9  # Solo 5 de las 9 clases tienen soporte > 0
    assert len(m_b.confusion_matrix) == 9
    assert len(m_b.confusion_matrix[0]) == 9

    sig = paired_significance_test(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        n_bootstrap=1000,
        seed=42,
        use_present_classes_only=True,
    )
    assert sig.delta_b_minus_a > 0.0
    assert sig.ci_95_low > 0.0
    assert sig.mcnemar_b_only_correct > sig.mcnemar_a_only_correct
    assert sig.p_value_bootstrap < 0.05
    assert sig.is_significant_95 is True

    # Compatibilidad con compute_hf_metrics
    import numpy as np

    fake_logits = np.eye(9)[[LABEL_TO_ID[y] for y in y_pred_b]]
    fake_labels = np.array([LABEL_TO_ID[y] for y in y_true])
    hf_dict = compute_hf_metrics((fake_logits, fake_labels))
    assert abs(hf_dict["f1_macro"] - m_b.macro_f1_all_9) < 1e-6


def test_tfidf_training_and_prediction_smoke(tmp_path: Path) -> None:
    """Valida el flujo extremo a extremo de entrenamiento y predicción del baseline TF-IDF."""
    instances = [
        make_synthetic_instance(idx=i + 1, label=CANONICAL_9_CLASSES[i % 3])
        for i in range(18)
    ]
    cfg = ExperimentConfig(
        experiment_name="test_tfidf_b",
        model_family="tfidf",
        seed=42,
        paths=PathsConfig(output_dir=tmp_path),
        input_representation=InputRepresentationConfig(configuration="B", strip_cite_tags=True),
        tfidf=TfidfModelConfig(ngram_min=1, ngram_max=2, min_df=1, max_df=1.0),
    )

    artifact = run_training_experiment(
        config=cfg,
        train_instances=instances[:12],
        dev_instances=instances[12:15],
        test_instances=instances[15:],
        enable_mlflow=False,
    )
    assert (tmp_path / "test_tfidf_b" / "tfidf_pipeline.joblib").exists()
    assert (tmp_path / "test_tfidf_b" / "metrics_summary.json").exists()
    assert artifact.operational_metrics_a7.total_params > 0
    assert len(artifact.dev_predictions) == 3
    assert artifact.test_predictions is not None and len(artifact.test_predictions) == 3

    import joblib

    loaded_pipe = joblib.load(tmp_path / "test_tfidf_b" / "tfidf_pipeline.joblib")
    records = predict_tfidf_batch(loaded_pipe, instances[15:], cfg.input_representation)
    assert len(records) == 3
    assert set(records[0].probabilities.keys()) == set(CANONICAL_9_CLASSES)


def test_weighted_sequence_trainer_and_lora_smoke(tmp_path: Path) -> None:
    """Valida WeightedSequenceTrainer con un BertForSequenceClassification diminuto en memoria y PEFT/LoRA."""
    instances = [
        make_synthetic_instance(idx=i + 1, label=CANONICAL_9_CLASSES[i % 3])
        for i in range(12)
    ]
    tiny_bert_cfg = BertConfig(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=len(CANONICAL_9_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    tiny_model = BertForSequenceClassification(tiny_bert_cfg)

    # Envolvemos con PEFT/LoRA (SEQ_CLS) para verificar que count_model_parameters reduzca trainable_params
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=8,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
    )
    lora_model = get_peft_model(tiny_model, peft_cfg)
    tot_p, train_p, ratio_pct = count_model_parameters(lora_model)
    assert 0 < train_p < tot_p
    assert ratio_pct < 50.0

    tokenizer = DummyWordTokenizer()
    cfg = ExperimentConfig(
        experiment_name="test_tiny_encoder",
        model_family="encoder",
        seed=42,
        device="cpu",
        paths=PathsConfig(output_dir=tmp_path),
        input_representation=InputRepresentationConfig(configuration="B", max_seq_length=48),
        encoder=EncoderModelConfig(
            pretrained_model_name_or_path="dummy-local",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            loss_type="focal",
            class_weight_scheme="sqrt_balanced",
            dynamic_padding=False,
        ),
    )

    artifact = run_training_experiment(
        config=cfg,
        train_instances=instances[:8],
        dev_instances=instances[8:10],
        test_instances=instances[10:],
        tokenizer=tokenizer,
        model=lora_model,
        enable_mlflow=False,
    )
    assert artifact.operational_metrics_a7.trainable_params == train_p
    assert artifact.test_metrics_a6 is not None

    preds = predict_transformer_batch(
        model=lora_model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instances=instances[10:],
        rep_config=cfg.input_representation.model_copy(update={"enforce_per_chunk_budget": False}),
        batch_size=2,
        device="cpu",
    )
    assert len(preds) == 2
    assert abs(sum(preds[0].probabilities.values()) - 1.0) < 1e-4


def test_prompts_and_llm_api_predictor_with_cache_and_invalid_format(tmp_path: Path) -> None:
    """Valida Forma 2 de RespuestaLLM, prompts A5 y medición de invalid_format_rate en A7."""
    inst_1 = make_synthetic_instance(idx=1, label="Comparison", citing_section="Experiments")
    inst_2 = make_synthetic_instance(idx=2, label="Background", citing_section="Introduction")
    rep_cfg = InputRepresentationConfig(configuration="B")

    # 1. Verificación de prompts genérico vs section_conditioned
    msgs_cond = build_llm_messages(inst_1, rep_cfg, prompt_mode="section_conditioned", few_shot_examples=[inst_2])
    assert len(msgs_cond) == 4  # system + 1 few-shot (user/assistant) + user
    assert "Experiments:" in msgs_cond[-1]["content"]

    msgs_gen = build_llm_messages(inst_1, rep_cfg, prompt_mode="generic")
    assert "Experiments:" not in msgs_gen[-1]["content"]

    # 2. Parseo JSON válido e inválido (Forma 2 con alias 'Further Reading')
    valid_json = LLMResponse.model_validate(
        {
            "predicted_label": "Comparison",
            "scores": {c: (0.9 if c == "Comparison" else 0.01) for c in CANONICAL_9_CLASSES},
        }
    ).model_dump_json(by_alias=True)
    parsed, is_ok = parse_llm_json_response(f"```json\n{valid_json}\n```")
    assert is_ok is True
    assert parsed.predicted_label == "Comparison"
    assert len(parsed.scores.to_prob_vector()) == 9

    _, is_bad = parse_llm_json_response("Not a valid JSON string")
    assert is_bad is False

    # 3. Mock del cliente OpenAI devolviendo 1 respuesta válida y 1 malformada
    responses: List[str] = [valid_json, "INVALID_TEXT_OUTPUT"]
    call_counter = {"count": 0}

    class FakeCompletions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            text = responses[call_counter["count"]]
            call_counter["count"] += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                usage=SimpleNamespace(prompt_tokens=200, completion_tokens=50),
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    cache_file = tmp_path / "llm_cache.jsonl"

    exp_cfg = ExperimentConfig(
        experiment_name="test_llm_api",
        model_family="llm_api",
        paths=PathsConfig(output_dir=tmp_path),
        input_representation=rep_cfg,
        llm_api=LLMApiConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            enforce_structured_output=False,
            cache_path=cache_file,
        ),
    )

    records, a6, a7 = run_llm_api_experiment(
        config=exp_cfg,
        eval_instances=[inst_1, inst_2],
        custom_client=fake_client,
    )
    assert len(records) == 2
    assert a7.invalid_format_rate == 0.5  # 1 válida y 1 inválida
    assert a7.estimated_api_cost_usd > 0.0
    assert cache_file.exists()

    # Segunda llamada usa el caché en disco sin invocar de nuevo al cliente
    _, _, _ = run_llm_api_experiment(
        config=exp_cfg,
        eval_instances=[inst_1, inst_2],
        custom_client=fake_client,
    )
    assert call_counter["count"] == 2


def test_gemini_native_sdk_path_with_thinking_budget(tmp_path: Path) -> None:
    """Valida el camino nativo de Google GenAI (Gemini) con ThinkingConfig(thinking_budget=0)."""
    inst = make_synthetic_instance(idx=10, label="Basis", citing_section="Methodology")
    rep_cfg = InputRepresentationConfig(configuration="A")

    parsed_resp = LLMResponse.model_validate(
        {
            "predicted_label": "Basis",
            "scores": {c: (0.95 if c == "Basis" else 0.01) for c in CANONICAL_9_CLASSES},
        }
    )
    captured_configs: list = []

    class FakeGeminiModels:
        def generate_content(self, *, model, contents, config):  # type: ignore[no-untyped-def]
            captured_configs.append((model, contents, config))
            return SimpleNamespace(
                parsed=parsed_resp,
                text=parsed_resp.model_dump_json(by_alias=True),
                usage_metadata=SimpleNamespace(
                    prompt_token_count=180,
                    candidates_token_count=45,
                    thoughts_token_count=0,
                ),
            )

    fake_gemini_client = SimpleNamespace(models=FakeGeminiModels())
    exp_cfg = ExperimentConfig(
        experiment_name="test_gemini_native",
        model_family="llm_api",
        paths=PathsConfig(output_dir=tmp_path),
        input_representation=rep_cfg,
        llm_api=LLMApiConfig(
            provider="gemini",
            model_name="gemini-2.5-flash",
            api_key_env="GEMINI_API_KEY",
            thinking_budget=0,
            enforce_structured_output=True,
            cache_path=tmp_path / "gemini_cache.jsonl",
        ),
    )

    records, a6, a7 = run_llm_api_experiment(
        config=exp_cfg,
        eval_instances=[inst],
        custom_client=fake_gemini_client,
    )
    assert len(records) == 1
    assert records[0].predicted_label == "Basis"
    assert a6.macro_f1_present == 1.0
    assert a7.invalid_format_rate == 0.0
    assert len(captured_configs) == 1
    assert captured_configs[0][2].thinking_config.thinking_budget == 0


def test_prepare_inference_citing_context_in_the_wild() -> None:
    """Valida la inserción de <cite> y enmascaramiento de citas secundarias con [OTHERCIT]."""
    from src.modeling.predict import prepare_inference_citing_context

    raw_text = (
        "Recent works study audio representations (Park and Glass, 2008; Devlin et al., 2018). "
        "Different from Baevski et al. (2019), we optimize a bi-directional model [12]."
    )
    processed = prepare_inference_citing_context(raw_text, "Baevski et al. (2019)")
    assert "<cite>Baevski et al. (2019)</cite>" in processed
    assert "[OTHERCIT]" in processed
    assert "Park and Glass" not in processed
    assert "Devlin et al." not in processed
    assert "[12]" not in processed
    # Si la cita objetivo no está presente, preserva el texto sin alteraciones
    assert prepare_inference_citing_context(raw_text, "NonExistent (2025)") == raw_text


def test_inference_request_and_polymorphic_predictors(tmp_path: Path) -> None:
    """Valida la arquitectura Strategy de BasePredictor, InferenceRequest y Factory (A8)."""
    import pytest
    from src.modeling.predict import (
        InferenceRequest,
        TfidfPredictor,
        TransformerPredictor,
        create_predictor,
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # 1. InferenceRequest desde instancia y pre-procesamiento
    inst = make_synthetic_instance(idx=99, label="Comparison")
    req = InferenceRequest.from_instance(inst)
    assert req.citing_context == inst.citation_context
    cited_title = inst.cited_paper.title if hasattr(inst.cited_paper, "title") else inst.cited_paper["title"]
    assert req.cited_title == cited_title

    # 2. TfidfPredictor con InferenceRequest
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression()),
    ])
    pipe.fit(
        ["Context text background", "Comparison text model", "Usage methods pipeline"],
        [LABEL_TO_ID["Background"], LABEL_TO_ID["Comparison"], LABEL_TO_ID["Application"]],
    )
    rep_a = InputRepresentationConfig(configuration="A")
    tfidf_pred = TfidfPredictor(pipeline=pipe, rep_config=rep_a)

    rec = tfidf_pred.predict(
        citing_context="Some work (Devlin et al., 2018) proposes BERT.",
        target_citation="Devlin et al., 2018",
        cited_title="BERT: Pre-training of Deep Bidirectional Transformers",
        cited_abstract="We introduce BERT...",
    )
    assert rec.predicted_label in CANONICAL_9_CLASSES
    assert rec.confidence > 0.0
    assert len(rec.probabilities) == 9

    # 3. Validación de evidencia faltante en Config A
    with pytest.raises(ValueError, match="Configuración A requiere"):
        tfidf_pred.predict(
            citing_context="Context only without cited paper",
            target_citation="Ref",
        )

    # 4. TransformerPredictor con modelo y tokenizer diminutos
    tiny_bert_cfg = BertConfig(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=len(CANONICAL_9_CLASSES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    tiny_model = BertForSequenceClassification(tiny_bert_cfg)
    tokenizer = DummyWordTokenizer()

    rep_b = InputRepresentationConfig(configuration="B")
    tf_pred = TransformerPredictor(
        model=tiny_model,
        tokenizer=tokenizer,
        rep_config=rep_b,
    )
    rec_b = tf_pred.predict(
        citing_context="Recent model outperforms baselines <cite>Ref</cite>.",
        cited_chunks=["We achieve SOTA accuracy on GLUE."],
    )
    assert rec_b.predicted_label in CANONICAL_9_CLASSES
    assert rec_b.confidence > 0.0

    # 5. create_predictor factory
    cfg_exp = ExperimentConfig(
        experiment_name="test_factory",
        model_family="tfidf",
        paths=PathsConfig(output_dir=tmp_path),
        input_representation=rep_a,
    )
    p1 = create_predictor(cfg_exp, model_or_pipe=pipe, use_cache=True)
    p2 = create_predictor(cfg_exp, model_or_pipe=pipe, use_cache=True)
    assert p1 is p2




