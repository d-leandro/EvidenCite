"""Pruebas del modo generativo de LoRA (LLM_MODO=generativo, TaskType.CAUSAL_LM)."""

import json
from pathlib import Path
import types

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.config.schema import (
    CANONICAL_9_CLASSES,
    ExperimentConfig,
    InputRepresentationConfig,
    LoraModelConfig,
    PathsConfig,
)
from src.modeling.__main__ import construir_config
from src.modeling.lora.generativo import (
    AjustesPrompt,
    collate_fn_test,
    collate_fn_train,
    ids_del_prompt,
    interpretar_respuesta,
    renderizar_prompt,
    respuesta_objetivo,
    tokenize_for_testing,
    tokenize_for_training,
)
from src.modeling.lora.modelo import build_lora_classifier, create_smoke_model
from src.modeling.prompts import build_llm_messages
from src.modeling.train import GenerativeLoraTrainer, run_training_experiment
from tests.unit.test_modeling_dataset_and_loss import make_synthetic_instance

CHATML = (
    "{%- for m in messages %}{{'<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n'}}{%- endfor %}"
    "{%- if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{'<think>\\n\\n</think>\\n\\n'}}{%- endif %}"
    "{%- endif %}"
)
SIN_SISTEMA = (
    "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for m in messages %}{{ '<start_of_turn>' + m['role'] + '\\n' + m['content'] + '<end_of_turn>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)


def _instancias(n: int = 18):
    return [make_synthetic_instance(idx=i + 1, label=CANONICAL_9_CLASSES[i % 9], chunk_text_multiplier=1 + i % 3)
            for i in range(n)]


@pytest.fixture(scope="module")
def tokenizador(tmp_path_factory: pytest.TempPathFactory):
    """El tokenizador del modelo de humo, entrenado con los prompts de A5 (como hace train.py en modo generativo)."""
    rep = InputRepresentationConfig(configuration="B")
    textos = [renderizar_prompt(build_llm_messages(x, rep), None, False)[0] + respuesta_objetivo(x)
              for x in _instancias()]
    carpeta = create_smoke_model(textos, tmp_path_factory.mktemp("humo"))
    return AutoTokenizer.from_pretrained(carpeta), carpeta


def test_llm_modo_elige_task_type_y_nombre() -> None:
    config, _ = construir_config({"LLM_MODO": "generativo", "MODO_PROMPT": "section_conditioned",
                                  "PLANTILLA_CHAT": "0", "MODELO": "Qwen/Qwen3-4B-Base"})
    assert config.llm_lora.task_type == "CAUSAL_LM"
    assert config.llm_lora.prompt_mode == "section_conditioned"
    assert config.llm_lora.use_chat_template is False
    assert config.experiment_name.startswith("llm_lora_gen_Qwen3-4B-Base_")
    assert construir_config({"LLM_MODO": "clasificador"})[0].llm_lora.task_type == "SEQ_CLS"
    assert construir_config({})[0].llm_lora.task_type == "SEQ_CLS"  # sin LLM_MODO, el flujo de siempre
    config, _ = construir_config({"LLM_MODO": "generativo", "MODO": "mini"})
    assert config.input_representation.max_seq_length == 1024  # el prompt de sistema no cabe en 256
    with pytest.raises(ValueError, match="LLM_MODO"):
        construir_config({"LLM_MODO": "generador"})


def test_renderizar_prompt_con_y_sin_plantilla(tokenizador) -> None:
    tok, _ = tokenizador
    mensajes = build_llm_messages(make_synthetic_instance(), InputRepresentationConfig(configuration="A"))
    texto, con_plantilla = renderizar_prompt(mensajes, tok, usar_plantilla=True)  # el humo no trae plantilla
    assert not con_plantilla and texto.startswith(mensajes[0]["content"]) and texto.endswith("\n\n[Assistant]\n")
    tok.chat_template = CHATML
    try:
        texto, con_plantilla = renderizar_prompt(mensajes, tok)
        assert con_plantilla and texto.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert renderizar_prompt(mensajes, tok, usar_plantilla=False)[1] is False  # PLANTILLA_CHAT=0
        tok.chat_template = SIN_SISTEMA  # como Gemma 2: el sistema pasa al primer turno del usuario
        texto, _ = renderizar_prompt(mensajes, tok)
        assert texto.startswith("<start_of_turn>user\n" + mensajes[0]["content"])
    finally:
        tok.chat_template = None


def test_tokenizacion_entrenamiento_y_prueba(tokenizador) -> None:
    tok, _ = tokenizador
    ajustes = AjustesPrompt(InputRepresentationConfig(configuration="B"))
    inst = make_synthetic_instance(label="Further Reading")
    tr = tokenize_for_training(inst, tok, ajustes)
    te = tokenize_for_testing(inst, tok, ajustes)
    respuesta = [t for t in tr["labels"] if t != -100]
    assert tok.decode(respuesta) == "Further Reading" + tok.eos_token  # la pérdida solo ve la respuesta y el EOS
    assert tr["input_ids"][: len(te["input_ids"])] == te["input_ids"]  # mismo prompt al entrenar y al generar
    assert te["etiqueta"] == "Further Reading" and tr["clase"] == CANONICAL_9_CLASSES.index("Further Reading")


def test_recorte_acorta_la_evidencia(tokenizador) -> None:
    tok, _ = tokenizador
    inst = make_synthetic_instance(chunk_text_multiplier=6)
    rep = InputRepresentationConfig(configuration="B")
    completo, recortado = ids_del_prompt(inst, tok, AjustesPrompt(rep))
    assert not recortado
    ids, recortado = ids_del_prompt(inst, tok, AjustesPrompt(rep, max_tokens=len(completo) - 20))
    assert recortado and len(ids) <= len(completo) - 20
    assert tok.decode(ids).endswith("\n\n[Assistant]\n")  # el final del prompt no se pierde
    with pytest.raises(ValueError, match="MAX_LONGITUD"):
        ids_del_prompt(inst, tok, AjustesPrompt(rep, max_tokens=40))


def test_collates_rellenan_a_derecha_e_izquierda(tokenizador) -> None:
    tok, _ = tokenizador
    ajustes = AjustesPrompt(InputRepresentationConfig(configuration="B"))
    insts = _instancias(3)
    lote = collate_fn_train([tokenize_for_training(x, tok, ajustes) for x in insts], pad_id=0, pad_to_multiple_of=8)
    assert lote["input_ids"].shape[1] % 8 == 0 and lote["attention_mask"][:, 0].all()
    assert (lote["labels"][lote["attention_mask"] == 0] == -100).all()  # el relleno no cuenta en la pérdida
    lote = collate_fn_test([tokenize_for_testing(x, tok, ajustes) for x in insts], pad_id=0, pad_to_multiple_of=8)
    assert lote["attention_mask"][:, -1].all()  # todas las filas terminan en la última columna
    assert lote["etiquetas"] == [x.label for x in insts]


def test_interpretar_respuesta() -> None:
    assert interpretar_respuesta("Background") == "Background"
    assert interpretar_respuesta(" further reading.") == "Further Reading"
    assert interpretar_respuesta('"Gap"\nporque...') == "Gap"
    assert interpretar_respuesta("Backgroun") is None and interpretar_respuesta("") is None


def test_perdida_solo_en_la_respuesta_igual_a_la_referencia(tokenizador) -> None:
    tok, carpeta = tokenizador
    model = build_lora_classifier(LoraModelConfig(pretrained_model_name_or_path=carpeta, task_type="CAUSAL_LM",
                                                  target_modules=["all-linear"]))
    assert type(model).__name__ == "PeftModelForCausalLM"
    model.eval()  # sin dropout de LoRA: las dos cuentas deben dar lo mismo
    for nombre, p in model.named_parameters():
        if "lora_B" in nombre:
            p.data.normal_(0, 0.5)
    ajustes = AjustesPrompt(InputRepresentationConfig(configuration="B"))
    insts = _instancias(9)
    lote = collate_fn_train([tokenize_for_training(x, tok, ajustes) for x in insts], pad_id=tok.pad_token_id)
    pesos = torch.linspace(0.5, 2.0, 9)
    falso = types.SimpleNamespace(class_weights=pesos, loss_type="weighted_ce", autocast_dtype=None,
                                  args=types.SimpleNamespace(device=torch.device("cpu")), _recorta_logits=True)
    perdida = GenerativeLoraTrainer.compute_loss(falso, model, dict(lote))
    logits = model(input_ids=lote["input_ids"], attention_mask=lote["attention_mask"]).logits.float()
    por_fila = []
    for i in range(len(insts)):
        objetivo = lote["labels"][i, 1:]
        cuenta = objetivo != -100
        por_fila.append(F.cross_entropy(logits[i, :-1][cuenta], objetivo[cuenta]))
    w = pesos[lote["clase"]]
    referencia = (w * torch.stack(por_fila)).sum() / w.sum()
    assert torch.allclose(perdida, referencia, atol=1e-5)


def test_entrenamiento_generativo_de_punta_a_punta(tmp_path: Path) -> None:
    insts = _instancias(27)
    cfg = ExperimentConfig(
        experiment_name="gen_humo",
        model_family="llm_lora",
        seed=42,
        device="cpu",
        paths=PathsConfig(output_dir=tmp_path),
        input_representation=InputRepresentationConfig(configuration="A", max_seq_length=1024),
        llm_lora=LoraModelConfig(pretrained_model_name_or_path="humo", task_type="CAUSAL_LM", num_train_epochs=1,
                                 per_device_train_batch_size=4, per_device_eval_batch_size=8),
    )
    artifact = run_training_experiment(config=cfg, train_instances=insts[:18], dev_instances=insts[18:23],
                                       test_instances=insts[23:], enable_mlflow=False)
    assert artifact.test_metrics_a6 is not None and len(artifact.test_predictions) == 4
    assert set(artifact.test_predictions) <= set(CANONICAL_9_CLASSES)
    assert 0.0 <= artifact.operational_metrics_a7.invalid_format_rate <= 1.0
    corrida = tmp_path / "gen_humo"
    adaptador = json.loads((corrida / "best_model" / "adapter_config.json").read_text(encoding="utf-8"))
    assert adaptador["task_type"] == "CAUSAL_LM" and not adaptador.get("modules_to_save")
    filas = [json.loads(linea) for linea in (corrida / "generaciones.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [f["split"] for f in filas].count("test") == 4 and {"generado", "prediccion"} <= set(filas[0])
    assert any("eval_f1_macro" in h for h in artifact.training_history)  # el early stopping tuvo su métrica