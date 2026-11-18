"""Modo generativo de la rama LoRA (LLM_MODO=generativo, PEFT TaskType.CAUSAL_LM).

En el modo clasificador (SEQ_CLS) una cabeza lineal nueva lee el último token y entrega 9 probabilidades. En el modo
generativo no hay cabeza nueva: el modelo recibe el mismo prompt que los LLM por API (build_llm_messages de
src/modeling/prompts.py, que no se toca) y aprende a escribir la etiqueta. La comparación con A5 es 1 a 1: el mismo
texto de entrada, la misma taxonomía y la misma regla para una respuesta inválida.

- Entrenamiento (tokenize_for_training + collate_fn_train): prompt + etiqueta + EOS, con relleno a la derecha. La
  pérdida solo cuenta la etiqueta y el EOS (labels = -100 en el prompt y en el relleno).
- Evaluación (tokenize_for_testing + collate_fn_test + generar_respuestas): solo el prompt, con relleno a la
  izquierda; generate() voraz escribe la respuesta e interpretar_respuesta() la lleva a una de las 9 clases. Una
  respuesta que no es ninguna cuenta como Background y suma en invalid_format_rate, como en parse_llm_json_response.
"""

import contextlib
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.config.schema import CANONICAL_9_CLASSES, LABEL_TO_ID, InputRepresentationConfig, resolve_canonical_label
from src.data.schemas import DatasetInstance
from src.modeling.dataset import format_instance_parts
from src.modeling.prompts import build_llm_messages

# La etiqueta que asigna parse_llm_json_response (predict.py) cuando la API no cumple el formato
ETIQUETA_SI_INVALIDA = "Background"
# Las plantillas de Llama 3.x escriben la fecha del día en el prompt de sistema: con una fija, el prompt de una
# instancia es el mismo al entrenar y al evaluar otro día. Las plantillas que no la usan ignoran el argumento.
FECHA_FIJA = "26 Jul 2024"


@dataclass(frozen=True)
class AjustesPrompt:
    """Cómo se arma y se recorta el prompt de una instancia."""

    rep_config: InputRepresentationConfig  # A, B o context_only: la misma de la corrida por API
    prompt_mode: str = "generic"  # generic o section_conditioned, como llm_api.prompt_mode
    usar_plantilla: bool = True  # plantilla de chat del tokenizador (PLANTILLA_CHAT)
    max_tokens: Optional[int] = None  # tope del prompt: MAX_LONGITUD menos la respuesta; None, sin recorte


def renderizar_prompt(
    mensajes: Sequence[Dict[str, str]], tokenizer: Any, usar_plantilla: bool = True
) -> Tuple[str, bool]:
    """Texto del prompt y si salió de la plantilla de chat del tokenizador.

    Con plantilla, el texto termina en el encabezado del turno del asistente (add_generation_prompt): ahí empieza a
    escribir el modelo. enable_thinking=False y thinking=False cierran el razonamiento de Qwen3, Qwen3.5 y Granite;
    date_string fija la fecha de Llama 3.x. Si la plantilla no admite el rol system (Gemma 2), el sistema va al inicio
    del primer turno del usuario.
    Sin plantilla (modelo de humo, PLANTILLA_CHAT=0): texto plano con los rótulos [User]/[Assistant] que usa
    _call_gemini_native (predict.py) para la API de Gemini.
    """
    if usar_plantilla and getattr(tokenizer, "chat_template", None):
        opciones = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False, thinking=False,
                        date_string=FECHA_FIJA)
        try:
            return tokenizer.apply_chat_template(list(mensajes), **opciones), True
        except Exception:  # p. ej. TemplateError «System role not supported»
            sistema = "\n\n".join(m["content"] for m in mensajes if m["role"] == "system")
            resto = [dict(m) for m in mensajes if m["role"] != "system"]
            resto[0]["content"] = f"{sistema}\n\n{resto[0]['content']}"
            return tokenizer.apply_chat_template(resto, **opciones), True
    partes = [m["content"] for m in mensajes if m["role"] == "system"]
    partes += [f"[{'User' if m['role'] == 'user' else 'Assistant'}]\n{m['content']}"
               for m in mensajes if m["role"] != "system"]
    return "\n\n".join(partes + ["[Assistant]\n"]), False


def respuesta_objetivo(instancia: DatasetInstance) -> str:
    """Lo que el modelo aprende a escribir: el nombre canónico de la clase, sin probabilidades."""
    etiqueta = resolve_canonical_label(instancia.label)
    if etiqueta is None:
        raise ValueError(f"Instancia {instancia.instance_id} tiene etiqueta inválida o nula: {instancia.label!r}")
    return etiqueta


def tokens_de_respuesta(tokenizer: Any) -> int:
    """Tokens de la respuesta más larga (la etiqueta más larga y el EOS). Es max_new_tokens al generar y lo que se
    reserva de MAX_LONGITUD para la respuesta."""
    return max(len(tokenizer(e, add_special_tokens=False)["input_ids"]) for e in CANONICAL_9_CLASSES) + 1


def ids_del_prompt(instancia: DatasetInstance, tokenizer: Any, ajustes: AjustesPrompt) -> Tuple[List[int], bool]:
    """Tokens del prompt de la instancia y si hubo que recortarlo.

    Se tokeniza el texto completo; sin add_special_tokens si viene de la plantilla, que ya trae su BOS. Si no cabe en
    ajustes.max_tokens, se acorta el final de la evidencia del citado (como only_second en el clasificador) y se vuelve
    a tokenizar: el prompt de sistema, el contexto citante y el encabezado del asistente nunca se recortan.
    """
    mensajes = build_llm_messages(instancia, ajustes.rep_config, prompt_mode=ajustes.prompt_mode)  # zero-shot
    texto, con_plantilla = renderizar_prompt(mensajes, tokenizer, ajustes.usar_plantilla)

    def tokenizar(t: str) -> List[int]:
        return list(tokenizer(t, add_special_tokens=not con_plantilla)["input_ids"])

    ids = tokenizar(texto)
    limite = ajustes.max_tokens
    if limite is None or len(ids) <= limite:
        return ids, False
    _, citado = format_instance_parts(instancia, ajustes.rep_config, tokenizer=None)  # la evidencia, como en el prompt
    inicio = texto.rfind(citado) if citado else -1
    if inicio >= 0:
        antes, despues = texto[:inicio], texto[inicio + len(citado):]
        citado_ids = tokenizer(citado, add_special_tokens=False)["input_ids"]
        while len(ids) > limite and citado_ids:
            citado_ids = citado_ids[: max(0, len(citado_ids) - (len(ids) - limite))]
            ids = tokenizar(antes + tokenizer.decode(citado_ids, skip_special_tokens=True).strip() + despues)
    if len(ids) > limite:
        raise ValueError(
            f"El prompt de {instancia.instance_id} ocupa {len(ids)} tokens aun sin la evidencia del citado y no cabe "
            f"en {limite} (MAX_LONGITUD menos la respuesta): sube MAX_LONGITUD o no lo fijes"
        )
    return ids, True


def tokenize_for_training(instancia: DatasetInstance, tokenizer: Any, ajustes: AjustesPrompt) -> Dict[str, Any]:
    """Prompt + respuesta + EOS. labels = -100 en el prompt: la pérdida solo cuenta la etiqueta y el EOS, que le
    enseña a detenerse. Prompt y respuesta se tokenizan por separado: así el prompt queda con los mismos tokens que en
    tokenize_for_testing, y la frontera entre los dos no depende de cómo se fusionen al tokenizar juntos."""
    prompt_ids, recortado = ids_del_prompt(instancia, tokenizer, ajustes)
    etiqueta = respuesta_objetivo(instancia)
    # Sin espacio inicial: el prompt termina en salto de línea (encabezado del asistente o "[Assistant]\n")
    answer_ids = list(tokenizer(etiqueta, add_special_tokens=False)["input_ids"])
    eos = tokenizer.eos_token_id

    input_ids = prompt_ids + answer_ids + [eos]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + answer_ids + [eos]

    assert len(input_ids) == len(labels) == len(attention_mask)
    assert input_ids[-1] == eos
    assert labels[len(prompt_ids)] == answer_ids[0]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "clase": LABEL_TO_ID[etiqueta], "recortado": recortado}


def tokenize_for_testing(instancia: DatasetInstance, tokenizer: Any, ajustes: AjustesPrompt) -> Dict[str, Any]:
    """Solo el prompt, que es lo que recibe el modelo para generar. La etiqueta va aparte, para medir."""
    prompt_ids, recortado = ids_del_prompt(instancia, tokenizer, ajustes)
    attention_mask = [1] * len(prompt_ids)

    assert len(prompt_ids) == len(attention_mask)
    assert prompt_ids[-1] != tokenizer.eos_token_id  # NO debe terminar en EOS: el modelo creería que ya respondió
    return {"input_ids": prompt_ids, "attention_mask": attention_mask,
            "etiqueta": respuesta_objetivo(instancia), "recortado": recortado}


def _largo_del_lote(batch: Sequence[Dict[str, Any]], pad_to_multiple_of: Optional[int]) -> int:
    max_len = max(len(x["input_ids"]) for x in batch)
    if pad_to_multiple_of:
        m = pad_to_multiple_of
        max_len = ((max_len + m - 1) // m) * m
    return max_len


def collate_fn_train(
    batch: Sequence[Dict[str, Any]], pad_id: int, pad_to_multiple_of: Optional[int] = 8
) -> Dict[str, Any]:
    """Relleno a la DERECHA hasta un múltiplo de pad_to_multiple_of (tensor cores; con torch.compile, menos formas
    distintas). El relleno lleva attention_mask = 0 y labels = -100: ni se atiende ni cuenta en la pérdida. "clase"
    lleva el id de la clase de cada fila, para el peso de su clase en la pérdida (PESOS_CLASE)."""
    max_len = _largo_del_lote(batch, pad_to_multiple_of)
    ids, attn, lbl = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * pad)
        attn.append(x["attention_mask"] + [0] * pad)
        lbl.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(lbl, dtype=torch.long),
        "clase": torch.tensor([x["clase"] for x in batch], dtype=torch.long),
    }


def collate_fn_test(
    batch: Sequence[Dict[str, Any]], pad_id: int, pad_to_multiple_of: Optional[int] = 8
) -> Dict[str, Any]:
    """Relleno a la IZQUIERDA: todas las filas terminan en la última columna, así generate() escribe la respuesta de
    cada una justo después de su prompt. Con relleno a la derecha, una fila corta seguiría escribiendo después de su
    relleno."""
    max_len = _largo_del_lote(batch, pad_to_multiple_of)
    ids, attn = [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        ids.append([pad_id] * pad + x["input_ids"])
        attn.append([0] * pad + x["attention_mask"])
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "etiquetas": [x["etiqueta"] for x in batch],  # lista de textos, no tensor
    }


class PromptsGenerativos(Dataset):
    """Instancias tokenizadas con tokenize_for_training (entrenamiento=True) o tokenize_for_testing.

    Todas se tokenizan al crear el dataset: un prompt que no cabe falla antes de entrenar, y longitudes() (que usan
    group_by_length y el orden de evaluación) no repite trabajo.
    """

    def __init__(self, instancias: Sequence[DatasetInstance], tokenizer: Any, ajustes: AjustesPrompt,
                 entrenamiento: bool) -> None:
        if tokenizer.eos_token_id is None:
            raise ValueError("El tokenizador no tiene eos_token: el modo generativo lo usa para cerrar la respuesta")
        tokenizar = tokenize_for_training if entrenamiento else tokenize_for_testing
        self.items = [tokenizar(inst, tokenizer, ajustes) for inst in instancias]
        self.recortados = sum(item["recortado"] for item in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]

    def longitudes(self) -> List[int]:
        return [len(item["input_ids"]) for item in self.items]


def interpretar_respuesta(texto: str) -> Optional[str]:
    """La clase canónica que escribió el modelo, o None si no es ninguna de las 9 (formato inválido).

    Mira la primera línea y tolera mayúsculas, espacios y comillas o puntuación alrededor ("further reading." vale).
    """
    lineas = texto.strip().splitlines()
    candidata = " ".join(lineas[0].strip(" \t\"'`.,;:").split()).casefold() if lineas else ""
    return next((e for e in CANONICAL_9_CLASSES if e.casefold() == candidata), None)


@torch.no_grad()
def generar_respuestas(
    model: torch.nn.Module,
    prompts: PromptsGenerativos,
    tokenizer: Any,
    batch_size: int,
    max_new_tokens: int,
    autocast_dtype: Optional[torch.dtype] = None,
    pad_to_multiple_of: Optional[int] = 8,
) -> Tuple[List[str], float]:
    """Genera con decodificación voraz la respuesta a cada prompt de `prompts` (un PromptsGenerativos de prueba).

    Devuelve los textos generados, en el orden de `prompts`, y los segundos que tomó. Los lotes van del prompt más
    largo al más corto: rellenan menos y, si la memoria no alcanza, falla el primer lote y no el último.
    """
    model.eval()
    device = next(p for p in model.parameters() if p.device.type != "meta").device
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    largos = prompts.longitudes()
    orden = sorted(range(len(prompts)), key=lambda i: -largos[i])
    loader = DataLoader(Subset(prompts, orden), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn_test(b, pad_id, pad_to_multiple_of))
    contexto = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
    textos = [""] * len(prompts)
    hechos = 0
    inicio = time.perf_counter()
    for lote in loader:
        input_ids = lote["input_ids"].to(device)
        with contexto:
            salida = model.generate(
                input_ids=input_ids,
                attention_mask=lote["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,  # voraz: siempre la misma respuesta, como temperature=0 en la API
                num_beams=1,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                use_cache=True,  # la config del modelo la apagó para entrenar
            )
        nuevos = salida[:, input_ids.shape[1]:]  # con relleno a la izquierda, lo nuevo empieza aquí en todas las filas
        for texto in tokenizer.batch_decode(nuevos, skip_special_tokens=True):
            textos[orden[hechos]] = texto
            hechos += 1
    return textos, max(1e-9, time.perf_counter() - inicio)