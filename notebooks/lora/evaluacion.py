"""Evaluación: predicciones, macro-F1, intervalo bootstrap, ECE y métricas por clase y por fuente (celda 9)."""

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from opciones_torch import contexto_autocast
from preprocesamiento import ETIQUETAS, mover


@torch.no_grad()
def predict(model, loader, opt: dict):
    """Pasa un split por el modelo y devuelve (logits, etiquetas, índices) como arrays de numpy.

    Pone el modelo en modo evaluación (sin dropout) y al terminar lo deja en el modo en que estaba. Usa no_grad y
    no inference_mode porque se llama a mitad del entrenamiento: un tensor creado en inference_mode no se puede
    usar después en el grafo de gradientes.
    """
    was_training = model.training
    model.eval()
    logits, labels, indices = [], [], []
    for batch in loader:
        batch = mover(batch, opt)
        with contexto_autocast(opt):
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        logits.append(output.float().cpu())
        labels.append(batch["labels"].cpu())
        indices.append(batch["indices"].cpu())
    model.train(was_training)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), torch.cat(indices).numpy()


def macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    """Macro-F1 sobre las clases presentes en el gold (dev no tiene Evidence ni Originator)."""
    return float(f1_score(labels, preds, labels=sorted(set(labels.tolist())), average="macro", zero_division=0))


# Las 5 clases con ejemplos en test; dev tiene además Future Work y Further Reading
CLASES_MACRO_5 = ("Background", "Uses", "Comparison", "Extends", "Motivation")


def metricas_comparables(labels: np.ndarray, preds: np.ndarray) -> dict:
    """F1-macro-9, F1-macro-5 y F1 ponderado, definidos como en la rama SciBERT (02_scibert_ab) para comparar.

    - f1_macro_9: promedio sobre las 9 etiquetas; las que no tienen ejemplos en el split cuentan 0.
    - f1_macro_5: promedio sobre CLASES_MACRO_5, en cualquier split. En test es igual a macro_f1 y al
      f1_macro_present de SciBERT.
    - f1_weighted: promedio ponderado por el soporte de cada clase.
    """
    nueve, cinco = list(range(len(ETIQUETAS))), [ETIQUETAS.index(c) for c in CLASES_MACRO_5]
    return {"f1_macro_9": float(f1_score(labels, preds, labels=nueve, average="macro", zero_division=0)),
            "f1_macro_5": float(f1_score(labels, preds, labels=cinco, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0))}


def bootstrap_ci(labels: np.ndarray, preds: np.ndarray, n: int = 1000, seed: int = 0) -> list[float]:
    """Intervalo de confianza del 95 % del macro-F1 por bootstrap (remuestreo de instancias con reemplazo)."""
    rng = np.random.default_rng(seed)
    scores = [macro_f1(labels[idx], preds[idx]) for idx in (rng.integers(0, len(labels), len(labels)) for _ in range(n))]
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """ECE: diferencia media entre confianza y acierto, ponderada por intervalos de confianza."""
    confidence, correct = probs.max(1), (probs.argmax(1) == labels).astype(float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def compute_metrics(logits: np.ndarray, labels: np.ndarray, sources: list[str], n_bootstrap: int) -> tuple[dict, np.ndarray]:
    """Calcula todas las métricas de un split a partir de los logits y devuelve (métricas, probabilidades).

    Incluye macro-F1 (con intervalo bootstrap si n_bootstrap > 0), accuracy, ECE, P/R/F1 por clase, macro-F1 por
    fuente, matriz de confusión y cuántas predicciones hubo de cada clase.
    """
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = probs.argmax(1)
    present = sorted(set(labels.tolist()))
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=present, zero_division=0)
    sources = np.array(sources)
    metrics = {
        "n": int(len(labels)), "macro_f1": macro_f1(labels, preds), **metricas_comparables(labels, preds),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1_ic95": bootstrap_ci(labels, preds, n_bootstrap) if n_bootstrap else None,
        "ece": expected_calibration_error(probs, labels),
        "clases_evaluadas": [ETIQUETAS[k] for k in present],
        "por_clase": {ETIQUETAS[k]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "soporte": int(s[i])}
                      for i, k in enumerate(present)},
        "por_fuente": {src: {"n": int((sources == src).sum()),
                             "macro_f1": macro_f1(labels[sources == src], preds[sources == src]),
                             "accuracy": float((preds[sources == src] == labels[sources == src]).mean())}
                       for src in sorted(set(sources.tolist()))},
        "confusion": confusion_matrix(labels, preds, labels=list(range(len(ETIQUETAS)))).tolist(),
        "predicciones_por_clase": {ETIQUETAS[k]: int(v) for k, v in enumerate(np.bincount(preds, minlength=len(ETIQUETAS))) if v},
    }
    return metrics, probs


def evaluate(model, loader, items: list[dict], opt: dict, n_bootstrap: int):
    """Evaluación final de un split: devuelve (métricas, predicciones por instancia con sus probabilidades)."""
    t0 = time.perf_counter()
    logits, labels, indices = predict(model, loader, opt)
    seconds = time.perf_counter() - t0
    sources = [items[i]["fuente"] for i in indices]
    metrics, probs = compute_metrics(logits, labels, sources, n_bootstrap)
    metrics["segundos"] = seconds
    metrics["ejemplos_por_s"] = len(labels) / max(seconds, 1e-9)
    predictions = sorted(({"instance_id": items[i]["instance_id"], "fuente": items[i]["fuente"],
                           "etiqueta": ETIQUETAS[int(labels[j])], "prediccion": ETIQUETAS[int(probs[j].argmax())],
                           "probs": {e: round(float(probs[j][k]), 5) for k, e in enumerate(ETIQUETAS)}}
                          for j, i in enumerate(indices)), key=lambda d: d["instance_id"])
    return metrics, predictions


def validate(model, loader, opt: dict) -> dict:
    """Evaluación rápida de dev durante el entrenamiento: macro-F1 (criterio del early stopping), accuracy y pérdida."""
    logits, labels, _ = predict(model, loader, opt)
    preds = logits.argmax(1)
    loss = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    return {"macro_f1": macro_f1(labels, preds), **metricas_comparables(labels, preds),
            "accuracy": float((preds == labels).mean()), "perdida": loss}


def print_metrics(m: dict, name: str) -> None:
    """Imprime las métricas de un split: resumen, tabla por clase, métricas por fuente y predicciones por clase."""
    ci = f" (IC95 {m['macro_f1_ic95'][0]:.3f}–{m['macro_f1_ic95'][1]:.3f})" if m["macro_f1_ic95"] else ""
    print(f"  {name}: macro-F1 {m['macro_f1']:.4f}{ci} · accuracy {m['accuracy']:.4f} · ECE {m['ece']:.3f} · "
          f"n {m['n']} · {m['ejemplos_por_s']:.1f} ej/s")
    print(f"  F1-macro-9 {m['f1_macro_9']:.4f} · F1-macro-5 {m['f1_macro_5']:.4f} · F1 ponderado {m['f1_weighted']:.4f} "
          f"(el macro-F1 de arriba promedia las {len(m['clases_evaluadas'])} clases presentes)")
    print(f"  {'clase':<17}{'P':>7}{'R':>7}{'F1':>7}{'n':>6}")
    for label, v in m["por_clase"].items():
        print(f"  {label:<17}{v['precision']:>7.3f}{v['recall']:>7.3f}{v['f1']:>7.3f}{v['soporte']:>6}")
    for source, v in m["por_fuente"].items():
        print(f"  fuente {source}: macro-F1 {v['macro_f1']:.4f} · accuracy {v['accuracy']:.4f} · n {v['n']}")
    print(f"  predicciones por clase: {m['predicciones_por_clase']}")
