"""Figuras de A4 y diagnóstico sobre el test revisado para el artículo IEEE.

Genera en reports/figures/paper/ (en formato PDF vectorial y PNG 400 DPI):
  1. fig12_a4_definitivos_gold.{pdf,png}      — 1 columna (3,48 in): modelos definitivos (test automático → Gold).
  2. fig12_a4_rondas_seleccion.{pdf,png}      — 2 columnas (7,16 in): (a) Ronda 1 y (b) Ronda 2 (Anexos).
  3. fig13_diagnostico_test_revisado.{pdf,png} — 2 columnas (7,16 in): diagnóstico Confirmadas + Dumbbell Corregidas (Anexos).

Entradas:
  - reports/a4/r1_comparacion.csv, r2_comparacion.csv, final_comparacion.csv, test_revisado_por_modelo.csv
  - data/labeled/test.jsonl, data/labeled/test_panel.jsonl y models/llm_api_cache/ (para referencias A5 0-Shot Config B)

Uso:  python scripts/generar_figuras_a4.py [--datos reports/a4] [--salida reports/figures/paper]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Paleta homogenizada con Frontera de Pareto (scripts/generar_figuras_paper.py): azul #1F6AA5 y rojo #C53030.
AZUL, NARANJA = "#1F6AA5", "#C53030"
TINTA, TINTA_2, TINTA_3 = "#0b0b0b", "#52514e", "#8a8983"
REJILLA, BANDA = "#e3e2de", "#f1f0ec"
MARCA = {"A": dict(color=AZUL, marker="o"), "B": dict(color=NARANJA, marker="s")}

NOMBRES = {
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Phi-4-mini-instruct": "Phi-4-mini",
    "SmolLM3-3B": "SmolLM3-3B",
    "granite-4.1-3b": "Granite-4.1-3B",
    "granite-4.2-3b": "Granite-4.2-3B",
    "Qwen3.5-2B": "Qwen3.5-2B",
    "Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "Hunyuan-4B-Instruct": "Hunyuan-4B",
    "gemma-4-E4B-it": "Gemma-4-E4B",
    "gemma-4-E2B-it": "Gemma-4-E2B",
    "gemma-4-26B-A4B-it": "Gemma-4-26B-A4B",
    "Llama-3.1-Nemotron-Nano-4B-v1.1": "Nemotron-Nano-4B",
    "Nanbeige4.1-3B": "Nanbeige4.1-3B",
}
DESCARTADOS_R1 = {"Hunyuan-4B-Instruct", "gemma-4-E4B-it", "gemma-4-E2B-it", "gemma-4-26B-A4B-it"}
CANDIDATOS_R2 = {"Llama-3.2-3B-Instruct", "granite-4.1-3b", "granite-4.2-3b", "Qwen3.5-2B", "Qwen3.5-4B"}

# Macro-F1 sobre el Gold de corridas cuyo CSV trae la métrica frente al test automático: mismas predicciones,
# recalificadas contra el test revisado.
#   - Qwen3-0.6B clasificador y Granite-4.2-8B: tabla de pruebas pareadas de consolidacion_metricas_a6_a7.ipynb.
#   - Qwen3-0.6B generativo: celda 10c de analizar_resultados.ipynb (test revisado).
GOLD_RECALIFICADO = {
    "Qwen3-0.6B-Base_lora_A_lr1e-4": 0.4563,
    "Qwen3-0.6B-Base_lora_B_lr1e-4": 0.4420,
    "Qwen3-0.6B-Base_lora_A_lr1e-4_gen": 0.434,
    "Qwen3-0.6B-Base_lora_B_lr1e-4_gen": 0.436,
    "granite-4.2-8b_lora_A_lr1e-4": 0.4529,
    "granite-4.2-8b_lora_B_lr1e-4": 0.4626,
}
# Corridas evaluadas directamente sobre el Gold (datos del tag test-revisado): su F1 de test ya es el del Gold.
EN_GOLD = (
    "tfidf_logreg_A_lrna", "tfidf_logreg_B_lrna",
    "scibert_scivocab_uncased_encoder_A_lr2e-5", "scibert_scivocab_uncased_encoder_A_lr2e-5_sin_modelo",
    "scibert_scivocab_uncased_encoder_B_lr2e-5",
    "granite-4.2-8b_lora_A_lr1e-4_test_revisado", "granite-4.2-8b_lora_B_lr1e-4_test_revisado",
)
# Referencias de A5 sobre el Gold (Tabla maestra del artículo): 0-Shot, Config A y B.
REFERENCIAS_API = {"GPT-4.1-mini 0-Shot": (0.4493, 0.4664), "Gemini-3.8-Flash 0-Shot": (0.5900, 0.5996)}

F1, INF, SUP = "F1-macro-9 test", "IC95 test inf.", "IC95 test sup."


def coma(valor: float, _pos=None, decimales: int = 2) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def estilo(ax, xlim, paso: float) -> None:
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MultipleLocator(paso))
    ax.xaxis.set_major_formatter(FuncFormatter(coma))
    ax.grid(axis="x", color=REJILLA, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=6.3, colors=TINTA_2, length=0, pad=2.5)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(TINTA_3)
    ax.spines["bottom"].set_linewidth(0.5)


def panel_ronda1(ax, r1: pd.DataFrame) -> None:
    orden = r1.groupby("modelo")[F1].mean().sort_values(ascending=False).index.tolist()
    y = {m: i for i, m in enumerate(orden)}
    desc = [y[m] for m in orden if m in DESCARTADOS_R1]
    ax.axhspan(min(desc) - 0.5, max(desc) + 0.5, color=BANDA, zorder=0)
    ax.text(0.8635, max(desc) + 0.36, "descartados", fontsize=5.8, color=TINTA_2, ha="right", va="bottom",
            style="italic")
    for _, fila in r1.iterrows():
        yy = y[fila["modelo"]] + (-0.17 if fila["config"] == "A" else 0.17)
        m = MARCA[fila["config"]]
        ax.plot([fila[INF], fila[SUP]], [yy, yy], color=m["color"], linewidth=0.7, alpha=0.55, solid_capstyle="round")
        ax.plot(fila[F1], yy, linestyle="none", markersize=3.6, markeredgecolor="white", markeredgewidth=0.5, **m)
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([NOMBRES[m] + ("*" if m == "gemma-4-26B-A4B-it" else "") for m in orden])
    for etiqueta, m in zip(ax.get_yticklabels(), orden):
        etiqueta.set_color(TINTA_2 if m in DESCARTADOS_R1 else TINTA)
    ax.set_ylim(len(orden) - 0.45, -0.6)
    estilo(ax, (0.765, 0.865), 0.02)
    ax.set_xlabel("Macro-F1 de test (etiquetas previas) e IC95", fontsize=6.3, color=TINTA_2, labelpad=2)


def panel_ronda2(ax, r2: pd.DataFrame) -> None:
    medias = r2.groupby("modelo")[F1].mean().sort_values(ascending=False)
    orden = medias.index.tolist()
    y = {m: i for i, m in enumerate(orden)}
    siete = medias.iloc[:7]
    ax.axvspan(siete.min(), siete.max(), color=BANDA, zorder=0)
    for _, fila in r2.iterrows():
        yy = y[fila["modelo"]] + (-0.17 if fila["config"] == "A" else 0.17)
        ax.plot(fila[F1], yy, linestyle="none", markersize=3.0, markeredgecolor="white", markeredgewidth=0.4,
                alpha=0.9, **MARCA[fila["config"]])
    for m, media in medias.items():
        ax.plot([media, media], [y[m] - 0.36, y[m] + 0.36], color=TINTA, linewidth=1.1, solid_capstyle="butt", zorder=4)
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([NOMBRES[m] for m in orden])
    for etiqueta, m in zip(ax.get_yticklabels(), orden):
        etiqueta.set_fontweight("bold" if m in CANDIDATOS_R2 else "normal")
        etiqueta.set_color(TINTA)
    ax.set_ylim(len(orden) - 0.45, -0.6)
    estilo(ax, (0.7175, 0.7585), 0.01)
    ax.set_xlabel("Macro-F1 de test (etiquetas automáticas)", fontsize=6.3, color=TINTA_2, labelpad=2)


def panel_definitivos(ax, fin: pd.DataFrame) -> None:
    grupos = [
        ("TF-IDF + RL", lambda c: c.startswith("tfidf")),
        ("SciBERT (110M)", lambda c: c.startswith("scibert")),
        ("Qwen3-0.6B clasif.", lambda c: c.startswith("Qwen3-0.6B") and not c.endswith("_gen")),
        ("Qwen3-0.6B gener.", lambda c: c.startswith("Qwen3-0.6B") and c.endswith("_gen")),
        ("Granite-4.2-8B", lambda c: c.startswith("granite-4.2-8b")),
    ]
    for banda, (nombre, (lo, hi)) in zip((0, 1), REFERENCIAS_API.items()):
        ax.axvspan(lo, hi, color=BANDA, zorder=0)
        ax.axvline((lo + hi) / 2, color=TINTA_3, linewidth=0.6, zorder=1)
    for i, (_, filtro) in enumerate(grupos):
        filas = fin[fin["corrida"].map(filtro)]
        for config in ("A", "B"):
            yy = i + (-0.19 if config == "A" else 0.19)
            m = MARCA[config]
            auto = [f[F1] for _, f in filas[filas["config"] == config].iterrows() if f["corrida"] not in EN_GOLD]
            gold = [f[F1] for _, f in filas[filas["config"] == config].iterrows() if f["corrida"] in EN_GOLD]
            gold += [GOLD_RECALIFICADO[c] for c in filas[filas["config"] == config]["corrida"] if c in GOLD_RECALIFICADO]
            if auto and gold:
                ax.plot([min(gold), max(auto)], [yy, yy], color=TINTA_3, linewidth=0.6, zorder=2)
            for v in auto:
                ax.plot(v, yy, linestyle="none", marker=m["marker"], markersize=3.8, markerfacecolor="white",
                        markeredgecolor=m["color"], markeredgewidth=0.9, zorder=3)
            for v in gold:
                ax.plot(v, yy, linestyle="none", markersize=3.8, markeredgecolor="white", markeredgewidth=0.5,
                        zorder=4, **m)
    ax.set_yticks(range(len(grupos)))
    ax.set_yticklabels([g[0] for g in grupos], color=TINTA)
    ax.set_ylim(len(grupos) - 0.4, -0.62)
    estilo(ax, (0.30, 0.80), 0.1)
    ax.set_xlabel("Macro-F1 de test", fontsize=6.3, color=TINTA_2, labelpad=2)
    gpt, gem = REFERENCIAS_API["GPT-4.1-mini 0-Shot"], REFERENCIAS_API["Gemini-3.8-Flash 0-Shot"]
    # Rótulos de las bandas de referencia, sobre el eje y hacia fuera para que no se pisen
    arriba = ax.get_xaxis_transform()
    ax.text(gpt[1], 1.02, "GPT-4.1-mini", transform=arriba, fontsize=5.8, color=TINTA_2, ha="right", va="bottom")
    ax.text(gem[0], 1.02, "Gemini-3.8-Flash", transform=arriba, fontsize=5.8, color=TINTA_2, ha="left", va="bottom")


def leyenda(fig) -> None:
    def punto(config, relleno=True, **kw):
        m = MARCA[config]
        return Line2D([], [], linestyle="none", marker=m["marker"], markersize=4,
                      markerfacecolor=m["color"] if relleno else "white", markeredgecolor=m["color"] if not relleno
                      else "white", markeredgewidth=0.9 if not relleno else 0.5, **kw)
    gris = dict(linestyle="none", marker="o", markersize=4)
    elementos = [
        punto("A", label="Config A (título + resumen)"),
        punto("B", label="Config B (Top-3 fragmentos)"),
        Line2D([], [], color=TINTA, linewidth=1.1, label="media por modelo (b)"),
        Line2D([], [], markerfacecolor="white", markeredgecolor=TINTA_2, markeredgewidth=0.9,
               label="test automático (c)", **gris),
        Line2D([], [], markerfacecolor=TINTA_2, markeredgecolor="white", markeredgewidth=0.5,
               label="test Gold humano (c)", **gris),
    ]
    fig.legend(handles=elementos, loc="lower center", ncol=5, frameon=False, fontsize=6.3, handletextpad=0.3,
               columnspacing=1.6, bbox_to_anchor=(0.5, -0.005), labelcolor=TINTA_2)


# Referencias de A5 sobre Confirmadas (n=1.278) y Corregidas (n=1.520) en 0-Shot Config B.
# Se recalculan desde data/labeled/{test,test_panel}.jsonl y models/llm_api_cache/ si existen.
A5_DIAGNOSTICO_FALLBACK = {
    "GPT-4.1-mini 0s-B (A5)": (0.6980, 0.2816, 0.3901),
    "Gemini-3.8-Flash 0s-B (A5)": (0.8279, 0.1947, 0.5632),
}


def calcular_diagnostico_a5() -> dict[str, tuple[float, float, float]]:
    """Calcula (conf_acc, corr_auto, corr_gold) para GPT-4.1-mini y Gemini-3.8-Flash 0-Shot Config B."""
    ruta_auto = Path("data/labeled/test_panel.jsonl")
    ruta_gold = Path("data/labeled/test.jsonl")
    cache_dir = Path("models/llm_api_cache")
    if not (ruta_auto.is_file() and ruta_gold.is_file() and cache_dir.is_dir()):
        return dict(A5_DIAGNOSTICO_FALLBACK)

    try:
        from src.config.schema import resolve_canonical_label
    except Exception:
        return dict(A5_DIAGNOSTICO_FALLBACK)

    def leer(ruta: Path) -> dict[str, str]:
        res: dict[str, str] = {}
        with ruta.open("r", encoding="utf-8") as fh:
            for linea in fh:
                if linea.strip():
                    d = json.loads(linea)
                    res[d["instance_id"]] = resolve_canonical_label(d["label"])
        return res

    auto = leer(ruta_auto)
    gold = leer(ruta_gold)
    ids = sorted(set(auto) & set(gold))
    conf_ids = [i for i in ids if auto[i] == gold[i]]
    corr_ids = [i for i in ids if auto[i] != gold[i]]

    objetivos = [
        ("GPT-4.1-mini 0s-B (A5)", cache_dir / "gpt41mini_config_b.jsonl", "B::generic::gpt-4.1-mini"),
        ("Gemini-3.8-Flash 0s-B (A5)", cache_dir / "gemini38flash_config_b.jsonl", "B::generic::gemini-3.8-flash"),
    ]
    salida: dict[str, tuple[float, float, float]] = {}
    for etiqueta, archivo, sufijo in objetivos:
        if not archivo.is_file():
            salida[etiqueta] = A5_DIAGNOSTICO_FALLBACK[etiqueta]
            continue
        preds: dict[str, str] = {}
        with archivo.open("r", encoding="utf-8") as fh:
            for linea in fh:
                if not linea.strip():
                    continue
                d = json.loads(linea)
                k = d.get("cache_key", "")
                if k.endswith(f"::{sufijo}"):
                    iid = k.split("::", 1)[0]
                    preds[iid] = d["record"]["predicted_label"]
        if len(set(preds) & set(ids)) < len(ids):
            salida[etiqueta] = A5_DIAGNOSTICO_FALLBACK[etiqueta]
            continue
        conf_acc = sum(preds[i] == gold[i] for i in conf_ids) / len(conf_ids)
        corr_auto = sum(preds[i] == auto[i] for i in corr_ids) / len(corr_ids)
        corr_gold = sum(preds[i] == gold[i] for i in corr_ids) / len(corr_ids)
        salida[etiqueta] = (conf_acc, corr_auto, corr_gold)
    return salida


def pct_coma(valor: float, _pos=None, decimales: int = 0) -> str:
    return f"{valor * 100:.{decimales}f}%".replace(".", ",")


def guardar_fig(fig, salida: Path, nombre: str) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida / f"{nombre}.png", dpi=400)
    plt.close(fig)
    print(f"Figura escrita en {salida}/{nombre}.png")


def generar_definitivos_1col(fin: pd.DataFrame, salida: Path) -> None:
    """Genera solo el panel de modelos definitivos (test automático -> Gold) a 1 columna IEEE (3,48 in)."""
    fig = plt.figure(figsize=(3.48, 2.38))
    ax = fig.add_axes([0.335, 0.285, 0.635, 0.575])
    panel_definitivos(ax, fin)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.set_xlabel("Macro-F1 de test", fontsize=6.6, color=TINTA_2, labelpad=2.5)

    def punto(config, relleno=True, **kw):
        m = MARCA[config]
        return Line2D(
            [], [], linestyle="none", marker=m["marker"], markersize=4.0,
            markerfacecolor=m["color"] if relleno else "white",
            markeredgecolor=m["color"] if not relleno else "white",
            markeredgewidth=0.9 if not relleno else 0.5, **kw,
        )

    gris = dict(linestyle="none", marker="o", markersize=4.0)
    elementos = [
        punto("A", label="Config A (título + resumen)"),
        punto("B", label="Config B (Top-3 fragmentos)"),
        Line2D([], [], markerfacecolor="white", markeredgecolor=TINTA_2, markeredgewidth=0.9,
               label="test automático", **gris),
        Line2D([], [], markerfacecolor=TINTA_2, markeredgecolor="white", markeredgewidth=0.5,
               label="test Gold humano", **gris),
    ]
    fig.legend(
        handles=elementos, loc="lower center", ncol=2, frameon=False, fontsize=6.2,
        handletextpad=0.35, columnspacing=1.1, labelspacing=0.3,
        bbox_to_anchor=(0.51, 0.005), labelcolor=TINTA_2,
    )
    guardar_fig(fig, salida, "fig12_a4_definitivos_gold")


def generar_rondas_2col(r1: pd.DataFrame, r2: pd.DataFrame, salida: Path) -> None:
    """Genera los paneles (a) Ronda 1 y (b) Ronda 2 a ancho completo IEEE (7,16 in) para Anexos."""
    fig = plt.figure(figsize=(7.16, 2.48))
    paneles = [
        (0.015, 0.165, 0.305, "(a) Ronda 1: familias de hasta 4B", panel_ronda1, r1),
        (0.515, 0.660, 0.320, "(b) Ronda 2: 3 semillas × Config A/B", panel_ronda2, r2),
    ]
    for x_titulo, x_eje, ancho, titulo, dibujar, tabla in paneles:
        ax = fig.add_axes([x_eje, 0.235, ancho, 0.645])
        dibujar(ax, tabla)
        ax.tick_params(axis="both", labelsize=6.6)
        ax.set_xlabel(ax.get_xlabel(), fontsize=6.6, color=TINTA_2, labelpad=2.5)
        fig.text(x_titulo, 0.945, titulo, fontsize=7.6, fontweight="bold", color=TINTA, ha="left", va="center")

    def punto(config, **kw):
        m = MARCA[config]
        return Line2D(
            [], [], linestyle="none", marker=m["marker"], markersize=4.2,
            markerfacecolor=m["color"], markeredgecolor="white", markeredgewidth=0.5, **kw,
        )

    elementos = [
        punto("A", label="Config A (título + resumen)"),
        punto("B", label="Config B (Top-3 fragmentos)"),
        Line2D([], [], color=TINTA, linewidth=1.2, label="media por modelo (b)"),
    ]
    fig.legend(
        handles=elementos, loc="lower center", ncol=3, frameon=False, fontsize=6.6,
        handletextpad=0.35, columnspacing=1.8, bbox_to_anchor=(0.5, 0.005), labelcolor=TINTA_2,
    )
    guardar_fig(fig, salida, "fig12_a4_rondas_seleccion")


def generar_diagnostico_test_revisado(diag: pd.DataFrame, salida: Path) -> None:
    """Genera Figura 9 mejorada: 2 paneles alineados (Confirmadas + Dumbbell Corregidas) A4 vs. A5."""
    c_conf = "confirmadas: acierta"
    c_auto = "corregidas: predice la automática"
    c_rev = "corregidas: predice la revisada"

    def clasificar_grupo(fila: pd.Series) -> str:
        m = str(fila["modelo"])
        if m == "granite-4.2-8b":
            return "Granite-4.2-8B (A4)"
        if m == "Qwen3-0.6B-Base":
            return "Qwen3-0.6B (clasif./gen.)"
        if "scibert" in m.lower():
            return "SciBERT 110M (A4)"
        return "R2 LoRA 2B–4B (20 mod.)"

    diag = diag.copy()
    diag["grupo"] = diag.apply(clasificar_grupo, axis=1)

    grupos_a4 = [
        "Granite-4.2-8B (A4)",
        "R2 LoRA 2B–4B (20 mod.)",
        "Qwen3-0.6B (clasif./gen.)",
        "SciBERT 110M (A4)",
    ]
    a5_vals = calcular_diagnostico_a5()
    grupos_a5 = ["GPT-4.1-mini 0s-B (A5)", "Gemini-3.8-Flash 0s-B (A5)"]
    etiquetas_y = grupos_a4 + grupos_a5

    # Coordenadas verticales con una ligera separación visual entre A4 y A5
    y_pos = [0.0, 1.0, 2.0, 3.0, 4.35, 5.35]
    y_sep = 3.675

    fig = plt.figure(figsize=(7.16, 2.74))
    ax_l = fig.add_axes([0.205, 0.225, 0.200, 0.620])
    ax_r = fig.add_axes([0.455, 0.225, 0.415, 0.620])

    fig.text(0.205, 0.945, "(a) Confirmadas (n = 1.278)", fontsize=7.4, fontweight="bold", color=TINTA, ha="left", va="center")
    fig.text(0.455, 0.945, "(b) Corregidas (n = 1.520): automática (○) vs. revisada (●)", fontsize=7.4, fontweight="bold", color=TINTA, ha="left", va="center")

    # Banda gris del rango global de los 28 modelos A4 en Confirmadas
    a4_conf_min, a4_conf_max = diag[c_conf].min(), diag[c_conf].max()
    ax_l.axvspan(a4_conf_min, a4_conf_max, color=BANDA, zorder=0)
    ax_l.text(
        (a4_conf_min + a4_conf_max) / 2, -0.62,
        f"A4: {a4_conf_min * 100:.1f}–{a4_conf_max * 100:.1f}%".replace(".", ","),
        fontsize=5.8, color=TINTA_2, ha="center", va="bottom",
    )

    # Bandas grises sutiles de los 28 modelos A4 en Corregidas
    a4_auto_min, a4_auto_max = diag[c_auto].min(), diag[c_auto].max()
    a4_rev_min, a4_rev_max = diag[c_rev].min(), diag[c_rev].max()
    ax_r.axvspan(a4_auto_min, a4_auto_max, color=BANDA, zorder=0)
    ax_r.axvspan(a4_rev_min, a4_rev_max, color=BANDA, zorder=0)
    ax_r.text(
        (a4_auto_min + a4_auto_max) / 2, -0.62,
        f"A4 auto: {a4_auto_min * 100:.1f}–{a4_auto_max * 100:.1f}%".replace(".", ","),
        fontsize=5.7, color=TINTA_2, ha="center", va="bottom",
    )
    ax_r.text(
        (a4_rev_min + a4_rev_max) / 2, -0.62,
        "A4 rev.: " + f"{a4_rev_min * 100:.1f}–{a4_rev_max * 100:.1f}%".replace(".", ","),
        fontsize=5.7, color=TINTA_2, ha="center", va="bottom",
    )

    # Separador horizontal entre supervisados A4 y API A5
    for ax in (ax_l, ax_r):
        ax.axhline(y_sep, color=TINTA_3, linewidth=0.6, linestyle="--", alpha=0.7, zorder=1)

    # Dibujar familias A4 (puntos individuales tenues + media destacada)
    for idx, grp in enumerate(grupos_a4):
        yy = y_pos[idx]
        sub = diag[diag["grupo"] == grp]
        n_pts = len(sub)
        offsets = np.linspace(-0.18, 0.18, n_pts) if n_pts > 1 else np.array([0.0])

        # Panel izquierdo: Confirmadas
        vals_conf = np.sort(sub[c_conf].to_numpy())
        ax_l.plot(vals_conf, yy + offsets, "o", linestyle="none", markersize=2.6,
                  color=AZUL, alpha=0.35, markeredgewidth=0, zorder=2)
        m_conf = float(sub[c_conf].mean())
        ax_l.plot([0.50, m_conf], [yy, yy], color=AZUL, linewidth=1.6, alpha=0.28, solid_capstyle="round", zorder=2)
        ax_l.plot(m_conf, yy, "o", markersize=4.6, color=AZUL, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax_l.text(m_conf + 0.018, yy, f"{m_conf * 100:.1f}%".replace(".", ","),
                  fontsize=6.1, color=TINTA, va="center", ha="left")

        # Panel derecho: Dumbbell Corregidas (naranja = sesgo hacia automática)
        vals_auto = np.sort(sub[c_auto].to_numpy())
        vals_rev = np.sort(sub[c_rev].to_numpy())
        ax_r.plot(vals_auto, yy + offsets, "o", linestyle="none", markersize=2.4,
                  markerfacecolor="white", markeredgecolor=NARANJA, markeredgewidth=0.5, alpha=0.45, zorder=2)
        ax_r.plot(vals_rev, yy + offsets, "o", linestyle="none", markersize=2.4,
                  color=NARANJA, alpha=0.32, markeredgewidth=0, zorder=2)

        m_auto = float(sub[c_auto].mean())
        m_rev = float(sub[c_rev].mean())
        delta_pp = (m_rev - m_auto) * 100.0
        ax_r.plot([m_rev, m_auto], [yy, yy], color=NARANJA, linewidth=1.8, solid_capstyle="round", zorder=3)
        ax_r.plot(m_auto, yy, "o", markersize=5.0, markerfacecolor="white",
                  markeredgecolor=NARANJA, markeredgewidth=1.1, zorder=4)
        ax_r.plot(m_rev, yy, "o", markersize=5.0, markerfacecolor=NARANJA,
                  markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax_r.text(m_rev - 0.018, yy, f"{m_rev * 100:.1f}%".replace(".", ","),
                  fontsize=5.9, color=NARANJA, va="center", ha="right", fontweight="bold")
        ax_r.text(m_auto + 0.018, yy, f"{m_auto * 100:.1f}%".replace(".", ","),
                  fontsize=5.9, color=TINTA_2, va="center", ha="left")
        ax_r.text(0.815, yy, f"{delta_pp:+.1f} pp".replace(".", ",").replace("-", "−"),
                  fontsize=6.0, color=NARANJA, va="center", ha="left", fontweight="bold")

    # Dibujar referencias A5 (azul = alineación con revisada)
    for j, grp in enumerate(grupos_a5):
        yy = y_pos[len(grupos_a4) + j]
        m_conf, m_auto, m_rev = a5_vals[grp]
        delta_pp = (m_rev - m_auto) * 100.0

        # Panel izquierdo
        ax_l.plot([0.50, m_conf], [yy, yy], color=TINTA_2, linewidth=1.6, alpha=0.22, solid_capstyle="round", zorder=2)
        ax_l.plot(m_conf, yy, "D", markersize=4.4, color=TINTA, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax_l.text(m_conf + 0.018, yy, f"{m_conf * 100:.1f}%".replace(".", ","),
                  fontsize=6.1, color=TINTA, va="center", ha="left", fontweight="bold")

        # Panel derecho: Dumbbell invertido (azul = revisada > automática)
        ax_r.plot([m_auto, m_rev], [yy, yy], color=AZUL, linewidth=1.9, solid_capstyle="round", zorder=3)
        ax_r.plot(m_auto, yy, "D", markersize=4.6, markerfacecolor="white",
                  markeredgecolor=AZUL, markeredgewidth=1.1, zorder=4)
        ax_r.plot(m_rev, yy, "D", markersize=4.8, markerfacecolor=AZUL,
                  markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax_r.text(m_auto - 0.018, yy, f"{m_auto * 100:.1f}%".replace(".", ","),
                  fontsize=5.9, color=TINTA_2, va="center", ha="right")
        ax_r.text(m_rev + 0.018, yy, f"{m_rev * 100:.1f}%".replace(".", ","),
                  fontsize=5.9, color=AZUL, va="center", ha="left", fontweight="bold")
        ax_r.text(0.815, yy, f"{delta_pp:+.1f} pp".replace(".", ","),
                  fontsize=6.0, color=AZUL, va="center", ha="left", fontweight="bold")

    # Encabezado de columna Δ (brecha revisada - automática)
    ax_r.text(0.815, -0.62, "Δ (rev. − auto.)", fontsize=5.8, color=TINTA_2, ha="left", va="bottom", fontweight="bold")

    # Formato de ejes
    for ax in (ax_l, ax_r):
        ax.set_ylim(5.95, -0.82)
        ax.grid(axis="x", color=REJILLA, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=6.3, colors=TINTA_2, length=0, pad=2.5)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(TINTA_3)
        ax.spines["bottom"].set_linewidth(0.5)

    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(etiquetas_y, color=TINTA, fontsize=6.6)
    for lbl, name in zip(ax_l.get_yticklabels(), etiquetas_y):
        if "(A5)" in name or "Granite" in name:
            lbl.set_fontweight("bold")
    ax_l.set_xlim(0.50, 1.01)
    ax_l.xaxis.set_major_locator(MultipleLocator(0.10))
    ax_l.xaxis.set_major_formatter(FuncFormatter(pct_coma))
    ax_l.set_xlabel("% acierto en confirmadas", fontsize=6.4, color=TINTA_2, labelpad=2.5)

    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels([])
    ax_r.set_xlim(0.0, 0.80)
    ax_r.xaxis.set_major_locator(MultipleLocator(0.20))
    ax_r.xaxis.set_major_formatter(FuncFormatter(pct_coma))
    ax_r.set_xlabel("% de instancias corregidas (el resto predice una 3.ª clase)", fontsize=6.4, color=TINTA_2, labelpad=2.5)

    # Leyenda inferior limpia
    elementos = [
        Line2D([], [], linestyle="none", marker="o", markersize=4.6, markerfacecolor="white",
               markeredgecolor=TINTA_2, markeredgewidth=1.0, label="repite etiqueta automática (○)"),
        Line2D([], [], linestyle="none", marker="o", markersize=4.6, markerfacecolor=TINTA_2,
               markeredgecolor="white", markeredgewidth=0.5, label="acierta etiqueta revisada (●)"),
        Line2D([], [], color=NARANJA, linewidth=1.8, label="A4 supervisados (sesgo hacia automática)"),
        Line2D([], [], color=AZUL, linewidth=1.8, marker="D", markersize=3.8,
               markerfacecolor=AZUL, markeredgecolor="white", markeredgewidth=0.4,
               label="A5 vía API (alineado con revisada)"),
    ]
    fig.legend(
        handles=elementos, loc="lower center", ncol=2, frameon=False, fontsize=6.2,
        handletextpad=0.4, columnspacing=1.5, labelspacing=0.25,
        bbox_to_anchor=(0.52, 0.005), labelcolor=TINTA_2,
    )
    guardar_fig(fig, salida, "fig13_diagnostico_test_revisado")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datos", default="reports/a4", help="carpeta con CSVs de A4")
    parser.add_argument("--salida", default="reports/figures/paper", help="carpeta de las figuras del artículo")
    args = parser.parse_args()
    datos, salida = Path(args.datos), Path(args.salida)
    r1 = pd.read_csv(datos / "r1_comparacion.csv")
    r2 = pd.read_csv(datos / "r2_comparacion.csv")
    fin = pd.read_csv(datos / "final_comparacion.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})

    # 1. Solo Definitivos (1 columna, cuerpo principal)
    generar_definitivos_1col(fin, salida)

    # 2. Rondas R1 + R2 (2 columnas, Anexos)
    generar_rondas_2col(r1, r2, salida)

    # 3. Diagnóstico sobre el test revisado (Confirmadas + Dumbbell Corregidas, Anexos)
    diag_csv = datos / "test_revisado_por_modelo.csv"
    if diag_csv.is_file():
        diag = pd.read_csv(diag_csv)
        generar_diagnostico_test_revisado(diag, salida)
    else:
        print(f"Aviso: {diag_csv} no encontrado; se omite fig13_diagnostico_test_revisado.")


if __name__ == "__main__":
    main()
