"""Figura de A4 para el artículo (reports/figures/paper/fig12_a4_seleccion_y_techo.{pdf,png}).

Tres paneles a ancho de página IEEE (7,16 in):
  (a) Ronda 1: Macro-F1 de test por modelo y configuración, con IC95 (comparacion.csv de `comparacion_modelos`).
  (b) Ronda 2: las 58 corridas (3 semillas x A/B) y la media por modelo (comparacion.csv de `comparacion_modelos_v2`).
  (c) Modelos definitivos: Macro-F1 frente al test automático (labeled-v2) y frente al Gold humano (test-revisado).

Entradas: los comparacion.csv que exporta notebooks/analizar_resultados.ipynb con GUARDAR=1, copiados en reports/a4/.
Los valores sobre el Gold que no están en esos CSV (mismas predicciones recalificadas contra el test revisado) van en
GOLD_RECALIFICADO, con su procedencia.

Uso:  python scripts/generar_figuras_a4.py [--datos reports/a4] [--salida reports/figures/paper]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Paleta: dos series (Config A y Config B) validadas para daltonismo sobre fondo blanco; la forma repite la identidad.
AZUL, NARANJA = "#2a78d6", "#eb6834"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datos", default="reports/a4", help="carpeta con r1_, r2_ y final_comparacion.csv")
    parser.add_argument("--salida", default="reports/figures/paper", help="carpeta de las figuras del artículo")
    args = parser.parse_args()
    datos, salida = Path(args.datos), Path(args.salida)
    r1 = pd.read_csv(datos / "r1_comparacion.csv")
    r2 = pd.read_csv(datos / "r2_comparacion.csv")
    fin = pd.read_csv(datos / "final_comparacion.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.16, 2.36))
    # [izquierda de las etiquetas de fila, izquierda del eje, ancho del eje], en fracción del ancho de la figura
    paneles = [
        (0.006, 0.136, 0.186, "(a) Ronda 1: familias de hasta 4B", panel_ronda1, r1),
        (0.340, 0.458, 0.186, "(b) Ronda 2: 3 semillas × Config A/B", panel_ronda2, r2),
        (0.664, 0.782, 0.190, "(c) Definitivos: test automático → Gold", panel_definitivos, fin),
    ]
    for x_titulo, x_eje, ancho, titulo, dibujar, tabla in paneles:
        ax = fig.add_axes([x_eje, 0.245, ancho, 0.630])
        dibujar(ax, tabla)
        fig.text(x_titulo, 0.955, titulo, fontsize=7.2, fontweight="bold", color=TINTA, ha="left", va="center")
    leyenda(fig)
    salida.mkdir(parents=True, exist_ok=True)
    for extension, extra in (("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(salida / f"fig12_a4_seleccion_y_techo.{extension}", **extra)
    print(f"Figura escrita en {salida}/fig12_a4_seleccion_y_techo.pdf y .png")


if __name__ == "__main__":
    main()
