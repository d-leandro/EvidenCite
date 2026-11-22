"""Genera las 4 figuras empíricas A5 optimizadas para publicación IEEE doble columna.

Guarda las figuras en ``reports/figures/paper/`` sin modificar ``reports/figures/a5/``.
Todas las cifras se calculan directamente desde los artefactos reales:
- ``data/labeled/test.jsonl``
- ``models/llm_api_cache/*.jsonl``
- ``models/llm_api_*/metrics_summary.json``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "reports" / "figures" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_9_CLASSES = [
    "Background",
    "Gap",
    "Basis",
    "Comparison",
    "Application",
    "Improvement",
    "Evidence",
    "Originator",
    "Further Reading",
]
SHORT_CLASS_LABELS = {
    "Background": "Background",
    "Gap": "Gap",
    "Basis": "Basis",
    "Comparison": "Comparison",
    "Application": "Application",
    "Improvement": "Improvement",
    "Evidence": "Evidence",
    "Originator": "Originator",
    "Further Reading": "Further Read.",
}
LABEL_TO_ID = {c: i for i, c in enumerate(CANONICAL_9_CLASSES)}

VARIANTS = [
    ("gpt41_zs_a",  "GPT 0s-A",  "llm_api_gpt41mini_zeroshot_config_a",   "gpt41mini_config_a.jsonl",     "gpt-4.1-mini",     0, "A", ""),
    ("gpt41_zs_b",  "GPT 0s-B",  "llm_api_gpt41mini_zeroshot_config_b",   "gpt41mini_config_b.jsonl",     "gpt-4.1-mini",     0, "B", ""),
    ("gpt41_fs_a",  "GPT 9s-A",  "llm_api_gpt41mini_fewshot_config_a",    "gpt41mini_config_a.jsonl",     "gpt-4.1-mini",     9, "A", "::fs9"),
    ("gpt41_fs_b",  "GPT 9s-B",  "llm_api_gpt41mini_fewshot_config_b",    "gpt41mini_config_b.jsonl",     "gpt-4.1-mini",     9, "B", "::fs9"),
    ("gemini_zs_a", "Gem 0s-A",  "llm_api_gemini38flash_zeroshot_config_a", "gemini38flash_config_a.jsonl", "gemini-3.8-flash", 0, "A", ""),
    ("gemini_zs_b", "Gem 0s-B",  "llm_api_gemini38flash_zeroshot_config_b", "gemini38flash_config_b.jsonl", "gemini-3.8-flash", 0, "B", ""),
    ("gemini_fs_a", "Gem 9s-A",  "llm_api_gemini38flash_fewshot_config_a",  "gemini38flash_config_a.jsonl", "gemini-3.8-flash", 9, "A", "::fs9"),
    ("gemini_fs_b", "Gem 9s-B",  "llm_api_gemini38flash_fewshot_config_b",  "gemini38flash_config_b.jsonl", "gemini-3.8-flash", 9, "B", "::fs9"),
]

COLORS = {
    "gpt41_zs_a":  "#2B6CB0",
    "gpt41_zs_b":  "#0D9488",
    "gpt41_fs_a":  "#64748B",
    "gpt41_fs_b":  "#8B5CF6",
    "gemini_zs_a": "#D97706",
    "gemini_zs_b": "#C53030",
    "gemini_fs_a": "#7C3AED",
    "gemini_fs_b": "#DB2777",
}


def fmt_es(val: float, dec: int = 2) -> str:
    return f"{val:.{dec}f}".replace(".", ",")


def fmt_int_es(val: int) -> str:
    return f"{int(val):,}".replace(",", ".")


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11.5,
        "axes.titlesize": 13.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "legend.fontsize": 11.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def map_macro_section(sec) -> str:
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return "Sin sección"
    s = str(sec).lower()
    if any(k in s for k in ("intro", "motivation", "overview", "prelimin", "backgr")):
        return "Intro / Prelim"
    if any(k in s for k in ("related", "literat", "prior", "previous")):
        return "Related Work"
    if any(k in s for k in ("method", "approach", "model", "architect", "algorith", "framework", "system", "implement", "train", "setup")):
        return "Method / Setup"
    if any(k in s for k in ("experim", "evaluat", "result", "ablation", "discuss", "analysis", "benchmark", "compar")):
        return "Experiments"
    if any(k in s for k in ("conclu", "future", "limit", "summary")):
        return "Conclusion"
    return "Otras"


def agreement_stratum(row) -> str:
    ls = row["prior_label_source"] if (row["label_source"] == "human_review" and pd.notna(row["prior_label_source"])) else row["label_source"]
    pa = row["panel_agreement"]
    if ls == "human_gold":
        return "Human Gold\n(MultiCite)"
    if ls == "llm_panel" and pa is not None and pa >= 0.99:
        return "Panel\nUnánime 3/3"
    if ls == "llm_panel":
        return "Panel\nMayoría 2/3"
    return "Needs Review\n(Desacuerdo)"


def load_data():
    test_path = BASE_DIR / "data" / "labeled" / "test.jsonl"
    test_raw = [json.loads(ln) for ln in test_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n_test = len(test_raw)
    y_true = [r["label"] for r in test_raw]

    gold_df = pd.DataFrame([
        {
            "instance_id": r["instance_id"],
            "y_true": r["label"],
            "y_true_idx": LABEL_TO_ID[r["label"]],
            "citing_section": (r.get("citing_paper") or {}).get("rhetorical_section"),
            "label_source": (r.get("annotation_metadata") or {}).get("label_source"),
            "prior_label_source": (r.get("annotation_metadata") or {}).get("prior_label_source"),
            "panel_agreement": (r.get("annotation_metadata") or {}).get("panel_agreement"),
        }
        for r in test_raw
    ])
    gold_df["macro_section"] = gold_df["citing_section"].map(map_macro_section)
    gold_df["stratum"] = gold_df.apply(agreement_stratum, axis=1)

    caches_by_file = {}
    for fname in sorted({v[3] for v in VARIANTS}):
        fpath = BASE_DIR / "models" / "llm_api_cache" / fname
        recs = {}
        with fpath.open("r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    obj = json.loads(ln)
                    recs[obj["cache_key"]] = obj["record"]
        caches_by_file[fname] = recs

    runs = {}
    summary_rows = []
    rows_all = []
    for vid, short_lbl, exp_dir_name, cache_fname, model_name, shots, cfg, suffix in VARIANTS:
        exp_dir = BASE_DIR / "models" / exp_dir_name
        ms = json.loads((exp_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        a6 = ms["test_metrics_a6"]
        a7 = ms["operational_metrics_a7"]
        cache_map = caches_by_file[cache_fname]
        ordered_recs = [cache_map[f"{iid}::{cfg}::generic::{model_name}{suffix}"] for iid in gold_df["instance_id"]]
        runs[vid] = {"label": short_lbl, "a6": a6, "a7": a7, "records": ordered_recs}

        lat_arr = np.array([r["latency_ms"] for r in ordered_recs], dtype=np.float64)
        summary_rows.append({
            "vid": vid,
            "Variante": short_lbl,
            "Modelo": model_name,
            "Shots": shots,
            "Config": cfg,
            "Macro-F1": a6["macro_f1_all_9"],
            "Weighted-F1": a6["weighted_f1"],
            "Accuracy": a6["micro_f1"],
            "Costo Neto ($)": a7["estimated_api_cost_usd"],
            "Costo Bruto ($)": a7["gross_api_cost_usd"],
            "Costo Neto / 10k ($)": (a7["estimated_api_cost_usd"] / n_test) * 10000.0,
            "Lat p50 (ms)": float(np.median(lat_arr)),
        })
        for idx, r in enumerate(ordered_recs):
            rows_all.append({
                "vid": vid,
                "Variante": short_lbl,
                "Modelo": model_name,
                "Shots": shots,
                "Config": cfg,
                "test_idx": idx,
                "is_correct": int(r["predicted_label"] == y_true[idx]),
                "input_tokens": int(r["input_tokens"]),
                "cached_input_tokens": int(r["cached_input_tokens"]),
                "uncached_input_tokens": int(r["input_tokens"] - r["cached_input_tokens"]),
                "output_tokens": int(r["output_tokens"]),
                "thoughts_tokens": int(r["thoughts_tokens"]),
                "latency_ms": float(r["latency_ms"]),
                "macro_section": gold_df.iloc[idx]["macro_section"],
                "stratum": gold_df.iloc[idx]["stratum"],
            })

    return gold_df, runs, pd.DataFrame(summary_rows), pd.DataFrame(rows_all)


def plot_fig02(runs, df_summary) -> None:
    """Fig. 5 del manuscrito: Heatmap F1 por clase + Dot/Lollipop global (Opción 5-2)."""
    from matplotlib.lines import Line2D

    support_dict = runs["gpt41_zs_a"]["a6"]["per_class_support"]
    cls_order = sorted(CANONICAL_9_CLASSES, key=lambda c: support_dict[c], reverse=True)
    mat_f1 = np.array(
        [[runs[v[0]]["a6"]["per_class_f1"][c] for c in cls_order] for v in VARIANTS],
        dtype=float,
    )

    fig = plt.figure(figsize=(12.8, 3.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.36, 1.05], wspace=0.16)

    # Panel A: Heatmap F1 por clase
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(mat_f1, cmap="YlGnBu", aspect="auto", vmin=0.15, vmax=0.95)
    ax0.set_xticks(np.arange(len(cls_order)))
    ax0.set_xticklabels(
        [f"{SHORT_CLASS_LABELS[c]} ({fmt_int_es(support_dict[c])})" for c in cls_order],
        rotation=32, ha="right", rotation_mode="anchor", fontsize=10.2,
    )
    ax0.set_yticks(np.arange(len(VARIANTS)))
    ax0.set_yticklabels([v[1] for v in VARIANTS], fontsize=10.2)
    ax0.axhline(3.5, color="white", linewidth=2.2)
    ax0.set_title("(a) F1-Score por clase canónica (por soporte)", pad=8, fontsize=11.5)

    for i in range(mat_f1.shape[0]):
        for j in range(mat_f1.shape[1]):
            val = mat_f1[i, j]
            ax0.text(
                j, i, fmt_es(val, 2),
                ha="center", va="center", fontsize=10.2,
                color="white" if val > 0.60 else "#111111",
                fontweight="bold",
            )
    cbar = fig.colorbar(im, ax=ax0, fraction=0.032, pad=0.025)
    cbar.set_label("F1-Score", fontsize=10.2)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_es(x, 1)))
    cbar.ax.tick_params(labelsize=10.0)

    # Panel B: Dot / Lollipop plot conectado horizontal (mismo eje Y que el heatmap)
    ax1 = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(len(VARIANTS))

    # Banda suave resaltando el óptimo global (Gem 0s-B, fila 5)
    ax1.axhspan(4.56, 5.44, color="#FEF2F2", alpha=0.95, zorder=0)
    ax1.axhline(3.5, color="#CBD5E1", linestyle="--", linewidth=1.1, zorder=1)

    c_macro = "#1F6AA5"
    c_acc   = "#D97706"
    c_wf1   = "#0D9488"

    for idx, row in df_summary.iterrows():
        m_f1 = row["Macro-F1"]
        acc  = row["Accuracy"]
        w_f1 = row["Weighted-F1"]
        is_opt = row["vid"] == "gemini_zs_b"
        fw = "bold" if is_opt else "normal"

        # Riel conector horizontal entre mínimo y máximo de las 3 métricas
        ax1.plot(
            [min(m_f1, acc, w_f1), max(m_f1, acc, w_f1)], [idx, idx],
            color="#94A3B8" if not is_opt else "#475569",
            linewidth=2.0 if not is_opt else 2.5, zorder=2,
        )
        # Marcadores diferenciados
        ax1.scatter(m_f1, idx, s=88 if is_opt else 72, marker="o", color=c_macro, edgecolors="#1A202C", linewidth=0.9, zorder=4)
        ax1.scatter(acc,  idx, s=84 if is_opt else 68, marker="^", color=c_acc,   edgecolors="#1A202C", linewidth=0.9, zorder=5)
        ax1.scatter(w_f1, idx, s=80 if is_opt else 64, marker="s", color=c_wf1,   edgecolors="#1A202C", linewidth=0.9, zorder=4)

        # Etiquetas limpias a izquierda (Macro-F1) y derecha (Weighted-F1)
        ax1.text(m_f1 - 0.013, idx, fmt_es(m_f1, 3), va="center", ha="right", fontsize=10.5, fontweight=fw, color="#1E3A5F")
        ax1.text(w_f1 + 0.013, idx, fmt_es(w_f1, 3), va="center", ha="left",  fontsize=10.5, fontweight=fw, color="#0F5149")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([])
    ax1.set_ylim(-0.5, len(VARIANTS) - 0.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0.32, 0.785)
    ax1.set_xticks([0.35, 0.45, 0.55, 0.65, 0.75])
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_es(x, 2)))
    ax1.set_xlabel("Puntaje")
    ax1.set_title("(b) Desempeño global (N = 2.798)", pad=8, fontsize=11.5)
    ax1.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)

    leg_b = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c_macro, markeredgecolor="#1A202C", markersize=6.8, label="Macro-F1"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=c_acc,   markeredgecolor="#1A202C", markersize=6.8, label="Accuracy"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=c_wf1,   markeredgecolor="#1A202C", markersize=6.4, label="Weighted-F1"),
    ]
    ax1.legend(
        handles=leg_b, loc="upper center", bbox_to_anchor=(0.49, -0.22),
        ncol=3, frameon=True, framealpha=0.95, fontsize=10.0,
        columnspacing=0.85, handletextpad=0.35,
    )

    fig.savefig(OUT_DIR / "fig02_desempeno_global_y_por_clase.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig03_split(gold_df, runs, df_all) -> None:
    """Genera por separado: (1) matrices de confusión A5 para Anexo A, (2) Dumbbell por macro-sección (1 col. cuerpo)."""
    from matplotlib.lines import Line2D

    short_9 = ["Bkg", "Gap", "Bas", "Cmp", "App", "Imp", "Evi", "Org", "F.R."]
    color_gpt = "#1F6AA5"
    color_gem = "#C53030"

    # --- 1. Matrices de confusión A5 (Anexo A) ---
    fig_cm = plt.figure(figsize=(11.8, 4.75))
    gs_cm = fig_cm.add_gridspec(1, 2, wspace=0.34)
    for col_idx, (vid, title_txt, cmap_name, title_color) in enumerate([
        ("gpt41_zs_b",  "(a) GPT-4.1-mini 0s-B (% fila)",   "Blues", "#1A4D7A"),
        ("gemini_zs_b", "(b) Gemini-3.8-Flash 0s-B (% fila)", "Reds",  "#9B2226"),
    ]):
        ax = fig_cm.add_subplot(gs_cm[0, col_idx])
        cm = np.array(runs[vid]["a6"]["confusion_matrix"], dtype=float)
        cm_pct = (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)) * 100.0
        im = ax.imshow(cm_pct, cmap=cmap_name, vmin=0.0, vmax=100.0)
        ax.set_title(title_txt, pad=10, color=title_color)
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels(short_9, rotation=38, ha="right", rotation_mode="anchor", fontsize=10.5)
        ax.set_yticklabels(short_9, fontsize=10.5)
        ax.set_xlabel("Predicción")
        if col_idx == 0:
            ax.set_ylabel("Etiqueta real")
        for i in range(9):
            for j in range(9):
                val = cm_pct[i, j]
                if val >= 8.0 or i == j:
                    ax.text(
                        j, i, f"{int(round(val))}",
                        ha="center", va="center", fontsize=10.5,
                        fontweight="bold" if i == j else "normal",
                        color="white" if val > 52.0 else "#111111",
                    )
        cbar = fig_cm.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
        cbar.set_ticks([0, 25, 50, 75, 100])
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
        cbar.ax.tick_params(labelsize=9.8)
        if col_idx == 1:
            cbar.set_label("% fila", fontsize=10.5)

    fig_cm.savefig(OUT_DIR / "fig03ab_matrices_confusion_a5.png", bbox_inches="tight")
    plt.close(fig_cm)

    # --- 2. Dumbbell pareado A -> B por macro-sección citante (1 columna en cuerpo principal) ---
    fig_s, ax2 = plt.subplots(figsize=(5.7, 3.30))
    sec_counts = gold_df["macro_section"].value_counts()
    sec_order = ["Related Work", "Intro / Prelim", "Experiments", "Method / Setup", "Otras", "Conclusion", "Sin sección"]
    sec_order = [s for s in sec_order if s in sec_counts.index]
    x_s = np.arange(len(sec_order))

    def sec_acc(vid: str) -> list[float]:
        return [
            df_all[(df_all["vid"] == vid) & (df_all["macro_section"] == s)]["is_correct"].mean() * 100.0
            for s in sec_order
        ]

    gpt_a = sec_acc("gpt41_zs_a")
    gpt_b = sec_acc("gpt41_zs_b")
    gem_a = sec_acc("gemini_zs_a")
    gem_b = sec_acc("gemini_zs_b")

    offset = 0.14
    for i in range(len(sec_order)):
        ax2.axvline(i, color="#E2E8F0", linewidth=0.9, zorder=0)
        xg = i - offset
        ax2.plot([xg, xg], [gpt_a[i], gpt_b[i]], color=color_gpt, linewidth=2.2, zorder=2)
        ax2.scatter(xg, gpt_a[i], s=72, marker="o", facecolors="white", edgecolors=color_gpt, linewidth=1.8, zorder=4)
        ax2.scatter(xg, gpt_b[i], s=80, marker="o", facecolors=color_gpt, edgecolors="#1A202C", linewidth=1.0, zorder=5)

        xm = i + offset
        ax2.plot([xm, xm], [gem_a[i], gem_b[i]], color=color_gem, linewidth=2.2, zorder=2)
        ax2.scatter(xm, gem_a[i], s=72, marker="o", facecolors="white", edgecolors=color_gem, linewidth=1.8, zorder=4)
        ax2.scatter(xm, gem_b[i], s=80, marker="o", facecolors=color_gem, edgecolors="#1A202C", linewidth=1.0, zorder=5)

    ax2.annotate(
        "+10,3 pp",
        xy=(-offset, gpt_b[0] + 0.8),
        xytext=(0.40, 60.5),
        fontsize=10.8, fontweight="bold", color=color_gpt,
        arrowprops=dict(arrowstyle="->", color=color_gpt, lw=1.4),
        zorder=6,
    )

    ax2.set_xticks(x_s)
    ax2.set_xticklabels(
        [f"{s} ({fmt_int_es(sec_counts[s])})" for s in sec_order],
        rotation=30, ha="right", rotation_mode="anchor", fontsize=10.4,
    )
    ax2.set_xlim(-0.55, len(sec_order) - 0.45)
    ax2.set_ylabel("Accuracy (%)", fontsize=11.0)
    ax2.set_ylim(25, 84)
    ax2.set_title("Exactitud A $\\rightarrow$ B por macro-sección (0-Shot)", pad=9, fontsize=11.8)
    ax2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    leg_c = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_gpt, markeredgecolor="#1A202C", markersize=7.0, label="GPT-4.1-mini"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_gem, markeredgecolor="#1A202C", markersize=7.0, label="Gemini-3.8-Flash"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#334155", markeredgecolor="#1A202C", markersize=7.0, label="Config B (local)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",   markeredgecolor="#334155", markeredgewidth=1.8, markersize=7.0, label="Config A (global)"),
    ]
    ax2.legend(
        handles=leg_c, loc="lower center", ncol=2,
        frameon=True, framealpha=0.95, fontsize=9.4,
        handletextpad=0.35, columnspacing=0.85, borderpad=0.38,
    )

    fig_s.savefig(OUT_DIR / "fig03c_exactitud_por_macro_seccion.png", bbox_inches="tight")
    plt.close(fig_s)


def plot_fig08(df_summary, df_all) -> None:
    """Fig. 7 del manuscrito: Tokens + Costo anidado Bruto/Neto + Latencia Hit vs Miss (Opción 7-1)."""
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.75))
    x_v = np.arange(len(VARIANTS))
    x_lbls = [f"{'GPT' if 'gpt' in v[0] else 'Gem'}\n{v[5]}s-{v[6]}" for v in VARIANTS]

    # Panel A: Millones de tokens (paleta sobria daltónico-segura + separador GPT | Gemini)
    ax0 = axes[0]
    uncached_m = [df_all[df_all["vid"] == v[0]]["uncached_input_tokens"].sum() / 1e6 for v in VARIANTS]
    cached_m   = [df_all[df_all["vid"] == v[0]]["cached_input_tokens"].sum() / 1e6 for v in VARIANTS]
    out_th_m   = [(df_all[df_all["vid"] == v[0]]["output_tokens"].sum() + df_all[df_all["vid"] == v[0]]["thoughts_tokens"].sum()) / 1e6 for v in VARIANTS]
    pct_cache  = [df_all[df_all["vid"] == v[0]]["cached_input_tokens"].sum() / df_all[df_all["vid"] == v[0]]["input_tokens"].sum() * 100.0 for v in VARIANTS]

    ax0.bar(x_v, uncached_m, width=0.66, color="#5B7C99", label="Entrada sin caché", zorder=3)
    ax0.bar(x_v, cached_m, width=0.66, bottom=uncached_m, color="#0D9488", label="Entrada cacheada", zorder=3)
    ax0.bar(x_v, out_th_m, width=0.66, bottom=np.array(uncached_m) + np.array(cached_m), color="#E07A5F", label="Salida + pensam.", zorder=3)
    ax0.axvline(3.5, color="#CBD5E1", linestyle="--", linewidth=1.1, zorder=1)

    for i, pct in enumerate(pct_cache):
        tot = uncached_m[i] + cached_m[i] + out_th_m[i]
        if pct >= 4.0:
            ax0.text(i, tot + 0.55, f"{fmt_es(pct, 0)}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#0F766E")
    ax0.set_xticks(x_v)
    ax0.set_xticklabels(x_lbls, fontsize=10.5)
    ax0.set_ylabel("Millones de tokens")
    ax0.set_title("(a) Tokens y tasa de caché", pad=10)
    ax0.set_ylim(0, 31.0)
    ax0.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax0.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10.2)

    # Panel B: Barras agrupadas lado a lado Costo Bruto vs. Neto
    ax1 = axes[1]
    w_b = 0.36
    ax1.bar(
        x_v - w_b / 2, df_summary["Costo Bruto ($)"], width=w_b,
        color="#E2E8F0", edgecolor="#64748B", linewidth=1.0,
        label="Bruto (sin caché)", zorder=3,
    )
    ax1.bar(
        x_v + w_b / 2, df_summary["Costo Neto ($)"], width=w_b,
        color="#1F6AA5", edgecolor="#1A202C", linewidth=0.7,
        label="Neto (con caché)", zorder=3,
    )
    ax1.axvline(3.5, color="#CBD5E1", linestyle="--", linewidth=1.1, zorder=1)

    for idx, r in df_summary.iterrows():
        gross_v = r["Costo Bruto ($)"]
        net_v = r["Costo Neto ($)"]
        if gross_v - net_v >= 1.5:
            sav_pct = (1.0 - net_v / gross_v) * 100.0
            ax1.text(
                idx - w_b / 2, gross_v + 0.42, f"-{int(round(sav_pct))}%",
                ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#0F766E",
            )
            ax1.text(
                idx + 0.02, net_v + 0.38, f"${fmt_es(net_v, 1)}",
                ha="left", va="bottom", fontsize=8.8, fontweight="bold", color="#1E3A5F",
                zorder=5,
            )
        else:
            ax1.text(
                idx, max(gross_v, net_v) + 0.42, f"${fmt_es(net_v, 1)}",
                ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#1E3A5F",
            )

    ax1.set_xticks(x_v)
    ax1.set_xticklabels(x_lbls, fontsize=10.5)
    ax1.set_xlim(-0.6, 7.85)
    ax1.set_ylabel("Costo corrida (USD, N = 2.798)")
    ax1.set_title("(b) Costo bruto vs. neto (etiqueta = neto)", pad=10)
    ax1.set_ylim(0, 21.5)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: fmt_es(y, 1)))
    ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10.2)

    # Panel C: Latencia p50 Cache Miss vs Cache Hit (paleta daltónico-segura gris/teal)
    ax2 = axes[2]
    fs_vids = ["gpt41_fs_a", "gpt41_fs_b", "gemini_fs_a", "gemini_fs_b"]
    pos = np.arange(len(fs_vids))
    hit_meds, miss_meds = [], []
    for v in fs_vids:
        sub = df_all[df_all["vid"] == v]
        hit_meds.append(sub[sub["cached_input_tokens"] > 0]["latency_ms"].median() / 1000.0)
        miss_meds.append(sub[sub["cached_input_tokens"] == 0]["latency_ms"].median() / 1000.0)

    w_c = 0.34
    ax2.bar(pos - w_c / 2 - 0.02, miss_meds, width=w_c, color="#64748B", edgecolor="#334155", linewidth=0.7, label="Cache Miss (p50)", zorder=3)
    ax2.bar(pos + w_c / 2 + 0.02, hit_meds,  width=w_c, color="#0D9488", edgecolor="#0F766E", linewidth=0.7, label="Cache Hit (p50)",  zorder=3)
    ax2.axvline(1.5, color="#CBD5E1", linestyle="--", linewidth=1.1, zorder=1)

    for i in pos:
        ax2.text(i - w_c / 2 - 0.06, miss_meds[i] + 0.07, f"{fmt_es(miss_meds[i], 2)} s", ha="center", va="bottom", fontsize=9.8, color="#334155")
        ax2.text(i + w_c / 2 + 0.06, hit_meds[i] + 0.07,  f"{fmt_es(hit_meds[i], 2)} s",  ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#0F766E")

    ax2.set_xticks(pos)
    ax2.set_xticklabels(["GPT\n9s-A", "GPT\n9s-B", "Gem\n9s-A", "Gem\n9s-B"], fontsize=10.5)
    ax2.set_ylabel("Latencia mediana p50 (s)")
    ax2.set_title("(c) Latencia con caché (9-Shot)", pad=10)
    ax2.set_ylim(0, 3.35)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_es(x, 1)))
    ax2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10.2)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_DIR / "fig08_tokens_cache_eficiencia_y_costos.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig11_split(gold_df, df_summary, df_all) -> None:
    """Genera por separado: (1) Frontera de Pareto (1 col. cuerpo), (2) Sensibilidad por estrato A1/A2 (Anexo A)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    color_gpt = "#1F6AA5"
    color_gem = "#C53030"

    # --- 1. Frontera de Pareto (1 columna en cuerpo principal) ---
    fig_p, ax0 = plt.subplots(figsize=(6.3, 3.90))

    label_specs = {
        "gpt41_zs_a":  ((-10, -16), "right"),
        "gpt41_zs_b":  ((-8, 24),   "right"),
        "gpt41_fs_a":  ((-10, -20), "right"),
        "gpt41_fs_b":  ((12, 12),   "left"),
        "gemini_zs_a": ((-16, -8),  "right"),
        "gemini_zs_b": ((-8, 16),   "right"),
        "gemini_fs_a": ((0, -32),   "center"),
        "gemini_fs_b": ((0, 16),    "center"),
    }

    pareto_sorted = df_summary.sort_values(["Costo Neto / 10k ($)", "Macro-F1"], ascending=[True, False])
    pareto_pts = []
    best_f1 = -1.0
    for _, rr in pareto_sorted.iterrows():
        if rr["Macro-F1"] > best_f1:
            pareto_pts.append((rr["Costo Neto / 10k ($)"], rr["Macro-F1"]))
            best_f1 = rr["Macro-F1"]

    xlim_min, xlim_max = 2.6, 82.0
    ylim_min, ylim_max = 0.365, 0.665

    dom_x = [p[0] for p in pareto_pts] + [xlim_max]
    dom_y = [p[1] for p in pareto_pts] + [pareto_pts[-1][1]]
    ax0.fill_between(dom_x, dom_y, ylim_min, color="#94A3B8", alpha=0.13, zorder=1)
    ax0.plot(
        [pareto_pts[-1][0], xlim_max], [pareto_pts[-1][1], pareto_pts[-1][1]],
        linestyle=":", color="#94A3B8", linewidth=1.1, zorder=2,
    )
    ax0.plot(
        [p[0] for p in pareto_pts], [p[1] for p in pareto_pts],
        linestyle="--", color="#334155", linewidth=1.6, zorder=3,
    )

    for _, r in df_summary.iterrows():
        vid = r["vid"]
        is_gpt = "gpt" in vid
        is_0s = "_zs_" in vid
        fam_color = color_gpt if is_gpt else color_gem
        x_val = r["Costo Neto / 10k ($)"]
        y_val = r["Macro-F1"]

        if vid == "gemini_zs_b":
            ax0.scatter(
                x_val, y_val,
                s=240, marker="o", facecolors="none",
                edgecolors=fam_color, linewidth=1.35, alpha=0.65, zorder=4,
            )

        ax0.scatter(
            x_val, y_val,
            s=110, marker="o",
            facecolors=fam_color if is_0s else "white",
            edgecolors="#1A202C" if is_0s else fam_color,
            linewidth=1.15 if is_0s else 2.0,
            zorder=5,
        )
        lbl_pt = f"{r['Variante']}\n(${fmt_es(x_val, 2)}; {fmt_es(y_val, 3)})"
        (dx, dy), ha_align = label_specs[vid]
        ax0.annotate(
            lbl_pt,
            xy=(x_val, y_val),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha_align,
            fontsize=10.3,
            fontweight="bold" if vid in ("gemini_zs_b", "gpt41_zs_b") else "normal",
            arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.85),
            zorder=6,
        )

    ax0.set_xscale("log")
    ax0.set_xlim(xlim_min, xlim_max)
    ax0.set_ylim(ylim_min, ylim_max)
    ax0.set_xticks([3, 5, 10, 20, 40, 60])
    ax0.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x)}"))
    ax0.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: fmt_es(y, 2)))
    ax0.set_xlabel("Costo neto por 10.000 citas (USD, escala log)")
    ax0.set_ylabel("Macro-F1 (9 clases)")
    ax0.set_title("Frontera de Pareto: costo neto vs. Macro-F1", pad=9)
    ax0.grid(True, which="major", linestyle="--", alpha=0.32, zorder=0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_gpt, markeredgecolor="#1A202C", markersize=7.2, label="GPT-4.1-mini"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_gem, markeredgecolor="#1A202C", markersize=7.2, label="Gemini-3.8-Flash"),
        Line2D([0], [0], color="#334155", linestyle="--", linewidth=1.5, label="Frontera de Pareto"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#334155", markeredgecolor="#1A202C", markersize=7.2, label="0-Shot (Pareto)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#334155", markeredgewidth=1.8, markersize=7.2, label="9-Shot (dominado)"),
        Patch(facecolor="#94A3B8", alpha=0.22, edgecolor="none", label="Región dominada"),
    ]
    ax0.legend(
        handles=legend_handles, loc="lower right", ncol=2,
        frameon=True, framealpha=0.95, fontsize=8.1,
        handletextpad=0.32, columnspacing=0.65, borderpad=0.35,
    )

    fig_p.savefig(OUT_DIR / "fig11a_frontera_pareto.png", bbox_inches="tight")
    plt.close(fig_p)

    # --- 2. Sensibilidad sobre Gold Humano por estrato previo A1/A2 (Anexo A) ---
    fig_e, ax1 = plt.subplots(figsize=(7.6, 4.4))
    strata_order = [
        "Human Gold\n(MultiCite)",
        "Panel\nUnánime 3/3",
        "Panel\nMayoría 2/3",
        "Needs Review\n(Desacuerdo)",
    ]
    st_counts = gold_df["stratum"].value_counts()
    xs = np.arange(len(strata_order))

    color_gem_soft = "#D96B6B"
    color_gpt_soft = "#6B9AC4"

    plot_specs = [
        ("gemini_zs_b", "Gem 0s-B", color_gem,      "-",  "o", 2.5, 6.8, False),
        ("gpt41_zs_b",  "GPT 0s-B", color_gpt,      "-",  "o", 2.5, 6.8, False),
        ("gemini_zs_a", "Gem 0s-A", color_gem,      ":",  "o", 1.9, 6.2, True),
        ("gpt41_zs_a",  "GPT 0s-A", color_gpt,      ":",  "o", 1.9, 6.2, True),
        ("gemini_fs_b", "Gem 9s-B", color_gem_soft, "--", "s", 1.7, 5.6, True),
        ("gpt41_fs_b",  "GPT 9s-B", color_gpt_soft, "--", "s", 1.7, 5.6, True),
    ]
    for vid, lbl, col, ls, mk, lw, ms, hollow in plot_specs:
        vals = [
            df_all[(df_all["vid"] == vid) & (df_all["stratum"] == st)]["is_correct"].mean() * 100.0
            for st in strata_order
        ]
        ax1.plot(
            xs, vals,
            marker=mk, linestyle=ls, linewidth=lw, markersize=ms,
            color=col,
            markerfacecolor="white" if hollow else col,
            markeredgecolor=col if hollow else "#1A202C",
            markeredgewidth=1.7 if hollow else 0.9,
            label=lbl,
            zorder=4 if not hollow else 3,
        )

    nr_gem = df_all[(df_all["vid"] == "gemini_zs_b") & (df_all["stratum"] == "Needs Review\n(Desacuerdo)")]["is_correct"].mean() * 100.0
    ax1.annotate(
        f"{fmt_es(nr_gem, 1)}%",
        xy=(3, nr_gem + 0.5), xytext=(2.48, nr_gem + 4.8),
        fontsize=10.8, fontweight="bold", color=color_gem,
        arrowprops=dict(arrowstyle="->", color=color_gem, lw=1.4),
    )

    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{st}\n(n={fmt_int_es(st_counts[st])})" for st in strata_order], fontsize=10.5)
    ax1.set_ylabel("Accuracy sobre Gold A3 (%)")
    ax1.set_ylim(37, 80)
    ax1.set_title("Exactitud sobre el Gold humano por estrato previo A1/A2", pad=10)
    ax1.grid(linestyle="--", alpha=0.32, zorder=0)
    ax1.legend(
        loc="lower left", ncol=3,
        frameon=True, framealpha=0.95, fontsize=9.8,
        columnspacing=0.9, handletextpad=0.4, borderpad=0.45,
    )

    fig_e.savefig(OUT_DIR / "fig11b_exactitud_por_estrato.png", bbox_inches="tight")
    plt.close(fig_e)


def main() -> None:
    setup_style()
    gold_df, runs, df_summary, df_all = load_data()
    plot_fig02(runs, df_summary)
    plot_fig03_split(gold_df, runs, df_all)
    plot_fig08(df_summary, df_all)
    plot_fig11_split(gold_df, df_summary, df_all)
    print("Figuras IEEE regeneradas exitosamente en:", OUT_DIR)


if __name__ == "__main__":
    main()
