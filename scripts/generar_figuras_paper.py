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
    """Fig. 5 del manuscrito: Heatmap F1 por clase + Barras globales."""
    support_dict = runs["gpt41_zs_a"]["a6"]["per_class_support"]
    cls_order = sorted(CANONICAL_9_CLASSES, key=lambda c: support_dict[c], reverse=True)
    mat_f1 = np.array(
        [[runs[v[0]]["a6"]["per_class_f1"][c] for c in cls_order] for v in VARIANTS],
        dtype=float,
    )

    fig = plt.figure(figsize=(13.2, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.58, 1.0], wspace=0.22)

    # Panel A: Heatmap F1 por clase
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(mat_f1, cmap="YlGnBu", aspect="auto", vmin=0.15, vmax=0.95)
    ax0.set_xticks(np.arange(len(cls_order)))
    ax0.set_xticklabels(
        [f"{SHORT_CLASS_LABELS[c]} ({fmt_int_es(support_dict[c])})" for c in cls_order],
        rotation=35, ha="right", rotation_mode="anchor", fontsize=11.0,
    )
    ax0.set_yticks(np.arange(len(VARIANTS)))
    ax0.set_yticklabels([v[1] for v in VARIANTS], fontsize=11.0)
    ax0.axhline(3.5, color="white", linewidth=2.4)
    ax0.set_title("(a) F1-Score por clase canónica (por soporte)", pad=10)

    for i in range(mat_f1.shape[0]):
        for j in range(mat_f1.shape[1]):
            val = mat_f1[i, j]
            ax0.text(
                j, i, fmt_es(val, 2),
                ha="center", va="center", fontsize=11.0,
                color="white" if val > 0.57 else "#111111",
                fontweight="bold",
            )
    cbar = fig.colorbar(im, ax=ax0, fraction=0.032, pad=0.025)
    cbar.set_label("F1-Score", fontsize=11.0)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_es(x, 1)))
    cbar.ax.tick_params(labelsize=11.0)

    # Panel B: Barras horizontales globales
    ax1 = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(len(VARIANTS))
    h = 0.27
    ax1.barh(y_pos - h, df_summary["Macro-F1"],    height=h, color="#2B5C8F", label="Macro-F1")
    ax1.barh(y_pos,     df_summary["Weighted-F1"], height=h, color="#3B9C6C", label="Weighted-F1")
    ax1.barh(y_pos + h, df_summary["Accuracy"],    height=h, color="#D97706", label="Accuracy")
    ax1.axhline(3.5, color="#CCCCCC", linestyle="--", linewidth=1.0)

    for idx, row in df_summary.iterrows():
        fw = "bold" if row["vid"] == "gemini_zs_b" else "normal"
        ax1.text(row["Macro-F1"] + 0.007, idx - h, fmt_es(row["Macro-F1"], 3), va="center", fontsize=11.0, fontweight=fw, color="#1A365D")
        ax1.text(row["Weighted-F1"] + 0.007, idx + 0.04, fmt_es(row["Weighted-F1"], 3), va="center", fontsize=11.0, fontweight=fw, color="#1E4D36")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([])
    ax1.invert_yaxis()
    ax1.set_xlim(0.35, 0.78)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_es(x, 2)))
    ax1.set_xlabel("Puntaje")
    ax1.set_title("(b) Desempeño global (N = 2.798)", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=3, frameon=True, framealpha=0.95, fontsize=11.0, columnspacing=0.9, handletextpad=0.4)

    fig.savefig(OUT_DIR / "fig02_desempeno_global_y_por_clase.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig03(gold_df, runs, df_all) -> None:
    """Fig. 6 del manuscrito: Matrices de confusión 0s-B (% fila) + Exactitud por macro-sección."""
    fig = plt.figure(figsize=(15.6, 5.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.18, 1.32], wspace=0.48)

    short_9 = ["Bkg", "Gap", "Bas", "Cmp", "App", "Imp", "Evi", "Org", "F.R."]

    for col_idx, (vid, title_txt) in enumerate([
        ("gpt41_zs_b",  "(a) GPT-4.1-mini 0s-B (% fila)"),
        ("gemini_zs_b", "(b) Gemini-3.8-Flash 0s-B (% fila)"),
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        cm = np.array(runs[vid]["a6"]["confusion_matrix"], dtype=float)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=0.95)
        ax.set_title(title_txt, pad=10)
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels(short_9, rotation=38, ha="right", rotation_mode="anchor", fontsize=11.0)
        ax.set_yticklabels(short_9, fontsize=11.0)
        ax.set_xlabel("Predicción")
        if col_idx == 0:
            ax.set_ylabel("Etiqueta real")
        for i in range(9):
            for j in range(9):
                val = cm_norm[i, j]
                if val >= 0.08 or i == j:
                    txt = f"{int(round(val * 100))}"
                    ax.text(
                        j, i, txt,
                        ha="center", va="center", fontsize=11.0,
                        fontweight="bold" if i == j else "normal",
                        color="white" if val > 0.48 else "#111111",
                    )
        if col_idx == 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.044, pad=0.035)
            cbar.set_label("Proporción", fontsize=11.0)
            cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_es(x, 1)))
            cbar.ax.tick_params(labelsize=11.0)

    # Panel C: Exactitud por macro-sección citante (0-Shot)
    ax2 = fig.add_subplot(gs[0, 2])
    sec_counts = gold_df["macro_section"].value_counts()
    sec_order = ["Related Work", "Intro / Prelim", "Experiments", "Method / Setup", "Otras", "Conclusion", "Sin sección"]
    sec_order = [s for s in sec_order if s in sec_counts.index]
    x_s = np.arange(len(sec_order))

    focus_styles = [
        ("gemini_zs_b", "Gem 0s-B", "-",  "D", 2.4, 6.2),
        ("gemini_zs_a", "Gem 0s-A", "--", "o", 2.0, 5.5),
        ("gpt41_zs_b",  "GPT 0s-B", "-",  "s", 2.4, 5.8),
        ("gpt41_zs_a",  "GPT 0s-A", "--", "^", 2.0, 5.5),
    ]
    for vid, lbl, ls, mk, lw, ms in focus_styles:
        vals = [
            df_all[(df_all["vid"] == vid) & (df_all["macro_section"] == s)]["is_correct"].mean() * 100.0
            for s in sec_order
        ]
        ax2.plot(x_s, vals, marker=mk, linestyle=ls, linewidth=lw, markersize=ms, color=COLORS[vid], label=lbl)

    rw_b = df_all[(df_all["vid"] == "gpt41_zs_b") & (df_all["macro_section"] == "Related Work")]["is_correct"].mean() * 100.0
    ax2.annotate(
        "+10,3 pp",
        xy=(0, rw_b), xytext=(0.40, 61.5),
        fontsize=11.0, fontweight="bold", color="#0D9488",
        arrowprops=dict(arrowstyle="->", color="#0D9488", lw=1.5),
    )

    ax2.set_xticks(x_s)
    ax2.set_xticklabels(
        [f"{s} ({fmt_int_es(sec_counts[s])})" for s in sec_order],
        rotation=32, ha="right", rotation_mode="anchor", fontsize=11.0,
    )
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(26, 84)
    ax2.set_title("(c) Exactitud por macro-sección (0-Shot)", pad=10)
    ax2.grid(linestyle="--", alpha=0.35)
    ax2.legend(loc="lower center", ncol=2, frameon=True, framealpha=0.95, fontsize=11.0)

    fig.savefig(OUT_DIR / "fig03_confusiones_criticas_y_seccion_citante.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig08(df_summary, df_all) -> None:
    """Fig. 7 del manuscrito: Tokens + Costo Bruto/Neto + Latencia Hit vs Miss."""
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))
    x_v = np.arange(len(VARIANTS))
    x_lbls = [f"{'GPT' if 'gpt' in v[0] else 'Gem'}\n{v[5]}s-{v[6]}" for v in VARIANTS]

    # Panel A: Millones de tokens
    ax0 = axes[0]
    uncached_m = [df_all[df_all["vid"] == v[0]]["uncached_input_tokens"].sum() / 1e6 for v in VARIANTS]
    cached_m   = [df_all[df_all["vid"] == v[0]]["cached_input_tokens"].sum() / 1e6 for v in VARIANTS]
    out_th_m   = [(df_all[df_all["vid"] == v[0]]["output_tokens"].sum() + df_all[df_all["vid"] == v[0]]["thoughts_tokens"].sum()) / 1e6 for v in VARIANTS]
    pct_cache  = [df_all[df_all["vid"] == v[0]]["cached_input_tokens"].sum() / df_all[df_all["vid"] == v[0]]["input_tokens"].sum() * 100.0 for v in VARIANTS]

    ax0.bar(x_v, uncached_m, color="#D97706", label="Entrada sin caché")
    ax0.bar(x_v, cached_m, bottom=uncached_m, color="#0D9488", label="Entrada cacheada")
    ax0.bar(x_v, out_th_m, bottom=np.array(uncached_m) + np.array(cached_m), color="#6B46C1", label="Salida + pensam.")
    for i, pct in enumerate(pct_cache):
        tot = uncached_m[i] + cached_m[i] + out_th_m[i]
        if pct >= 4.0:
            ax0.text(i, tot + 0.55, f"{fmt_es(pct, 0)}%", ha="center", va="bottom", fontsize=11.0, fontweight="bold", color="#0F766E")
    ax0.set_xticks(x_v)
    ax0.set_xticklabels(x_lbls, fontsize=11.0)
    ax0.set_ylabel("Millones de tokens")
    ax0.set_title("(a) Tokens y tasa de caché", pad=10)
    ax0.set_ylim(0, 31.0)
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=11.0)

    # Panel B: Costo Bruto vs Neto USD
    ax1 = axes[1]
    w = 0.38
    ax1.bar(x_v - w / 2, df_summary["Costo Bruto ($)"], width=w, color="#BFDBFE", edgecolor="#1D4ED8", linewidth=0.9, label="Bruto (sin caché)")
    ax1.bar(x_v + w / 2, df_summary["Costo Neto ($)"],  width=w, color="#1D4ED8", label="Neto (con caché)")
    for idx, r in df_summary.iterrows():
        gross_v = r["Costo Bruto ($)"]
        net_v = r["Costo Neto ($)"]
        if gross_v - net_v >= 1.5:
            # Barras con ahorro grande: colocar neto sobre barra azul oscura y % ahorro sobre la barra bruta
            sav_pct = (1.0 - net_v / gross_v) * 100.0
            ax1.text(
                idx + w / 2 + 0.10, net_v + 0.35, f"${fmt_es(net_v, 1)}",
                ha="center", va="bottom", fontsize=10.0, fontweight="bold", color="#1E3A8A",
                bbox=dict(facecolor="white", alpha=0.90, edgecolor="none", pad=0.6),
            )
            ax1.text(idx - w / 2, gross_v + 0.35, f"-{int(round(sav_pct))}%", ha="center", va="bottom", fontsize=10.0, fontweight="bold", color="#0F766E")
        else:
            ax1.text(idx, max(gross_v, net_v) + 0.42, f"${fmt_es(net_v, 1)}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1E3A8A")
    ax1.set_xticks(x_v)
    ax1.set_xticklabels(x_lbls, fontsize=11.0)
    ax1.set_ylabel("Costo corrida (USD, N = 2.798)")
    ax1.set_title("(b) Costo bruto vs. neto (etiqueta = neto)", pad=10)
    ax1.set_ylim(0, 21.5)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: fmt_es(y, 1)))
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=11.0)

    # Panel C: Latencia p50 Cache Miss vs Cache Hit (9-Shot)
    ax2 = axes[2]
    fs_vids = ["gpt41_fs_a", "gpt41_fs_b", "gemini_fs_a", "gemini_fs_b"]
    pos = np.arange(len(fs_vids))
    hit_meds, miss_meds = [], []
    for v in fs_vids:
        sub = df_all[df_all["vid"] == v]
        hit_meds.append(sub[sub["cached_input_tokens"] > 0]["latency_ms"].median() / 1000.0)
        miss_meds.append(sub[sub["cached_input_tokens"] == 0]["latency_ms"].median() / 1000.0)
    ax2.bar(pos - w / 2, miss_meds, width=w, color="#DC2626", alpha=0.85, label="Cache Miss (p50)")
    ax2.bar(pos + w / 2, hit_meds,  width=w, color="#16A34A", alpha=0.88, label="Cache Hit (p50)")
    for i in pos:
        ax2.text(i - w / 2 - 0.03, miss_meds[i] + 0.07, f"{fmt_es(miss_meds[i], 2)} s", ha="center", va="bottom", fontsize=10.5)
        ax2.text(i + w / 2 + 0.05, hit_meds[i] + 0.07,  f"{fmt_es(hit_meds[i], 2)} s",  ha="center", va="bottom", fontsize=10.5)
    ax2.set_xticks(pos)
    ax2.set_xticklabels(["GPT\n9s-A", "GPT\n9s-B", "Gem\n9s-A", "Gem\n9s-B"], fontsize=11.0)
    ax2.set_ylabel("Latencia mediana p50 (s)")
    ax2.set_title("(c) Latencia con caché (9-Shot)", pad=10)
    ax2.set_ylim(0, 3.35)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_es(x, 1)))
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=11.0)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_DIR / "fig08_tokens_cache_eficiencia_y_costos.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig11(gold_df, df_summary, df_all) -> None:
    """Fig. 8 del manuscrito: Frontera de Pareto + Sensibilidad por estrato A1/A2."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9))

    # Panel A: Frontera de Pareto (Alternativa A: círculos uniformes + sombreado región dominada)
    ax0 = axes[0]
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    label_specs = {
        "gpt41_zs_a":  ((-8, -32),  "right"),
        "gpt41_zs_b":  ((-8, 28),   "right"),
        "gpt41_fs_a":  ((-6, -34),  "center"),
        "gpt41_fs_b":  ((12, 12),   "left"),
        "gemini_zs_a": ((-16, -8),  "right"),
        "gemini_zs_b": ((-8, 18),   "right"),
        "gemini_fs_a": ((0, -36),   "center"),
        "gemini_fs_b": ((0, 18),    "center"),
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

    # Sombreado suave de la región estrictamente dominada (bajo la frontera y a la derecha del óptimo)
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

    color_gpt = "#1F6AA5"
    color_gem = "#C53030"

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
                s=255, marker="o", facecolors="none",
                edgecolors=fam_color, linewidth=1.35, alpha=0.65, zorder=4,
            )

        ax0.scatter(
            x_val, y_val,
            s=115, marker="o",
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
            fontsize=10.6,
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
    ax0.set_title("(a) Frontera de Pareto costo vs. Macro-F1", pad=10)
    ax0.grid(True, which="major", linestyle="--", alpha=0.32, zorder=0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_gpt, markeredgecolor="#1A202C", markersize=7.5, label="GPT-4.1-mini"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_gem, markeredgecolor="#1A202C", markersize=7.5, label="Gemini-3.8-Flash"),
        Line2D([0], [0], color="#334155", linestyle="--", linewidth=1.5, label="Frontera de Pareto"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#334155", markeredgecolor="#1A202C", markersize=7.5, label="0-Shot (Pareto)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#334155", markeredgewidth=1.8, markersize=7.5, label="9-Shot (dominado)"),
        Patch(facecolor="#94A3B8", alpha=0.22, edgecolor="none", label="Región dominada"),
    ]
    ax0.legend(
        handles=legend_handles, loc="lower right", ncol=2,
        frameon=True, framealpha=0.95, fontsize=9.0,
        handletextpad=0.4, columnspacing=0.9, borderpad=0.45,
    )

    # Panel B: Sensibilidad sobre Gold Humano por estrato previo A1/A2
    ax1 = axes[1]
    strata_order = [
        "Human Gold\n(MultiCite)",
        "Panel\nUnánime 3/3",
        "Panel\nMayoría 2/3",
        "Needs Review\n(Desacuerdo)",
    ]
    st_counts = gold_df["stratum"].value_counts()
    xs = np.arange(len(strata_order))

    plot_specs = [
        ("gemini_zs_b", "Gem 0s-B", "-",  "D", 2.5, 6.5),
        ("gemini_zs_a", "Gem 0s-A", ":",  "o", 1.9, 5.0),
        ("gemini_fs_b", "Gem 9s-B", "--", "s", 1.9, 5.0),
        ("gpt41_zs_b",  "GPT 0s-B", "-",  "s", 2.3, 6.0),
        ("gpt41_zs_a",  "GPT 0s-A", ":",  "^", 1.9, 5.0),
        ("gpt41_fs_b",  "GPT 9s-B", "--", "v", 1.9, 5.0),
    ]
    for vid, lbl, ls, mk, lw, ms in plot_specs:
        vals = [
            df_all[(df_all["vid"] == vid) & (df_all["stratum"] == st)]["is_correct"].mean() * 100.0
            for st in strata_order
        ]
        ax1.plot(xs, vals, marker=mk, linestyle=ls, linewidth=lw, markersize=ms, color=COLORS[vid], label=lbl)

    nr_gem = df_all[(df_all["vid"] == "gemini_zs_b") & (df_all["stratum"] == "Needs Review\n(Desacuerdo)")]["is_correct"].mean() * 100.0
    ax1.annotate(
        f"{fmt_es(nr_gem, 1)}%",
        xy=(3, nr_gem), xytext=(2.48, nr_gem + 4.5),
        fontsize=11.0, fontweight="bold", color="#C53030",
        arrowprops=dict(arrowstyle="->", color="#C53030", lw=1.4),
    )

    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{st}\n(n={fmt_int_es(st_counts[st])})" for st in strata_order], fontsize=11.0)
    ax1.set_ylabel("Accuracy sobre Gold A3 (%)")
    ax1.set_ylim(38, 80)
    ax1.set_title("(b) Exactitud por estrato previo A1/A2", pad=10)
    ax1.grid(linestyle="--", alpha=0.35)
    ax1.legend(loc="lower left", ncol=3, frameon=True, framealpha=0.95, fontsize=11.0)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_DIR / "fig11_frontera_pareto_y_sensibilidad_panel.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    gold_df, runs, df_summary, df_all = load_data()
    plot_fig02(runs, df_summary)
    plot_fig03(gold_df, runs, df_all)
    plot_fig08(df_summary, df_all)
    plot_fig11(gold_df, df_summary, df_all)
    print("4 figuras IEEE regeneradas exitosamente en:", OUT_DIR)


if __name__ == "__main__":
    main()
