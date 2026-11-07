"""Figuras de análisis de error, matrices de confusión comparativas A vs B (A6) y frontera operativa (A7)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from src.config.schema import CANONICAL_9_CLASSES
from src.modeling.evaluate import OperationalMetricsA7, TechnicalMetricsA6


def _normalize_cm_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    return np.where(row_sums > 0.0, cm.astype(np.float64) / np.maximum(row_sums, 1.0), 0.0)


def plot_ab_confusion_matrices(
    eval_a: TechnicalMetricsA6,
    eval_b: TechnicalMetricsA6,
    class_names: Optional[Sequence[str]] = None,
    output_path: Optional[Path] = None,
) -> Figure:
    """Grafica un panel triple: Matriz Normalizada A, Matriz Normalizada B y Diferencial (B - A)."""
    labels = list(class_names) if class_names is not None else list(CANONICAL_9_CLASSES)
    cm_a = _normalize_cm_rows(np.asarray(eval_a.confusion_matrix, dtype=np.float64))
    cm_b = _normalize_cm_rows(np.asarray(eval_b.confusion_matrix, dtype=np.float64))
    cm_diff = cm_b - cm_a

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=150)
    panels = [
        (axes[0], cm_a, f"Config A (Macro-F1={eval_a.macro_f1_all_9:.3f})", "Blues", 0.0, 1.0),
        (axes[1], cm_b, f"Config B (Macro-F1={eval_b.macro_f1_all_9:.3f})", "Blues", 0.0, 1.0),
        (axes[2], cm_diff, "Diferencia Normalizada (Config B - Config A)", "RdYlGn", -0.5, 0.5),
    ]

    n_cls = len(labels)
    for ax, matrix, panel_title, cmap_name, vmin, vmax in panels:
        im = ax.imshow(matrix, interpolation="nearest", cmap=cmap_name, vmin=vmin, vmax=vmax)
        ax.set_title(panel_title, fontsize=10.5, fontweight="bold")
        ax.set_xticks(np.arange(n_cls))
        ax.set_yticks(np.arange(n_cls))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicción", fontsize=9)
        ax.set_ylabel("Etiqueta Real", fontsize=9)

        for i in range(n_cls):
            for j in range(n_cls):
                val = matrix[i, j]
                if abs(val) >= 0.01 or i == j:
                    text_str = f"{val:+.2f}" if vmin < 0 else f"{val:.2f}"
                    ax.text(
                        j,
                        i,
                        text_str,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if abs(val) < 0.65 else "white",
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_per_class_f1_comparison(
    results_by_config: Dict[str, TechnicalMetricsA6],
    output_path: Optional[Path] = None,
    title: str = "Comparación de F1-Score por Clase Canónica (Actividad A6)",
) -> Figure:
    """Genera un gráfico de barras horizontales comparando el F1 por clase entre configuraciones."""
    config_names = list(results_by_config.keys())
    num_classes = len(CANONICAL_9_CLASSES)
    y = np.arange(num_classes)
    height = 0.8 / max(1, len(config_names))

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=150)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for c_idx, cfg_name in enumerate(config_names):
        metrics = results_by_config[cfg_name]
        f1_vals = [metrics.per_class_f1.get(cls_name, 0.0) for cls_name in CANONICAL_9_CLASSES]
        offset = (c_idx - (len(config_names) - 1) / 2.0) * height
        ax.barh(
            y + offset,
            f1_vals,
            height=height,
            label=f"{cfg_name} (Macro-F1={metrics.macro_f1_all_9:.3f})",
            color=palette[c_idx % len(palette)],
            alpha=0.88,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(CANONICAL_9_CLASSES, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("F1-Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_a6_a7_efficiency_frontier(
    records: Sequence[Tuple[str, TechnicalMetricsA6, OperationalMetricsA7]],
    output_path: Optional[Path] = None,
    title: str = "Frontera de Eficiencia Técnica vs. Operativa (Actividades A6 y A7)",
) -> Figure:
    """Grafica el compromiso entre Latencia de Inferencia (ms/instancia) y Macro-F1."""
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=150)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    for idx, (label, a6, a7) in enumerate(records):
        lat = max(1e-3, a7.inference_latency_ms_per_instance)
        f1 = a6.macro_f1_all_9

        effective_params = max(a7.total_params, a7.trainable_params)
        is_api = a7.model_family == "llm_api" or effective_params == 0

        if is_api:
            bubble_size = 280.0
            marker_shape = "D"
            param_desc = "API Cloud"
        else:
            bubble_size = max(120.0, min(950.0, 100.0 + np.log10(max(1, effective_params)) * 85.0))
            marker_shape = "o"
            param_desc = (
                f"{effective_params / 1e9:.1f}B"
                if effective_params >= 1e9
                else f"{effective_params / 1e6:.0f}M"
            )

        color = palette[idx % len(palette)]
        ax.scatter(
            [lat],
            [f1],
            s=bubble_size,
            marker=marker_shape,
            color=color,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.2,
            label=f"{label} ({lat:.1f} ms, F1={f1:.3f}, {param_desc})",
        )
        ax.annotate(
            label,
            xy=(lat, f1),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Latencia por Inferencia (ms / instancia, escala log)", fontsize=10)
    ax.set_ylabel("Macro-F1 (9 Clases Canónicas)", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=8.5, loc="best")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_curves(
    training_history: Sequence[Dict[str, Any]],
    output_path: Optional[Path] = None,
    title: str = "Curvas de Entrenamiento y Validación por Época",
) -> Figure:
    """Grafica la evolución de la pérdida y de eval_f1_macro a partir de trainer.state.log_history."""
    train_epochs: List[float] = []
    train_losses: List[float] = []
    eval_epochs: List[float] = []
    eval_losses: List[float] = []
    eval_f1s: List[float] = []

    for entry in training_history:
        ep = entry.get("epoch")
        if ep is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            train_epochs.append(float(ep))
            train_losses.append(float(entry["loss"]))
        if "eval_loss" in entry:
            eval_epochs.append(float(ep))
            eval_losses.append(float(entry["eval_loss"]))
        if "eval_f1_macro" in entry:
            eval_f1s.append(float(entry["eval_f1_macro"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=150)

    if train_epochs:
        axes[0].plot(train_epochs, train_losses, marker="o", label="Train Loss", color="#1f77b4")
    if eval_epochs and eval_losses:
        axes[0].plot(eval_epochs, eval_losses, marker="s", label="Val Loss", color="#d62728")
    axes[0].set_title("Función de Pérdida", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].grid(linestyle="--", alpha=0.35)
    if train_epochs or eval_epochs:
        axes[0].legend()

    if eval_epochs and eval_f1s:
        axes[1].plot(
            eval_epochs[: len(eval_f1s)],
            eval_f1s,
            marker="^",
            color="#2ca02c",
            label="Val Macro-F1 (9 clases)",
        )
        axes[1].legend()
    axes[1].set_title("Evolución de Macro-F1 en Validación", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(linestyle="--", alpha=0.35)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_a7_economic_frontier(
    records: Sequence[Tuple[str, TechnicalMetricsA6, OperationalMetricsA7]],
    target_citations: int = 10_000,
    test_set_size: int = 2798,
    output_path: Optional[Path] = None,
    title: str = "Frontera Económica A7: Costo Financiero (USD por 10.000 Citas) vs. Macro-F1",
) -> Figure:
    """Grafica el compromiso entre el Costo Financiero en Producción (USD) y la Calidad Predictiva (Macro-F1).

    Permite contrastar la viabilidad de despliegue entre modelos locales fine-tuned (costo marginal ~0)
    y modelos comerciales vía API de pago por token.
    """
    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=150)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    for idx, (label, a6, a7) in enumerate(records):
        f1 = a6.macro_f1_all_9
        is_api = a7.model_family == "llm_api" or a7.total_params == 0

        # Costo proyectado en USD para target_citations
        if is_api and a7.estimated_api_cost_usd > 0.0:
            cost_usd = (a7.estimated_api_cost_usd / max(1, test_set_size)) * target_citations
            marker_shape = "D"
            bubble_size = 280.0
            type_tag = "API Cloud"
        else:
            # Costo marginal en GPU propia / CPU amortizada
            cost_usd = 0.05 if a7.model_family == "llm_lora" else 0.01
            marker_shape = "o"
            effective_params = max(a7.total_params, a7.trainable_params)
            bubble_size = max(120.0, min(950.0, 100.0 + np.log10(max(1, effective_params)) * 85.0))
            param_str = (
                f"{effective_params / 1e9:.1f}B"
                if effective_params >= 1e9
                else f"{effective_params / 1e6:.0f}M"
            )
            type_tag = f"Local ({param_str})"

        color = palette[idx % len(palette)]
        cost_plot = max(1e-2, cost_usd)

        ax.scatter(
            [cost_plot],
            [f1],
            s=bubble_size,
            marker=marker_shape,
            color=color,
            alpha=0.85,
            edgecolors="black",
            linewidth=1.2,
            label=f"{label} (${cost_usd:.2f}/10k, F1={f1:.3f}, {type_tag})",
        )
        ax.annotate(
            f"{label}\n(${cost_usd:.2f})",
            xy=(cost_plot, f1),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        f"Costo Financiero Estimado (USD por {target_citations:,} citas en producción, escala log)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_ylabel("Macro-F1 (9 Clases Canónicas)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
