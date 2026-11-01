"""Visualización de distribuciones de clases canónicas y longitudes de secuencias en tokens (ADR-04)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from src.config.schema import (
    CANONICAL_9_CLASSES,
    InputRepresentationConfig,
    resolve_canonical_label,
)
from src.data.schemas import DatasetInstance
from src.modeling.dataset import format_instance_parts


def plot_class_distribution(
    splits_instances: Dict[str, Sequence[DatasetInstance]],
    output_path: Optional[Path] = None,
    title: str = "Distribución de las 9 Clases Canónicas por Partición",
) -> Figure:
    """Genera un gráfico de barras agrupadas con el conteo por clase canónica en cada split (train/dev/test)."""
    split_names = list(splits_instances.keys())
    num_classes = len(CANONICAL_9_CLASSES)
    x = np.arange(num_classes)
    width = 0.8 / max(1, len(split_names))

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    palette = ["#2b5c8f", "#d95f02", "#2ca02c", "#7570b3"]

    for s_idx, split_name in enumerate(split_names):
        instances = splits_instances[split_name]
        counts = {cls_name: 0 for cls_name in CANONICAL_9_CLASSES}
        for inst in instances:
            canonical = resolve_canonical_label(inst.label)
            if canonical in counts:
                counts[canonical] += 1
        values = [counts[c] for c in CANONICAL_9_CLASSES]
        offset = (s_idx - (len(split_names) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=f"{split_name} (n={len(instances)})",
            color=palette[s_idx % len(palette)],
            alpha=0.88,
        )
        for bar in bars:
            height = int(bar.get_height())
            if height > 0:
                ax.annotate(
                    f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2.0, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Número de Instancias", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CANONICAL_9_CLASSES, rotation=28, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_token_length_distributions(
    instances: Sequence[DatasetInstance],
    tokenizer: Any,
    max_seq_length: int = 512,
    enforce_per_chunk_budget: bool = True,
    output_path: Optional[Path] = None,
) -> Figure:
    """Compara la distribución de longitudes en tokens para Solo Contexto, Config A y Config B (ADR-04)."""
    modes: List[tuple[str, str, str]] = [
        ("context_only", "Solo Contexto (Ablación)", "#7f7f7f"),
        ("A", "Configuración A (Título + Abstract)", "#1f77b4"),
        ("B", "Configuración B (Top-3 Chunks)", "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=150)

    for mode_key, label_text, color in modes:
        rep_cfg = InputRepresentationConfig(
            configuration=mode_key,  # type: ignore[arg-type]
            max_seq_length=max_seq_length,
            enforce_per_chunk_budget=enforce_per_chunk_budget,
        )
        lengths: List[int] = []
        for inst in instances:
            text_citing, text_cited = format_instance_parts(
                inst,
                rep_cfg,
                tokenizer=tokenizer if enforce_per_chunk_budget else None,
            )
            citing_ids = tokenizer.encode(text_citing, add_special_tokens=False)
            if text_cited is not None:
                cited_ids = tokenizer.encode(text_cited, add_special_tokens=False)
                total_len = len(citing_ids) + len(cited_ids) + 3
            else:
                total_len = len(citing_ids) + 2
            lengths.append(total_len)

        arr = np.asarray(lengths, dtype=np.float64)
        trunc_pct = float(np.mean(arr > max_seq_length) * 100.0) if len(arr) > 0 else 0.0
        mean_len = float(np.mean(arr)) if len(arr) > 0 else 0.0
        ax.hist(
            arr,
            bins=min(30, max(5, len(arr) // 2)),
            alpha=0.45,
            color=color,
            edgecolor=color,
            label=f"{label_text} (media={mean_len:.0f}, >{max_seq_length}: {trunc_pct:.1f}%)",
        )

    ax.axvline(
        max_seq_length,
        color="#b2182b",
        linestyle="--",
        linewidth=2.0,
        label=f"Presupuesto máximo ({max_seq_length} tokens - ADR-04)",
    )
    ax.set_title("Distribución de Longitud en Tokens por Configuración de Entrada", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitud Total de Secuencia (Tokens WordPiece/BPE)", fontsize=10)
    ax.set_ylabel("Frecuencia (Instancias)", fontsize=10)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig

