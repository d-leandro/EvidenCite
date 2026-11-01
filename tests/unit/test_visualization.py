"""Pruebas unitarias para el paquete de visualización científica (src.visualization)."""

from pathlib import Path

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.config.schema import CANONICAL_9_CLASSES
from src.modeling.evaluate import OperationalMetricsA7, compute_a6_metrics
from src.visualization import (
    plot_a6_a7_efficiency_frontier,
    plot_ab_confusion_matrices,
    plot_class_distribution,
    plot_per_class_f1_comparison,
    plot_token_length_distributions,
    plot_training_curves,
)
from tests.unit.test_modeling_dataset_and_loss import (
    DummyWordTokenizer,
    make_synthetic_instance,
)


def test_all_visualization_functions_generate_figures_and_pngs(tmp_path: Path) -> None:
    """Verifica que todas las funciones de src/visualization/ generen Figure y persistan PNGs."""
    instances = [
        make_synthetic_instance(idx=i + 1, label=CANONICAL_9_CLASSES[i % 9])
        for i in range(18)
    ]
    splits = {"train": instances[:10], "dev": instances[10:14], "test": instances[14:]}

    # 1. Distribución de clases
    p1 = tmp_path / "class_dist.png"
    fig1 = plot_class_distribution(splits, output_path=p1)
    assert isinstance(fig1, Figure) and p1.exists() and p1.stat().st_size > 0
    plt.close(fig1)

    # 2. Distribución de longitud de tokens (ADR-04)
    p2 = tmp_path / "token_lengths.png"
    fig2 = plot_token_length_distributions(
        instances,
        tokenizer=DummyWordTokenizer(),
        max_seq_length=64,
        output_path=p2,
    )
    assert isinstance(fig2, Figure) and p2.exists() and p2.stat().st_size > 0
    plt.close(fig2)

    # 3. Matrices de confusión comparativas A vs B y F1 por clase
    y_true = [inst.label for inst in instances]
    y_pred_a = [CANONICAL_9_CLASSES[(i + 1) % 9] for i in range(18)]
    y_pred_b = [inst.label for inst in instances]
    eval_a = compute_a6_metrics(y_true, y_pred_a)  # type: ignore[arg-type]
    eval_b = compute_a6_metrics(y_true, y_pred_b)  # type: ignore[arg-type]

    p3 = tmp_path / "ab_cm.png"
    fig3 = plot_ab_confusion_matrices(eval_a, eval_b, output_path=p3)
    assert isinstance(fig3, Figure) and p3.exists() and p3.stat().st_size > 0
    plt.close(fig3)

    p4 = tmp_path / "per_class_f1.png"
    fig4 = plot_per_class_f1_comparison({"Config A": eval_a, "Config B": eval_b}, output_path=p4)
    assert isinstance(fig4, Figure) and p4.exists() and p4.stat().st_size > 0
    plt.close(fig4)

    # 4. Frontera de eficiencia operativa A6 vs A7
    op_a = OperationalMetricsA7(
        model_family="tfidf",
        configuration="A",
        total_params=45000,
        trainable_params=45000,
        trainable_ratio_pct=100.0,
        inference_latency_ms_per_instance=0.35,
        throughput_instances_per_sec=2857.0,
    )
    op_b = OperationalMetricsA7(
        model_family="encoder",
        configuration="B",
        total_params=109_000_000,
        trainable_params=109_000_000,
        trainable_ratio_pct=100.0,
        inference_latency_ms_per_instance=8.2,
        throughput_instances_per_sec=121.9,
    )
    p5 = tmp_path / "frontier.png"
    fig5 = plot_a6_a7_efficiency_frontier(
        [("TF-IDF (A)", eval_a, op_a), ("SciBERT (B)", eval_b, op_b)],
        output_path=p5,
    )
    assert isinstance(fig5, Figure) and p5.exists() and p5.stat().st_size > 0
    plt.close(fig5)

    # 5. Curvas de entrenamiento
    history = [
        {"epoch": 1.0, "loss": 1.95},
        {"epoch": 1.0, "eval_loss": 1.62, "eval_f1_macro": 0.42},
        {"epoch": 2.0, "loss": 1.31},
        {"epoch": 2.0, "eval_loss": 1.18, "eval_f1_macro": 0.64},
    ]
    p6 = tmp_path / "curves.png"
    fig6 = plot_training_curves(history, output_path=p6)
    assert isinstance(fig6, Figure) and p6.exists() and p6.stat().st_size > 0
    plt.close(fig6)

