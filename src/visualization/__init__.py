"""Paquete de visualización científica para distribuciones de datos (A1/A4) y análisis de errores/eficiencia (A6/A7)."""

from src.visualization.data_distributions import (
    plot_class_distribution,
    plot_token_length_distributions,
)
from src.visualization.error_analysis import (
    plot_a6_a7_efficiency_frontier,
    plot_a7_economic_frontier,
    plot_ab_confusion_matrices,
    plot_per_class_f1_comparison,
    plot_training_curves,
)

__all__ = [
    "plot_a6_a7_efficiency_frontier",
    "plot_a7_economic_frontier",
    "plot_ab_confusion_matrices",
    "plot_class_distribution",
    "plot_per_class_f1_comparison",
    "plot_token_length_distributions",
    "plot_training_curves",
]
