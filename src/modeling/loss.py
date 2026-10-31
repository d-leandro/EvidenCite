"""Funciones de pérdida y esquemas de ponderación de clases (Cross-Entropy, Weighted CE y Focal Loss)."""

from collections import Counter
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.config.schema import CANONICAL_9_CLASSES, LABEL_TO_ID, resolve_canonical_label


def compute_class_weights(
    labels: Sequence[Union[int, str]],
    scheme: Literal["none", "balanced", "sqrt_balanced"] = "balanced",
    num_classes: int = len(CANONICAL_9_CLASSES),
) -> Optional[torch.Tensor]:
    """Calcula el vector de pesos por clase sobre la partición de entrenamiento.

    Args:
        labels: Secuencia de etiquetas enteras (0..8) o nombres de clases canónicas.
        scheme:
            - 'none': Retorna None (sin ponderación).
            - 'balanced': Frecuencia inversa estándar N / (K_activas * n_c).
            - 'sqrt_balanced': Raíz cuadrada suavizada sqrt(N / n_c) normalizada a media 1.0
              sobre las clases presentes para prevenir gradientes extremos en clases minoritarias.
        num_classes: Número total de clases canónicas (9).
    """
    if scheme == "none":
        return None

    int_labels: list[int] = []
    for lbl in labels:
        if isinstance(lbl, (int, np.integer)):
            int_labels.append(int(lbl))
        else:
            resolved = resolve_canonical_label(str(lbl))
            if resolved is not None and resolved in LABEL_TO_ID:
                int_labels.append(LABEL_TO_ID[resolved])

    if not int_labels:
        return torch.ones(num_classes, dtype=torch.float32)

    counts = Counter(int_labels)
    total = float(len(int_labels))
    present_classes = [c for c in range(num_classes) if counts.get(c, 0) > 0]
    k_present = float(len(present_classes)) if present_classes else float(num_classes)

    weights = np.ones(num_classes, dtype=np.float32)
    if scheme == "balanced":
        for c in range(num_classes):
            n_c = counts.get(c, 0)
            if n_c > 0:
                weights[c] = total / (k_present * float(n_c))
            else:
                weights[c] = 0.0
    elif scheme == "sqrt_balanced":
        raw = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            n_c = counts.get(c, 0)
            if n_c > 0:
                raw[c] = float(np.sqrt(total / float(n_c)))
        mean_active = float(np.mean([raw[c] for c in present_classes])) if present_classes else 1.0
        if mean_active > 0:
            weights = raw / mean_active
    else:
        raise ValueError(f"Esquema de pesos no soportado: {scheme}")

    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Implementación diferenciable de Multi-Class Focal Loss con soporte para pesos por clase."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight.clone().detach().float())
        else:
            self.weight = None
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        targets_flat = targets.view(-1)
        log_pt = log_probs.gather(dim=-1, index=targets_flat.unsqueeze(1)).squeeze(1)
        pt = probs.gather(dim=-1, index=targets_flat.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - pt).clamp(min=0.0) ** self.gamma
        loss = -focal_term * log_pt

        if self.weight is not None:
            alpha_t = self.weight.to(logits.device).gather(dim=0, index=targets_flat)
            loss = alpha_t * loss

        if self.reduction == "mean":
            if self.weight is not None:
                denom = alpha_t.sum().clamp(min=1e-8)
                return loss.sum() / denom
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss_fn(
    loss_type: Literal["cross_entropy", "weighted_ce", "focal"] = "weighted_ce",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """Construye el módulo de pérdida PyTorch configurado para WeightedSequenceTrainer."""
    weights_on_device = (
        class_weights.to(device) if (class_weights is not None and device is not None) else class_weights
    )
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights_on_device)
    if loss_type == "focal":
        return FocalLoss(weight=weights_on_device, gamma=focal_gamma)
    raise ValueError(f"Tipo de función de pérdida no soportado: {loss_type}")

