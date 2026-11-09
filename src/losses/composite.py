"""Weighted composite loss for multiclass CamVid segmentation.

The component losses are built through the existing project loss factory so
that their ignore-index and implementation details stay consistent with the
individual-loss experiments.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from src.losses import build_loss


COMPOSITE_COMPONENTS = ("cross_entropy", "dice", "lovasz")


class CompositeSegmentationLoss(nn.Module):
    """Weighted sum of Cross-Entropy, Dice and Lovasz-Softmax losses.

    Args:
        num_classes: Number of semantic classes.
        ignore_index: Mask label excluded from all component losses.
        weights: Non-negative component weights. The weights must sum to 1.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        weights: Mapping[str, float],
    ) -> None:
        super().__init__()

        missing = set(COMPOSITE_COMPONENTS) - set(weights)
        extra = set(weights) - set(COMPOSITE_COMPONENTS)
        if missing or extra:
            raise ValueError(
                "Composite weights must contain exactly these keys: "
                f"{', '.join(COMPOSITE_COMPONENTS)}. "
                f"Missing={sorted(missing)}, extra={sorted(extra)}."
            )

        checked: dict[str, float] = {}
        for name in COMPOSITE_COMPONENTS:
            value = float(weights[name])
            if not torch.isfinite(torch.tensor(value)).item() or value < 0:
                raise ValueError(
                    f"Weight for {name!r} must be finite and non-negative."
                )
            checked[name] = value

        if abs(sum(checked.values()) - 1.0) > 1e-6:
            raise ValueError(
                "Composite loss weights must sum to 1.0; received "
                f"{sum(checked.values()):.8f}."
            )
        if not any(value > 0 for value in checked.values()):
            raise ValueError("At least one composite loss weight must be positive.")

        self.weights = checked
        self.losses = nn.ModuleDict(
            {
                name: build_loss(
                    loss_name=name,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )
                for name in COMPOSITE_COMPONENTS
            }
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = logits.new_zeros(())
        for name in COMPOSITE_COMPONENTS:
            weight = self.weights[name]
            if weight == 0.0:
                continue
            component_value = self.losses[name](logits, targets)
            total = total + weight * component_value
        return total
