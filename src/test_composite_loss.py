"""Smoke tests for the weighted composite segmentation loss."""

import torch

from src.losses.composite import (
    COMPOSITE_COMPONENTS,
    CompositeSegmentationLoss,
)


def main() -> None:
    torch.manual_seed(7)
    weights = {
        "cross_entropy": 0.34,
        "dice": 0.33,
        "lovasz": 0.33,
    }
    criterion = CompositeSegmentationLoss(
        num_classes=11,
        ignore_index=255,
        weights=weights,
    )

    logits = torch.randn(2, 11, 24, 32, requires_grad=True)
    targets = torch.randint(0, 11, (2, 24, 32), dtype=torch.long)
    targets[:, :2, :3] = 255

    loss = criterion(logits, targets)
    assert loss.ndim == 0, f"Expected scalar loss; got {tuple(loss.shape)}"
    assert torch.isfinite(loss).item(), "Composite loss is not finite."
    loss.backward()
    assert logits.grad is not None, "No gradients were produced."
    assert torch.isfinite(logits.grad).all().item(), "Gradients are not finite."

    print("Composite components:", ", ".join(COMPOSITE_COMPONENTS))
    print("Weights:", weights)
    print(f"Composite loss: {loss.detach().item():.6f}")
    print("Gradient check: PASS")

    try:
        CompositeSegmentationLoss(
            num_classes=11,
            ignore_index=255,
            weights={
                "cross_entropy": 0.5,
                "dice": 0.5,
                "lovasz": 0.5,
            },
        )
    except ValueError:
        print("Invalid weight-sum check: PASS")
    else:
        raise AssertionError("Invalid weights should have been rejected.")


if __name__ == "__main__":
    main()
