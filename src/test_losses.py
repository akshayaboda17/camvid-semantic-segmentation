
"""
Unit tests for all five CamVid segmentation losses.

Run:
    python -m src.test_losses
"""

import torch

from src.losses import (
    LOSS_NAMES,
    build_loss,
)


def main() -> None:

    torch.manual_seed(42)

    num_classes = 11
    batch_size = 2
    height = 32
    width = 40

    print("=" * 65)
    print("TESTING CAMVID SEGMENTATION LOSS FUNCTIONS")
    print("=" * 65)

    # Simulated model output.
    base_logits = torch.randn(
        batch_size,
        num_classes,
        height,
        width,
    )

    # Simulated ground-truth segmentation masks.
    targets = torch.randint(
        0,
        num_classes,
        (batch_size, height, width),
    )

    # Introduce ignored pixels.
    targets[:, 0:3, 0:5] = 255
    targets[:, 10:12, 10:15] = 255

    # -----------------------------------------------------
    # 1. TEST EACH LOSS AND ITS GRADIENTS
    # -----------------------------------------------------

    for loss_name in LOSS_NAMES:

        print(f"\nTesting {loss_name} loss...")

        logits = base_logits.clone().requires_grad_(True)

        criterion = build_loss(
            loss_name,
            num_classes=num_classes,
            ignore_index=255,
        )

        loss = criterion(
            logits,
            targets,
        )

        assert loss.ndim == 0, (
            f"{loss_name} must return a scalar."
        )

        assert torch.isfinite(loss), (
            f"{loss_name} produced a non-finite loss."
        )

        loss.backward()

        assert logits.grad is not None, (
            f"{loss_name} did not produce gradients."
        )

        assert torch.isfinite(logits.grad).all(), (
            f"{loss_name} produced non-finite gradients."
        )

        print(f"  Loss value: {loss.item():.6f}")
        print("  Scalar output: Passed")
        print("  Backpropagation: Passed")
        print("  Gradient validity: Passed")

    # -----------------------------------------------------
    # 2. TEST IGNORE-ONLY TARGETS
    # -----------------------------------------------------

    print("\nTesting all-ignored masks...")

    all_ignored_targets = torch.full(
        (batch_size, height, width),
        255,
        dtype=torch.long,
    )

    for loss_name in LOSS_NAMES:

        logits = base_logits.clone().requires_grad_(True)

        criterion = build_loss(
            loss_name,
            num_classes=num_classes,
            ignore_index=255,
        )

        loss = criterion(
            logits,
            all_ignored_targets,
        )

        assert torch.isfinite(loss), (
            f"{loss_name} failed on all-ignored targets."
        )

        assert abs(loss.item()) < 1e-7, (
            f"{loss_name} should return zero when all pixels "
            "are ignored."
        )

        loss.backward()

        assert logits.grad is not None

        assert torch.isfinite(logits.grad).all()

        print(f"  {loss_name}: Passed")

    # -----------------------------------------------------
    # 3. TEST LOSS FACTORY ERROR HANDLING
    # -----------------------------------------------------

    print("\nTesting invalid loss name...")

    try:

        build_loss(
            "unknown_loss",
            num_classes=num_classes,
        )

    except ValueError:
        print("  Invalid loss name correctly rejected.")

    else:
        raise AssertionError(
            "Unknown loss name was not rejected."
        )

    print("\n" + "=" * 65)
    print("ALL FIVE LOSS FUNCTIONS PASSED.")
    print("=" * 65)


if __name__ == "__main__":
    main()
