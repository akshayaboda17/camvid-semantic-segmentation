
"""
Unit test for semantic segmentation metrics.

Run:
python -m src.test_metrics

The synthetic example has known TP, FP and FN values,
so we can verify that the metric formulas are correct.
"""

import numpy as np
import torch

from src.metrics import (
    SegmentationMetrics,
    print_metrics,
)


def main() -> None:

    print("=" * 60)
    print("TESTING SEGMENTATION METRICS")
    print("=" * 60)

    # Three semantic classes in this small example.
    # 255 is an ignored pixel.

    targets = torch.tensor([
        [
            [0, 0, 1],
            [1, 255, 2],
        ]
    ])

    predictions = torch.tensor([
        [
            [0, 1, 1],
            [0, 0, 2],
        ]
    ])

    # -----------------------------------------------------
    # 1. TEST INTEGER CLASS PREDICTIONS
    # -----------------------------------------------------

    metric_calculator = SegmentationMetrics(
        num_classes=3,
        ignore_index=255,
        boundary_tolerance=1,
    )

    metric_calculator.update(
        predictions,
        targets,
    )

    results = metric_calculator.compute()

    print("\nResults from synthetic example:")
    print_metrics(results)

    # -----------------------------------------------------
    # 2. VERIFY KNOWN MATHEMATICAL RESULTS
    # -----------------------------------------------------

    expected_miou = 5.0 / 9.0
    expected_dice = 2.0 / 3.0
    expected_pixel_accuracy = 3.0 / 5.0

    assert np.isclose(
        results["mean_iou"],
        expected_miou,
        atol=1e-6,
    ), "Mean IoU calculation is incorrect."

    assert np.isclose(
        results["mean_dice"],
        expected_dice,
        atol=1e-6,
    ), "Mean Dice calculation is incorrect."

    assert np.isclose(
        results["pixel_accuracy"],
        expected_pixel_accuracy,
        atol=1e-6,
    ), "Pixel Accuracy calculation is incorrect."

    print("\nKnown-value mathematical checks passed.")

    # -----------------------------------------------------
    # 3. TEST MODEL LOGITS INPUT
    # -----------------------------------------------------

    # Convert class predictions into artificial model logits.
    logits = torch.nn.functional.one_hot(
        predictions,
        num_classes=3,
    ).permute(0, 3, 1, 2).float() * 10.0

    logits_metrics = SegmentationMetrics(
        num_classes=3,
        ignore_index=255,
        boundary_tolerance=1,
    )

    logits_metrics.update(
        logits,
        targets,
    )

    logits_results = logits_metrics.compute()

    assert np.isclose(
        logits_results["mean_iou"],
        expected_miou,
        atol=1e-6,
    ), "Logit-based prediction calculation is incorrect."

    print("Model-logit input test passed.")

    # -----------------------------------------------------
    # 4. TEST IGNORE INDEX
    # -----------------------------------------------------

    # The pixel whose ground truth is 255 must not
    # contribute to the confusion matrix.

    total_evaluated_pixels = int(
        results["confusion_matrix"].sum()
    )

    assert total_evaluated_pixels == 5, (
        "Ignore-index handling is incorrect."
    )

    print("Ignore-index test passed.")

    # -----------------------------------------------------
    # 5. TEST BOUNDARY METRIC
    # -----------------------------------------------------

    boundary_score = results["mean_boundary_f1"]

    assert (
        np.isfinite(boundary_score)
        and 0.0 <= boundary_score <= 1.0
    ), "Boundary F-score is invalid."

    print("Boundary metric test passed.")

    # -----------------------------------------------------
    # 6. TEST RESET
    # -----------------------------------------------------

    metric_calculator.reset()

    assert int(
        metric_calculator.confusion_matrix.sum()
    ) == 0, "Reset did not clear the confusion matrix."

    print("Reset test passed.")

    print("\n" + "=" * 60)
    print("ALL SEGMENTATION METRIC TESTS PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
