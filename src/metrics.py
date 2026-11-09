
"""
Semantic Segmentation Evaluation Metrics for CamVid.

Metrics:
- Per-class IoU
- Mean IoU
- Per-class and mean Dice/F1
- Per-class and mean Precision
- Per-class and mean Recall
- Pixel Accuracy
- Per-class and mean Boundary F-score

Supported predictions:
- Logits: [N, C, H, W]
- Class IDs: [N, H, W]
- Single image: [H, W]

Ground-truth ignore index: 255
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


CLASS_NAMES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Sidewalk",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
]


class SegmentationMetrics:

    def __init__(
        self,
        num_classes: int = 11,
        ignore_index: int = 255,
        boundary_tolerance: int = 2,
    ) -> None:

        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        if boundary_tolerance < 0:
            raise ValueError("Boundary tolerance cannot be negative.")

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.boundary_tolerance = boundary_tolerance

        # Rows: ground-truth classes.
        # Columns: predicted classes.
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.int64,
        )

        # Boundary statistics accumulated across images.
        self.boundary_pred_total = np.zeros(
            num_classes, dtype=np.int64
        )
        self.boundary_gt_total = np.zeros(
            num_classes, dtype=np.int64
        )
        self.boundary_matched_pred = np.zeros(
            num_classes, dtype=np.int64
        )
        self.boundary_matched_gt = np.zeros(
            num_classes, dtype=np.int64
        )

    # -------------------------------------------------
    # 1. PREPARE PREDICTIONS
    # -------------------------------------------------

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:

        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())

        return torch.as_tensor(value)

    def _prepare_predictions(self, predictions: Any) -> torch.Tensor:

        predictions = self._to_tensor(predictions)

        # Convert model logits into predicted class IDs.
        if predictions.ndim == 4:

            if predictions.shape[1] != self.num_classes:
                raise ValueError(
                    f"Expected {self.num_classes} output channels, "
                    f"received {predictions.shape[1]}."
                )

            predictions = predictions.argmax(dim=1)

        elif predictions.ndim == 2:
            predictions = predictions.unsqueeze(0)

        elif predictions.ndim != 3:
            raise ValueError(
                "Predictions must have shape [H,W], "
                "[N,H,W] or [N,C,H,W]."
            )

        return predictions.detach().to(
            device="cpu",
            dtype=torch.int64,
        )

    # -------------------------------------------------
    # 2. PREPARE GROUND-TRUTH MASKS
    # -------------------------------------------------

    def _prepare_targets(self, targets: Any) -> torch.Tensor:

        targets = self._to_tensor(targets)

        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        elif targets.ndim == 2:
            targets = targets.unsqueeze(0)

        elif targets.ndim != 3:
            raise ValueError(
                "Targets must have shape [H,W], "
                "[N,H,W] or [N,1,H,W]."
            )

        return targets.detach().to(
            device="cpu",
            dtype=torch.int64,
        )

    # -------------------------------------------------
    # 3. EXTRACT THE BOUNDARY OF ONE SEMANTIC CLASS
    # -------------------------------------------------

    @staticmethod
    def _class_boundary(
        labels: torch.Tensor,
        valid: torch.Tensor,
        class_id: int,
    ) -> torch.Tensor:

        class_mask = (labels == class_id) & valid

        boundary = torch.zeros_like(
            class_mask,
            dtype=torch.bool,
        )

        # Compare horizontal neighbours.
        if labels.shape[1] > 1:

            pair_valid = valid[:, :-1] & valid[:, 1:]

            different = (
                (labels[:, :-1] != labels[:, 1:])
                & pair_valid
            )

            boundary[:, :-1] |= (
                different & class_mask[:, :-1]
            )

            boundary[:, 1:] |= (
                different & class_mask[:, 1:]
            )

        # Compare vertical neighbours.
        if labels.shape[0] > 1:

            pair_valid = valid[:-1, :] & valid[1:, :]

            different = (
                (labels[:-1, :] != labels[1:, :])
                & pair_valid
            )

            boundary[:-1, :] |= (
                different & class_mask[:-1, :]
            )

            boundary[1:, :] |= (
                different & class_mask[1:, :]
            )

        return boundary

    # -------------------------------------------------
    # 4. ALLOW A FIXED SPATIAL TOLERANCE WHEN MATCHING
    #    PREDICTED AND GROUND-TRUTH BOUNDARIES
    # -------------------------------------------------

    def _dilate_boundary(
        self,
        boundary: torch.Tensor,
    ) -> torch.Tensor:

        radius = self.boundary_tolerance

        if radius == 0:
            return boundary

        kernel_size = 2 * radius + 1

        boundary_float = boundary.float()[None, None]

        dilated = F.max_pool2d(
            boundary_float,
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        )

        return dilated[0, 0] > 0

    # -------------------------------------------------
    # 5. ACCUMULATE BOUNDARY STATISTICS
    # -------------------------------------------------

    def _update_boundary_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:

        for sample_index in range(predictions.shape[0]):

            pred = predictions[sample_index]
            target = targets[sample_index]
            valid_sample = valid[sample_index]

            for class_id in range(self.num_classes):

                pred_boundary = self._class_boundary(
                    pred, valid_sample, class_id
                )

                gt_boundary = self._class_boundary(
                    target, valid_sample, class_id
                )

                pred_dilated = self._dilate_boundary(
                    pred_boundary
                )

                gt_dilated = self._dilate_boundary(
                    gt_boundary
                )

                matched_pred = (
                    pred_boundary & gt_dilated
                )

                matched_gt = (
                    gt_boundary & pred_dilated
                )

                self.boundary_pred_total[class_id] += int(
                    pred_boundary.sum().item()
                )

                self.boundary_gt_total[class_id] += int(
                    gt_boundary.sum().item()
                )

                self.boundary_matched_pred[class_id] += int(
                    matched_pred.sum().item()
                )

                self.boundary_matched_gt[class_id] += int(
                    matched_gt.sum().item()
                )

    # -------------------------------------------------
    # 6. UPDATE METRICS WITH A BATCH
    # -------------------------------------------------

    def update(
        self,
        predictions: Any,
        targets: Any,
    ) -> None:

        predictions = self._prepare_predictions(predictions)
        targets = self._prepare_targets(targets)

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction shape {tuple(predictions.shape)} "
                f"does not match target shape {tuple(targets.shape)}."
            )

        # Ignore only the explicitly designated ignore index.
        # Any other invalid ground-truth value is an error.
        invalid_targets = (
            (targets != self.ignore_index)
            & (
                (targets < 0)
                | (targets >= self.num_classes)
            )
        )

        if invalid_targets.any():
            bad_ids = torch.unique(
                targets[invalid_targets]
            ).tolist()

            raise ValueError(
                f"Invalid ground-truth class IDs: {bad_ids}"
            )

        valid = targets != self.ignore_index

        if not valid.any():
            return

        valid_predictions = predictions[valid]
        valid_targets = targets[valid]

        invalid_predictions = (
            (valid_predictions < 0)
            | (valid_predictions >= self.num_classes)
        )

        if invalid_predictions.any():
            raise ValueError(
                "Predictions contain class IDs outside "
                f"0 to {self.num_classes - 1}."
            )

        # Encode target-prediction combinations.
        encoded = (
            valid_targets * self.num_classes
            + valid_predictions
        )

        batch_confusion = torch.bincount(
            encoded,
            minlength=self.num_classes ** 2,
        ).reshape(
            self.num_classes,
            self.num_classes,
        )

        self.confusion_matrix += batch_confusion

        self._update_boundary_metrics(
            predictions,
            targets,
            valid,
        )

    # -------------------------------------------------
    # 7. SAFE AVERAGE
    # -------------------------------------------------

    @staticmethod
    def _safe_mean(values: np.ndarray) -> float:

        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            return float("nan")

        return float(finite_values.mean())

    # -------------------------------------------------
    # 8. COMPUTE FINAL METRICS
    # -------------------------------------------------

    def compute(self) -> dict[str, Any]:

        cm = self.confusion_matrix.numpy().astype(
            np.float64
        )

        tp = np.diag(cm)

        actual = cm.sum(axis=1)
        predicted = cm.sum(axis=0)

        fp = predicted - tp
        fn = actual - tp

        union = tp + fp + fn

        # Exclude only classes absent from both actual
        # and predicted labels.
        present = union > 0

        # IoU.
        iou = np.full(
            self.num_classes, np.nan
        )

        np.divide(
            tp,
            union,
            out=iou,
            where=present,
        )

        # Dice / F1.
        dice_denominator = 2 * tp + fp + fn

        dice = np.full(
            self.num_classes, np.nan
        )

        np.divide(
            2 * tp,
            dice_denominator,
            out=dice,
            where=present,
        )

        # Precision.
        precision = np.zeros(
            self.num_classes
        )

        np.divide(
            tp,
            predicted,
            out=precision,
            where=predicted > 0,
        )

        precision[~present] = np.nan

        # Recall.
        recall = np.zeros(
            self.num_classes
        )

        np.divide(
            tp,
            actual,
            out=recall,
            where=actual > 0,
        )

        recall[~present] = np.nan

        # Pixel accuracy.
        total_pixels = cm.sum()

        pixel_accuracy = (
            float(tp.sum() / total_pixels)
            if total_pixels > 0
            else float("nan")
        )

        # Boundary Precision and Recall.
        bp = np.zeros(self.num_classes)
        br = np.zeros(self.num_classes)

        boundary_present = (
            self.boundary_pred_total
            + self.boundary_gt_total
        ) > 0

        np.divide(
            self.boundary_matched_pred,
            self.boundary_pred_total,
            out=bp,
            where=self.boundary_pred_total > 0,
        )

        np.divide(
            self.boundary_matched_gt,
            self.boundary_gt_total,
            out=br,
            where=self.boundary_gt_total > 0,
        )

        # If a class has a boundary in only one map,
        # its boundary F-score is zero, not undefined.
        boundary_f1 = np.full(
            self.num_classes,
            np.nan
        )

        boundary_f1[boundary_present] = 0.0

        denominator = bp + br

        np.divide(
            2 * bp * br,
            denominator,
            out=boundary_f1,
            where=(
                boundary_present
                & (denominator > 0)
            ),
        )

        # Per-class results.
        class_metrics = []

        for class_id in range(self.num_classes):

            class_name = (
                CLASS_NAMES[class_id]
                if self.num_classes == len(CLASS_NAMES)
                else f"class_{class_id}"
            )

            class_metrics.append({
                "class_id": class_id,
                "class_name": class_name,
                "actual_pixels": int(actual[class_id]),
                "predicted_pixels": int(predicted[class_id]),
                "iou": float(iou[class_id]),
                "dice": float(dice[class_id]),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "boundary_f1": float(boundary_f1[class_id]),
            })

        return {
            "mean_iou": self._safe_mean(iou),
            "mean_dice": self._safe_mean(dice),
            "mean_precision": self._safe_mean(precision),
            "mean_recall": self._safe_mean(recall),
            "pixel_accuracy": pixel_accuracy,
            "mean_boundary_f1": self._safe_mean(boundary_f1),
            "per_class": class_metrics,
            "confusion_matrix": cm.astype(np.int64),
        }

    # -------------------------------------------------
    # 9. RESET ACCUMULATED VALUES
    # -------------------------------------------------

    def reset(self) -> None:

        self.confusion_matrix.zero_()

        self.boundary_pred_total.fill(0)
        self.boundary_gt_total.fill(0)
        self.boundary_matched_pred.fill(0)
        self.boundary_matched_gt.fill(0)


# ---------------------------------------------------------
# 10. PRINT RESULTS
# ---------------------------------------------------------

def print_metrics(results: dict[str, Any]) -> None:

    print("\n" + "=" * 70)
    print("SEMANTIC SEGMENTATION EVALUATION")
    print("=" * 70)

    aggregate = [
        ("Mean IoU", results["mean_iou"]),
        ("Mean Dice / F1", results["mean_dice"]),
        ("Mean Precision", results["mean_precision"]),
        ("Mean Recall", results["mean_recall"]),
        ("Pixel Accuracy", results["pixel_accuracy"]),
        ("Mean Boundary F-score", results["mean_boundary_f1"]),
    ]

    for name, value in aggregate:

        if np.isfinite(value):
            print(f"{name:28s}: {value * 100:.2f}%")
        else:
            print(f"{name:28s}: N/A")

    print("\nPER-CLASS METRICS")
    print("-" * 70)

    print(
        f"{'Class':15s}"
        f"{'IoU':>10s}"
        f"{'Dice':>10s}"
        f"{'Precision':>12s}"
        f"{'Recall':>10s}"
    )

    for item in results["per_class"]:

        def display(value: float) -> str:
            if np.isfinite(value):
                return f"{value * 100:.2f}%"
            return "N/A"

        print(
            f"{item['class_name']:15s}"
            f"{display(item['iou']):>10s}"
            f"{display(item['dice']):>10s}"
            f"{display(item['precision']):>12s}"
            f"{display(item['recall']):>10s}"
        )

    print("\nConfusion matrix shape:",
          results["confusion_matrix"].shape)

    print("=" * 70)
