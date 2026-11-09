
"""
Five loss functions for multiclass CamVid segmentation.

Implemented:
1. Cross-Entropy Loss
2. Dice Loss
3. Focal Loss
4. Lovasz-Softmax Loss
5. Distance-transform Boundary Loss

Expected inputs:
    logits:  [N, C, H, W]
    targets: [N, H, W]

Ignored target ID:
    255
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------
# 1. COMMON INPUT VALIDATION
# ---------------------------------------------------------

def validate_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:

    if logits.ndim != 4:
        raise ValueError(
            "Logits must have shape [N, C, H, W]."
        )

    if targets.ndim != 3:
        raise ValueError(
            "Targets must have shape [N, H, W]."
        )

    if logits.shape[0] != targets.shape[0]:
        raise ValueError("Batch sizes do not match.")

    if logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} classes, "
            f"received {logits.shape[1]}."
        )

    if logits.shape[-2:] != targets.shape[-2:]:
        raise ValueError(
            "Logit and target spatial dimensions do not match."
        )


# ---------------------------------------------------------
# 2. CROSS-ENTROPY LOSS
# ---------------------------------------------------------

class CrossEntropySegmentationLoss(nn.Module):
    """
    Standard multiclass pixel-classification loss.
    """

    def __init__(
        self,
        ignore_index: int = 255,
    ) -> None:

        super().__init__()

        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        valid = targets != self.ignore_index

        if not valid.any():
            return logits.sum() * 0.0

        return F.cross_entropy(
            logits.float(),
            targets.long(),
            ignore_index=self.ignore_index,
        )


# ---------------------------------------------------------
# 3. DICE LOSS
# ---------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Multiclass soft Dice loss.

    Dice = 2 * Intersection / (Prediction + Ground truth)

    Loss = 1 - Mean Dice
    """

    def __init__(
        self,
        num_classes: int = 11,
        ignore_index: int = 255,
        smooth: float = 1.0,
    ) -> None:

        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        validate_inputs(
            logits,
            targets,
            self.num_classes,
        )

        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )

        valid = (
            targets != self.ignore_index
        )

        if not valid.any():
            return logits.sum() * 0.0

        safe_targets = targets.masked_fill(
            ~valid,
            0,
        )

        one_hot = F.one_hot(
            safe_targets.long(),
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).float()

        valid_mask = valid.unsqueeze(1).float()

        probabilities = (
            probabilities * valid_mask
        )

        one_hot = one_hot * valid_mask

        # Accumulate over batch and spatial dimensions.
        dimensions = (0, 2, 3)

        intersection = (
            probabilities * one_hot
        ).sum(dim=dimensions)

        prediction_sum = probabilities.sum(
            dim=dimensions
        )

        target_sum = one_hot.sum(
            dim=dimensions
        )

        denominator = (
            prediction_sum + target_sum
        )

        # Include a class if it occurs in either prediction
        # or ground truth. Exclude only absent-from-both classes.
        present = denominator > 0

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice[present].mean()


# ---------------------------------------------------------
# 4. FOCAL LOSS
# ---------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss.

    Focal = (1 - pt)^gamma * CrossEntropy

    Gamma controls how strongly easy pixels are down-weighted.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        gamma: float = 2.0,
    ) -> None:

        super().__init__()

        if gamma < 0:
            raise ValueError("Gamma must be non-negative.")

        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        valid = targets != self.ignore_index

        if not valid.any():
            return logits.sum() * 0.0

        cross_entropy = F.cross_entropy(
            logits.float(),
            targets.long(),
            ignore_index=self.ignore_index,
            reduction="none",
        )

        probabilities = torch.exp(
            -cross_entropy
        )

        focal = (
            (1.0 - probabilities) ** self.gamma
        ) * cross_entropy

        return focal[valid].mean()


# ---------------------------------------------------------
# 5. LOVASZ-SOFTMAX LOSS
# ---------------------------------------------------------

def lovasz_gradient(
    sorted_foreground: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the discrete Lovasz extension gradient.
    """

    foreground_sum = sorted_foreground.sum()

    intersection = (
        foreground_sum
        - sorted_foreground.float().cumsum(0)
    )

    union = (
        foreground_sum
        + (1.0 - sorted_foreground.float()).cumsum(0)
    )

    jaccard = 1.0 - intersection / union

    if jaccard.numel() > 1:
        jaccard = torch.cat([
            jaccard[:1],
            jaccard[1:] - jaccard[:-1],
        ])

    return jaccard


class LovaszSoftmaxLoss(nn.Module):
    """
    Multiclass Lovasz-Softmax loss.

    Computes a Lovasz surrogate of the Jaccard/IoU error.
    Classes absent from the target are skipped.
    """

    def __init__(
        self,
        num_classes: int = 11,
        ignore_index: int = 255,
    ) -> None:

        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        validate_inputs(
            logits,
            targets,
            self.num_classes,
        )

        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )

        # Rearrange to [N*H*W, C].
        probabilities = probabilities.permute(
            0, 2, 3, 1
        ).reshape(-1, self.num_classes)

        labels = targets.reshape(-1).long()

        valid = labels != self.ignore_index

        if not valid.any():
            return logits.sum() * 0.0

        probabilities = probabilities[valid]
        labels = labels[valid]

        invalid_labels = (
            (labels < 0)
            | (labels >= self.num_classes)
        )

        if invalid_labels.any():
            raise ValueError(
                "Invalid ground-truth class IDs found."
            )

        class_losses = []

        for class_id in range(self.num_classes):

            foreground = (
                labels == class_id
            ).float()

            # Average over classes present in the target.
            if foreground.sum() == 0:
                continue

            errors = (
                foreground
                - probabilities[:, class_id]
            ).abs()

            sorted_errors, permutation = torch.sort(
                errors,
                descending=True,
            )

            sorted_foreground = foreground[
                permutation
            ]

            gradient = lovasz_gradient(
                sorted_foreground
            )

            class_loss = torch.dot(
                sorted_errors,
                gradient,
            )

            class_losses.append(class_loss)

        if not class_losses:
            return logits.sum() * 0.0

        return torch.stack(class_losses).mean()


# ---------------------------------------------------------
# 6. DISTANCE-TRANSFORM BOUNDARY LOSS
# ---------------------------------------------------------

class BoundaryLoss(nn.Module):
    """
    Multiclass distance-transform boundary loss.

    Signed distance map:
        Negative inside a ground-truth class.
        Positive outside the class.

    Loss:
        1 + mean(sum_c(probability_c * signed_distance_c))

    A constant is added to keep the reported loss non-negative;
    it does not change its gradients.

    Distance maps are computed from ground-truth masks with
    SciPy on CPU. Ignored pixels are excluded from the final
    loss reduction.
    """

    def __init__(
        self,
        num_classes: int = 11,
        ignore_index: int = 255,
    ) -> None:

        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def _create_distance_maps(
        self,
        targets: torch.Tensor,
    ) -> np.ndarray:

        targets_np = (
            targets.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        batch_size, height, width = targets_np.shape

        distance_maps = np.zeros(
            (
                batch_size,
                self.num_classes,
                height,
                width,
            ),
            dtype=np.float32,
        )

        diagonal = float(
            np.sqrt(height ** 2 + width ** 2)
        )

        for batch_index in range(batch_size):

            target = targets_np[batch_index]

            valid = target != self.ignore_index

            for class_id in range(self.num_classes):

                binary_mask = (
                    (target == class_id) & valid
                )

                # If the class is absent, assign a positive
                # distance throughout valid pixels.
                if not binary_mask.any():

                    distance_map = np.ones(
                        (height, width),
                        dtype=np.float32,
                    )

                else:

                    inside_distance = distance_transform_edt(
                        binary_mask
                    )

                    outside_distance = distance_transform_edt(
                        ~binary_mask
                    )

                    signed_distance = (
                        outside_distance - inside_distance
                    )

                    distance_map = np.clip(
                        signed_distance / diagonal,
                        -1.0,
                        1.0,
                    ).astype(np.float32)

                distance_map[~valid] = 0.0

                distance_maps[
                    batch_index,
                    class_id,
                ] = distance_map

        return distance_maps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        validate_inputs(
            logits,
            targets,
            self.num_classes,
        )

        valid = targets != self.ignore_index

        if not valid.any():
            return logits.sum() * 0.0

        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )

        distance_maps = self._create_distance_maps(
            targets
        )

        distance_tensor = torch.from_numpy(
            distance_maps
        ).to(
            device=logits.device,
            dtype=probabilities.dtype,
        )

        # Weighted sum of predicted class probabilities
        # and their ground-truth signed distance maps.
        boundary_energy = (
            probabilities * distance_tensor
        ).sum(dim=1)

        return 1.0 + boundary_energy[valid].mean()


# ---------------------------------------------------------
# 7. LOSS FACTORY
# ---------------------------------------------------------

LOSS_NAMES = (
    "cross_entropy",
    "dice",
    "focal",
    "lovasz",
    "boundary",
)


def build_loss(
    loss_name: str,
    num_classes: int = 11,
    ignore_index: int = 255,
) -> nn.Module:
    """
    Construct a loss using its registered name.
    """

    name = loss_name.lower().strip()

    if name == "cross_entropy":

        return CrossEntropySegmentationLoss(
            ignore_index=ignore_index,
        )

    if name == "dice":

        return DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    if name == "focal":

        return FocalLoss(
            ignore_index=ignore_index,
            gamma=2.0,
        )

    if name == "lovasz":

        return LovaszSoftmaxLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    if name == "boundary":

        return BoundaryLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    raise ValueError(
        f"Unknown loss '{loss_name}'. "
        f"Choose from {LOSS_NAMES}."
    )
