
"""
Test the CamVid preprocessing and DataLoader pipeline.

Run from the project root:

    python -m src.test_data_pipeline
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image

from src.datasets.camvid import (
    CLASS_NAMES,
    IGNORE_INDEX,
    create_dataloader,
)


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "results" / "preprocessing"

BATCH_SIZE = 2
IMAGE_SIZE = (192, 256)
RANDOM_SEED = 42

# Must match the normalization used in transforms.py.
MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

CLASS_COLORS = [
    (128, 128, 128),  # Sky
    (128, 0, 0),      # Building
    (192, 192, 128),  # Pole
    (128, 64, 128),   # Road
    (0, 0, 192),      # Sidewalk
    (128, 128, 0),    # Tree
    (192, 128, 128),  # SignSymbol
    (64, 64, 128),    # Fence
    (64, 0, 128),     # Car
    (64, 64, 0),      # Pedestrian
    (0, 128, 192),    # Bicyclist
]


# ---------------------------------------------------------
# 2. CONVERT MASK CLASS IDS TO DISPLAY COLOURS
# ---------------------------------------------------------

def colorize_mask(mask: np.ndarray) -> np.ndarray:

    height, width = mask.shape

    result = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for class_id, color in enumerate(CLASS_COLORS):
        result[mask == class_id] = color

    result[mask == IGNORE_INDEX] = (0, 0, 0)

    return result


# ---------------------------------------------------------
# 3. DENORMALIZE IMAGE FOR DISPLAY
# ---------------------------------------------------------

def denormalize_image(
    tensor: torch.Tensor
) -> np.ndarray:

    image = tensor.permute(1, 2, 0).numpy()

    image = image * STD + MEAN

    image = np.clip(image, 0.0, 1.0)

    return image


# ---------------------------------------------------------
# 4. TEST DATALOADER
# ---------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("TESTING CAMVID DATA PIPELINE")
    print("=" * 60)

    # Use training split but disable augmentation for
    # easier visual inspection of the original samples.
    train_loader = create_dataloader(
        split="train",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        augment_train=False,
        num_workers=0,
        seed=RANDOM_SEED,
    )

    print(f"\nNumber of training batches: {len(train_loader)}")

    # Load one batch.
    images, masks = next(iter(train_loader))

    print("\nTENSOR DETAILS")
    print("-" * 40)

    print("Image tensor shape:", tuple(images.shape))
    print("Image tensor dtype:", images.dtype)

    print("Mask tensor shape:", tuple(masks.shape))
    print("Mask tensor dtype:", masks.dtype)

    print("Image minimum:", images.min().item())
    print("Image maximum:", images.max().item())

    unique_ids = torch.unique(masks).tolist()

    print("Mask IDs present in this batch:", unique_ids)

    # -----------------------------------------------------
    # 5. BASIC VALIDATION CHECKS
    # -----------------------------------------------------

    expected_image_shape = (
        BATCH_SIZE,
        3,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
    )

    expected_mask_shape = (
        BATCH_SIZE,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
    )

    assert tuple(images.shape) == expected_image_shape, (
        "Incorrect image tensor shape."
    )

    assert tuple(masks.shape) == expected_mask_shape, (
        "Incorrect mask tensor shape."
    )

    assert images.dtype == torch.float32, (
        "Images must be float32."
    )

    assert masks.dtype == torch.int64, (
        "Masks must be int64."
    )

    valid_ids = set(range(len(CLASS_NAMES))) | {
        IGNORE_INDEX
    }

    actual_ids = set(int(x) for x in unique_ids)

    assert actual_ids.issubset(valid_ids), (
        f"Unexpected mask IDs: {actual_ids - valid_ids}"
    )

    assert torch.isfinite(images).all(), (
        "Images contain NaN or infinite values."
    )

    print("\nAll tensor and class-ID checks passed.")

    # -----------------------------------------------------
    # 6. VISUALIZE THE PROCESSED BATCH
    # -----------------------------------------------------

    number_of_samples = min(
        BATCH_SIZE,
        images.shape[0]
    )

    fig, axes = plt.subplots(
        number_of_samples,
        3,
        figsize=(14, 4 * number_of_samples),
        squeeze=False,
    )

    for i in range(number_of_samples):

        image = denormalize_image(images[i])

        mask = masks[i].numpy()

        colored_mask = colorize_mask(mask)

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Preprocessed RGB Image")

        axes[i, 1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axes[i, 1].set_title("Grayscale Class-ID Mask")

        axes[i, 2].imshow(colored_mask)
        axes[i, 2].set_title("Colour Visualization")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "data_pipeline_preview.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    print(f"\nVisualization saved to: {output_path}")

    plt.show()
    plt.close(fig)

    print("\nCAMVID DATA PIPELINE TEST COMPLETED.")


if __name__ == "__main__":
    main()
