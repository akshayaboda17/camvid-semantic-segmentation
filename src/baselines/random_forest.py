
"""
Random Forest Semantic Segmentation Baseline for CamVid.

Features:
    1. Red channel
    2. Green channel
    3. Blue channel
    4. Normalized horizontal coordinate
    5. Normalized vertical coordinate

Training:
    - Uses training split only.
    - Samples pixels separately by semantic class.
    - Excludes ignore label 255.
    - Limits samples to control memory usage.

Evaluation:
    - Uses validation split only.
    - Reports mIoU, Dice, Precision, Recall,
      Pixel Accuracy and Boundary F-score.
    - Saves class-wise metrics and visual predictions.

Run from the project root:

    python -m src.baselines.random_forest
"""

from __future__ import annotations

import csv
import json
import time

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from src.datasets.camvid import (
    CamVidDataset,
    CLASS_NAMES,
    NUM_CLASSES,
    IGNORE_INDEX,
)

from src.metrics import (
    SegmentationMetrics,
    print_metrics,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "random_forest"
MODEL_DIR = PROJECT_ROOT / "models"

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256

RANDOM_SEED = 42

# Sampling limits.
MAX_PIXELS_PER_CLASS_PER_IMAGE = 100
MAX_TRAIN_SAMPLES_PER_CLASS = 5000

# Random Forest baseline parameters.
RF_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": 18,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

FEATURE_NAMES = [
    "Red",
    "Green",
    "Blue",
    "X_position",
    "Y_position",
]

# 11-class colour map, matching the CamVid protocol.
CLASS_COLORS = [
    (128, 128, 128),   # Sky
    (128, 0, 0),       # Building
    (192, 192, 128),   # Pole
    (128, 64, 128),    # Road
    (0, 0, 192),       # Sidewalk
    (128, 128, 0),     # Tree
    (192, 128, 128),   # SignSymbol
    (64, 64, 128),     # Fence
    (64, 0, 128),      # Car
    (64, 64, 0),       # Pedestrian
    (0, 128, 192),     # Bicyclist
]


# =========================================================
# 2. LOAD AND RESIZE ONE IMAGE-MASK PAIR
# =========================================================

def load_resized_pair(
    image_path: Path,
    mask_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one RGB image and its grayscale segmentation mask.

    RGB image:
        Bilinear resizing.

    Segmentation mask:
        Nearest-neighbour resizing to preserve integer labels.
    """

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    with Image.open(mask_path) as mask_file:
        if mask_file.mode != "L":
            raise ValueError(
                f"Expected grayscale mask, got "
                f"{mask_file.mode}: {mask_path}"
            )

        mask = mask_file.copy()

    if image.size != mask.size:
        raise ValueError(
            f"Image/mask size mismatch: "
            f"{image.size} vs {mask.size}"
        )

    image = image.resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        resample=Image.Resampling.BILINEAR,
    )

    mask = mask.resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        resample=Image.Resampling.NEAREST,
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    mask_array = np.asarray(
        mask,
        dtype=np.uint8,
    )

    unique_ids = set(
        int(x) for x in np.unique(mask_array)
    )

    valid_ids = set(range(NUM_CLASSES)) | {
        IGNORE_INDEX
    }

    invalid_ids = unique_ids - valid_ids

    if invalid_ids:
        raise ValueError(
            f"Invalid mask IDs {sorted(invalid_ids)} "
            f"in {mask_path}"
        )

    return image_array, mask_array


# =========================================================
# 3. EXTRACT PIXEL FEATURES
# =========================================================

def extract_pixel_features(
    image_array: np.ndarray,
) -> np.ndarray:
    """
    Convert an image into one feature vector per pixel.

    Features:
        R, G, B, normalized X, normalized Y.

    Input:
        [H, W, 3]

    Output:
        [H*W, 5]
    """

    height, width, channels = image_array.shape

    if channels != 3:
        raise ValueError(
            "Expected a three-channel RGB image."
        )

    # Flatten RGB values.
    rgb_features = image_array.reshape(
        -1, 3
    )

    # Normalized horizontal positions: [-1, 1].
    x_coordinates = np.tile(
        np.linspace(
            -1.0,
            1.0,
            width,
            dtype=np.float32,
        ),
        height,
    )

    # Normalized vertical positions: [-1, 1].
    y_coordinates = np.repeat(
        np.linspace(
            -1.0,
            1.0,
            height,
            dtype=np.float32,
        ),
        width,
    )

    coordinate_features = np.column_stack(
        (
            x_coordinates,
            y_coordinates,
        )
    )

    features = np.concatenate(
        (
            rgb_features,
            coordinate_features,
        ),
        axis=1,
    )

    return np.ascontiguousarray(
        features,
        dtype=np.float32,
    )


# =========================================================
# 4. COLLECT CLASS-BALANCED TRAINING PIXELS
# =========================================================

def collect_training_samples(
    dataset: CamVidDataset,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Build a memory-conscious training sample.

    For each image:
        - Extract pixel features.
        - Separate pixels by ground-truth class.
        - Randomly sample at most the configured number
          of pixels per class per image.

    Then:
        - Combine samples across training images.
        - Apply a global cap per class.

    Only the training dataset is used here.
    """

    class_feature_chunks: dict[int, list[np.ndarray]] = {
        class_id: []
        for class_id in range(NUM_CLASSES)
    }

    print("\nCOLLECTING TRAINING PIXEL SAMPLES")
    print("-" * 60)

    for image_index, (image_path, mask_path) in enumerate(
        dataset.samples,
        start=1,
    ):

        image_array, mask_array = load_resized_pair(
            image_path,
            mask_path,
        )

        features = extract_pixel_features(
            image_array
        )

        flattened_mask = mask_array.reshape(-1)

        for class_id in range(NUM_CLASSES):

            positions = np.flatnonzero(
                flattened_mask == class_id
            )

            if positions.size == 0:
                continue

            sample_count = min(
                positions.size,
                MAX_PIXELS_PER_CLASS_PER_IMAGE,
            )

            if positions.size > sample_count:

                selected_positions = rng.choice(
                    positions,
                    size=sample_count,
                    replace=False,
                )

            else:
                selected_positions = positions

            class_feature_chunks[class_id].append(
                features[selected_positions]
            )

        if image_index % 50 == 0:

            print(
                f"Processed {image_index}/"
                f"{len(dataset)} training images."
            )

    # -----------------------------------------------------
    # Apply global per-class sample cap.
    # -----------------------------------------------------

    final_class_features = []
    final_labels = []

    sampling_rows = []

    print("\nFINAL TRAINING SAMPLE COUNTS")
    print("-" * 60)

    for class_id in range(NUM_CLASSES):

        chunks = class_feature_chunks[class_id]

        if chunks:
            class_features = np.concatenate(
                chunks,
                axis=0,
            )
        else:
            class_features = np.empty(
                (0, len(FEATURE_NAMES)),
                dtype=np.float32,
            )

        total_available = len(class_features)

        final_count = min(
            total_available,
            MAX_TRAIN_SAMPLES_PER_CLASS,
        )

        if total_available > final_count:

            selected_indices = rng.choice(
                total_available,
                size=final_count,
                replace=False,
            )

            class_features = class_features[
                selected_indices
            ]

        if final_count == 0:
            print(
                f"WARNING: No training pixels found for "
                f"class {class_id} ({CLASS_NAMES[class_id]})."
            )

        final_class_features.append(
            class_features
        )

        final_labels.append(
            np.full(
                final_count,
                class_id,
                dtype=np.int64,
            )
        )

        sampling_rows.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "candidate_samples": total_available,
            "selected_samples": final_count,
        })

        print(
            f"{class_id:2d}. "
            f"{CLASS_NAMES[class_id]:12s}: "
            f"{final_count:,} selected"
        )

    X_train = np.concatenate(
        final_class_features,
        axis=0,
    )

    y_train = np.concatenate(
        final_labels,
        axis=0,
    )

    if len(X_train) == 0:
        raise RuntimeError(
            "No valid training samples were collected."
        )

    # Shuffle the balanced sample.
    permutation = rng.permutation(
        len(X_train)
    )

    X_train = X_train[permutation]
    y_train = y_train[permutation]

    return X_train, y_train, sampling_rows


# =========================================================
# 5. COLOURIZE A SEGMENTATION MASK
# =========================================================

def colorize_mask(
    mask: np.ndarray,
) -> np.ndarray:

    height, width = mask.shape

    color_image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    for class_id, color in enumerate(CLASS_COLORS):

        color_image[
            mask == class_id
        ] = color

    # Ignored pixels are shown as black.
    color_image[
        mask == IGNORE_INDEX
    ] = (0, 0, 0)

    return color_image


# =========================================================
# 6. VALIDATION
# =========================================================

def evaluate_random_forest(
    model: RandomForestClassifier,
    validation_dataset: CamVidDataset,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    """
    Evaluate on validation data only.

    Test split is deliberately never loaded by this script.
    """

    metric_calculator = SegmentationMetrics(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        boundary_tolerance=2,
    )

    inference_times = []

    visualization_items = []

    print("\nVALIDATING RANDOM FOREST")
    print("-" * 60)

    for index, (image_path, mask_path) in enumerate(
        validation_dataset.samples,
        start=1,
    ):

        image_array, gt_mask = load_resized_pair(
            image_path,
            mask_path,
        )

        # Include feature extraction and classifier prediction
        # in the reported per-image inference time.
        start_time = time.perf_counter()

        features = extract_pixel_features(
            image_array
        )

        predicted_flat = model.predict(
            features
        )

        elapsed = time.perf_counter() - start_time

        inference_times.append(
            elapsed
        )

        prediction = predicted_flat.reshape(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )

        # Update accumulated metric statistics.
        metric_calculator.update(
            prediction,
            gt_mask,
        )

        # Retain a few samples for qualitative analysis.
        if len(visualization_items) < 5:

            visualization_items.append({
                "image": image_array,
                "ground_truth": gt_mask.copy(),
                "prediction": prediction.copy(),
                "name": image_path.name,
            })

        if index % 20 == 0:

            print(
                f"Evaluated {index}/"
                f"{len(validation_dataset)} images."
            )

    results = metric_calculator.compute()

    average_inference_seconds = float(
        np.mean(inference_times)
    )

    return (
        results,
        average_inference_seconds,
        visualization_items,
    )


# =========================================================
# 7. SAVE VALIDATION VISUALIZATIONS
# =========================================================

def save_prediction_visualizations(
    visualization_items: list[dict[str, Any]],
) -> None:

    if not visualization_items:
        return

    number_of_samples = len(
        visualization_items
    )

    fig, axes = plt.subplots(
        number_of_samples,
        3,
        figsize=(15, 4 * number_of_samples),
        squeeze=False,
    )

    for row, item in enumerate(
        visualization_items
    ):

        axes[row, 0].imshow(
            item["image"]
        )

        axes[row, 0].set_title(
            "Original RGB image"
        )

        axes[row, 1].imshow(
            colorize_mask(item["ground_truth"])
        )

        axes[row, 1].set_title(
            "Ground-truth segmentation"
        )

        axes[row, 2].imshow(
            colorize_mask(item["prediction"])
        )

        axes[row, 2].set_title(
            "Random Forest prediction"
        )

        for axis in axes[row]:
            axis.axis("off")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR / "validation_predictions.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved prediction visualization: {output_path}"
    )


# =========================================================
# 8. SAVE METRICS AND EXPERIMENT DETAILS
# =========================================================

def save_results(
    results: dict[str, Any],
    training_rows: list[dict[str, Any]],
    training_time: float,
    inference_time: float,
    model_path: Path,
    total_training_samples: int,
) -> None:

    # -----------------------------------------------------
    # Class-wise metrics CSV.
    # -----------------------------------------------------

    class_metrics_path = (
        RESULTS_DIR / "class_metrics.csv"
    )

    with class_metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results["per_class"][0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results["per_class"]
        )

    # -----------------------------------------------------
    # Sampling CSV.
    # -----------------------------------------------------

    sampling_path = (
        RESULTS_DIR / "training_sampling.csv"
    )

    with sampling_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_id",
                "class_name",
                "candidate_samples",
                "selected_samples",
            ],
        )

        writer.writeheader()
        writer.writerows(training_rows)

    # -----------------------------------------------------
    # Confusion matrix CSV.
    # -----------------------------------------------------

    np.savetxt(
        RESULTS_DIR / "confusion_matrix.csv",
        results["confusion_matrix"],
        delimiter=",",
        fmt="%d",
    )

    # -----------------------------------------------------
    # Overall experiment summary.
    # -----------------------------------------------------

    metrics_without_confusion = {
        key: value
        for key, value in results.items()
        if key != "confusion_matrix"
    }

    experiment_summary = {
        "experiment": "Random Forest CamVid baseline",
        "dataset": "CamVid 11-class protocol",
        "training_split": 367,
        "validation_split": 101,
        "test_split": "Not accessed",
        "input_size": [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ],
        "features": FEATURE_NAMES,
        "sampling_policy": {
            "maximum_per_class_per_image":
                MAX_PIXELS_PER_CLASS_PER_IMAGE,
            "maximum_per_class_over_training":
                MAX_TRAIN_SAMPLES_PER_CLASS,
        },
        "random_seed": RANDOM_SEED,
        "classifier_parameters": RF_PARAMETERS,
        "training_samples": total_training_samples,
        "training_time_seconds": training_time,
        "validation_inference_seconds_per_image":
            inference_time,
        "saved_model_size_bytes": (
            model_path.stat().st_size
        ),
        "validation_metrics": metrics_without_confusion,
        "boundary_metric_definition": (
            "Class-wise boundary F-score with two-pixel "
            "spatial tolerance at 256x192 resolution."
        ),
    }

    summary_path = (
        RESULTS_DIR / "experiment_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            experiment_summary,
            file,
            indent=4,
            allow_nan=False,
        )

    print("\nSaved:")
    print(class_metrics_path)
    print(sampling_path)
    print(summary_path)


# =========================================================
# 9. MAIN
# =========================================================

def main() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR / "random_forest_baseline.joblib"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print("=" * 70)
    print("CAMVID RANDOM FOREST SEMANTIC SEGMENTATION BASELINE")
    print("=" * 70)

    print(f"Input resolution: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"Features: {FEATURE_NAMES}")
    print(f"Random seed: {RANDOM_SEED}")

    # -----------------------------------------------------
    # Load training split only.
    # -----------------------------------------------------

    train_dataset = CamVidDataset(
        split="train",
        validate_mask_ids=True,
    )

    X_train, y_train, training_rows = (
        collect_training_samples(
            train_dataset,
            rng,
        )
    )

    print("\nTRAINING DATA DETAILS")
    print("-" * 60)

    print("Feature matrix shape:", X_train.shape)
    print("Label vector shape:", y_train.shape)
    print("Total training pixel samples:", len(y_train))

    print("\nTraining classifier...")

    classifier = RandomForestClassifier(
        **RF_PARAMETERS
    )

    training_start = time.perf_counter()

    classifier.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter() - training_start
    )

    print("\nRandom Forest training completed.")
    print(
        f"Training time: {training_time:.2f} seconds"
    )

    # Save the local model.
    joblib.dump(
        classifier,
        model_path,
        compress=3,
    )

    print(f"Saved model locally: {model_path}")

    # Feature importance is useful for understanding
    # the contributions of RGB and position features.
    importance_path = (
        RESULTS_DIR / "feature_importance.csv"
    )

    with importance_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "feature",
            "importance",
        ])

        for name, importance in zip(
            FEATURE_NAMES,
            classifier.feature_importances_,
        ):

            writer.writerow([
                name,
                float(importance),
            ])

    print("\nFEATURE IMPORTANCE")

    for name, importance in zip(
        FEATURE_NAMES,
        classifier.feature_importances_,
    ):

        print(
            f"{name:15s}: {importance:.4f}"
        )

    # -----------------------------------------------------
    # Load validation split only.
    # -----------------------------------------------------

    validation_dataset = CamVidDataset(
        split="validation",
        validate_mask_ids=True,
    )

    (
        validation_results,
        inference_time,
        visualization_items,
    ) = evaluate_random_forest(
        classifier,
        validation_dataset,
    )

    print_metrics(
        validation_results
    )

    print(
        f"\nAverage inference time per image: "
        f"{inference_time * 1000:.2f} ms"
    )

    # -----------------------------------------------------
    # Save outputs.
    # -----------------------------------------------------

    save_prediction_visualizations(
        visualization_items
    )

    save_results(
        results=validation_results,
        training_rows=training_rows,
        training_time=training_time,
        inference_time=inference_time,
        model_path=model_path,
        total_training_samples=len(y_train),
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST BASELINE COMPLETED")
    print("=" * 70)

    print("\nEvaluation Protocol:")
    print("Training: 367 images")
    print("Validation: 101 images")
    print("Test: Reserved for final evaluation")


if __name__ == "__main__":
    main()
