
"""
CamVid Dataset Audit and Exploratory Data Analysis

Checks:
1. Train/validation/test split files.
2. Image-mask correspondence.
3. Missing files and duplicate assignments.
4. Image/mask dimensions and modes.
5. Valid semantic mask IDs.
6. Training-set class-pixel distribution.
7. Sample image and segmentation-mask visualizations.

Run from the project root:
python src/data_audit.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "data" / "camvid_source"
OUTPUT_DIR = PROJECT_ROOT / "results" / "audit"

SPLITS = {
    "train": "camvid_train.txt",
    "validation": "camvid_val.txt",
    "test": "camvid_test.txt",
}

# Official 11-class grayscale mask mapping in this repository.
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

NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

VALID_MASK_IDS = set(range(NUM_CLASSES)) | {IGNORE_INDEX}

# RGB colours corresponding to the repository's 11-class palette.
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


# ---------------------------------------------------------
# 2. READ DATASET SPLIT FILES
# ---------------------------------------------------------

def read_split_file(split_name: str, filename: str) -> list[dict[str, Any]]:
    """
    Each non-empty line must contain:
    relative_image_path relative_mask_path
    """

    split_path = DATASET_ROOT / filename

    if not split_path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )

    records = []

    with split_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid entry in {filename}, "
                    f"line {line_number}: {line}"
                )

            image_rel, mask_rel = parts

            image_path = DATASET_ROOT / Path(image_rel.replace("/", "\\"))
            mask_path = DATASET_ROOT / Path(mask_rel.replace("/", "\\"))

            records.append({
                "split": split_name,
                "line_number": line_number,
                "image_relative_path": image_rel,
                "mask_relative_path": mask_rel,
                "image_path": image_path,
                "mask_path": mask_path,
            })

    return records


# ---------------------------------------------------------
# 3. AUDIT ONE SPLIT
# ---------------------------------------------------------

def audit_split(
    split_name: str,
    records: list[dict[str, Any]],
    training_pixel_counts: Counter,
    training_image_counts: Counter,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[int]]:

    image_modes = Counter()
    mask_modes = Counter()
    resolutions = Counter()
    mask_sizes = Counter()

    found_mask_ids = set()
    invalid_mask_ids = set()

    missing_images = 0
    missing_masks = 0
    dimension_mismatches = 0
    unreadable_pairs = 0

    valid_records = []

    for index, record in enumerate(records, start=1):

        image_path = record["image_path"]
        mask_path = record["mask_path"]

        issues = []

        if not image_path.is_file():
            missing_images += 1
            issues.append("missing_image")

        if not mask_path.is_file():
            missing_masks += 1
            issues.append("missing_mask")

        if issues:
            record["audit_status"] = ";".join(issues)
            continue

        try:
            # Read the original RGB image.
            with Image.open(image_path) as image:
                image_size = image.size
                image_mode = image.mode

            # Important: do not convert the mask to RGB or grayscale.
            # Preserve its original integer class IDs.
            with Image.open(mask_path) as mask:
                mask_size = mask.size
                mask_mode = mask.mode
                mask_array = np.array(mask)

        except Exception as error:
            unreadable_pairs += 1
            record["audit_status"] = f"unreadable: {error}"
            continue

        image_modes[image_mode] += 1
        mask_modes[mask_mode] += 1
        resolutions[str(image_size)] += 1
        mask_sizes[str(mask_size)] += 1

        pair_valid = True

        # Check image-mask dimensions.
        if image_size != mask_size:
            dimension_mismatches += 1
            issues.append("dimension_mismatch")
            pair_valid = False

        # Expected CamVidGray masks must be 2D integer-label arrays.
        if mask_array.ndim != 2:
            issues.append("mask_not_2d")
            invalid_mask_ids.add(-1)
            pair_valid = False

        else:
            unique_ids, pixel_counts = np.unique(
                mask_array,
                return_counts=True
            )

            found_mask_ids.update(
                int(value) for value in unique_ids
            )

            invalid = set(
                int(value) for value in unique_ids
            ) - VALID_MASK_IDS

            if invalid:
                invalid_mask_ids.update(invalid)
                issues.append(
                    "invalid_mask_ids:" + str(sorted(invalid))
                )
                pair_valid = False

            # Only training masks contribute to the class-imbalance
            # statistics used for development decisions.
            if split_name == "train":

                for class_id, count in zip(
                    unique_ids, pixel_counts
                ):
                    class_id = int(class_id)
                    count = int(count)

                    if class_id in range(NUM_CLASSES):
                        training_pixel_counts[class_id] += count
                        training_image_counts[class_id] += (
                            1 if count > 0 else 0
                        )

        if not pair_valid:
            pass
        else:
            valid_records.append(record)

        record["image_width"] = image_size[0]
        record["image_height"] = image_size[1]
        record["image_mode"] = image_mode
        record["mask_width"] = mask_size[0]
        record["mask_height"] = mask_size[1]
        record["mask_mode"] = mask_mode
        record["audit_status"] = (
            "OK" if not issues else ";".join(issues)
        )

        if index % 50 == 0:
            print(
                f"Audited {index}/{len(records)} "
                f"{split_name} pairs..."
            )

    summary = {
        "split": split_name,
        "listed_pairs": len(records),
        "valid_pairs": len(valid_records),
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "unreadable_pairs": unreadable_pairs,
        "dimension_mismatches": dimension_mismatches,
        "image_modes": dict(image_modes),
        "mask_modes": dict(mask_modes),
        "image_resolutions": dict(resolutions),
        "mask_resolutions": dict(mask_sizes),
        "mask_ids_found": sorted(found_mask_ids),
        "invalid_mask_ids": sorted(invalid_mask_ids),
    }

    return summary, valid_records, found_mask_ids


# ---------------------------------------------------------
# 4. CHECK DUPLICATES ACROSS THE THREE SPLITS
# ---------------------------------------------------------

def check_split_overlap(
    all_records: dict[str, list[dict[str, Any]]]
) -> list[dict[str, str]]:

    assigned_to = {}
    overlaps = []

    for split_name, records in all_records.items():

        for record in records:

            image_key = record["image_relative_path"].lower()

            if image_key in assigned_to:
                overlaps.append({
                    "image": image_key,
                    "first_split": assigned_to[image_key],
                    "second_split": split_name,
                })
            else:
                assigned_to[image_key] = split_name

    return overlaps


# ---------------------------------------------------------
# 5. SAVE CSV REPORTS
# ---------------------------------------------------------

def save_csv(
    filename: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:

    with filename.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------
# 6. CLASS-PIXEL DISTRIBUTION (TRAINING ONLY)
# ---------------------------------------------------------

def save_class_distribution(
    pixel_counts: Counter,
    image_counts: Counter,
) -> None:

    total_valid_pixels = sum(pixel_counts.values())

    if total_valid_pixels == 0:
        raise RuntimeError(
            "No valid training pixels were found."
        )

    rows = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = int(pixel_counts[class_id])

        percentage = (
            100.0 * count / total_valid_pixels
        )

        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "pixel_count": count,
            "percentage_of_valid_training_pixels": round(
                percentage, 6
            ),
            "training_images_containing_class": int(
                image_counts[class_id]
            ),
        })

    save_csv(
        OUTPUT_DIR / "training_class_pixel_distribution.csv",
        rows,
        [
            "class_id",
            "class_name",
            "pixel_count",
            "percentage_of_valid_training_pixels",
            "training_images_containing_class",
        ],
    )

    # Plot training class-pixel distribution.
    names = [row["class_name"] for row in rows]
    percentages = [
        row["percentage_of_valid_training_pixels"]
        for row in rows
    ]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, percentages)

    plt.title(
        "CamVid Training Set: Class-Pixel Distribution"
    )
    plt.xlabel("Semantic class")
    plt.ylabel("Percentage of valid training pixels")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_class_distribution.png",
        dpi=160
    )

    plt.close()


# ---------------------------------------------------------
# 7. VISUALIZE TRAINING IMAGE-MASK PAIRS
# ---------------------------------------------------------

def colorize_mask(mask_array: np.ndarray) -> np.ndarray:

    height, width = mask_array.shape

    color_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for class_id, color in enumerate(CLASS_COLORS):
        color_mask[mask_array == class_id] = color

    # Void / ignored pixels are displayed in black.
    color_mask[mask_array == IGNORE_INDEX] = (0, 0, 0)

    return color_mask


def save_sample_visualization(
    training_records: list[dict[str, Any]],
    maximum_samples: int = 3,
) -> None:

    samples = training_records[:maximum_samples]

    if not samples:
        print("No valid training pairs available for visualization.")
        return

    fig, axes = plt.subplots(
        len(samples),
        3,
        figsize=(15, 4 * len(samples)),
        squeeze=False
    )

    for row, record in enumerate(samples):

        image = np.array(
            Image.open(record["image_path"]).convert("RGB")
        )

        # Preserve the original grayscale class IDs.
        mask = np.array(
            Image.open(record["mask_path"])
        )

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Original RGB image")

        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("Original grayscale mask")

        axes[row, 2].imshow(colorize_mask(mask))
        axes[row, 2].set_title("11-class colour visualization")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "sample_image_mask_pairs.png",
        dpi=160
    )

    plt.close()


# ---------------------------------------------------------
# 8. MAIN AUDIT FUNCTION
# ---------------------------------------------------------

def main() -> None:

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("CAMVID DATASET AUDIT")
    print("=" * 65)

    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output folder: {OUTPUT_DIR}")

    all_records = {}
    split_summaries = []
    all_valid_records = {}
    all_found_mask_ids = set()

    training_pixel_counts = Counter()
    training_image_counts = Counter()

    # Read and audit each split.
    for split_name, split_file in SPLITS.items():

        print(f"\nReading {split_name} split...")

        records = read_split_file(
            split_name,
            split_file
        )

        all_records[split_name] = records

        summary, valid_records, found_ids = audit_split(
            split_name,
            records,
            training_pixel_counts,
            training_image_counts,
        )

        split_summaries.append(summary)
        all_valid_records[split_name] = valid_records

        all_found_mask_ids.update(found_ids)

        print(f"\n{split_name.upper()} SUMMARY")

        print(f"Listed pairs: {summary['listed_pairs']}")
        print(f"Valid pairs: {summary['valid_pairs']}")
        print(f"Missing images: {summary['missing_images']}")
        print(f"Missing masks: {summary['missing_masks']}")
        print(f"Dimension mismatches: {summary['dimension_mismatches']}")
        print(f"Image resolutions: {summary['image_resolutions']}")
        print(f"Mask modes: {summary['mask_modes']}")
        print(f"Mask IDs found: {summary['mask_ids_found']}")
        print(f"Invalid IDs: {summary['invalid_mask_ids']}")

    # Verify that split lists do not overlap.
    overlaps = check_split_overlap(all_records)

    # Save split summary.
    save_csv(
        OUTPUT_DIR / "split_summary.csv",
        split_summaries,
        [
            "split",
            "listed_pairs",
            "valid_pairs",
            "missing_images",
            "missing_masks",
            "unreadable_pairs",
            "dimension_mismatches",
            "image_modes",
            "mask_modes",
            "image_resolutions",
            "mask_resolutions",
            "mask_ids_found",
            "invalid_mask_ids",
        ],
    )

    # Save detailed image-mask audit.
    pair_rows = []

    for records in all_records.values():
        for record in records:
            pair_rows.append({
                key: value
                for key, value in record.items()
                if key not in ("image_path", "mask_path")
            })

    save_csv(
        OUTPUT_DIR / "file_pair_audit.csv",
        pair_rows,
        [
            "split",
            "line_number",
            "image_relative_path",
            "mask_relative_path",
            "image_width",
            "image_height",
            "image_mode",
            "mask_width",
            "mask_height",
            "mask_mode",
            "audit_status",
        ],
    )

    # Save training class-pixel distribution.
    save_class_distribution(
        training_pixel_counts,
        training_image_counts,
    )

    # Visualize only training samples.
    save_sample_visualization(
        all_valid_records["train"]
    )

    # Build final JSON summary.
    total_listed = sum(
        item["listed_pairs"] for item in split_summaries
    )

    total_valid = sum(
        item["valid_pairs"] for item in split_summaries
    )

    issues = []

    for item in split_summaries:

        if (
            item["missing_images"]
            or item["missing_masks"]
            or item["unreadable_pairs"]
            or item["dimension_mismatches"]
            or item["invalid_mask_ids"]
        ):
            issues.append(item["split"])

    if overlaps:
        issues.append("split_overlap")

    final_summary = {
        "dataset": "CamVid",
        "protocol": "11 semantic classes",
        "dataset_root": str(DATASET_ROOT),
        "total_listed_pairs": total_listed,
        "total_valid_pairs": total_valid,
        "semantic_classes": {
            str(i): name
            for i, name in enumerate(CLASS_NAMES)
        },
        "ignore_index": IGNORE_INDEX,
        "valid_mask_ids_found": sorted(all_found_mask_ids),
        "split_summaries": split_summaries,
        "overlapping_images": overlaps,
        "audit_issues": issues,
        "class_pixel_distribution_policy": (
            "Class-pixel counts and EDA plots use training masks only. "
            "Validation and test masks are structurally checked; "
            "test labels are not used for modelling or tuning."
        ),
    }

    with (
        OUTPUT_DIR / "audit_summary.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            final_summary,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED")
    print("=" * 65)

    print(f"Total listed pairs: {total_listed}")
    print(f"Total valid pairs: {total_valid}")
    print(f"Images appearing in multiple splits: {len(overlaps)}")

    print("\nTRAINING CLASS DISTRIBUTION")

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = training_pixel_counts[class_id]

        print(
            f"{class_id:2d}. {class_name:12s}: "
            f"{count:,} pixels"
        )

    print("\nGenerated files:")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f" - {path.name}")

    if issues:
        print("\nWARNING: Audit issues were detected.")
        print("Affected sections:", issues)
        print(
            "Inspect file_pair_audit.csv and "
            "audit_summary.json before training."
        )
    else:
        print("\nNo structural audit issues detected.")


if __name__ == "__main__":
    main()
