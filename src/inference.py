"""
Inference and Test-Time Augmentation (TTA) for CamVid.

Evaluates one trained checkpoint using:
1. Standard inference: original image only.
2. Horizontal-flip TTA: original + flipped image, averaged probabilities.

By default, evaluates the validation split. The test split should be
reserved for the final evaluation after all choices are complete.

Run from the project root:
    python -m src.inference --checkpoint results/training/unetplusplus_composite_seed42/checkpoints/best_model.pth

Optional:
    python -m src.inference --checkpoint PATH --split validation
    python -m src.inference --checkpoint PATH --split validation --tta-only
    python -m src.inference --checkpoint PATH --output-dir results/improvements/my_tta_eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.camvid import (
    create_dataloader,
    NUM_CLASSES,
    IGNORE_INDEX,
)
from src.models import build_model
from src.metrics import SegmentationMetrics, print_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "inference"


def make_json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare standard and horizontal-flip TTA inference on CamVid."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pth or another compatible training checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Dataset split. Keep test for final evaluation only.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional CamVid dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save metrics. Defaults to results/inference/<checkpoint_name>_<split>.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="use_amp",
        help="Disable CUDA mixed precision.",
    )
    parser.add_argument(
        "--tta-only",
        action="store_true",
        help="Run TTA only; by default both standard and TTA are compared.",
    )
    parser.set_defaults(use_amp=True)

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    return args


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the checkpoint's architecture and load its trained weights."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # weights_only=False is needed because the training checkpoint contains
    # optimizer, scheduler, scaler and configuration metadata as well.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The supplied checkpoint does not contain 'model_state_dict'."
        )

    config = checkpoint.get("config", {})
    model_name = config.get("model")

    if model_name not in ("unetplusplus", "pspnet"):
        raise ValueError(
            "Could not determine model architecture from checkpoint config. "
            f"Found model={model_name!r}."
        )

    model = build_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    return model, config


def synchronize_if_cuda(device: torch.device) -> None:
    """Ensure GPU work has completed before recording elapsed time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    use_tta: bool,
) -> tuple[dict[str, Any], float, float]:
    """
    Evaluate one complete split.

    TTA combines softmax probabilities from the original image and the
    horizontally flipped image. The flipped prediction is flipped back
    before averaging.
    """
    metrics = SegmentationMetrics(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        boundary_tolerance=2,
    )

    total_model_seconds = 0.0
    total_images = 0

    for batch_index, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        synchronize_if_cuda(device)
        batch_start = time.perf_counter()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(use_amp and device.type == "cuda"),
        ):
            original_logits = model(images)

            if original_logits.shape[-2:] != masks.shape[-2:]:
                original_logits = F.interpolate(
                    original_logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if use_tta:
                flipped_images = torch.flip(images, dims=[3])
                flipped_logits = model(flipped_images)

                if flipped_logits.shape[-2:] != masks.shape[-2:]:
                    flipped_logits = F.interpolate(
                        flipped_logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # Undo the input flip in the prediction.
                flipped_logits = torch.flip(flipped_logits, dims=[3])

                # Average probabilities, not raw logits.
                original_probabilities = torch.softmax(
                    original_logits.float(), dim=1
                )
                flipped_probabilities = torch.softmax(
                    flipped_logits.float(), dim=1
                )
                predictions = (
                    original_probabilities + flipped_probabilities
                ) / 2.0
            else:
                predictions = original_logits

        synchronize_if_cuda(device)
        total_model_seconds += time.perf_counter() - batch_start

        metrics.update(predictions, masks)
        total_images += int(images.shape[0])

        if batch_index % 25 == 0:
            print(f"  Evaluated batch {batch_index}/{len(loader)}")

    if total_images == 0:
        raise RuntimeError("No images were evaluated.")

    results = metrics.compute()
    seconds_per_image = total_model_seconds / total_images

    return results, total_model_seconds, seconds_per_image


def save_results(
    results: dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> None:
    """Save aggregate metrics, per-class metrics and confusion matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        key: value
        for key, value in results.items()
        if key not in ("confusion_matrix", "per_class")
    }

    with (output_dir / f"{prefix}_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(make_json_safe(summary), file, indent=4, allow_nan=False)

    with (output_dir / f"{prefix}_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        import csv

        writer = csv.DictWriter(
            file,
            fieldnames=list(results["per_class"][0].keys()),
        )
        writer.writeheader()
        writer.writerows(
            make_json_safe(row) for row in results["per_class"]
        )

    np.savetxt(
        output_dir / f"{prefix}_confusion_matrix.csv",
        results["confusion_matrix"],
        delimiter=",",
        fmt="%d",
    )


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("CAMVID INFERENCE: STANDARD vs HORIZONTAL-FLIP TTA")
    print("=" * 72)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", checkpoint_path)
    print("Split:", args.split)

    model, checkpoint_config = load_checkpoint(checkpoint_path, device)
    model_name = checkpoint_config["model"]

    input_size = checkpoint_config.get("input_size", (192, 256))
    input_size = tuple(int(value) for value in input_size)

    print("Model:", model_name)
    print("Input size:", input_size)
    print("Mixed precision:", args.use_amp and device.type == "cuda")

    loader = create_dataloader(
        split=args.split,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=input_size,
        augment_train=False,
        num_workers=args.num_workers,
        seed=int(checkpoint_config.get("seed", 42)),
        crop_mode="none",
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (
            DEFAULT_OUTPUT_ROOT
            / f"{checkpoint_path.parent.parent.name}_{args.split}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary: dict[str, Any] = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "number_of_images": len(loader.dataset),
        "input_size": input_size,
        "device": str(device),
        "mixed_precision": bool(args.use_amp and device.type == "cuda"),
        "test_split_accessed": args.split == "test",
        "methods": {},
    }

    methods = ("tta",) if args.tta_only else ("standard", "tta")

    for method in methods:
        use_tta = method == "tta"

        print("\n" + "-" * 72)
        print("Running:", "Horizontal-flip TTA" if use_tta else "Standard inference")
        print("-" * 72)

        results, total_seconds, seconds_per_image = evaluate(
            model=model,
            loader=loader,
            device=device,
            use_amp=args.use_amp,
            use_tta=use_tta,
        )

        print_metrics(results)

        print(f"\nModel inference time: {total_seconds:.3f} seconds")
        print(f"Model inference time per image: {seconds_per_image * 1000:.3f} ms")

        save_results(results, output_dir, method)

        run_summary["methods"][method] = {
            "metrics": {
                key: value
                for key, value in results.items()
                if key not in ("confusion_matrix", "per_class")
            },
            "total_model_inference_seconds": total_seconds,
            "model_inference_ms_per_image": seconds_per_image * 1000.0,
            "per_class_metrics_file": f"{method}_class_metrics.csv",
            "confusion_matrix_file": f"{method}_confusion_matrix.csv",
        }

    if "standard" in run_summary["methods"] and "tta" in run_summary["methods"]:
        standard = run_summary["methods"]["standard"]
        tta = run_summary["methods"]["tta"]

        run_summary["tta_minus_standard"] = {
            "mean_iou": (
                tta["metrics"]["mean_iou"]
                - standard["metrics"]["mean_iou"]
            ),
            "mean_dice": (
                tta["metrics"]["mean_dice"]
                - standard["metrics"]["mean_dice"]
            ),
            "pixel_accuracy": (
                tta["metrics"]["pixel_accuracy"]
                - standard["metrics"]["pixel_accuracy"]
            ),
            "mean_boundary_f1": (
                tta["metrics"]["mean_boundary_f1"]
                - standard["metrics"]["mean_boundary_f1"]
            ),
            "inference_time_ratio": (
                tta["total_model_inference_seconds"]
                / standard["total_model_inference_seconds"]
                if standard["total_model_inference_seconds"] > 0
                else None
            ),
        }

    with (output_dir / "inference_comparison.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            make_json_safe(run_summary),
            file,
            indent=4,
            allow_nan=False,
        )

    print("\n" + "=" * 72)
    print("INFERENCE EVALUATION COMPLETED")
    print("=" * 72)
    print("Results saved to:", output_dir)
    print("Dataset split:", args.split)
    print("No training or checkpoint weights were modified.")


if __name__ == "__main__":
    main()
