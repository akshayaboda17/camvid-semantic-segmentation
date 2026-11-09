
"""
Reusable training pipeline for CamVid semantic segmentation.

Models:
    - U-Net++
    - PSPNet

Supported loss functions:
    - Cross-Entropy
    - Dice
    - Focal
    - Lovasz-Softmax
    - Boundary

Features:
    - GPU training
    - Mixed precision (FP16)
    - AdamW optimizer
    - Learning-rate scheduler
    - Validation after every epoch
    - mIoU-based best checkpoint selection
    - Latest checkpoint saving for resuming
    - CSV training history
    - JSON experiment configuration and summary

Run from the project root:

    python -m src.train --model unetplusplus

    python -m src.train --model pspnet --loss cross_entropy

    python -m src.train --model unetplusplus --loss dice

"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.camvid import (
    create_dataloader,
    NUM_CLASSES,
    IGNORE_INDEX,
)

from src.models import (
    build_model,
    get_model_info,
)

from src.losses import (
    build_loss,
    LOSS_NAMES,
)

from src.metrics import SegmentationMetrics


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "results" / "training"
)

IMAGE_SIZE = (192, 256)

DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 10

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4

DEFAULT_SEED = 42

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

LOSS_DISPLAY_NAMES = {
    "cross_entropy": "Cross-Entropy",
    "dice": "Dice",
    "focal": "Focal",
    "lovasz": "Lovasz-Softmax",
    "boundary": "Boundary",
}

LOSS_DIRECTORY_NAMES = {
    "cross_entropy": "ce",
    "dice": "dice",
    "focal": "focal",
    "lovasz": "lovasz",
    "boundary": "boundary",
}


def normalize_saved_loss_name(loss_name: str | None) -> str:
    """Normalize loss names stored by older versions of train.py."""

    if loss_name in (None, "CrossEntropyLoss", "ce"):
        # Older checkpoints were created before selectable losses existed.
        # That version trained with Cross-Entropy only.
        return "cross_entropy"

    return loss_name


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

def seed_everything(seed: int) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings improve reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed: {seed}")


# =========================================================
# 3. SAFE JSON SERIALIZATION
# =========================================================

def make_json_safe(value: Any) -> Any:
    """
    Convert NumPy values and non-finite floats into
    JSON-compatible Python values.
    """

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# =========================================================
# 4. SAVE CSV TRAINING HISTORY
# =========================================================

def save_history(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:

    if not history:
        return

    fieldnames = list(history[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


# =========================================================
# 5. TRAINING EPOCH
# =========================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    """
    Train the model for one complete epoch.

    Returns mean loss weighted by valid pixel count.
    """

    model.train()

    total_loss = 0.0
    total_valid_pixels = 0

    for batch_index, (images, masks) in enumerate(
        loader,
        start=1,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        valid_pixels = int(
            (masks != IGNORE_INDEX).sum().item()
        )

        if valid_pixels == 0:
            continue

        optimizer.zero_grad(
            set_to_none=True
        )

        # Mixed-precision forward pass.
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(images)

            # Make sure output and mask dimensions match.
            if logits.shape[-2:] != masks.shape[-2:]:

                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = criterion(
                logits,
                masks,
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at batch "
                f"{batch_index}."
            )

        # Scaled backpropagation.
        scaler.scale(loss).backward()

        # Unscale before gradient clipping.
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += (
            loss.detach().float().item()
            * valid_pixels
        )

        total_valid_pixels += valid_pixels

        if batch_index % 25 == 0:

            print(
                f"  Training batch {batch_index}/"
                f"{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    if total_valid_pixels == 0:
        raise RuntimeError(
            "No valid training pixels were processed."
        )

    return total_loss / total_valid_pixels


# =========================================================
# 6. VALIDATION EPOCH
# =========================================================

@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, dict[str, Any]]:
    """
    Evaluate on the validation set.

    The test split is never loaded by this function.
    """

    model.eval()

    total_loss = 0.0
    total_valid_pixels = 0

    metric_calculator = SegmentationMetrics(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        boundary_tolerance=2,
    )

    for batch_index, (images, masks) in enumerate(
        loader,
        start=1,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        valid_pixels = int(
            (masks != IGNORE_INDEX).sum().item()
        )

        if valid_pixels == 0:
            continue

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(images)

            if logits.shape[-2:] != masks.shape[-2:]:

                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = criterion(
                logits,
                masks,
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss "
                f"at batch {batch_index}."
            )

        total_loss += (
            loss.float().item() * valid_pixels
        )

        total_valid_pixels += valid_pixels

        # Accumulate metrics over the entire validation set.
        metric_calculator.update(
            logits,
            masks,
        )

    if total_valid_pixels == 0:
        raise RuntimeError(
            "No valid validation pixels were processed."
        )

    mean_validation_loss = (
        total_loss / total_valid_pixels
    )

    results = metric_calculator.compute()

    return mean_validation_loss, results


# =========================================================
# 7. CHECKPOINT SAVING
# =========================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    scaler,
    epoch: int,
    best_miou: float,
    config: dict[str, Any],
) -> None:
    """
    Save model and optimizer state for recovery.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_miou": best_miou,
        "config": config,
    }

    torch.save(
        checkpoint,
        path,
    )


# =========================================================
# 8. SAVE VALIDATION METRICS
# =========================================================

def save_validation_metrics(
    results: dict[str, Any],
    output_dir: Path,
) -> None:

    results_without_matrix = {
        key: value
        for key, value in results.items()
        if key != "confusion_matrix"
    }

    safe_results = make_json_safe(
        results_without_matrix
    )

    with (
        output_dir / "best_validation_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            safe_results,
            file,
            indent=4,
            allow_nan=False,
        )

    np.savetxt(
        output_dir / "best_confusion_matrix.csv",
        results["confusion_matrix"],
        delimiter=",",
        fmt="%d",
    )

    class_rows = results["per_class"]

    if class_rows:

        with (
            output_dir / "best_class_metrics.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(class_rows[0].keys()),
            )

            writer.writeheader()

            writer.writerows([
                make_json_safe(row)
                for row in class_rows
            ])


# =========================================================
# 9. PARSE TRAINING ARGUMENTS
# =========================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train U-Net++ or PSPNet on CamVid."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["unetplusplus", "pspnet"],
        help="Segmentation architecture to train.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        choices=LOSS_NAMES,
        help=(
            "Loss function: cross_entropy, dice, focal, "
            "lovasz, or boundary. Default: cross_entropy."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="use_amp",
        help="Disable mixed precision.",
    )

    parser.set_defaults(
        use_amp=True
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("epochs must be positive.")

    if args.batch_size <= 0:
        parser.error("batch size must be positive.")

    if args.num_workers < 0:
        parser.error("num_workers cannot be negative.")

    if args.learning_rate <= 0:
        parser.error("learning rate must be positive.")

    return args


# =========================================================
# 10. MAIN TRAINING FUNCTION
# =========================================================

def main() -> None:

    args = parse_args()

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU not available. "
            "Run training in Google Colab with a GPU runtime."
        )

    device = torch.device("cuda")

    seed_everything(args.seed)

    use_amp = args.use_amp

    print("\n" + "=" * 70)
    print("CAMVID SEMANTIC SEGMENTATION TRAINING")
    print("=" * 70)

    print("Model:", args.model)
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Input size:", IMAGE_SIZE)
    print("Number of classes:", NUM_CLASSES)
    print("Optimizer: AdamW")
    print("Learning rate:", args.learning_rate)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Mixed precision:", use_amp)
    print("Loss:", LOSS_DISPLAY_NAMES.get(args.loss, args.loss))
    print("Ignore index:", IGNORE_INDEX)

    # -----------------------------------------------------
    # OUTPUT DIRECTORY
    # -----------------------------------------------------

    if args.output_dir:

        output_dir = Path(
            args.output_dir
        )

    else:

        output_dir = (
            DEFAULT_RESULTS_DIR
            / f"{args.model}_{LOSS_DIRECTORY_NAMES.get(args.loss, args.loss)}_seed{args.seed}"
        )

    checkpoint_dir = (
        output_dir / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # DATA LOADERS
    # -----------------------------------------------------

    print("\nLoading CamVid datasets...")

    train_loader = create_dataloader(
        split="train",
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=IMAGE_SIZE,
        augment_train=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    validation_loader = create_dataloader(
        split="validation",
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=IMAGE_SIZE,
        augment_train=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(validation_loader)
    )

    print(
        "Test split: Not accessed."
    )

    # -----------------------------------------------------
    # BUILD MODEL
    # -----------------------------------------------------

    print("\nBuilding model...")

    model = build_model(
        args.model,
        pretrained=True,
    ).to(device)

    model_info = get_model_info(
        model,
        args.model,
    )

    print(
        "Total parameters:",
        f"{model_info['total_parameters']:,}"
    )

    print(
        "Trainable parameters:",
        f"{model_info['trainable_parameters']:,}"
    )

    # -----------------------------------------------------
    # LOSS AND OPTIMIZER
    # -----------------------------------------------------

    criterion = build_loss(
        loss_name=args.loss,
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # -----------------------------------------------------
    # EXPERIMENT CONFIGURATION
    # -----------------------------------------------------

    config = {
        "model": args.model,
        "encoder": "resnet18",
        "encoder_weights": "imagenet",
        "classes": NUM_CLASSES,
        "input_size": IMAGE_SIZE,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "loss_display_name": LOSS_DISPLAY_NAMES.get(args.loss, args.loss),
        "ignore_index": IGNORE_INDEX,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "mixed_precision": use_amp,
        "dataset": "CamVid",
        "training_images": 367,
        "validation_images": 101,
        "test_accessed": False,
        "device": torch.cuda.get_device_name(0),
    }

    # -----------------------------------------------------
    # RESUME SUPPORT
    # -----------------------------------------------------

    start_epoch = 1
    best_miou = -float("inf")

    if args.resume:

        resume_path = Path(
            args.resume
        )

        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {resume_path}"
            )

        print("\nResuming from:", resume_path)

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        checkpoint_config = checkpoint.get("config", {})

        saved_model = checkpoint_config.get("model")
        if saved_model is not None and saved_model != args.model:
            raise ValueError(
                "Checkpoint model does not match the requested model. "
                f"Checkpoint: {saved_model}; requested: {args.model}."
            )

        saved_loss = normalize_saved_loss_name(
            checkpoint_config.get("loss")
        )
        if saved_loss != args.loss:
            raise ValueError(
                "Checkpoint loss does not match the requested loss. "
                f"Checkpoint: {saved_loss}; requested: {args.loss}. "
                "Use the same --loss value used for the original run."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        best_miou = checkpoint["best_miou"]

        print(
            "Resuming at epoch:",
            start_epoch,
        )

        print(
            "Previous best mIoU:",
            best_miou,
        )

    # Write the configuration only after any resume checkpoint has been
    # validated, so a mismatched resume request cannot overwrite metadata.
    with (
        output_dir / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # TRAINING HISTORY
    # -----------------------------------------------------

    history: list[dict[str, Any]] = []

    history_path = (
        output_dir / "history.csv"
    )

    if history_path.is_file() and args.resume:

        with history_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            history = [
                row for row in csv.DictReader(file)
            ]

    best_checkpoint = (
        checkpoint_dir / "best_model.pth"
    )

    last_checkpoint = (
        checkpoint_dir / "last_checkpoint.pth"
    )

    if last_checkpoint.is_file() and not args.resume:
        raise FileExistsError(
            f"A checkpoint already exists in {checkpoint_dir}. "
            "Use --resume with that checkpoint, or choose a new --output-dir."
        )

    # -----------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------

    total_training_start = time.perf_counter()

    print("\nStarting training...")

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        print("\n" + "-" * 70)
        print(
            f"EPOCH {epoch}/{args.epochs}"
        )
        print("-" * 70)

        epoch_start = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # One training pass through all training batches.
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        # One evaluation pass through validation batches.
        validation_loss, validation_results = (
            validate_one_epoch(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )
        )

        validation_miou = (
            validation_results["mean_iou"]
        )

        if not np.isfinite(validation_miou):
            raise RuntimeError(
                "Validation mIoU is not finite."
            )

        # The scheduler changes learning rate using validation mIoU.
        scheduler.step(
            validation_miou
        )

        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_time = (
            time.perf_counter() - epoch_start
        )

        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

        current_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_miou": validation_miou,
            "validation_dice": validation_results["mean_dice"],
            "validation_precision": validation_results["mean_precision"],
            "validation_recall": validation_results["mean_recall"],
            "validation_pixel_accuracy": validation_results["pixel_accuracy"],
            "validation_boundary_f1": validation_results["mean_boundary_f1"],
            "learning_rate": learning_rate,
            "epoch_time_seconds": epoch_time,
            "peak_gpu_memory_mb": peak_memory_mb,
        }

        history.append(
            current_row
        )

        save_history(
            history,
            history_path,
        )

        print("\nEPOCH RESULTS")

        print(f"Training loss: {train_loss:.5f}")
        print(f"Validation loss: {validation_loss:.5f}")

        print(
            f"Validation mIoU: "
            f"{validation_miou * 100:.2f}%"
        )

        print(
            f"Validation Dice: "
            f"{validation_results['mean_dice'] * 100:.2f}%"
        )

        print(
            f"Validation Pixel Accuracy: "
            f"{validation_results['pixel_accuracy'] * 100:.2f}%"
        )

        print(
            f"Learning rate: {learning_rate:.7f}"
        )

        print(
            f"Epoch time: {epoch_time:.2f} seconds"
        )

        print(
            f"Peak GPU memory: {peak_memory_mb:.2f} MB"
        )

        # -------------------------------------------------
        # SAVE BEST MODEL
        # -------------------------------------------------

        if validation_miou > best_miou:

            best_miou = validation_miou

            save_checkpoint(
                path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_miou=best_miou,
                config=config,
            )

            save_validation_metrics(
                validation_results,
                output_dir,
            )

            print(
                "\nNew best model saved. "
                f"Validation mIoU: {best_miou * 100:.2f}%"
            )

        # -------------------------------------------------
        # SAVE LATEST CHECKPOINT EVERY EPOCH
        # -------------------------------------------------

        save_checkpoint(
            path=last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_miou=best_miou,
            config=config,
        )

        print(
            "Latest checkpoint saved."
        )

    total_training_time = (
        time.perf_counter() - total_training_start
    )

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    # Sum recorded epoch durations so a resumed experiment includes all
    # completed epochs, not only the latest Colab session.
    recorded_epoch_times = [
        float(row["epoch_time_seconds"])
        for row in history
        if row.get("epoch_time_seconds") not in (None, "")
    ]
    total_recorded_training_time = float(sum(recorded_epoch_times))

    summary = {
        "model": args.model,
        "loss": args.loss,
        "loss_display_name": LOSS_DISPLAY_NAMES.get(args.loss, args.loss),
        "best_validation_miou": best_miou,
        "epochs_completed": len(history),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "total_training_time_seconds": total_recorded_training_time,
        "this_session_elapsed_seconds": total_training_time,
        "mean_epoch_time_seconds": (
            float(np.mean(recorded_epoch_times))
            if recorded_epoch_times else None
        ),
        "history_file": str(history_path),
        "test_split_accessed": False,
        "model_information": model_info,
    }

    with (
        output_dir / "training_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            make_json_safe(summary),
            file,
            indent=4,
            allow_nan=False,
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print("Model:", args.model)
    print("Loss:", LOSS_DISPLAY_NAMES.get(args.loss, args.loss))

    print(
        f"Best validation mIoU: "
        f"{best_miou * 100:.2f}%"
    )

    print(
        f"Total recorded epoch time: "
        f"{total_recorded_training_time / 60:.2f} minutes"
    )

    print("Results directory:", output_dir)

    print("\nThe test split was not accessed.")


# =========================================================
# 11. SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
