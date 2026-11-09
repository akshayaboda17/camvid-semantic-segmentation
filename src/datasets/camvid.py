
"""
CamVid Dataset Loader

Reads:
- camvid_train.txt
- camvid_val.txt
- camvid_test.txt

Each entry contains:
RGB image path + corresponding grayscale mask path.

Returns:
- Image tensor: [3, H, W], float32
- Mask tensor: [H, W], int64

Semantic classes: 0 to 10
Ignore index: 255
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .transforms import CamVidTransform


# ---------------------------------------------------------
# 1. DATASET CONSTANTS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "data" / "camvid_source"
)

SPLIT_FILES = {
    "train": "camvid_train.txt",
    "validation": "camvid_val.txt",
    "test": "camvid_test.txt",
}

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

VALID_MASK_IDS = set(range(NUM_CLASSES)) | {
    IGNORE_INDEX
}


# ---------------------------------------------------------
# 2. CAMVID DATASET CLASS
# ---------------------------------------------------------

class CamVidDataset(Dataset):
    """
    PyTorch Dataset for the CamVid 11-class protocol.
    """

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        split: str = "train",
        transform: CamVidTransform | None = None,
        validate_mask_ids: bool = True,
    ) -> None:

        if split not in SPLIT_FILES:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Choose from {list(SPLIT_FILES)}."
            )

        self.dataset_root = Path(
            dataset_root or DEFAULT_DATASET_ROOT
        ).resolve()

        self.split = split

        self.transform = (
            transform
            if transform is not None
            else CamVidTransform(augment=False)
        )

        self.validate_mask_ids = validate_mask_ids

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist: "
                f"{self.dataset_root}"
            )

        split_file = (
            self.dataset_root / SPLIT_FILES[split]
        )

        if not split_file.is_file():
            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

        self.samples: list[tuple[Path, Path]] = []

        # -------------------------------------------------
        # 3. READ IMAGE-MASK PAIRS FROM SPLIT FILE.
        # -------------------------------------------------

        with split_file.open(
            "r",
            encoding="utf-8-sig"
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid line {line_number} "
                        f"in {split_file.name}: {line}"
                    )

                image_relative, mask_relative = parts

                image_relative = image_relative.replace(
                    "\\", "/"
                )

                mask_relative = mask_relative.replace(
                    "\\", "/"
                )

                image_path = (
                    self.dataset_root
                    / Path(*image_relative.split("/"))
                )

                mask_path = (
                    self.dataset_root
                    / Path(*mask_relative.split("/"))
                )

                self.samples.append(
                    (image_path, mask_path)
                )

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {split_file}"
            )

        print(
            f"Loaded {len(self.samples)} "
            f"{self.split} image-mask pairs."
        )

    def __len__(self) -> int:
        """Return the number of image-mask pairs."""

        return len(self.samples)

    def __getitem__(
        self,
        index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load one image-mask pair and transform it.
        """

        image_path, mask_path = self.samples[index]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        # Load original RGB image.
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        # Read mask without changing its class IDs.
        with Image.open(mask_path) as mask_file:
            mask = mask_file.copy()

        if mask.mode != "L":
            raise ValueError(
                f"Expected grayscale mask (L mode), "
                f"but got {mask.mode}: {mask_path}"
            )

        mask_array = np.asarray(mask)

        if mask_array.ndim != 2:
            raise ValueError(
                f"Expected a 2D segmentation mask: "
                f"{mask_path}"
            )

        # Optional runtime verification of mask IDs.
        if self.validate_mask_ids:

            unique_ids = set(
                int(value)
                for value in np.unique(mask_array)
            )

            invalid_ids = (
                unique_ids - VALID_MASK_IDS
            )

            if invalid_ids:
                raise ValueError(
                    f"Invalid mask IDs {sorted(invalid_ids)} "
                    f"in {mask_path}"
                )

        # Apply cropping, resizing, augmentation and normalization.
        image_tensor, mask_tensor = self.transform(
            image,
            mask
        )

        return image_tensor, mask_tensor


# ---------------------------------------------------------
# 4. DETERMINISTIC WORKER SEEDING
# ---------------------------------------------------------

def seed_worker(worker_id: int) -> None:
    """
    Give each DataLoader worker a reproducible random seed.
    """

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------
# 5. CREATE A PYTORCH DATALOADER
# ---------------------------------------------------------

def create_dataloader(
    split: str,
    dataset_root: str | Path | None = None,
    batch_size: int = 2,
    image_size: tuple[int, int] = (192, 256),
    augment_train: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    crop_mode: str = "none",
    crop_size: tuple[int, int] = (360, 480),
) -> DataLoader:
    """
    Create a DataLoader for train, validation or test.

    Training:
        Shuffling + optional augmentation + selected crop mode.

    Validation / test:
        No augmentation, no cropping and no shuffling.

    crop_mode:
        "none"        - Original full-image preprocessing.
        "random"      - Random training crops.
        "class_aware" - Crops centred around rare-class pixels.
    """

    if split not in SPLIT_FILES:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Choose from {list(SPLIT_FILES)}."
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if crop_mode not in ("none", "random", "class_aware"):
        raise ValueError(
            "crop_mode must be 'none', 'random', or 'class_aware'."
        )

    is_training = split == "train"

    # Cropping and augmentation are restricted to training.
    effective_crop_mode = (
        crop_mode if is_training and augment_train else "none"
    )

    transform = CamVidTransform(
        image_size=image_size,
        augment=(
            is_training and augment_train
        ),
        crop_mode=effective_crop_mode,
        crop_size=crop_size,
    )

    dataset = CamVidDataset(
        dataset_root=dataset_root,
        split=split,
        transform=transform,
        validate_mask_ids=True,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=(
            seed_worker if num_workers > 0 else None
        ),
        generator=generator,
    )

    return loader
