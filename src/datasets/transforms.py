
"""
CamVid image and mask transformations.

Images:
- Optional random or class-aware cropping during training.
- Resize using bilinear interpolation.
- Apply optional training augmentation.
- Convert to float tensors.
- Normalize using ImageNet statistics.

Masks:
- Apply exactly the same spatial transformations as images.
- Resize using nearest-neighbour interpolation.
- Preserve integer class IDs.
- Convert to long tensors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from PIL import Image, ImageEnhance


# ImageNet normalization values.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# CamVid rare classes selected for class-aware cropping.
# 2: Pole
# 6: SignSymbol
# 7: Fence
# 9: Pedestrian
# 10: Bicyclist
RARE_CLASS_IDS = (2, 6, 7, 9, 10)


@dataclass
class CamVidTransform:
    """
    Apply identical spatial transformations to an RGB image
    and its corresponding segmentation mask.

    image_size: (height, width) of model input.
    crop_size: (height, width) of crop in original image.
    crop_mode: "none", "random", or "class_aware".

    Cropping is applied only when augment=True.
    """

    image_size: tuple[int, int] = (192, 256)
    augment: bool = False

    crop_mode: str = "none"
    crop_size: tuple[int, int] = (360, 480)
    rare_class_ids: tuple[int, ...] = RARE_CLASS_IDS

    def __post_init__(self) -> None:

        height, width = self.image_size
        crop_height, crop_width = self.crop_size

        if height <= 0 or width <= 0:
            raise ValueError("Image dimensions must be positive.")

        if crop_height <= 0 or crop_width <= 0:
            raise ValueError("Crop dimensions must be positive.")

        if self.crop_mode not in ("none", "random", "class_aware"):
            raise ValueError(
                "crop_mode must be 'none', 'random', or 'class_aware'."
            )

        if not self.rare_class_ids:
            raise ValueError("rare_class_ids cannot be empty.")

        self.mean = np.asarray(
            IMAGENET_MEAN,
            dtype=np.float32
        ).reshape(1, 1, 3)

        self.std = np.asarray(
            IMAGENET_STD,
            dtype=np.float32
        ).reshape(1, 1, 3)

    def _random_crop(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:

        image_width, image_height = image.size
        crop_height, crop_width = self.crop_size

        # Do not make a crop larger than the original image.
        crop_width = min(crop_width, image_width)
        crop_height = min(crop_height, image_height)

        # Randomly select the top-left corner.
        left = random.randint(0, image_width - crop_width)
        top = random.randint(0, image_height - crop_height)

        box = (
            left,
            top,
            left + crop_width,
            top + crop_height
        )

        return image.crop(box), mask.crop(box)

    def _class_aware_crop(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:

        mask_array = np.asarray(mask, dtype=np.int64)

        # Find rare classes actually present in this image.
        available_classes = [
            class_id
            for class_id in self.rare_class_ids
            if np.any(mask_array == class_id)
        ]

        # If no rare class is present, use an ordinary random crop.
        if not available_classes:
            return self._random_crop(image, mask)

        # Select one available rare class.
        selected_class = random.choice(available_classes)

        # Find all pixel locations belonging to that class.
        coordinates = np.argwhere(mask_array == selected_class)

        # Randomly select one pixel of the selected class.
        selected_pixel = coordinates[
            random.randrange(len(coordinates))
        ]

        y, x = int(selected_pixel[0]), int(selected_pixel[1])

        image_width, image_height = image.size
        crop_height, crop_width = self.crop_size

        crop_width = min(crop_width, image_width)
        crop_height = min(crop_height, image_height)

        # Centre the crop around the selected rare-class pixel.
        # Clamp the coordinates to remain inside the image.
        left = x - crop_width // 2
        top = y - crop_height // 2

        left = max(0, min(left, image_width - crop_width))
        top = max(0, min(top, image_height - crop_height))

        box = (
            left,
            top,
            left + crop_width,
            top + crop_height
        )

        return image.crop(box), mask.crop(box)

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        height, width = self.image_size

        # -------------------------------------------------
        # 1. Check image-mask alignment.
        # -------------------------------------------------

        if image.size != mask.size:
            raise ValueError(
                f"Image size {image.size} does not match "
                f"mask size {mask.size}."
            )

        # -------------------------------------------------
        # 2. Apply training-only cropping.
        # -------------------------------------------------

        # Validation and test transforms use augment=False.
        # Therefore, they always retain the original full image.
        if self.augment:

            if self.crop_mode == "random":
                image, mask = self._random_crop(image, mask)

            elif self.crop_mode == "class_aware":
                image, mask = self._class_aware_crop(image, mask)

        # -------------------------------------------------
        # 3. Resize image and mask.
        # -------------------------------------------------

        image = image.resize(
            (width, height),
            resample=Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (width, height),
            resample=Image.Resampling.NEAREST
        )

        # -------------------------------------------------
        # 4. Apply existing training augmentations.
        # -------------------------------------------------

        if self.augment:

            # Random horizontal flip.
            if random.random() < 0.5:
                image = image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

                mask = mask.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

            # Brightness adjustment.
            if random.random() < 0.5:
                brightness = random.uniform(0.9, 1.1)

                image = ImageEnhance.Brightness(
                    image
                ).enhance(brightness)

            # Contrast adjustment.
            if random.random() < 0.5:
                contrast = random.uniform(0.9, 1.1)

                image = ImageEnhance.Contrast(
                    image
                ).enhance(contrast)

            # Saturation adjustment.
            if random.random() < 0.5:
                saturation = random.uniform(0.9, 1.1)

                image = ImageEnhance.Color(
                    image
                ).enhance(saturation)

        # -------------------------------------------------
        # 5. Convert RGB image to NumPy array.
        # -------------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image_array = (
            image_array - self.mean
        ) / self.std

        # H x W x C -> C x H x W.
        image_array = np.transpose(
            image_array,
            (2, 0, 1)
        )

        image_array = np.ascontiguousarray(
            image_array,
            dtype=np.float32
        )

        # -------------------------------------------------
        # 6. Convert mask to integer class IDs.
        # -------------------------------------------------

        # Never normalize or divide mask values by 255.
        mask_array = np.asarray(
            mask,
            dtype=np.int64
        )

        mask_array = np.ascontiguousarray(
            mask_array
        )

        # -------------------------------------------------
        # 7. Convert arrays to PyTorch tensors.
        # -------------------------------------------------

        image_tensor = torch.from_numpy(image_array)

        mask_tensor = torch.from_numpy(
            mask_array
        ).long()

        return image_tensor, mask_tensor
