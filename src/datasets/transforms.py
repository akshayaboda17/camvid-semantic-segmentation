
"""
CamVid image and mask transformations.

Images:
- Resize using bilinear interpolation.
- Apply optional training augmentation.
- Convert to float tensors.
- Normalize using ImageNet statistics.

Masks:
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
# These are used when working with ImageNet-pretrained encoders.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CamVidTransform:
    """
    Apply the same spatial transformations to an RGB image
    and its corresponding segmentation mask.

    image_size uses (height, width) order.
    """

    image_size: tuple[int, int] = (192, 256)
    augment: bool = False

    def __post_init__(self) -> None:
        height, width = self.image_size

        if height <= 0 or width <= 0:
            raise ValueError("Image dimensions must be positive.")

        self.mean = np.asarray(
            IMAGENET_MEAN,
            dtype=np.float32
        ).reshape(1, 1, 3)

        self.std = np.asarray(
            IMAGENET_STD,
            dtype=np.float32
        ).reshape(1, 1, 3)

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        height, width = self.image_size

        # -------------------------------------------------
        # 1. Check that image and mask are aligned.
        # -------------------------------------------------

        if image.size != mask.size:
            raise ValueError(
                f"Image size {image.size} does not match "
                f"mask size {mask.size}."
            )

        # -------------------------------------------------
        # 2. Resize image and mask.
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
        # 3. Apply training-only augmentation.
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

            # Colour transformations affect only RGB images.
            # Masks must never receive colour transformations.
            if random.random() < 0.5:
                brightness = random.uniform(0.9, 1.1)

                image = ImageEnhance.Brightness(
                    image
                ).enhance(brightness)

            if random.random() < 0.5:
                contrast = random.uniform(0.9, 1.1)

                image = ImageEnhance.Contrast(
                    image
                ).enhance(contrast)

            if random.random() < 0.5:
                saturation = random.uniform(0.9, 1.1)

                image = ImageEnhance.Color(
                    image
                ).enhance(saturation)

        # -------------------------------------------------
        # 4. Convert RGB image to NumPy array.
        # -------------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # Normalize every RGB channel.
        image_array = (
            image_array - self.mean
        ) / self.std

        # Change layout from H x W x C to C x H x W.
        image_array = np.transpose(
            image_array,
            (2, 0, 1)
        )

        image_array = np.ascontiguousarray(
            image_array,
            dtype=np.float32
        )

        # -------------------------------------------------
        # 5. Convert mask to integer class IDs.
        # -------------------------------------------------

        # Do not divide by 255 or normalize the mask.
        mask_array = np.asarray(
            mask,
            dtype=np.int64
        )

        mask_array = np.ascontiguousarray(
            mask_array
        )

        # -------------------------------------------------
        # 6. Convert both arrays to PyTorch tensors.
        # -------------------------------------------------

        image_tensor = torch.from_numpy(
            image_array
        )

        mask_tensor = torch.from_numpy(
            mask_array
        ).long()

        return image_tensor, mask_tensor
