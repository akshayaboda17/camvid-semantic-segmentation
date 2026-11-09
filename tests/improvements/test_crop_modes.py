
import numpy as np
import torch
from PIL import Image

from src.datasets.transforms import CamVidTransform

# Create a synthetic 960 x 720 RGB image.
image = Image.fromarray(
    np.random.randint(0, 256, (720, 960, 3), dtype=np.uint8),
    mode="RGB"
)

# Create a mask with all 11 classes and ignore pixels.
mask_array = np.zeros((720, 960), dtype=np.uint8)

mask_array[100:200, 100:200] = 2
mask_array[250:350, 300:400] = 6
mask_array[400:500, 500:600] = 7
mask_array[200:300, 650:750] = 9
mask_array[300:400, 800:900] = 10
mask_array[0:20, :] = 255

mask = Image.fromarray(mask_array, mode="L")

for mode in ("none", "random", "class_aware"):

    transform = CamVidTransform(
        image_size=(192, 256),
        augment=True,
        crop_mode=mode,
        crop_size=(360, 480)
    )

    image_tensor, mask_tensor = transform(image, mask)

    print(f"\nCrop mode: {mode}")
    print("Image shape:", tuple(image_tensor.shape))
    print("Mask shape:", tuple(mask_tensor.shape))
    print("Mask dtype:", mask_tensor.dtype)
    print("Mask IDs:", torch.unique(mask_tensor).tolist())

    assert image_tensor.shape == (3, 192, 256)
    assert mask_tensor.shape == (192, 256)
    assert mask_tensor.dtype == torch.int64

print("\nALL CROP MODE TESTS PASSED.")
