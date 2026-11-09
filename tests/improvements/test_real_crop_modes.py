
import torch

from src.datasets.camvid import create_dataloader


def test_training_crop_modes():

    for mode in ("none", "random", "class_aware"):

        print("\n" + "=" * 55)
        print(f"TESTING TRAINING CROP MODE: {mode}")
        print("=" * 55)

        loader = create_dataloader(
            split="train",
            batch_size=2,
            image_size=(192, 256),
            augment_train=True,
            num_workers=0,
            seed=42,
            crop_mode=mode,
            crop_size=(360, 480),
        )

        images, masks = next(iter(loader))

        print("Image shape:", tuple(images.shape))
        print("Mask shape:", tuple(masks.shape))
        print("Image dtype:", images.dtype)
        print("Mask dtype:", masks.dtype)

        print(
            "Mask IDs:",
            torch.unique(masks).tolist()
        )

        assert images.shape == (2, 3, 192, 256)
        assert masks.shape == (2, 192, 256)
        assert images.dtype == torch.float32
        assert masks.dtype == torch.int64

    print("\nALL TRAINING CROP MODE TESTS PASSED.")


def test_validation_remains_unchanged():

    print("\n" + "=" * 55)
    print("TESTING VALIDATION PREPROCESSING")
    print("=" * 55)

    # Even if class-aware cropping is requested,
    # validation must use the original full-image pipeline.
    loader = create_dataloader(
        split="validation",
        batch_size=2,
        augment_train=True,
        num_workers=0,
        seed=42,
        crop_mode="class_aware",
    )

    images, masks = next(iter(loader))

    print("Validation image shape:", tuple(images.shape))
    print("Validation mask shape:", tuple(masks.shape))

    assert images.shape == (2, 3, 192, 256)
    assert masks.shape == (2, 192, 256)

    print("Validation preprocessing test PASSED.")


if __name__ == "__main__":

    test_training_crop_modes()
    test_validation_remains_unchanged()

    print("\nALL REAL DATASET TESTS PASSED.")
