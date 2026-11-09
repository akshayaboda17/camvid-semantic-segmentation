
"""
Smoke test for CamVid segmentation architectures.

Checks:
- Model construction
- Pretrained encoder initialization
- Input/output dimensions
- Total and trainable parameters
- GPU inference memory

Run on a CUDA-enabled runtime:

    python -m src.test_models
"""

import gc
import json
from pathlib import Path

import torch

from src.models import (
    build_model,
    get_model_info,
    MODEL_NAMES,
    IMAGE_SIZE,
)


def test_model(
    model_name: str,
    device: torch.device,
) -> dict:

    print("\n" + "=" * 60)
    print(f"TESTING {model_name.upper()}")
    print("=" * 60)

    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.reset_peak_memory_stats()

    # Build model with ImageNet-pretrained encoder.
    model = build_model(
        model_name,
        pretrained=True,
    ).to(device)

    model.eval()

    info = get_model_info(
        model,
        model_name,
    )

    print(
        f"Total parameters: "
        f"{info['total_parameters']:,}"
    )

    print(
        f"Trainable parameters: "
        f"{info['trainable_parameters']:,}"
    )

    # Create two sample RGB images.
    batch_size = 2

    dummy_input = torch.randn(
        batch_size,
        3,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        device=device,
    )

    torch.cuda.synchronize()

    # Forward pass without gradients.
    with torch.inference_mode():

        output = model(dummy_input)

    torch.cuda.synchronize()

    expected_shape = (
        batch_size,
        11,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
    )

    print("Input shape:", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))

    assert tuple(output.shape) == expected_shape, (
        f"Unexpected output shape for {model_name}: "
        f"{tuple(output.shape)}"
    )

    assert torch.isfinite(output).all(), (
        f"{model_name} produced NaN or infinite outputs."
    )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 2)
    )

    info["forward_output_shape"] = list(output.shape)
    info["forward_peak_memory_mb"] = round(
        peak_memory_mb,
        2,
    )
    info["forward_test_passed"] = True

    print(
        f"Forward-pass peak GPU memory: "
        f"{peak_memory_mb:.2f} MB"
    )

    print(f"{model_name} forward test passed.")

    # Free memory before testing the next model.
    del dummy_input
    del output
    del model

    gc.collect()
    torch.cuda.empty_cache()

    return info


def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Run this script in Colab "
            "with an NVIDIA GPU runtime."
        )

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)

    results = []

    for model_name in MODEL_NAMES:

        result = test_model(
            model_name,
            device,
        )

        results.append(result)

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "model_checks"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        output_dir / "model_smoke_test.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("ALL MODEL FORWARD TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("Saved model information to:", result_path)


if __name__ == "__main__":
    main()
