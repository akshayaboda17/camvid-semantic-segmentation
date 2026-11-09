
"""
Model factory for CamVid semantic segmentation.

Architectures:
1. U-Net++
2. PSPNet

Shared configuration:
- ResNet18 encoder
- ImageNet pretrained weights
- 3 input channels
- 11 output classes
- Raw logits (no final activation)
"""

from __future__ import annotations

import importlib

import torch
from torch import nn


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

NUM_CLASSES = 11
INPUT_CHANNELS = 3

ENCODER_NAME = "resnet18"
ENCODER_WEIGHTS = "imagenet"

IMAGE_SIZE = (192, 256)

MODEL_NAMES = (
    "unetplusplus",
    "pspnet",
)


# ---------------------------------------------------------
# BUILD MODELS
# ---------------------------------------------------------

def build_model(
    model_name: str,
    pretrained: bool = True,
) -> nn.Module:
    """
    Construct the requested segmentation architecture.

    pretrained=True:
        Initialize ResNet18 encoder using ImageNet weights.

    pretrained=False:
        Initialize encoder without pretrained weights.
    """

    name = model_name.lower().strip()

    if name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from {MODEL_NAMES}."
        )

    smp = importlib.import_module(
        "segmentation_models_pytorch"
    )

    encoder_weights = (
        ENCODER_WEIGHTS if pretrained else None
    )

    if name == "unetplusplus":

        model = smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights=encoder_weights,
            encoder_depth=5,
            in_channels=INPUT_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
        )

    elif name == "pspnet":

        model = smp.PSPNet(
            encoder_name=ENCODER_NAME,
            encoder_weights=encoder_weights,
            encoder_depth=3,
            in_channels=INPUT_CHANNELS,
            classes=NUM_CLASSES,
            activation=None,
            upsampling=8,
            psp_dropout=0.2,
        )

    return model


# ---------------------------------------------------------
# PARAMETER COUNTS
# ---------------------------------------------------------

def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """
    Return:
        total parameters
        trainable parameters
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

def get_model_info(
    model: nn.Module,
    model_name: str,
) -> dict:

    total, trainable = count_parameters(model)

    return {
        "model_name": model_name,
        "encoder": ENCODER_NAME,
        "pretrained_weights": ENCODER_WEIGHTS,
        "input_channels": INPUT_CHANNELS,
        "output_classes": NUM_CLASSES,
        "input_height": IMAGE_SIZE[0],
        "input_width": IMAGE_SIZE[1],
        "total_parameters": total,
        "trainable_parameters": trainable,
    }
