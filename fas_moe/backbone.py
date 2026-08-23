"""Backbone for the FAS pipeline: ResNet-50 (default) or ResNet-34.

The full network (conv1..fc removed) is used and fine-tuned, unlike the old
design which froze a stem->layer2 slice.  Standard ImageNet normalization is
applied by the model on [0, 255] uint8/float input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torchvision.models import (
    ResNet34_Weights,
    ResNet50_Weights,
    resnet34,
    resnet50,
)

BackboneName = Literal["resnet50", "resnet34"]

_BUILDERS = {
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
}

# Final conv output channels for each supported backbone.
OUT_CHANNELS = {"resnet50": 2048, "resnet34": 512}


def _cached_weights_path(backbone: str) -> Path | None:
    """Return a compatible torchvision checkpoint from the hub cache if present."""
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    patterns = {"resnet50": "resnet50-*.pth", "resnet34": "resnet34-*.pth"}[backbone]
    candidates = sorted(cache_dir.glob(patterns))
    return candidates[0] if candidates else None


def build_backbone(
    name: BackboneName = "resnet50",
    *,
    pretrained: bool = True,
    weights_path: str | Path | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Build the convolutional trunk without the ImageNet classifier head."""
    if name not in _BUILDERS:
        raise ValueError(f"unsupported backbone {name!r}; choose from {sorted(_BUILDERS)}")
    builder, default_weights = _BUILDERS[name]
    if pretrained:
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {str(k).removeprefix("module."): v for k, v in state.items()}
            network = builder(weights=None)
            missing, unexpected = network.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(f"backbone weights missing keys: {sorted(missing)[:5]}")
            if unexpected:
                raise RuntimeError(f"backbone weights unexpected keys: {sorted(unexpected)[:5]}")
        else:
            network = builder(weights=default_weights)
    else:
        network = builder(weights=None)
    trunk = nn.Sequential(
        network.conv1,
        network.bn1,
        network.relu,
        network.maxpool,
        network.layer1,
        network.layer2,
        network.layer3,
        network.layer4,
    )
    if dropout > 0.0:
        trunk.add_module("dropout", nn.Dropout2d(dropout))
    return trunk


__all__ = ["OUT_CHANNELS", "BackboneName", "build_backbone"]
