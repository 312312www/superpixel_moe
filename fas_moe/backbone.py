"""Shared spatial backbone for the minimal Superpixel-MoE model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def _cached_resnet50_weights() -> Path | None:
    """Find a compatible ResNet-50 checkpoint in the torch hub cache."""

    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    candidates = sorted(cache_dir.glob("resnet50-*.pth"))
    return candidates[0] if candidates else None


def _load_state_dict(network: nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    network.load_state_dict(state, strict=False)


class ResNet50Layer2(nn.Module):
    """ResNet-50 stem through layer2, returning a 32x32 feature map."""

    output_channels = 512

    def __init__(
        self,
        *,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if weights_path is not None:
            network = resnet50(weights=None)
            _load_state_dict(network, Path(weights_path))
        elif pretrained and (cached := _cached_resnet50_weights()) is not None:
            network = resnet50(weights=None)
            _load_state_dict(network, cached)
        else:
            network = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
        )
        self.freeze = bool(freeze)
        if self.freeze:
            for parameter in self.features.parameters():
                parameter.requires_grad_(False)
            self.features.eval()

    def train(self, mode: bool = True) -> "ResNet50Layer2":
        super().train(mode)
        if self.freeze:
            self.features.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected BCHW RGB input, got {tuple(images.shape)}")
        return self.features(images)


def build_backbone(
    *, pretrained: bool = True, weights_path: str | Path | None = None, freeze: bool = True
) -> ResNet50Layer2:
    """Construct the default shared backbone."""

    return ResNet50Layer2(pretrained=pretrained, weights_path=weights_path, freeze=freeze)


__all__ = ["ResNet50Layer2", "build_backbone"]
