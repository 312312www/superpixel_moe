"""Five-stage face anti-spoofing models with domain-generalization controls.

A: ResNet classifier.
B: A + global top-k Native MoE.
C: B + multi-scale SLIC tokens + one shared Naive (equal-weight) MoE.
D: C + normalized face-position encoding.
E: D + Fine/Medium/Coarse scale-specific Naive MoEs.

For LODO training, optional MixStyle and a gradient-reversal domain classifier
reduce source-domain style leakage without touching target labels.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import OUT_CHANNELS, build_backbone

PHASES = ("A", "B", "C", "D", "E")


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient.neg().mul(ctx.coefficient), None


def gradient_reverse(value: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(value, coefficient)


class MixStyle(nn.Module):
    """Mix feature-map channel statistics across source samples during training."""

    def __init__(self, probability: float, alpha: float) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0 or alpha <= 0.0:
            raise ValueError("MixStyle probability must be [0,1] and alpha positive")
        self.probability = float(probability)
        self.alpha = float(alpha)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0 or features.shape[0] < 2:
            return features
        if float(torch.rand((), device=features.device)) > self.probability:
            return features
        mean = features.mean(dim=(2, 3), keepdim=True)
        std = (features.var(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6).sqrt()
        normalized = (features - mean) / std
        beta = torch.distributions.Beta(self.alpha, self.alpha).sample((features.shape[0],)).to(features.device, features.dtype)
        beta = beta.reshape(-1, 1, 1, 1)
        permutation = torch.randperm(features.shape[0], device=features.device)
        mixed_mean = beta * mean + (1.0 - beta) * mean[permutation]
        mixed_std = beta * std + (1.0 - beta) * std[permutation]
        return normalized * mixed_std + mixed_mean


@dataclass(frozen=True)
class FASModelConfig:
    phase: str = "B"
    backbone: str = "resnet50"
    pretrained: bool = True
    weights_path: str | None = None
    image_size: int = 224
    num_classes: int = 2
    num_experts: int = 8
    top_k: int = 2
    expert_hidden_dim: int = 1024
    expert_dropout: float = 0.1
    moe_dropout: float = 0.1
    balance_loss_weight: float = 0.01
    superpixel_levels: tuple[int, ...] = (16, 9, 4)
    superpixel_token_dim: int = 256
    superpixel_hidden_dim: int = 512
    position_dim: int = 5
    freeze_batch_norm: bool = True
    mixstyle_prob: float = 0.0
    mixstyle_alpha: float = 0.1
    num_domains: int = 1
    domain_hidden_dim: int = 512
    head_hidden_dim: int | None = None
    head_dropout: float = 0.2

    def __post_init__(self) -> None:
        phase = str(self.phase).upper()
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {self.phase!r}")
        object.__setattr__(self, "phase", phase)
        levels = tuple(int(level) for level in self.superpixel_levels)
        if len(levels) != 3 or len(set(levels)) != 3 or any(level < 1 for level in levels):
            raise ValueError("superpixel_levels must contain three distinct positive values")
        object.__setattr__(self, "superpixel_levels", levels)
        if self.num_experts < 1 or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("num_experts >= 1 and 1 <= top_k <= num_experts required")
        if self.num_domains < 1:
            raise ValueError("num_domains must be positive")

    @property
    def use_global_moe(self) -> bool:
        return self.phase in ("B", "C", "D", "E")

    @property
    def use_superpixels(self) -> bool:
        return self.phase in ("C", "D", "E")

    @property
    def use_position_encoding(self) -> bool:
        return self.phase in ("D", "E")

    @property
    def use_scale_specific_moes(self) -> bool:
        return self.phase == "E"

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FASModelConfig":
        accepted = {field.name for field in fields(cls) if field.init}
        payload = {key: value for key, value in values.items() if key in accepted}
        if "superpixel_levels" in payload:
            payload["superpixel_levels"] = tuple(payload["superpixel_levels"])
        return cls(**payload)


class NativeMoE(nn.Module):
    """Sparse top-k routed expert MLP with a Switch-style balance penalty."""

    def __init__(self, channels: int, *, num_experts: int, top_k: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router = nn.Linear(channels, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(channels), nn.Linear(channels, hidden_dim), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(hidden_dim, channels),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim < 2:
            raise ValueError(f"expected [..., channels], got {tuple(tokens.shape)}")
        flat = tokens.reshape(-1, tokens.shape[-1])
        active_mask = (
            torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
            if token_mask is None
            else token_mask.reshape(-1).to(device=flat.device, dtype=torch.bool)
        )
        if active_mask.numel() != flat.shape[0]:
            raise ValueError("token_mask must match all token dimensions except channels")
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return torch.zeros_like(tokens), torch.zeros((), dtype=tokens.dtype, device=tokens.device)
        active = flat.index_select(0, active_indices)
        probabilities = F.softmax(self.router(active), dim=-1)
        top_probs, top_indices = probabilities.topk(self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(tokens.dtype).eps)
        routed = torch.zeros_like(active)
        for expert_index, expert in enumerate(self.experts):
            weights = torch.zeros(active.shape[0], dtype=active.dtype, device=active.device)
            for rank in range(self.top_k):
                weights = weights + (top_indices[:, rank] == expert_index).to(active.dtype) * top_probs[:, rank]
            selected = torch.nonzero(weights > 0, as_tuple=False).flatten()
            if selected.numel():
                contribution = expert(active.index_select(0, selected)) * weights.index_select(0, selected).unsqueeze(1)
                routed = torch.index_add(routed, 0, selected, contribution)
        output = torch.index_copy(torch.zeros_like(flat), 0, active_indices, routed).reshape_as(tokens)
        mean_probability = probabilities.mean(dim=0)
        hard_load = torch.zeros(self.num_experts, dtype=active.dtype, device=active.device)
        for rank in range(self.top_k):
            hard_load.scatter_add_(0, top_indices[:, rank], torch.ones_like(top_indices[:, rank], dtype=active.dtype))
        hard_load /= active.shape[0] * self.top_k
        balance = self.num_experts * torch.sum(mean_probability * hard_load)
        return output, balance


class NaiveMoE(nn.Module):
    """Equal-weight experts required by shared/scale-specific regional phases."""

    def __init__(self, channels: int, *, num_experts: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(channels), nn.Linear(channels, hidden_dim), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(hidden_dim, channels),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.stack([expert(tokens) for expert in self.experts], dim=0).mean(dim=0)
        if token_mask is not None:
            output = output * token_mask.to(output.device, output.dtype).unsqueeze(-1)
        return output, torch.zeros((), dtype=tokens.dtype, device=tokens.device)


def pool_superpixel_regions(feature_map: torch.Tensor, labels: torch.Tensor, expected_regions: int) -> torch.Tensor:
    """Vectorized average pooling from BCHW map into BxKxC superpixel tokens."""
    if feature_map.ndim != 4 or labels.ndim != 3 or labels.shape[0] != feature_map.shape[0]:
        raise ValueError("feature_map must be BCHW and labels must be matching BHW")
    batch, channels, height, width = feature_map.shape
    grid = F.interpolate(labels[:, None].float(), size=(height, width), mode="nearest").long()[:, 0]
    if int(grid.min()) < 0 or int(grid.max()) >= expected_regions:
        raise ValueError("superpixel labels must be in [0, expected_regions)")
    pixels = height * width
    indices = grid.reshape(batch, pixels)
    values = feature_map.permute(0, 2, 3, 1).reshape(batch, pixels, channels)
    sums = torch.zeros(batch, expected_regions, channels, dtype=feature_map.dtype, device=feature_map.device)
    sums.scatter_add_(1, indices[..., None].expand(-1, -1, channels), values)
    counts = torch.zeros(batch, expected_regions, 1, dtype=feature_map.dtype, device=feature_map.device)
    counts.scatter_add_(1, indices[..., None], torch.ones(batch, pixels, 1, dtype=feature_map.dtype, device=feature_map.device))
    pooled = sums / counts.clamp_min(1.0)
    fallback = feature_map.mean(dim=(2, 3)).unsqueeze(1).expand(-1, expected_regions, -1)
    return torch.where(counts > 0, pooled, fallback)


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(tokens.device, tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class FASModel(nn.Module):
    """ResNet FAS classifier implementing A--E with frozen source BN by default."""

    def __init__(self, config: FASModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or FASModelConfig()
        cfg = self.config
        if cfg.backbone not in OUT_CHANNELS:
            raise ValueError(f"unsupported backbone {cfg.backbone!r}")
        self.backbone = build_backbone(cfg.backbone, pretrained=cfg.pretrained, weights_path=cfg.weights_path)  # type: ignore[arg-type]
        channels = OUT_CHANNELS[cfg.backbone]
        self.mixstyle = MixStyle(cfg.mixstyle_prob, cfg.mixstyle_alpha)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.global_moe: NativeMoE | None = None
        if cfg.use_global_moe:
            self.global_moe = NativeMoE(channels, num_experts=cfg.num_experts, top_k=cfg.top_k, hidden_dim=cfg.expert_hidden_dim, dropout=cfg.expert_dropout)
            self.global_moe_dropout = nn.Dropout(cfg.moe_dropout)
        self.domain_classifier: nn.Module | None = None
        if cfg.num_domains > 1:
            self.domain_classifier = nn.Sequential(
                nn.Linear(channels, cfg.domain_hidden_dim), nn.GELU(), nn.Dropout(cfg.head_dropout), nn.Linear(cfg.domain_hidden_dim, cfg.num_domains)
            )

        self.region_projection: nn.Module | None = None
        self.position_encoder: nn.Module | None = None
        self.shared_region_moe: NaiveMoE | None = None
        self.scale_region_moes: nn.ModuleDict | None = None
        if cfg.use_superpixels:
            self.region_projection = nn.Sequential(
                nn.LayerNorm(channels), nn.Linear(channels, cfg.superpixel_token_dim), nn.GELU(), nn.LayerNorm(cfg.superpixel_token_dim)
            )
            if cfg.use_position_encoding:
                self.position_encoder = nn.Sequential(
                    nn.Linear(cfg.position_dim, cfg.superpixel_token_dim), nn.GELU(), nn.Linear(cfg.superpixel_token_dim, cfg.superpixel_token_dim)
                )
            if cfg.use_scale_specific_moes:
                self.scale_region_moes = nn.ModuleDict({
                    str(level): NaiveMoE(cfg.superpixel_token_dim, num_experts=cfg.num_experts, hidden_dim=cfg.superpixel_hidden_dim, dropout=cfg.expert_dropout)
                    for level in cfg.superpixel_levels
                })
            else:
                self.shared_region_moe = NaiveMoE(
                    cfg.superpixel_token_dim, num_experts=cfg.num_experts, hidden_dim=cfg.superpixel_hidden_dim, dropout=cfg.expert_dropout
                )
            self.region_dropout = nn.Dropout(cfg.moe_dropout)

        classifier_channels = channels + (len(cfg.superpixel_levels) * cfg.superpixel_token_dim if cfg.use_superpixels else 0)
        hidden = cfg.head_hidden_dim or max(channels // 2, cfg.superpixel_token_dim * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(classifier_channels), nn.Dropout(cfg.head_dropout), nn.Linear(classifier_channels, hidden),
            nn.GELU(), nn.Dropout(cfg.head_dropout), nn.Linear(hidden, cfg.num_classes),
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.config.freeze_batch_norm:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        *,
        superpixel_labels: Mapping[str, torch.Tensor] | None = None,
        superpixel_positions: Mapping[str, torch.Tensor] | None = None,
        superpixel_valid: Mapping[str, torch.Tensor] | None = None,
        domain_adversarial_coefficient: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected BCHW RGB input, got {tuple(images.shape)}")
        normalized = (images.float() / 255.0 - self.image_mean.to(images.device)) / self.image_std.to(images.device)
        feature_map = self.mixstyle(self.backbone(normalized))
        global_features = self.pool(feature_map).flatten(1)
        balances: list[torch.Tensor] = []
        if self.global_moe is not None:
            routed, balance = self.global_moe(global_features)
            global_features = self.global_moe_dropout(global_features + routed)
            balances.append(balance)

        details: dict[str, torch.Tensor] = {"global_features": global_features}
        if self.domain_classifier is not None:
            details["domain_logits"] = self.domain_classifier(gradient_reverse(global_features, domain_adversarial_coefficient))
        fusion = [global_features]
        if self.config.use_superpixels:
            if superpixel_labels is None or superpixel_valid is None:
                raise ValueError(f"phase {self.config.phase} requires superpixel labels and valid masks")
            if self.config.use_position_encoding and superpixel_positions is None:
                raise ValueError(f"phase {self.config.phase} requires superpixel positions")
            assert self.region_projection is not None
            for level in self.config.superpixel_levels:
                key = str(level)
                labels = superpixel_labels[key].to(images.device, non_blocking=True)
                valid = superpixel_valid[key].to(images.device, dtype=torch.bool, non_blocking=True)
                if tuple(valid.shape) != (images.shape[0], level):
                    raise ValueError(f"valid mask for {level} must be [B,{level}]")
                tokens = self.region_projection(pool_superpixel_regions(feature_map, labels, level))
                if self.position_encoder is not None:
                    assert superpixel_positions is not None
                    positions = superpixel_positions[key].to(images.device, dtype=tokens.dtype, non_blocking=True)
                    if tuple(positions.shape) != (images.shape[0], level, self.config.position_dim):
                        raise ValueError(f"position tensor for {level} has invalid shape")
                    tokens = tokens + self.position_encoder(positions)
                moe = self.scale_region_moes[key] if self.scale_region_moes is not None else self.shared_region_moe
                assert moe is not None
                routed, balance = moe(tokens, valid)
                tokens = self.region_dropout((tokens + routed) * valid.to(tokens.dtype).unsqueeze(-1))
                balances.append(balance)
                details[f"region_tokens_{key}"] = tokens
                fusion.append(_masked_mean(tokens, valid))

        fused = torch.cat(fusion, dim=1)
        details["features"] = fused
        details["balance_loss"] = torch.stack(balances).mean() if balances else torch.zeros((), dtype=fused.dtype, device=fused.device)
        return self.head(fused), details


def total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "FASModel", "FASModelConfig", "MixStyle", "NaiveMoE", "NativeMoE", "PHASES",
    "gradient_reverse", "pool_superpixel_regions", "total_parameters", "trainable_parameters",
]
