"""Minimal region-pooling and equal-weight MoE classifier."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Mapping
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .backbone import ResNet50Layer2, build_backbone
from .face_parts import NUM_FACE_PARTS
from .segmentation import SuperpixelConfig, SuperpixelViews, segment_views


@dataclass(frozen=True)
class SuperpixelMoEConfig:
    """One locked configuration for an A--E ablation model."""

    EXPERIMENT_SPECS: ClassVar[dict[str, tuple[bool, bool, str]]] = {
        "A": (False, False, "none"),
        "B": (False, False, "single"),
        "C": (True, False, "single"),
        "D": (True, True, "single"),
        "E": (True, True, "multiple"),
    }

    experiment: str = "E"
    levels: tuple[int, ...] = (128, 64, 16)
    feature_channels: int = 512
    position_dim: int = 5
    expert_hidden_dim: int = 256
    num_experts: int = 4
    num_classes: int = 2
    pretrained_backbone: bool = True
    freeze_backbone: bool = False
    freeze_batch_norm: bool = True
    weights_path: str | None = None
    landmark_model_path: str | None = "models/face_landmarker.task"
    landmark_cache_dir: str | None = "outputs/landmark_cache"
    slic_cache_dir: str | None = "outputs/slic_cache"
    require_slic_cache: bool = False
    require_landmark_cache: bool = False
    image_range: str = "auto"
    use_superpixel: bool = field(init=False)
    use_landmarks: bool = field(init=False)
    moe_mode: str = field(init=False)

    def __post_init__(self) -> None:
        experiment = str(self.experiment).upper()
        if experiment not in self.EXPERIMENT_SPECS:
            raise ValueError(f"experiment must be one of {sorted(self.EXPERIMENT_SPECS)}")
        object.__setattr__(self, "experiment", experiment)
        use_superpixel, use_landmarks, moe_mode = self.EXPERIMENT_SPECS[experiment]
        object.__setattr__(self, "use_superpixel", use_superpixel)
        object.__setattr__(self, "use_landmarks", use_landmarks)
        object.__setattr__(self, "moe_mode", moe_mode)
        if self.levels != (128, 64, 16):
            raise ValueError("the baseline levels are fixed at (128, 64, 16)")
        if self.image_range not in ("auto", "0-1/255", "0-1", "0-255"):
            raise ValueError("image_range must be one of auto, 0-1/255, 0-1, 0-255")
        if self.num_experts < 1 or self.expert_hidden_dim < 1:
            raise ValueError("num_experts and expert_hidden_dim must be positive")
        if self.require_slic_cache and not self.use_superpixel:
            raise ValueError("experiments A/B cannot require a SLIC cache")
        if self.require_landmark_cache and not self.use_landmarks:
            raise ValueError("only experiments D/E can require a Landmark cache")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SuperpixelMoEConfig":
        """Reconstruct an init-safe config from a serialized dataclass mapping."""

        accepted = {item.name for item in fields(cls) if item.init}
        payload = {key: value for key, value in values.items() if key in accepted}
        if "levels" in payload:
            payload["levels"] = tuple(payload["levels"])
        return cls(**payload)


class EqualWeightMoE(nn.Module):
    """Four independent experts whose outputs are averaged elementwise."""

    def __init__(self, channels: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(channels),
                    nn.Linear(channels, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, channels),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = torch.stack([expert(tokens) for expert in self.experts], dim=0)
        return outputs.mean(dim=0), outputs


def _labels_to_feature_grid(labels: np.ndarray, height: int, width: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(labels.astype(np.float32, copy=False)).to(device)
    return F.interpolate(tensor[None, None], size=(height, width), mode="nearest").long()[0, 0]


def pool_regions(
    feature_map: torch.Tensor, labels: np.ndarray, expected_regions: int
) -> torch.Tensor:
    """Average-pool a feature map by a source-resolution label image."""

    if feature_map.ndim != 3:
        raise ValueError(f"expected CHW feature map, got {tuple(feature_map.shape)}")
    if labels.ndim != 2 or int(labels.max()) + 1 != expected_regions:
        raise ValueError("labels must be a 2D contiguous map with expected_regions labels")
    channels, height, width = feature_map.shape
    grid = _labels_to_feature_grid(labels, height, width, feature_map.device).reshape(-1)
    values = feature_map.reshape(channels, -1).transpose(0, 1)
    sums = torch.zeros((expected_regions, channels), device=feature_map.device, dtype=feature_map.dtype)
    counts = torch.zeros(expected_regions, device=feature_map.device, dtype=feature_map.dtype)
    sums.index_add_(0, grid, values)
    counts.index_add_(0, grid, torch.ones_like(grid, dtype=feature_map.dtype))
    pooled = sums / counts.clamp_min(1.0).unsqueeze(1)
    missing = counts == 0
    if torch.any(missing):
        # Very small regions can disappear during nearest-neighbor downsampling.
        # Use the feature vector at that region's centroid as a deterministic fallback.
        for region in torch.nonzero(missing, as_tuple=False).flatten().tolist():
            region_y, region_x = np.nonzero(labels == region)
            cy = min(height - 1, max(0, int(round(float(region_y.mean()) * height / labels.shape[0]))))
            cx = min(width - 1, max(0, int(round(float(region_x.mean()) * width / labels.shape[1]))))
            pooled[region] = feature_map[:, cy, cx]
    return pooled


class SuperpixelMoE(nn.Module):
    """Shared backbone, three region views, and equal-weight per-scale MoE."""

    def __init__(self, config: SuperpixelMoEConfig | None = None) -> None:
        super().__init__()
        self.config = config or SuperpixelMoEConfig()
        self.backbone: ResNet50Layer2 = build_backbone(
            pretrained=self.config.pretrained_backbone,
            weights_path=self.config.weights_path,
            freeze=self.config.freeze_backbone,
            freeze_batch_norm=self.config.freeze_batch_norm,
        )
        self.position: nn.Module | None = None
        self.part_embedding: nn.Embedding | None = None
        self.token_norms: nn.ModuleDict | None = None
        self.global_moe: EqualWeightMoE | None = None
        self.shared_moe: EqualWeightMoE | None = None
        self.moes: nn.ModuleDict | None = None
        if self.config.use_superpixel:
            self.position = nn.Sequential(
                nn.Linear(self.config.position_dim, self.config.feature_channels // 4),
                nn.GELU(),
                nn.Linear(self.config.feature_channels // 4, self.config.feature_channels),
            )
            if self.config.use_landmarks:
                self.part_embedding = nn.Embedding(NUM_FACE_PARTS, self.config.feature_channels)
            self.token_norms = nn.ModuleDict(
                {str(level): nn.LayerNorm(self.config.feature_channels) for level in self.config.levels}
            )
            if self.config.moe_mode == "single":
                self.shared_moe = EqualWeightMoE(
                    self.config.feature_channels,
                    self.config.expert_hidden_dim,
                    self.config.num_experts,
                )
            else:
                self.moes = nn.ModuleDict(
                    {
                        str(level): EqualWeightMoE(
                            self.config.feature_channels,
                            self.config.expert_hidden_dim,
                            self.config.num_experts,
                        )
                        for level in self.config.levels
                    }
                )
        elif self.config.moe_mode == "single":
            self.global_moe = EqualWeightMoE(
                self.config.feature_channels,
                self.config.expert_hidden_dim,
                self.config.num_experts,
            )
        classifier_channels = (
            self.config.feature_channels * len(self.config.levels)
            if self.config.use_superpixel
            else self.config.feature_channels
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_channels),
            nn.Linear(classifier_channels, self.config.num_classes),
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(
        self,
        images: torch.Tensor,
        *,
        views: SuperpixelViews | list[SuperpixelViews] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected BCHW RGB input, got {tuple(images.shape)}")
        input_dtype = images.dtype
        input_was_integer = not torch.is_floating_point(images)
        images = images.float()
        if self.config.image_range == "0-1/255":
            images = images * (255.0 * 255.0)
        elif self.config.image_range == "0-1":
            images = images * 255.0
        elif self.config.image_range == "auto" and input_was_integer:
            maximum = float(images.detach().amax())
            if maximum > 255.0:
                try:
                    dtype_maximum = float(torch.iinfo(input_dtype).max)
                except TypeError:
                    dtype_maximum = maximum
                images = images * (255.0 / dtype_maximum)
        elif self.config.image_range == "auto":
            # Infer independently per sample so a mixed batch of [0,1] and
            # legacy [0,1/255] arrays is handled consistently.  A genuinely
            # dark canonical [0,255] float image remains inherently ambiguous;
            # callers can select ``image_range='0-255'`` for that case.
            maxima = images.detach().amax(dim=(1, 2, 3), keepdim=True)
            scale = torch.ones_like(maxima)
            scale = torch.where(maxima <= (1.0 / 255.0) + 1e-6, 255.0 * 255.0, scale)
            scale = torch.where(
                (maxima > (1.0 / 255.0) + 1e-6) & (maxima <= 1.5),
                torch.full_like(maxima, 255.0),
                scale,
            )
            images = images * scale
        if self.config.use_superpixel and views is None:
            segmentation_config = SuperpixelConfig(
                use_landmarks=self.config.use_landmarks,
                landmark_model_path=self.config.landmark_model_path,
                landmark_cache_dir=self.config.landmark_cache_dir,
                slic_cache_dir=self.config.slic_cache_dir,
                require_slic_cache=self.config.require_slic_cache,
                require_landmark_cache=self.config.require_landmark_cache,
                # The conversion above has produced canonical [0,255] values.
                image_range="0-255",
            )
            views = [
                segment_views(image.detach().cpu().permute(1, 2, 0).numpy(), segmentation_config)
                for image in images
            ]
        normalized = (images / 255.0 - self.image_mean.to(images.device)) / self.image_std.to(images.device)
        feature_maps = self.backbone(normalized)
        details: dict[str, torch.Tensor] = {}
        if not self.config.use_superpixel:
            pooled = feature_maps.mean(dim=(2, 3))
            details["global_features"] = pooled
            if self.global_moe is not None:
                pooled, expert_outputs = self.global_moe(pooled)
                details["global_experts"] = expert_outputs
            details["fused"] = pooled
            return self.classifier(pooled), details

        if views is None:
            raise RuntimeError("superpixel views were not generated")
        view_list = views if isinstance(views, list) else [views]
        if len(view_list) != images.shape[0]:
            if not isinstance(views, list):
                view_list = [views] * images.shape[0]
            else:
                raise ValueError("views list length must match the image batch size")
        assert self.position is not None and self.token_norms is not None
        scale_vectors = []
        for level in self.config.levels:
            token_batches = []
            for batch_index, view in enumerate(view_list):
                pooled = pool_regions(feature_maps[batch_index], view.labels[level], level)
                position = self.position(torch.from_numpy(view.positions[level]).to(images.device, dtype=pooled.dtype))
                distribution = torch.from_numpy(view.part_distributions[level]).to(
                    images.device, dtype=pooled.dtype
                )
                if distribution.shape != (level, NUM_FACE_PARTS):
                    raise ValueError(
                        f"part distribution for level {level} must have shape {(level, NUM_FACE_PARTS)}"
                    )
                if self.config.use_landmarks:
                    assert self.part_embedding is not None
                    part_encoding = distribution @ self.part_embedding.weight.to(dtype=pooled.dtype)
                else:
                    part_encoding = torch.zeros_like(pooled)
                tokens = self.token_norms[str(level)](pooled + position + part_encoding)
                token_batches.append(tokens)
            tokens = torch.stack(token_batches, dim=0)
            details[f"input_tokens_{level}"] = tokens
            moe = self.shared_moe if self.shared_moe is not None else self.moes[str(level)]  # type: ignore[index]
            moe_output, expert_outputs = moe(tokens)
            details[f"tokens_{level}"] = moe_output
            details[f"experts_{level}"] = expert_outputs
            scale_vectors.append(moe_output.mean(dim=1))
        fused = torch.cat(scale_vectors, dim=1)
        logits = self.classifier(fused)
        details["fused"] = fused
        return logits, details


__all__ = [
    "EqualWeightMoE",
    "SuperpixelMoE",
    "SuperpixelMoEConfig",
    "pool_regions",
]
