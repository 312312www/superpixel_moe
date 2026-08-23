"""Cached multi-scale SLIC superpixels for FAS stages C--E.

The cache is keyed by the source NPY file, row index, SLIC configuration and
schema.  It stores image-resolution label maps plus fixed-size geometry arrays;
this keeps C--E training deterministic and prevents SLIC from being recomputed
on every epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np
from skimage.segmentation import slic


SUPERPIXEL_SCHEMA = 1
DEFAULT_LEVELS = (16, 9, 4)  # Fine / Medium / Coarse at a 7x7 ResNet map.


@dataclass(frozen=True)
class SuperpixelConfig:
    levels: tuple[int, ...] = DEFAULT_LEVELS
    compactness: float = 10.0
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        levels = tuple(int(level) for level in self.levels)
        if len(levels) != 3 or any(level < 1 for level in levels) or len(set(levels)) != len(levels):
            raise ValueError("levels must contain exactly three distinct positive values")
        if self.compactness <= 0:
            raise ValueError("compactness must be positive")
        object.__setattr__(self, "levels", levels)


def _cache_path(source: str | Path, index: int, config: SuperpixelConfig) -> Path | None:
    if config.cache_dir is None:
        return None
    fingerprint = "|".join(
        (
            str(Path(source)),
            str(int(index)),
            str(SUPERPIXEL_SCHEMA),
            ",".join(map(str, config.levels)),
            repr(float(config.compactness)),
        )
    )
    return Path(config.cache_dir) / f"{sha256(fingerprint.encode('utf-8')).hexdigest()}.npz"


def _contiguous_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel arbitrary integer ids into contiguous [0, actual_regions)."""
    _, inverse = np.unique(labels.reshape(-1), return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32, copy=False)


def _positions(labels: np.ndarray, expected_regions: int) -> tuple[np.ndarray, np.ndarray]:
    """Return [K,5] geometry and [K] valid-region mask for a label map."""
    height, width = labels.shape
    positions = np.zeros((expected_regions, 5), dtype=np.float32)
    # Missing regions use neutral geometry and are ignored via `valid` later.
    positions[:, 0] = 0.5
    positions[:, 1] = 0.5
    positions[:, 3] = 1.0
    positions[:, 4] = 1.0
    valid = np.zeros(expected_regions, dtype=np.bool_)
    actual_regions = int(labels.max()) + 1
    for region in range(min(actual_regions, expected_regions)):
        ys, xs = np.nonzero(labels == region)
        if ys.size == 0:
            continue
        valid[region] = True
        positions[region] = np.asarray(
            [
                float(xs.mean()) / max(1, width - 1),
                float(ys.mean()) / max(1, height - 1),
                float(ys.size) / float(height * width),
                float(xs.max() - xs.min() + 1) / float(width),
                float(ys.max() - ys.min() + 1) / float(height),
            ],
            dtype=np.float32,
        )
    return positions, valid


def _segment_level(image: np.ndarray, level: int, compactness: float) -> np.ndarray:
    """Run deterministic SLIC and preserve at most `level` region ids.

    SLIC can occasionally emit fewer regions than requested.  This is valid:
    the model receives a validity mask and excludes absent slots when pooling.
    It can very rarely emit more; those are deterministically folded into the
    final available slot rather than producing an invalid tensor shape.
    """
    labels = slic(
        image,
        n_segments=int(level),
        compactness=float(compactness),
        sigma=0.0,
        start_label=0,
        enforce_connectivity=True,
        channel_axis=-1,
        convert2lab=True,
    )
    labels = _contiguous_labels(np.asarray(labels))
    actual = int(labels.max()) + 1
    if actual > level:
        # Folding surplus fragments is deterministic and keeps all spatial
        # pixels assigned without any silent region dropping.
        labels = np.minimum(labels, level - 1).astype(np.int32, copy=False)
        labels = _contiguous_labels(labels)
    return labels


def segment_image(image: np.ndarray, config: SuperpixelConfig) -> dict[str, np.ndarray]:
    """Compute fixed-shape label / geometry records for every configured scale."""
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {source.shape}")
    if source.dtype != np.uint8:
        source = np.clip(np.round(source), 0, 255).astype(np.uint8)
    result: dict[str, np.ndarray] = {"schema": np.asarray(SUPERPIXEL_SCHEMA, dtype=np.int64)}
    for level in config.levels:
        labels = _segment_level(source, level, config.compactness)
        positions, valid = _positions(labels, level)
        result[f"labels_{level}"] = labels.astype(np.int32, copy=False)
        result[f"positions_{level}"] = positions
        result[f"valid_{level}"] = valid
    return result


def cached_superpixels(
    image: np.ndarray,
    *,
    source: str | Path,
    index: int,
    config: SuperpixelConfig,
) -> dict[str, np.ndarray]:
    """Load a verified cache record or compute and atomically persist it."""
    path = _cache_path(source, index, config)
    required = {"schema"}
    for level in config.levels:
        required.update((f"labels_{level}", f"positions_{level}", f"valid_{level}"))
    if path is not None and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as cached:
                if set(cached.files) >= required and int(cached["schema"]) == SUPERPIXEL_SCHEMA:
                    values = {name: np.array(cached[name], copy=True) for name in required}
                    if all(
                        values[f"labels_{level}"].ndim == 2
                        and values[f"positions_{level}"].shape == (level, 5)
                        and values[f"valid_{level}"].shape == (level,)
                        for level in config.levels
                    ):
                        return values
        except (OSError, ValueError, KeyError):
            pass
    values = segment_image(image, config)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **values)
        temporary.replace(path)
    return values


def batch_superpixel_tensors(
    records: Mapping[str, object], levels: tuple[int, ...]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compatibility helper for callers assembling cache records manually."""
    labels = {str(level): np.asarray(records[f"labels_{level}"], dtype=np.int64) for level in levels}
    positions = {str(level): np.asarray(records[f"positions_{level}"], dtype=np.float32) for level in levels}
    valid = {str(level): np.asarray(records[f"valid_{level}"], dtype=np.bool_) for level in levels}
    return labels, positions, valid


__all__ = [
    "DEFAULT_LEVELS",
    "SUPERPIXEL_SCHEMA",
    "SuperpixelConfig",
    "batch_superpixel_tensors",
    "cached_superpixels",
    "segment_image",
]
