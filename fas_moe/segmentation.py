"""Independent 128/64/16 SLIC views for the Superpixel-MoE baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from skimage.color import rgb2lab
from skimage.segmentation import slic

from .features import FEATURE_NAMES, extract_region_features
from .face_parts import (
    NUM_FACE_PARTS,
    PART_NAMES,
    landmarks_to_part_masks,
    part_distributions_for_levels,
    unknown_part_distributions,
)
from .io import prepare_image
from .landmarks import detect_face_landmarks, model_identity, resolve_model_path


DEFAULT_LEVELS: Tuple[int, ...] = (128, 64, 16)


@dataclass(frozen=True)
class SuperpixelConfig:
    """Parameters for reproducible independent SLIC views."""

    image_size: Tuple[int, int] = (256, 256)
    levels: Tuple[int, ...] = DEFAULT_LEVELS
    compactness: float = 10.0
    sigma: float = 1.0
    max_num_iter: int = 10
    max_slic_attempts: int = 8
    use_landmarks: bool = True
    landmark_model_path: str | Path | None = Path("models/face_landmarker.task")
    landmark_cache_dir: str | Path | None = Path("outputs/landmark_cache")
    landmark_detection_confidence: float = 0.5
    landmark_presence_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not self.levels or tuple(sorted(self.levels, reverse=True)) != self.levels:
            raise ValueError("levels must be a non-empty tuple sorted from fine to coarse")
        if any(level < 2 for level in self.levels):
            raise ValueError("every level must contain at least two regions")
        if self.compactness <= 0 or self.sigma < 0 or self.max_num_iter < 1:
            raise ValueError("compactness, sigma and max_num_iter must be valid")
        if self.max_slic_attempts < 1:
            raise ValueError("max_slic_attempts must be positive")
        if not 0.0 <= self.landmark_detection_confidence <= 1.0:
            raise ValueError("landmark_detection_confidence must be in [0,1]")
        if not 0.0 <= self.landmark_presence_confidence <= 1.0:
            raise ValueError("landmark_presence_confidence must be in [0,1]")


@dataclass
class SuperpixelViews:
    """Artifacts for the three independent superpixel views."""

    image: np.ndarray
    labels: Dict[int, np.ndarray]
    features: Dict[int, np.ndarray]
    edges: Dict[int, np.ndarray]
    positions: Dict[int, np.ndarray]
    part_distributions: Dict[int, np.ndarray]
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def part_distribution(self) -> Dict[int, np.ndarray]:
        """Singular alias matching the mathematical quantity name."""

        return self.part_distributions


def _canonicalize(labels: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(np.asarray(labels), return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32, copy=False)


def _extract_edges(labels: np.ndarray) -> np.ndarray:
    """Return sorted undirected region-adjacency edges."""

    pairs: set[tuple[int, int]] = set()
    for left, right in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        different = left != right
        for first, second in zip(left[different].tolist(), right[different].tolist()):
            pairs.add((min(int(first), int(second)), max(int(first), int(second))))
    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(sorted(pairs), dtype=np.int32)


def _region_lab_means(labels: np.ndarray, lab: np.ndarray) -> np.ndarray:
    count = int(labels.max()) + 1
    flat = labels.ravel()
    area = np.bincount(flat, minlength=count).astype(np.float64)
    means = np.zeros((count, 3), dtype=np.float64)
    for channel in range(3):
        means[:, channel] = np.bincount(
            flat, weights=lab[..., channel].ravel(), minlength=count
        )
    return means / np.maximum(area[:, None], 1.0)


def _merge_to_target(labels: np.ndarray, image: np.ndarray, target: int) -> np.ndarray:
    """Merge adjacent regions until ``target`` remains.

    This is only a cardinality correction after SLIC.  It never merges
    non-adjacent regions and uses a deterministic Lab/size tie-break.
    """

    labels = _canonicalize(labels)
    lab = rgb2lab(image.astype(np.float32) / 255.0).astype(np.float32)
    while int(labels.max()) + 1 > target:
        count = int(labels.max()) + 1
        means = _region_lab_means(labels, lab)
        areas = np.bincount(labels.ravel(), minlength=count).astype(np.float64)
        candidates: list[tuple[float, int, int]] = []
        for first, second in _extract_edges(labels).tolist():
            color = (means[first] - means[second]) / np.array([100.0, 255.0, 255.0])
            color_cost = float(np.linalg.norm(color) / np.sqrt(3.0))
            size_cost = float(abs(areas[first] - areas[second]) / (areas[first] + areas[second]))
            candidates.append((0.9 * color_cost + 0.1 * size_cost, first, second))
        if not candidates:
            raise RuntimeError(f"region adjacency graph exhausted at {count} regions")
        _, first, second = min(candidates)
        labels[labels == second] = first
        labels = _canonicalize(labels)
    return labels


def _segment_level(image: np.ndarray, target: int, config: SuperpixelConfig) -> tuple[np.ndarray, int]:
    """Run connected SLIC with retries, then correct the exact count."""

    image_float = image.astype(np.float32) / 255.0
    requested = max(int(target), 2)
    last_count = 0
    for attempt in range(config.max_slic_attempts):
        labels = slic(
            image_float,
            n_segments=requested,
            compactness=float(config.compactness),
            sigma=float(config.sigma),
            max_num_iter=int(config.max_num_iter),
            enforce_connectivity=True,
            min_size_factor=0.25,
            max_size_factor=3.0,
            convert2lab=True,
            start_label=0,
            channel_axis=-1,
        )
        labels = _canonicalize(labels)
        last_count = int(labels.max()) + 1
        if last_count >= target:
            return _merge_to_target(labels, image, target), requested
        requested = max(requested + 1, int(np.ceil(requested * 1.35)))
    raise RuntimeError(
        f"SLIC could not produce {target} connected regions; last count was {last_count}"
    )


def _geometry_features(labels: np.ndarray) -> np.ndarray:
    """Return [centroid_x, centroid_y, area, bbox_width, bbox_height]."""

    height, width = labels.shape
    count = int(labels.max()) + 1
    result = np.empty((count, 5), dtype=np.float32)
    for region in range(count):
        yy, xx = np.nonzero(labels == region)
        if len(xx) == 0:
            raise RuntimeError(f"region {region} has no pixels")
        result[region] = (
            float(xx.mean()) / max(width - 1, 1),
            float(yy.mean()) / max(height - 1, 1),
            float(len(xx)) / float(height * width),
            float(xx.max() - xx.min() + 1) / float(width),
            float(yy.max() - yy.min() + 1) / float(height),
        )
    return result


def _landmark_cache_path(image: np.ndarray, config: SuperpixelConfig) -> Path | None:
    if config.landmark_cache_dir is None:
        return None
    payload = {
        "version": 2,
        "shape": list(image.shape),
        "levels": list(config.levels),
        "compactness": config.compactness,
        "sigma": config.sigma,
        "max_num_iter": config.max_num_iter,
        "max_slic_attempts": config.max_slic_attempts,
        "model": model_identity(config.landmark_model_path),
        "detection_confidence": config.landmark_detection_confidence,
        "presence_confidence": config.landmark_presence_confidence,
    }
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(image).tobytes())
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return Path(config.landmark_cache_dir) / f"{digest.hexdigest()}.npz"


def _load_part_cache(
    path: Path, levels: tuple[int, ...]
) -> tuple[dict[int, np.ndarray], bool, str] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            distributions = {level: np.asarray(cache[f"part_{level}"], dtype=np.float32) for level in levels}
            detected = bool(int(cache["detected"]))
            reason = str(cache["reason"].item())
        for level, distribution in distributions.items():
            if distribution.shape != (level, NUM_FACE_PARTS):
                return None
            if not np.isfinite(distribution).all() or not np.allclose(distribution.sum(axis=1), 1.0):
                return None
        return distributions, detected, reason
    except (OSError, ValueError, KeyError):
        return None


def _save_part_cache(
    path: Path,
    distributions: dict[int, np.ndarray],
    points: np.ndarray,
    detected: bool,
    reason: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    payload: dict[str, object] = {
        "landmarks": np.asarray(points, dtype=np.float32),
        "detected": np.asarray(int(detected), dtype=np.uint8),
        "reason": np.asarray(reason),
    }
    payload.update({f"part_{level}": value for level, value in distributions.items()})
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _part_distributions(
    image: np.ndarray,
    labels: dict[int, np.ndarray],
    config: SuperpixelConfig,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    if not config.use_landmarks:
        return unknown_part_distributions(labels), {
            "landmarks_enabled": False,
            "landmarks_detected": False,
            "landmark_reason": "landmarks disabled",
            "landmark_cache_hit": False,
        }
    cache_path = _landmark_cache_path(image, config)
    if cache_path is not None and (cached := _load_part_cache(cache_path, config.levels)) is not None:
        distributions, detected, reason = cached
        return distributions, {
            "landmarks_enabled": True,
            "landmarks_detected": detected,
            "landmark_reason": reason,
            "landmark_cache_hit": True,
            "landmark_cache_path": str(cache_path.resolve()),
        }
    result = detect_face_landmarks(
        image,
        config.landmark_model_path,
        detection_confidence=config.landmark_detection_confidence,
        presence_confidence=config.landmark_presence_confidence,
    )
    if result.detected:
        masks = landmarks_to_part_masks(result.points, image.shape[:2])
        distributions = part_distributions_for_levels(labels, masks)
    else:
        distributions = unknown_part_distributions(labels)
    # Do not persist configuration/dependency failures; a corrected setup should retry immediately.
    cacheable_failure = result.reason == "no face detected"
    if cache_path is not None and (result.detected or cacheable_failure):
        _save_part_cache(cache_path, distributions, result.points, result.detected, result.reason)
    return distributions, {
        "landmarks_enabled": True,
        "landmarks_detected": result.detected,
        "landmark_reason": result.reason,
        "landmark_cache_hit": False,
        "landmark_cache_path": str(cache_path.resolve()) if cache_path is not None else None,
        "landmark_model_path": str(resolve_model_path(config.landmark_model_path))
        if resolve_model_path(config.landmark_model_path) is not None
        else None,
    }


def segment_views(
    image: np.ndarray, config: SuperpixelConfig | None = None
) -> SuperpixelViews:
    """Generate the 128/64/16 SLIC views and region descriptors."""

    config = config or SuperpixelConfig()
    prepared, preparation_metadata = prepare_image(image, config.image_size)
    labels: Dict[int, np.ndarray] = {}
    features: Dict[int, np.ndarray] = {}
    edges: Dict[int, np.ndarray] = {}
    positions: Dict[int, np.ndarray] = {}
    requested_segments: Dict[str, int] = {}
    for level in config.levels:
        level_labels, requested = _segment_level(prepared, level, config)
        labels[level] = level_labels
        features[level] = extract_region_features(prepared, level_labels)
        edges[level] = _extract_edges(level_labels)
        positions[level] = _geometry_features(level_labels)
        requested_segments[str(level)] = requested

    part_distributions, landmark_metadata = _part_distributions(prepared, labels, config)

    metadata: Dict[str, object] = {
        **preparation_metadata,
        "levels": list(config.levels),
        "compactness": float(config.compactness),
        "sigma": float(config.sigma),
        "max_num_iter": int(config.max_num_iter),
        "feature_names": list(FEATURE_NAMES),
        "position_names": [
            "centroid_x",
            "centroid_y",
            "area_ratio",
            "bbox_width_ratio",
            "bbox_height_ratio",
        ],
        "requested_slic_segments": requested_segments,
        "part_names": list(PART_NAMES),
        **landmark_metadata,
    }
    return SuperpixelViews(prepared, labels, features, edges, positions, part_distributions, metadata)


__all__ = [
    "DEFAULT_LEVELS",
    "FEATURE_NAMES",
    "SuperpixelConfig",
    "SuperpixelViews",
    "segment_views",
]
