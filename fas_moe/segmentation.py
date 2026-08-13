"""Independent 128/64/16 SLIC views for the Superpixel-MoE baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
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
from .io import IMAGE_RANGE_CHOICES, prepare_image
from .landmarks import detect_face_landmarks, model_identity, resolve_model_path


DEFAULT_LEVELS: Tuple[int, ...] = (128, 64, 16)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLIC_CACHE_SCHEMA = 1


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
    # SLIC artifacts are expensive during training.  Keep this opt-out cache
    # separate from the landmark cache so either cache can be invalidated
    # independently.  Relative paths are resolved against the project root.
    slic_cache_dir: str | Path | None = Path("outputs/slic_cache")
    # ``auto`` is convenient for ordinary images, while callers that already
    # restored a float array to canonical ``[0,255]`` should pass ``0-255``.
    # This removes the unavoidable ambiguity of a very dark float image.
    image_range: str = "auto"

    def __post_init__(self) -> None:
        if self.image_range not in IMAGE_RANGE_CHOICES:
            raise ValueError(
                f"image_range must be one of {IMAGE_RANGE_CHOICES}, got {self.image_range!r}"
            )
        if not self.levels or tuple(sorted(self.levels, reverse=True)) != self.levels:
            raise ValueError("levels must be a non-empty tuple sorted from fine to coarse")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("levels must not contain duplicates")
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


def _slic_cache_path(image: np.ndarray, config: SuperpixelConfig) -> Path | None:
    """Return the deterministic cache path for one prepared image.

    The key includes every parameter that can affect labels or their derived
    descriptors.  The image is already normalized/resized by ``prepare_image``
    when this helper is called, which means equivalent inputs share a cache
    entry while changes in source normalization cannot leak into the result.
    """

    if config.slic_cache_dir is None:
        return None
    cache_dir = Path(str(config.slic_cache_dir).replace("\\", "/")).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = _PROJECT_ROOT / cache_dir
    image = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    payload = {
        "schema": SLIC_CACHE_SCHEMA,
        "shape": list(image.shape),
        "image_size": list(config.image_size),
        "levels": list(config.levels),
        "compactness": float(config.compactness),
        "sigma": float(config.sigma),
        "max_num_iter": int(config.max_num_iter),
        "max_slic_attempts": int(config.max_slic_attempts),
        "feature_names": list(FEATURE_NAMES),
        "position_schema": 1,
    }
    digest = hashlib.sha256()
    digest.update(image.tobytes())
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return cache_dir / f"{digest.hexdigest()}.npz"


def _validated_integer_array(value: np.ndarray, *, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    """Return an int32 copy only when an on-disk integer array is lossless."""

    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        return None
    if shape is not None and array.shape != shape:
        return None
    converted = array.astype(np.int32, copy=False)
    if not np.array_equal(array, converted):
        return None
    return converted


def _load_slic_cache(
    path: Path | None,
    image_shape: tuple[int, ...],
    levels: tuple[int, ...],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[str, int],
] | None:
    """Load and validate all SLIC artifacts from one cache entry.

    A cache is treated as disposable derived data: any malformed, truncated,
    stale, or internally inconsistent entry simply becomes a cache miss.
    ``allow_pickle=False`` keeps loading limited to numeric arrays.
    """

    if path is None or not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            version = np.asarray(cache["schema"])
            if version.shape != () or int(version) != SLIC_CACHE_SCHEMA:
                return None
            cached_shape = np.asarray(cache["image_shape"], dtype=np.int64).reshape(-1)
            if tuple(int(value) for value in cached_shape) != tuple(image_shape):
                return None
            cached_levels = np.asarray(cache["levels"], dtype=np.int64).reshape(-1)
            if tuple(int(value) for value in cached_levels) != tuple(levels):
                return None
            cached_feature_names = np.asarray(cache["feature_names"])
            if tuple(str(value) for value in cached_feature_names.reshape(-1)) != tuple(FEATURE_NAMES):
                return None
            position_schema = np.asarray(cache["position_schema"])
            if position_schema.shape != () or int(position_schema) != 1:
                return None
            requested_values = _validated_integer_array(np.asarray(cache["requested"]), shape=(len(levels),))
            if requested_values is None or np.any(requested_values < 1):
                return None

            labels: dict[int, np.ndarray] = {}
            features: dict[int, np.ndarray] = {}
            edges: dict[int, np.ndarray] = {}
            positions: dict[int, np.ndarray] = {}
            for index, level in enumerate(levels):
                level_labels = _validated_integer_array(np.asarray(cache[f"labels_{level}"]))
                if level_labels is None or level_labels.shape != tuple(image_shape[:2]):
                    return None
                if not np.array_equal(np.unique(level_labels), np.arange(level, dtype=np.int32)):
                    return None

                level_features = np.asarray(cache[f"features_{level}"])
                if (
                    level_features.shape != (level, len(FEATURE_NAMES))
                    or level_features.dtype.kind not in "fiu"
                    or not np.isfinite(level_features).all()
                ):
                    return None
                level_features = level_features.astype(np.float32, copy=False)
                if not np.isfinite(level_features).all():
                    return None

                level_edges = _validated_integer_array(np.asarray(cache[f"edges_{level}"]))
                if level_edges is None or level_edges.ndim != 2 or level_edges.shape[1] != 2:
                    return None
                if level_edges.size:
                    if np.any(level_edges[:, 0] >= level_edges[:, 1]):
                        return None
                    if np.any(level_edges < 0) or np.any(level_edges >= level):
                        return None
                    if np.unique(level_edges, axis=0).shape[0] != level_edges.shape[0]:
                        return None
                # Adjacency is cheap to check and catches a partially written
                # or hand-edited cache that otherwise has plausible shapes.
                if not np.array_equal(level_edges, _extract_edges(level_labels)):
                    return None

                level_positions = np.asarray(cache[f"positions_{level}"])
                if (
                    level_positions.shape != (level, 5)
                    or level_positions.dtype.kind not in "fiu"
                    or not np.isfinite(level_positions).all()
                ):
                    return None
                level_positions = level_positions.astype(np.float32, copy=False)
                if not np.isfinite(level_positions).all():
                    return None
                # Geometry descriptors have a bounded, normalized contract;
                # reject plausible-looking but stale/corrupted numeric blobs.
                if (
                    np.any(level_positions < 0.0)
                    or np.any(level_positions > 1.0 + 1e-5)
                    or np.any(level_positions[:, 2] <= 0.0)
                    or not np.isclose(float(level_positions[:, 2].sum()), 1.0, atol=2e-4)
                ):
                    return None

                labels[level] = level_labels
                features[level] = level_features
                edges[level] = level_edges
                positions[level] = level_positions
                # ``index`` is intentionally consumed here so a malformed
                # requested array cannot silently change level ordering.
                if int(requested_values[index]) < level:
                    return None
            requested = {str(level): int(requested_values[index]) for index, level in enumerate(levels)}
        return labels, features, edges, positions, requested
    except Exception:
        # Corrupt ZIP members, interrupted writes, and old schema entries are
        # all recoverable by recomputing the derived artifacts.
        return None


def _save_slic_cache(
    path: Path,
    image_shape: tuple[int, ...],
    levels: tuple[int, ...],
    labels: dict[int, np.ndarray],
    features: dict[int, np.ndarray],
    edges: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    requested: dict[str, int],
) -> bool:
    """Persist SLIC artifacts with a unique temporary file and atomic replace."""

    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp.npz", dir=str(path.parent)
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        payload: dict[str, object] = {
            "schema": np.asarray(SLIC_CACHE_SCHEMA, dtype=np.int64),
            "image_shape": np.asarray(image_shape, dtype=np.int64),
            "levels": np.asarray(levels, dtype=np.int64),
            "requested": np.asarray([requested[str(level)] for level in levels], dtype=np.int64),
            "feature_names": np.asarray(FEATURE_NAMES),
            "position_schema": np.asarray(1, dtype=np.int64),
        }
        for level in levels:
            payload[f"labels_{level}"] = np.asarray(labels[level], dtype=np.int32)
            payload[f"features_{level}"] = np.asarray(features[level], dtype=np.float32)
            payload[f"edges_{level}"] = np.asarray(edges[level], dtype=np.int32)
            payload[f"positions_{level}"] = np.asarray(positions[level], dtype=np.float32)
        # Supplying an open stream avoids NumPy's implicit '.npz' suffix and
        # lets us flush the complete ZIP before publishing it.
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
        return True
    except Exception:
        # Cache persistence is best effort: a read-only volume, quota limit,
        # interrupted filesystem operation, or serializer failure must not
        # make otherwise valid segmentation unusable.
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _landmark_cache_path(image: np.ndarray, config: SuperpixelConfig) -> Path | None:
    if config.landmark_cache_dir is None:
        return None
    # Accept paths copied from Windows command lines when the same checkout is
    # executed under Linux/WSL.  ``Path`` on POSIX treats backslashes as normal
    # characters, so normalize them before resolving the project-relative
    # cache directory (the SLIC cache follows the same rule).
    cache_dir = Path(str(config.landmark_cache_dir).replace("\\", "/")).expanduser()
    # Runtime defaults are project-relative rather than process-cwd-relative.
    # This matters when a Linux service/cron job starts from ``/`` (or another
    # read-only directory): cache creation must still land beside the checkout.
    if not cache_dir.is_absolute():
        cache_dir = _PROJECT_ROOT / cache_dir
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
    return cache_dir / f"{digest.hexdigest()}.npz"


def _load_part_cache(
    path: Path, levels: tuple[int, ...]
) -> tuple[dict[int, np.ndarray], bool, str] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            # Validate every scalar and numeric member before converting it.
            # Cache files are disposable derived data, so malformed or stale
            # entries should become misses rather than surfacing as inference
            # failures.  ``allow_pickle=False`` also keeps object payloads out.
            points = np.asarray(cache["landmarks"])
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or points.dtype.kind not in "fiu"
                or not np.isfinite(points).all()
                or np.any(points < 0.0)
                or np.any(points > 1.0 + 1e-5)
            ):
                return None

            detected_value = np.asarray(cache["detected"])
            if detected_value.shape != () or detected_value.dtype.kind not in "biu":
                return None
            detected_int = int(detected_value)
            if detected_int not in (0, 1):
                return None
            detected = bool(detected_int)

            reason_value = np.asarray(cache["reason"])
            if reason_value.shape != () or reason_value.dtype.kind not in "SU":
                return None
            reason = reason_value.item()
            if isinstance(reason, bytes):
                reason = reason.decode("utf-8", errors="replace")
            if not isinstance(reason, str):
                return None

            distributions: dict[int, np.ndarray] = {}
            for level in levels:
                raw_distribution = np.asarray(cache[f"part_{level}"])
                if (
                    raw_distribution.shape != (level, NUM_FACE_PARTS)
                    or raw_distribution.dtype.kind not in "fiu"
                ):
                    return None
                distribution = raw_distribution.astype(np.float32, copy=False)
                if (
                    not np.isfinite(distribution).all()
                    or np.any(distribution < -1e-6)
                    or np.any(distribution > 1.0 + 1e-5)
                    or not np.allclose(distribution.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
                ):
                    return None
                distributions[level] = distribution
        return distributions, detected, reason
    except Exception:
        return None


def _save_part_cache(
    path: Path,
    distributions: dict[int, np.ndarray],
    points: np.ndarray,
    detected: bool,
    reason: str,
) -> bool:
    """Persist landmark artifacts with a unique temporary file and publish atomically."""

    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp.npz", dir=str(path.parent)
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        payload: dict[str, object] = {
            "landmarks": np.asarray(points, dtype=np.float32),
            "detected": np.asarray(int(bool(detected)), dtype=np.uint8),
            "reason": np.asarray(str(reason)),
        }
        payload.update({f"part_{level}": np.asarray(value, dtype=np.float32) for level, value in distributions.items()})
        # Write through an open stream so NumPy does not append another suffix;
        # flush/fsync completes the archive before the atomic publication.
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
        return True
    except Exception:
        # Landmark data is a best-effort cache.  A read-only volume or a race
        # with cleanup should not interrupt otherwise valid inference.
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


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
    prepared, preparation_metadata = prepare_image(image, config.image_size, config.image_range)
    labels: Dict[int, np.ndarray] = {}
    features: Dict[int, np.ndarray] = {}
    edges: Dict[int, np.ndarray] = {}
    positions: Dict[int, np.ndarray] = {}
    requested_segments: Dict[str, int] = {}
    slic_cache_path = _slic_cache_path(prepared, config)
    cached = _load_slic_cache(slic_cache_path, prepared.shape, config.levels)
    slic_cache_hit = cached is not None
    if cached is not None:
        labels, features, edges, positions, requested_segments = cached
    else:
        for level in config.levels:
            level_labels, requested = _segment_level(prepared, level, config)
            labels[level] = level_labels
            features[level] = extract_region_features(prepared, level_labels)
            edges[level] = _extract_edges(level_labels)
            positions[level] = _geometry_features(level_labels)
            requested_segments[str(level)] = requested
        if slic_cache_path is not None:
            _save_slic_cache(
                slic_cache_path,
                prepared.shape,
                config.levels,
                labels,
                features,
                edges,
                positions,
                requested_segments,
            )

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
        "slic_cache_enabled": slic_cache_path is not None,
        "slic_cache_hit": slic_cache_hit,
        "slic_cache_path": str(slic_cache_path.resolve()) if slic_cache_path is not None else None,
        "part_names": list(PART_NAMES),
        **landmark_metadata,
    }
    return SuperpixelViews(prepared, labels, features, edges, positions, part_distributions, metadata)


__all__ = [
    "DEFAULT_LEVELS",
    "FEATURE_NAMES",
    "SLIC_CACHE_SCHEMA",
    "SuperpixelConfig",
    "SuperpixelViews",
    "segment_views",
]
