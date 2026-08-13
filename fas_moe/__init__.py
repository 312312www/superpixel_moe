"""Minimal Superpixel-MoE baseline for face anti-spoofing."""

from .features import FEATURE_NAMES, extract_region_features
from .checkpoint import checkpoint_state, load_checkpoint, validate_checkpoint
from .face_parts import (
    NUM_FACE_PARTS,
    PART_NAMES,
    landmarks_to_part_masks,
    part_distribution_for_labels,
    unknown_part_distributions,
)
from .io import infer_image_range, load_input, prepare_image, restore_image_range
from .landmarks import FaceLandmarkResult, detect_face_landmarks
from .model import EqualWeightMoE, SuperpixelMoE, SuperpixelMoEConfig, pool_regions
from .segmentation import (
    DEFAULT_LEVELS,
    SuperpixelConfig,
    SuperpixelViews,
    SLIC_CACHE_SCHEMA,
    segment_views,
)

__all__ = [
    "DEFAULT_LEVELS",
    "FEATURE_NAMES",
    "NUM_FACE_PARTS",
    "PART_NAMES",
    "EqualWeightMoE",
    "checkpoint_state",
    "SuperpixelConfig",
    "SuperpixelMoE",
    "SuperpixelMoEConfig",
    "SuperpixelViews",
    "SLIC_CACHE_SCHEMA",
    "extract_region_features",
    "FaceLandmarkResult",
    "detect_face_landmarks",
    "landmarks_to_part_masks",
    "load_input",
    "infer_image_range",
    "load_checkpoint",
    "pool_regions",
    "prepare_image",
    "restore_image_range",
    "part_distribution_for_labels",
    "segment_views",
    "unknown_part_distributions",
    "validate_checkpoint",
]
