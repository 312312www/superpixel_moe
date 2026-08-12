"""Minimal Superpixel-MoE baseline for face anti-spoofing."""

from .features import FEATURE_NAMES, extract_region_features
from .face_parts import (
    NUM_FACE_PARTS,
    PART_NAMES,
    landmarks_to_part_masks,
    part_distribution_for_labels,
    unknown_part_distributions,
)
from .io import load_input, prepare_image
from .landmarks import FaceLandmarkResult, detect_face_landmarks
from .model import EqualWeightMoE, SuperpixelMoE, SuperpixelMoEConfig, pool_regions
from .segmentation import (
    DEFAULT_LEVELS,
    SuperpixelConfig,
    SuperpixelViews,
    segment_views,
)

__all__ = [
    "DEFAULT_LEVELS",
    "FEATURE_NAMES",
    "NUM_FACE_PARTS",
    "PART_NAMES",
    "EqualWeightMoE",
    "SuperpixelConfig",
    "SuperpixelMoE",
    "SuperpixelMoEConfig",
    "SuperpixelViews",
    "extract_region_features",
    "FaceLandmarkResult",
    "detect_face_landmarks",
    "landmarks_to_part_masks",
    "load_input",
    "pool_regions",
    "prepare_image",
    "part_distribution_for_labels",
    "segment_views",
    "unknown_part_distributions",
]
