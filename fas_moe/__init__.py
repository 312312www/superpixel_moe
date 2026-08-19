"""Minimal Superpixel-MoE baseline for face anti-spoofing."""

from .features import FEATURE_NAMES, extract_region_features
from .checkpoint import checkpoint_state, load_checkpoint, validate_checkpoint
from .datasets import (
    DATASET_SPECS,
    LODO_PROTOCOLS,
    FixedDomainClassBatchSampler,
    ManifestFASDataset,
    NpyBinaryFASDataset,
    generate_lodo_manifests,
    load_manifest,
)
from .face_parts import (
    NUM_FACE_PARTS,
    PART_NAMES,
    landmarks_to_part_masks,
    part_distribution_for_labels,
    unknown_part_distributions,
)
from .io import infer_image_range, load_input, prepare_image, restore_image_range
from .landmarks import FaceLandmarkResult, detect_face_landmarks
from .metrics import evaluate_scores, select_macro_hter_threshold
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
    "DATASET_SPECS",
    "FEATURE_NAMES",
    "NUM_FACE_PARTS",
    "PART_NAMES",
    "EqualWeightMoE",
    "FixedDomainClassBatchSampler",
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
    "load_manifest",
    "LODO_PROTOCOLS",
    "ManifestFASDataset",
    "NpyBinaryFASDataset",
    "pool_regions",
    "prepare_image",
    "generate_lodo_manifests",
    "restore_image_range",
    "part_distribution_for_labels",
    "segment_views",
    "unknown_part_distributions",
    "validate_checkpoint",
    "evaluate_scores",
    "select_macro_hter_threshold",
]
