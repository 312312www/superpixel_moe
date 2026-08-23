"""FAS stages A--E: backbone, Native MoE and multi-scale superpixel variants."""

from .backbone import OUT_CHANNELS, build_backbone
from .checkpoint import checkpoint_state, load_checkpoint, validate_checkpoint
from .data import (
    DATASET_CODES,
    DATASETS,
    SPECS,
    DatasetSpec,
    FASDataset,
    available_datasets,
    dataset_available,
    dataset_records,
    generate_manifests,
    load_manifest,
    manifest_sha256,
    subject_split,
)
from .metrics import (
    equal_error_rate,
    error_rates,
    evaluate_scores,
    roc_auc,
    select_macro_hter_threshold,
    threshold_candidates,
)
from .model import (
    FASModel,
    FASModelConfig,
    MixStyle,
    NaiveMoE,
    NativeMoE,
    PHASES,
    gradient_reverse,
    pool_superpixel_regions,
    total_parameters,
    trainable_parameters,
)
from .superpixels import DEFAULT_LEVELS, SUPERPIXEL_SCHEMA, SuperpixelConfig, cached_superpixels, segment_image

__all__ = [
    "DATASET_CODES", "DATASETS", "DatasetSpec", "OUT_CHANNELS", "SPECS", "FASDataset",
    "FASModel", "FASModelConfig", "MixStyle", "NaiveMoE", "NativeMoE", "PHASES", "SuperpixelConfig",
    "DEFAULT_LEVELS", "SUPERPIXEL_SCHEMA", "available_datasets", "build_backbone", "cached_superpixels",
    "checkpoint_state", "dataset_available", "dataset_records", "equal_error_rate", "error_rates",
    "evaluate_scores", "generate_manifests", "gradient_reverse", "load_checkpoint", "load_manifest",
    "manifest_sha256", "pool_superpixel_regions", "roc_auc", "segment_image",
    "select_macro_hter_threshold", "subject_split", "threshold_candidates", "total_parameters",
    "trainable_parameters", "validate_checkpoint",
]
