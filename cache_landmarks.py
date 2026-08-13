"""Precompute MediaPipe landmarks and per-scale part distributions for an NPY FAS dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fas_moe import SuperpixelConfig, infer_image_range, restore_image_range, segment_views
from train_moe import DATASET_PREFIXES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_PREFIXES), default="CASIA-FASD")
    parser.add_argument("--landmark-model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/landmark_cache"))
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--image-range",
        choices=("auto", "0-1/255", "0-1", "0-255"),
        default="auto",
        help="numeric range of cached RGB NPY arrays (auto detects from values)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit_samples is not None and args.limit_samples < 0:
        raise ValueError("limit-samples must be nonnegative when provided")
    folder = args.dataset_root / "domain-generalization" / args.dataset
    prefix = DATASET_PREFIXES[args.dataset]
    array_paths = [
        ("live", folder / f"{prefix}_images_live.npy"),
        ("spoof", folder / f"{prefix}_images_spoof.npy"),
    ]
    arrays = [(kind, np.load(path, mmap_mode="r", allow_pickle=False)) for kind, path in array_paths]
    effective_range = args.image_range
    if effective_range == "auto":
        # Infer once over the complete pair so dark [0,1] frames are not
        # mistaken for the legacy [0,1/255] encoding on a per-image basis.
        maxima = [float(np.asarray(array.max())) for _, array in arrays]
        if any(np.issubdtype(array.dtype, np.integer) and max_value > 255.0 for (_, array), max_value in zip(arrays, maxima)):
            effective_range = "auto"
        else:
            probe = np.asarray([max(maxima)], dtype=np.float32)
            effective_range, _ = infer_image_range(probe)
    config = SuperpixelConfig(
        use_landmarks=True,
        landmark_model_path=args.landmark_model,
        landmark_cache_dir=args.cache_dir,
        # ``restore_image_range`` above returns canonical [0,255] values.
        image_range="0-255",
    )
    processed = detected = cache_hits = 0
    remaining = args.limit_samples
    try:
        for kind, array in arrays:
            count = len(array)
            if remaining is not None:
                count = min(count, max(0, remaining))
            for index in range(count):
                image, _ = restore_image_range(np.asarray(array[index]), effective_range)
                views = segment_views(image, config)
                processed += 1
                detected += int(bool(views.metadata["landmarks_detected"]))
                cache_hits += int(bool(views.metadata["landmark_cache_hit"]))
                print(
                    f"{kind} {index + 1}/{count}: detected={views.metadata['landmarks_detected']} "
                    f"cache_hit={views.metadata['landmark_cache_hit']}"
                )
                if remaining is not None:
                    remaining -= 1
            if remaining is not None and remaining <= 0:
                break
    finally:
        for _, mapped in arrays:
            if isinstance(mapped, np.memmap):
                mapped._mmap.close()
    print(f"Processed: {processed}; detected: {detected}; cache hits: {cache_hits}")
    print(f"Cache: {args.cache_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
