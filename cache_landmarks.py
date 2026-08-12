"""Precompute MediaPipe landmarks and per-scale part distributions for an NPY FAS dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fas_moe import SuperpixelConfig, segment_views
from train_moe import DATASET_PREFIXES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_PREFIXES), default="CASIA-FASD")
    parser.add_argument("--landmark-model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/landmark_cache"))
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    folder = args.dataset_root / "domain-generalization" / args.dataset
    prefix = DATASET_PREFIXES[args.dataset]
    arrays = [
        ("live", np.load(folder / f"{prefix}_images_live.npy", mmap_mode="r", allow_pickle=False)),
        ("spoof", np.load(folder / f"{prefix}_images_spoof.npy", mmap_mode="r", allow_pickle=False)),
    ]
    config = SuperpixelConfig(
        use_landmarks=True,
        landmark_model_path=args.landmark_model,
        landmark_cache_dir=args.cache_dir,
    )
    processed = detected = cache_hits = 0
    remaining = args.limit_samples
    for kind, array in arrays:
        count = len(array)
        if remaining is not None:
            count = min(count, max(0, remaining))
        for index in range(count):
            image = np.clip(
                np.asarray(array[index], dtype=np.float32) * (255.0 * 255.0), 0.0, 255.0
            )
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
    print(f"Processed: {processed}; detected: {detected}; cache hits: {cache_hits}")
    print(f"Cache: {args.cache_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
