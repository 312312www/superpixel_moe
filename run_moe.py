"""Run one image through the minimal Superpixel-MoE baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from fas_moe import (
    SuperpixelConfig,
    SuperpixelMoE,
    SuperpixelMoEConfig,
    load_checkpoint,
    load_input,
    segment_views,
)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JPG/PNG/BMP/TIFF image or NPY array")
    parser.add_argument("--index", type=int, default=0, help="sample index for batched NPY")
    parser.add_argument("--output", type=Path, default=Path("outputs/moe_demo"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="optional train_moe checkpoint")
    parser.add_argument("--weights-path", type=Path, default=None, help="optional local ResNet-50 weights")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--landmarks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--landmark-model", type=Path, default=Path("models/face_landmarker.task"),
        help="MediaPipe face_landmarker.task",
    )
    parser.add_argument("--landmark-cache-dir", type=Path, default=Path("outputs/landmark_cache"))
    parser.add_argument("--slic-cache-dir", type=Path, default=Path("outputs/slic_cache"))
    parser.add_argument(
        "--image-range",
        choices=("auto", "0-1/255", "0-1", "0-255"),
        default="auto",
        help="numeric range of an NPY RGB input; use 0-255 for dark canonical float data",
    )
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image, input_metadata = load_input(args.input, args.index)
    views = segment_views(
        image,
        SuperpixelConfig(
            use_landmarks=args.landmarks,
            landmark_model_path=args.landmark_model,
            landmark_cache_dir=args.landmark_cache_dir,
            slic_cache_dir=args.slic_cache_dir,
            image_range=args.image_range,
        ),
    )
    device = _device(args.device)
    model = SuperpixelMoE(
        SuperpixelMoEConfig(
            pretrained_backbone=args.pretrained,
            weights_path=str(args.weights_path) if args.weights_path else None,
            use_landmarks=args.landmarks,
            landmark_model_path=str(args.landmark_model) if args.landmark_model else None,
            landmark_cache_dir=str(args.landmark_cache_dir) if args.landmark_cache_dir else None,
            slic_cache_dir=str(args.slic_cache_dir) if args.slic_cache_dir else None,
            # ``views.image`` has already been normalized by segment_views to
            # canonical uint8 [0,255]; do not apply the source-range scale a
            # second time in the model.
            image_range="0-255",
        )
    ).to(device)
    checkpoint_report: dict[str, object] | None = None
    if args.checkpoint is not None:
        # Preflight validates every key/shape and structural config field, then
        # performs an exact strict load.  Incompatible checkpoints stop before
        # any logits are produced.
        checkpoint_report = load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        logits, details = model(
            torch.from_numpy(views.image).permute(2, 0, 1).unsqueeze(0).float().to(device),
            views=views,
        )
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(views.image).save(output / "input.png")
    for level in sorted(views.labels, reverse=True):
        np.save(output / f"labels_{level:03d}.npy", views.labels[level])
        np.save(output / f"features_{level:03d}.npy", views.features[level])
        np.save(output / f"edges_{level:03d}.npy", views.edges[level])
        np.save(output / f"positions_{level:03d}.npy", views.positions[level])
        np.save(output / f"part_distribution_{level:03d}.npy", views.part_distributions[level])
        np.save(output / f"tokens_{level:03d}.npy", details[f"tokens_{level}"][0].cpu().numpy())

    summary = {
        **views.metadata,
        **input_metadata,
        "device": str(device),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_validation": checkpoint_report,
        "pretrained_backbone": bool(args.pretrained),
        "landmarks_enabled": bool(args.landmarks),
        "levels": {str(level): int(np.unique(views.labels[level]).size) for level in views.labels},
        "logits": logits[0].cpu().tolist(),
        "probabilities": {"spoof": float(probabilities[0]), "live": float(probabilities[1])},
        "runtime_seconds": round(time.perf_counter() - started, 4),
        "note": "Probabilities are not meaningful until a trained checkpoint is supplied.",
    }
    _save_json(output / "summary.json", summary)
    print(f"Output: {output.resolve()}")
    print(f"Device: {device}")
    print(f"Levels: {summary['levels']}")
    print(f"Logits: {summary['logits']}")
    print("Forward: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
