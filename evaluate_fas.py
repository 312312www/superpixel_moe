"""Evaluate any saved A--E checkpoint on a manifest split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fas_moe.checkpoint import load_checkpoint
from fas_moe.data import FASDataset, load_manifest
from fas_moe.metrics import evaluate_scores
from fas_moe.model import FASModel, FASModelConfig
from fas_moe.superpixels import SuperpixelConfig
from train_fas import fas_collate, make_transforms, collect_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--superpixel-cache-dir", type=Path, default=Path("outputs/superpixel_cache"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    manifest = load_manifest(args.manifest)
    if manifest.get("status") == "pending_dataset":
        raise RuntimeError("the requested manifest is a pending OULU-NPU placeholder")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = FASModelConfig.from_dict(payload["model_config"])
    model = FASModel(config).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    superpixel_config = (
        SuperpixelConfig(levels=config.superpixel_levels, cache_dir=args.superpixel_cache_dir)
        if config.use_superpixels
        else None
    )
    dataset = FASDataset(
        manifest["records"],
        split=args.split,
        transform=make_transforms(config.image_size, False),
        superpixel_config=superpixel_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=fas_collate,
    )
    labels, scores, domains, loss = collect_scores(model, loader, device)
    metrics = evaluate_scores(labels, scores, domains, threshold=float(payload["threshold"]))
    metrics.update(
        {
            "loss": loss,
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "phase": config.phase,
            "model_config": asdict(config),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    macro = metrics["macro"]
    print(
        f"[eval {args.split}] phase={config.phase} AUC={macro['auc']:.4f} HTER={macro['hter']:.4f} "
        f"APCER={macro['apcer']:.4f} BPCER={macro['bpcer']:.4f} threshold={metrics['threshold']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
