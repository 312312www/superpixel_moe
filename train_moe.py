"""Run a deliberately short CASIA-FASD training smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from fas_moe import SuperpixelMoE, SuperpixelMoEConfig


DATASET_PREFIXES = {
    "CASIA-FASD": "casia",
    "Idiap Replay-Attack": "replay",
    "MSU-MFSD": "MSU",
    "OULU-NPU": "Oulu",
}


class NpyBinaryFASDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Lazy RGB loader for one cached domain-generalization dataset."""

    def __init__(self, dataset_root: str | Path, dataset: str, limit_samples: int | None = None) -> None:
        if dataset not in DATASET_PREFIXES:
            raise ValueError(f"unsupported dataset {dataset!r}; choose from {sorted(DATASET_PREFIXES)}")
        folder = Path(dataset_root) / "domain-generalization" / dataset
        prefix = DATASET_PREFIXES[dataset]
        self.live = np.load(folder / f"{prefix}_images_live.npy", mmap_mode="r", allow_pickle=False)
        self.spoof = np.load(folder / f"{prefix}_images_spoof.npy", mmap_mode="r", allow_pickle=False)
        if self.live.ndim != 4 or self.live.shape[-1] != 3 or self.spoof.ndim != 4 or self.spoof.shape[-1] != 3:
            raise ValueError("cached RGB arrays must have NHWC shape")
        self.live_count = int(self.live.shape[0])
        self.spoof_count = int(self.spoof.shape[0])
        total = self.live_count + self.spoof_count
        self.length = min(total, int(limit_samples)) if limit_samples is not None else total

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < self.live_count:
            array = np.asarray(self.live[index], dtype=np.float32)
            label = 1
        else:
            spoof_index = index - self.live_count
            if spoof_index >= self.spoof_count:
                raise IndexError(index)
            array = np.asarray(self.spoof[spoof_index], dtype=np.float32)
            label = 0
        # The cached RGB files are in [0, 1/255]. Restore [0, 255].
        image = torch.from_numpy(array * (255.0 * 255.0)).permute(2, 0, 1).contiguous()
        image = image.clamp(0.0, 255.0)
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_PREFIXES), default="CASIA-FASD")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--landmarks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--landmark-model", type=Path, default=Path("models/face_landmarker.task"),
        help="MediaPipe face_landmarker.task",
    )
    parser.add_argument("--landmark-cache-dir", type=Path, default=Path("outputs/landmark_cache"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train_smoke"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.max_steps < 1:
        raise ValueError("batch-size, epochs and max-steps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = NpyBinaryFASDataset(args.dataset_root, args.dataset, args.limit_samples)
    if len(dataset) < 2:
        raise RuntimeError("dataset must contain at least two samples")
    labels = np.asarray([1] * min(dataset.live_count, len(dataset)) + [0] * max(0, len(dataset) - dataset.live_count))
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    sample_weights = np.asarray([1.0 / max(counts[label], 1.0) for label in labels], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True, num_workers=0)
    device = _device(args.device)
    model = SuperpixelMoE(
        SuperpixelMoEConfig(
            pretrained_backbone=args.pretrained,
            freeze_backbone=not args.train_backbone,
            weights_path=str(args.weights_path) if args.weights_path else None,
            use_landmarks=args.landmarks,
            landmark_model_path=str(args.landmark_model) if args.landmark_model else None,
            landmark_cache_dir=str(args.landmark_cache_dir) if args.landmark_cache_dir else None,
        )
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate
    )
    history: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            if step >= args.max_steps:
                break
            images = batch["image"].to(device)
            labels_tensor = batch["label"].to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            item = {
                "epoch": epoch,
                "step": step,
                "batch_size": int(images.shape[0]),
                "loss": float(loss.detach().cpu()),
            }
            history.append(item)
            print(item)
    if not history:
        raise RuntimeError("no complete batch was available; reduce batch-size or increase limit-samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "levels": [128, 64, 16],
                "pretrained_backbone": args.pretrained,
                "freeze_backbone": not args.train_backbone,
                "use_landmarks": args.landmarks,
                "landmark_model_path": str(args.landmark_model) if args.landmark_model else None,
            },
            "history": history,
        },
        args.output_dir / "checkpoint.pt",
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {(args.output_dir / 'checkpoint.pt').resolve()}")
    print("Training smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
