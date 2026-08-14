"""Run a deliberately short CASIA-FASD training smoke test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from fas_moe import SuperpixelMoE, SuperpixelMoEConfig, restore_image_range


DATASET_PREFIXES = {
    "CASIA-FASD": "casia",
    "Idiap Replay-Attack": "replay",
    "MSU-MFSD": "MSU",
    "OULU-NPU": "Oulu",
}


class NpyBinaryFASDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Lazy RGB loader for one cached domain-generalization dataset."""

    def __init__(
        self,
        dataset_root: str | Path,
        dataset: str,
        limit_samples: int | None = None,
        image_range: str = "auto",
    ) -> None:
        if dataset not in DATASET_PREFIXES:
            raise ValueError(f"unsupported dataset {dataset!r}; choose from {sorted(DATASET_PREFIXES)}")
        if limit_samples is not None and int(limit_samples) < 0:
            raise ValueError("limit_samples must be nonnegative when provided")
        folder = Path(dataset_root) / "domain-generalization" / dataset
        prefix = DATASET_PREFIXES[dataset]
        self.live_path = folder / f"{prefix}_images_live.npy"
        self.spoof_path = folder / f"{prefix}_images_spoof.npy"
        live = np.load(self.live_path, mmap_mode="r", allow_pickle=False)
        spoof = np.load(self.spoof_path, mmap_mode="r", allow_pickle=False)
        try:
            live_shape, live_dtype = live.shape, live.dtype
            spoof_shape, spoof_dtype = spoof.shape, spoof.dtype
            live_max = float(np.asarray(live.max()))
            spoof_max = float(np.asarray(spoof.max()))
        finally:
            for mapped in (live, spoof):
                if isinstance(mapped, np.memmap):
                    mapped._mmap.close()
        if (
            len(live_shape) != 4
            or live_shape[-1] != 3
            or len(spoof_shape) != 4
            or spoof_shape[-1] != 3
        ):
            raise ValueError("cached RGB arrays must have NHWC shape")
        if live_shape[1:] != spoof_shape[1:]:
            raise ValueError(
                f"live/spoof image shapes must match, got {live_shape[1:]} and {spoof_shape[1:]}"
            )
        if not np.issubdtype(live_dtype, np.number) or not np.issubdtype(spoof_dtype, np.number):
            raise TypeError("cached RGB arrays must use numeric dtypes")
        if np.issubdtype(live_dtype, np.integer) != np.issubdtype(spoof_dtype, np.integer):
            raise ValueError(
                f"live/spoof arrays must use compatible numeric dtypes, got {live_dtype} and {spoof_dtype}"
            )
        if live_shape[0] == 0 or spoof_shape[0] == 0:
            raise ValueError("cached RGB arrays must contain at least one sample per class")
        self.live_shape, self.live_dtype = tuple(live_shape), np.dtype(live_dtype)
        self.spoof_shape, self.spoof_dtype = tuple(spoof_shape), np.dtype(spoof_dtype)
        self.live_count = int(live_shape[0])
        self.spoof_count = int(spoof_shape[0])
        if image_range not in ("auto", "0-1/255", "0-1", "0-255"):
            raise ValueError("image_range must be one of auto, 0-1/255, 0-1, 0-255")
        self.image_range = image_range
        self.detected_range: str | None = None
        if image_range == "auto":
            # Infer once over both memmaps.  Per-frame inference would confuse
            # unusually dark [0,1] images with the legacy [0,1/255] format.
            combined_max = max(live_max, spoof_max)
            if np.issubdtype(self.live_dtype, np.integer):
                self.detected_range = "0-255" if combined_max <= 255.0 else "auto"
            elif combined_max <= (1.0 / 255.0) + 1e-6:
                self.detected_range = "0-1/255"
            elif combined_max <= 1.0 + 1e-6:
                self.detected_range = "0-1"
            elif combined_max <= 255.0 + 1e-6:
                self.detected_range = "0-255"
            else:
                raise ValueError(f"unsupported dataset image range: max={combined_max}")
        total = self.live_count + self.spoof_count
        self.length = min(total, int(limit_samples)) if limit_samples is not None else total

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < self.live_count:
            array = self._read_sample(self.live_path, index)
            label = 1
        else:
            spoof_index = index - self.live_count
            if spoof_index >= self.spoof_count:
                raise IndexError(index)
            array = self._read_sample(self.spoof_path, spoof_index)
            label = 0
        # Dataset mirrors can contain the original [0,1/255] cache or ordinary
        # [0,1]/[0,255] RGB arrays.  Keep conversion in one tested helper.
        restored, _ = restore_image_range(array, self.detected_range or self.image_range)
        image = torch.from_numpy(restored).permute(2, 0, 1).contiguous()
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}

    @staticmethod
    def _read_sample(path: Path, index: int) -> np.ndarray:
        """Read one memmapped sample and close the file immediately.

        Keeping a memmap on ``self`` prevents Windows from deleting a temporary
        dataset directory after a test or short training run.  A per-item map
        retains lazy I/O while making file lifetime explicit on every platform.
        """

        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            return np.array(mapped[index], copy=True)
        finally:
            if isinstance(mapped, np.memmap):
                mapped._mmap.close()


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
    parser.add_argument(
        "--image-range",
        choices=("auto", "0-1/255", "0-1", "0-255"),
        default="auto",
        help="numeric range of cached RGB NPY arrays (auto detects from values)",
    )
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
    parser.add_argument("--slic-cache-dir", type=Path, default=Path("outputs/slic_cache"))
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
    dataset = NpyBinaryFASDataset(
        args.dataset_root, args.dataset, args.limit_samples, image_range=args.image_range
    )
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
            slic_cache_dir=str(args.slic_cache_dir) if args.slic_cache_dir else None,
            # Dataset.__getitem__ restores each sample to canonical [0,255]
            # before it reaches the model.  Keep the model/segmentation path
            # explicit so dark canonical tensors are not re-inferred as [0,1].
            image_range="0-255",
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
            # Persist every model field so inference can reject semantic
            # mismatches even when tensor names/shapes happen to coincide.
            "model_config": asdict(model.config),
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
