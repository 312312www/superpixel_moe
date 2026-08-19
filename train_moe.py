"""Train A--E Superpixel-MoE ablations in smoke or locked LODO mode."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from fas_moe import (
    DATASET_SPECS,
    FixedDomainClassBatchSampler,
    ManifestFASDataset,
    NpyBinaryFASDataset,
    SuperpixelMoE,
    SuperpixelMoEConfig,
    evaluate_scores,
    load_checkpoint,
    load_manifest,
    select_macro_hter_threshold,
)


DATASET_PREFIXES = {name: spec.prefix for name, spec in DATASET_SPECS.items()}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="locked OCI_M/OMI_C/OCM_I/ICM_O JSON; omit for a single-dataset smoke run",
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_PREFIXES), default="CASIA-FASD")
    parser.add_argument("--experiment", choices=tuple("ABCDE"), default="E")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--image-range", choices=("auto", "0-1/255", "0-1", "0-255"), default="0-1/255",
        help="numeric range of the source RGB NPY arrays",
    )
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--module-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="legacy alias that overrides module LR (and the smoke-run backbone LR)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--freeze-batch-norm", action=argparse.BooleanOptionalAction, default=True,
        help="train convolution weights while keeping every ResNet BatchNorm in eval with frozen affine params",
    )
    parser.add_argument(
        "--landmark-model", type=Path, default=Path("models/face_landmarker.task"),
        help="MediaPipe face_landmarker.task",
    )
    parser.add_argument("--landmark-cache-dir", type=Path, default=Path("outputs/landmark_cache"))
    parser.add_argument("--slic-cache-dir", type=Path, default=Path("outputs/slic_cache"))
    parser.add_argument(
        "--allow-cache-miss", action="store_true",
        help="allow SLIC/MediaPipe computation; formal manifest runs are strict by default",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-test", action="store_true", help="do not reveal/evaluate the target domain")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train_smoke"))
    return parser


def _model(args: argparse.Namespace, *, formal: bool) -> SuperpixelMoE:
    experiment = str(args.experiment).upper()
    uses_superpixel = experiment in "CDE"
    uses_landmarks = experiment in "DE"
    strict = formal and not args.allow_cache_miss
    config = SuperpixelMoEConfig(
        experiment=experiment,
        pretrained_backbone=args.pretrained,
        freeze_backbone=not args.train_backbone,
        freeze_batch_norm=args.freeze_batch_norm,
        weights_path=str(args.weights_path) if args.weights_path else None,
        landmark_model_path=str(args.landmark_model) if args.landmark_model else None,
        landmark_cache_dir=str(args.landmark_cache_dir) if args.landmark_cache_dir else None,
        slic_cache_dir=str(args.slic_cache_dir) if args.slic_cache_dir else None,
        require_slic_cache=bool(strict and uses_superpixel),
        require_landmark_cache=bool(strict and uses_landmarks),
        image_range="0-255",
    )
    return SuperpixelMoE(config)


def _optimizer(model: SuperpixelMoE, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in backbone}
    modules = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    module_lr = float(args.learning_rate or args.module_learning_rate)
    backbone_lr = float(args.learning_rate or args.backbone_learning_rate)
    groups: list[dict[str, Any]] = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
    if modules:
        groups.append({"params": modules, "lr": module_lr, "name": "new_modules"})
    if not groups:
        raise RuntimeError("model has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))


def _checkpoint_payload(
    model: SuperpixelMoE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    best_metric: float,
    threshold: float | None,
    protocol: str | None,
    seed: int,
    manifest_sha256: str | None,
    history: list[dict[str, Any]],
    git_commit: str | None,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_config": asdict(model.config),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "validation_threshold": None if threshold is None else float(threshold),
        "protocol": protocol,
        "seed": int(seed),
        "manifest_sha256": manifest_sha256,
        "history": history,
        "git_commit": git_commit,
    }


def _collect_scores(
    model: SuperpixelMoE, loader: DataLoader[Any], device: torch.device
) -> tuple[list[int], list[float], list[str], float]:
    was_training = model.training
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    domains: list[str] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            logits, _ = model(images)
            losses.append(float(criterion(logits, targets).detach().cpu()))
            labels.extend(int(value) for value in targets.cpu().tolist())
            scores.extend(float(value) for value in torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            batch_domains = batch.get("domain", ["single"] * len(targets))
            domains.extend(str(value) for value in batch_domains)
    if was_training:
        model.train()
    return labels, scores, domains, float(np.mean(losses))


def _train_step_loop(
    model: SuperpixelMoE,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp_enabled: bool,
) -> tuple[float, int, int]:
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    samples = 0
    optimizer_updates = 0
    model.train()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits, _ = model(images)
            loss = criterion(logits, targets)
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not amp_enabled or float(scaler.get_scale()) >= scale_before:
            optimizer_updates += 1
        losses.append(float(loss.detach().cpu()))
        samples += int(images.shape[0])
    if not losses:
        raise RuntimeError("no complete training batch was available")
    if optimizer_updates < 1:
        raise RuntimeError("AMP skipped every optimizer update because gradients overflowed")
    return float(np.mean(losses)), samples, optimizer_updates


def _run_smoke(args: argparse.Namespace, device: torch.device) -> int:
    dataset = NpyBinaryFASDataset(
        args.dataset_root, args.dataset, args.limit_samples, image_range=args.image_range
    )
    if len(dataset) < 2:
        raise RuntimeError("dataset must contain at least two samples")
    labels = np.asarray([1] * dataset.live_count + [0] * dataset.spoof_count)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    sampler_generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights, len(dataset), replacement=True, generator=sampler_generator
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True, num_workers=0
    )
    model = _model(args, formal=False).to(device)
    optimizer = _optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=1024.0)
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(loader):
            if args.max_steps is not None and step >= args.max_steps:
                break
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, _ = model(images)
                loss = nn.functional.cross_entropy(logits, targets)
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if amp_enabled and float(scaler.get_scale()) < scale_before:
                raise RuntimeError("AMP skipped the smoke-test optimizer update due to overflow")
            item = {
                "epoch": epoch,
                "step": step,
                "batch_size": int(images.shape[0]),
                "loss": float(loss.detach().cpu()),
            }
            history.append(item)
            print(item)
        scheduler.step()
    if not history:
        raise RuntimeError("no training step ran; reduce batch size or increase limit-samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent
    payload = _checkpoint_payload(
        model, optimizer, scheduler, scaler,
        epoch=args.epochs - 1,
        best_metric=float("nan"), threshold=None, protocol=None,
        seed=args.seed, manifest_sha256=None, history=history,
        git_commit=_git_commit(project_root),
    )
    torch.save(payload, args.output_dir / "checkpoint.pt")
    _write_json(args.output_dir / "history.json", history)
    _write_json(args.output_dir / "config.json", vars(args))
    print(f"Device: {device}")
    print(f"Experiment: {model.config.experiment}")
    print(f"Checkpoint: {(args.output_dir / 'checkpoint.pt').resolve()}")
    print("Training smoke: PASS")
    return 0


def _run_formal(args: argparse.Namespace, device: torch.device) -> int:
    assert args.manifest is not None
    manifest = load_manifest(args.manifest)
    dataset_root = args.dataset_root.resolve()
    records = manifest["records"]
    train_dataset = ManifestFASDataset(dataset_root, records, split="train", image_range=args.image_range)
    val_dataset = ManifestFASDataset(dataset_root, records, split="val", image_range=args.image_range)
    test_dataset = ManifestFASDataset(dataset_root, records, split="test", image_range=args.image_range)
    total_batches = math.ceil(len(train_dataset) / args.batch_size)
    if args.max_steps is not None:
        total_batches = min(total_batches, args.max_steps)
    batch_sampler = FixedDomainClassBatchSampler(
        train_dataset.records, batch_size=args.batch_size, seed=args.seed, num_batches=total_batches
    )
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = _model(args, formal=True).to(device)
    optimizer = _optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=1024.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent
    git_commit = _git_commit(project_root)
    manifest_hash = _manifest_sha256(args.manifest)
    history: list[dict[str, Any]] = []
    batch_order: list[dict[str, Any]] = []
    best_auc = -float("inf")
    best_hter = float("inf")
    best_threshold: float | None = None

    for epoch in range(args.epochs):
        batch_sampler.set_epoch(epoch)
        planned = batch_sampler.batches()
        sample_ids = [
            [train_dataset.records[index]["sample_id"] for index in batch]
            for batch in planned
        ]
        order_text = json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":"))
        batch_order.append(
            {
                "epoch": epoch,
                "sha256": hashlib.sha256(order_text.encode("utf-8")).hexdigest(),
                "batches": sample_ids,
            }
        )
        train_loss, train_samples, optimizer_updates = _train_step_loop(
            model, train_loader, optimizer, scaler, device, amp_enabled=amp_enabled
        )
        val_labels, val_scores, val_domains, val_loss = _collect_scores(model, val_loader, device)
        threshold_report = select_macro_hter_threshold(val_labels, val_scores, val_domains)
        val_metrics = evaluate_scores(
            val_labels, val_scores, val_domains, threshold=float(threshold_report["threshold"])
        )
        item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_samples": train_samples,
            "optimizer_updates": optimizer_updates,
            "val_loss": val_loss,
            "val_macro_auc": val_metrics["macro"]["auc"],
            "val_macro_hter": val_metrics["macro"]["hter"],
            "threshold": threshold_report,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        }
        history.append(item)
        print(item)
        current_auc = float(val_metrics["macro"]["auc"])
        current_hter = float(val_metrics["macro"]["hter"])
        improved = current_auc > best_auc + 1e-12 or (
            abs(current_auc - best_auc) <= 1e-12 and current_hter < best_hter - 1e-12
        )
        if improved:
            best_auc, best_hter = current_auc, current_hter
            best_threshold = float(threshold_report["threshold"])
            torch.save(
                _checkpoint_payload(
                    model, optimizer, scheduler, scaler, epoch=epoch,
                    best_metric=best_auc, threshold=best_threshold,
                    protocol=str(manifest["protocol"]), seed=args.seed,
                    manifest_sha256=manifest_hash, history=history, git_commit=git_commit,
                ),
                args.output_dir / "best_auc.pt",
            )
            _write_json(args.output_dir / "metrics_val.json", val_metrics)
        scheduler.step()
        torch.save(
            _checkpoint_payload(
                model, optimizer, scheduler, scaler, epoch=epoch,
                best_metric=best_auc, threshold=best_threshold,
                protocol=str(manifest["protocol"]), seed=args.seed,
                manifest_sha256=manifest_hash, history=history, git_commit=git_commit,
            ),
            args.output_dir / "last.pt",
        )
        _write_json(args.output_dir / "history.json", history)
        _write_json(args.output_dir / "batch_order.json", batch_order)

    if best_threshold is None:
        raise RuntimeError("training produced no best validation checkpoint")
    if not args.skip_test:
        load_checkpoint(
            model,
            args.output_dir / "best_auc.pt",
            map_location=device,
            expected_metadata={
                "protocol": str(manifest["protocol"]),
                "manifest_sha256": manifest_hash,
                "seed": int(args.seed),
            },
        )
        test_labels, test_scores, test_domains, test_loss = _collect_scores(model, test_loader, device)
        test_metrics = evaluate_scores(test_labels, test_scores, test_domains, threshold=best_threshold)
        test_metrics["loss"] = test_loss
        test_metrics["threshold_source"] = "source-validation minimum Macro-HTER"
        _write_json(args.output_dir / "metrics_test.json", test_metrics)

    config = dict(vars(args))
    config.update(
        {
            "protocol": manifest["protocol"],
            "manifest_sha256": manifest_hash,
            "model_config": asdict(model.config),
            "evaluation_unit": "one local NPY array row",
            "augmentation": "none",
            "git_commit": git_commit,
        }
    )
    _write_json(args.output_dir / "config.json", config)
    (args.output_dir / "manifest.json").write_bytes(args.manifest.read_bytes())
    _write_json(
        args.output_dir / "environment.json",
        {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": str(device), "git_commit": git_commit,
        },
    )
    (args.output_dir / "git_commit.txt").write_text(
        str(git_commit or "unknown") + "\n", encoding="utf-8"
    )
    print(f"Protocol: {manifest['protocol']}")
    print(f"Experiment: {model.config.experiment}")
    print(f"Best source-val Macro-AUC: {best_auc:.6f}")
    print(f"Best checkpoint: {(args.output_dir / 'best_auc.pt').resolve()}")
    print("Formal training pipeline: PASS")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch-size and epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive when supplied")
    if args.backbone_learning_rate <= 0 or args.module_learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rates must be positive and weight decay nonnegative")
    _set_seed(args.seed)
    device = _device(args.device)
    if args.manifest is None:
        return _run_smoke(args, device)
    return _run_formal(args, device)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DATASET_PREFIXES", "NpyBinaryFASDataset", "build_parser", "main"]
