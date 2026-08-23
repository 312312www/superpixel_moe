"""Train and evaluate FAS stages A--E.

A: ResNet classifier.
B: A + global top-k Native MoE.
C: B + multi-scale SLIC tokens + shared Naive MoE.
D: C + normalized region-position encoding.
E: D + Fine/Medium/Coarse scale-specific Naive MoEs.

LODO training optionally enables class/domain-balanced batches, MixStyle and a
DANN gradient-reversal auxiliary loss. Target-domain labels are never used for
model selection or threshold selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torchvision.transforms as T

from fas_moe.checkpoint import load_checkpoint
from fas_moe.data import FASDataset, load_manifest, manifest_sha256
from fas_moe.metrics import evaluate_scores, select_macro_hter_threshold
from fas_moe.model import FASModel, FASModelConfig, total_parameters, trainable_parameters
from fas_moe.superpixels import SuperpixelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    parser.add_argument("--phase", choices=("A", "B", "C", "D", "E"), required=True)
    parser.add_argument("--backbone", choices=("resnet50", "resnet34"), default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5, help="new-module learning rate")
    parser.add_argument("--backbone-lr", type=float, default=None, help="default: lr * 0.2")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=1024)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--augment", choices=("standard", "strong"), default="standard")
    parser.add_argument("--superpixel-cache-dir", type=Path, default=Path("outputs/superpixel_cache"))
    parser.add_argument("--mixstyle-prob", type=float, default=None)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    parser.add_argument("--domain-loss-weight", type=float, default=None)
    parser.add_argument("--domain-warmup-epochs", type=int, default=5)
    parser.add_argument("--domain-balanced-batches", action=argparse.BooleanOptionalAction, default=True)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(
    image_size: int,
    train: bool,
    augment: str = "standard",
    *,
    preserve_geometry: bool = False,
):
    """Keep C--E geometry fixed; A/B can also use crop/flip augmentation."""
    if not train:
        return T.Compose([T.ToPILImage(), T.Resize((image_size, image_size)), T.ToTensor(), T.Lambda(lambda x: x * 255.0)])
    color_ops: list[Any]
    if augment == "strong":
        color_ops = [
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
    else:
        color_ops = [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)]
    if preserve_geometry:
        return T.Compose([T.ToPILImage(), T.Resize((image_size, image_size)), *color_ops, T.ToTensor(), T.Lambda(lambda x: x * 255.0)])
    crop = (
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33))
        if augment == "strong"
        else T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25))
    )
    return T.Compose([T.ToPILImage(), crop, T.RandomHorizontalFlip(), *color_ops, T.ToTensor(), T.Lambda(lambda x: x * 255.0)])


def fas_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "dataset": [str(item["dataset"]) for item in batch],
        "subject_id": torch.stack([item["subject_id"] for item in batch]),
        "sample_id": [str(item["sample_id"]) for item in batch],
    }
    if "superpixel_labels" in batch[0]:
        keys = sorted(batch[0]["superpixel_labels"])
        output["superpixel_labels"] = {key: torch.stack([item["superpixel_labels"][key] for item in batch]) for key in keys}
        output["superpixel_positions"] = {key: torch.stack([item["superpixel_positions"][key] for item in batch]) for key in keys}
        output["superpixel_valid"] = {key: torch.stack([item["superpixel_valid"][key] for item in batch]) for key in keys}
    return output


class BalancedDomainClassBatchSampler(Sampler[list[int]]):
    """Deterministic replacement sampling balanced across source domain/class groups."""

    def __init__(self, records: Sequence[dict[str, Any]], *, batch_size: int, seed: int, num_batches: int) -> None:
        self.records = list(records)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.num_batches = int(num_batches)
        self.epoch = 0
        self.groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.groups[(str(record["dataset"]), int(record["label"]))].append(index)
        if not self.groups or any(not values for values in self.groups.values()):
            raise ValueError("every domain/class group needs at least one record")
        self.keys = sorted(self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
        pools: dict[tuple[str, int], list[int]] = {}
        offsets: dict[tuple[str, int], int] = {}

        def draw(key: tuple[str, int]) -> int:
            pool = pools.get(key)
            offset = offsets.get(key, 0)
            if pool is None or offset >= len(pool):
                pool = list(self.groups[key])
                rng.shuffle(pool)
                pools[key] = pool
                offset = 0
            offsets[key] = offset + 1
            return pool[offset]

        base, remainder = divmod(self.batch_size, len(self.keys))
        for batch_index in range(self.num_batches):
            quotas = {key: base for key in self.keys}
            for offset in range(remainder):
                quotas[self.keys[(batch_index * max(1, remainder) + offset) % len(self.keys)]] += 1
            batch = [draw(key) for key in self.keys for _ in range(quotas[key])]
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def build_loaders(
    manifest: dict[str, Any],
    image_size: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    augment: str,
    *,
    superpixel_config: SuperpixelConfig | None,
    domain_balanced: bool,
):
    records = manifest["records"]
    preserve_geometry = superpixel_config is not None
    train_dataset = FASDataset(
        records, split="train", transform=make_transforms(image_size, True, augment, preserve_geometry=preserve_geometry), superpixel_config=superpixel_config
    )
    val_dataset = FASDataset(records, split="val", transform=make_transforms(image_size, False), superpixel_config=superpixel_config)
    test_dataset = FASDataset(records, split="test", transform=make_transforms(image_size, False), superpixel_config=superpixel_config)
    common = {"num_workers": num_workers, "collate_fn": fas_collate, "pin_memory": False}
    source_domains = {str(record["dataset"]) for record in train_dataset.records}
    if domain_balanced:
        sampler = BalancedDomainClassBatchSampler(
            train_dataset.records,
            batch_size=batch_size,
            seed=seed,
            num_batches=math.ceil(len(train_dataset) / batch_size),
        )
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, **common)
    else:
        labels = np.asarray([int(record["label"]) for record in train_dataset.records])
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError(f"training split must contain both classes, counts={counts.tolist()}")
        weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(labels), replacement=True, generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, **common)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, sorted(source_domains)


def _forward_batch(
    model: FASModel,
    batch: dict[str, Any],
    device: torch.device,
    *,
    domain_adversarial_coefficient: float = 0.0,
):
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    kwargs: dict[str, Any] = {"domain_adversarial_coefficient": domain_adversarial_coefficient}
    if "superpixel_labels" in batch:
        kwargs["superpixel_labels"] = batch["superpixel_labels"]
        kwargs["superpixel_positions"] = batch["superpixel_positions"]
        kwargs["superpixel_valid"] = batch["superpixel_valid"]
    logits, details = model(images, **kwargs)
    return labels, logits, details


def collect_scores(model: FASModel, loader: DataLoader, device: torch.device):
    was_training = model.training
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    domains: list[str] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            targets, logits, _ = _forward_batch(model, batch, device)
            losses.append(float(nn.functional.cross_entropy(logits, targets).cpu()))
            labels.extend(int(value) for value in targets.cpu().tolist())
            scores.extend(float(value) for value in torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            domains.extend(batch["dataset"])
    if was_training:
        model.train()
    return labels, scores, domains, float(np.mean(losses))


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _optimizer(model: FASModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_params = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in backbone_params}
    other_params = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in backbone_ids]
    backbone_lr = float(args.backbone_lr if args.backbone_lr is not None else args.lr * 0.2)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": float(args.lr)},
        ],
        weight_decay=float(args.weight_decay),
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("epochs, batch-size and eval-every must be positive")
    set_seed(args.seed)
    device = _resolve_device(args.device)
    manifest = load_manifest(args.manifest)
    if manifest.get("status") == "pending_dataset":
        raise RuntimeError(f"manifest is a placeholder; add the missing dataset first: {args.manifest}")
    is_cross_domain = str(manifest["protocol"]) == "lodo"
    source_domains = sorted({str(record["dataset"]) for record in manifest["records"] if record["split"] == "train"})
    mixstyle_prob = float(args.mixstyle_prob) if args.mixstyle_prob is not None else (0.5 if is_cross_domain else 0.0)
    domain_loss_weight = float(args.domain_loss_weight) if args.domain_loss_weight is not None else (0.05 if is_cross_domain and len(source_domains) > 1 else 0.0)
    domain_index = {name: index for index, name in enumerate(source_domains)}

    cfg = FASModelConfig(
        phase=args.phase,
        backbone=args.backbone,
        pretrained=True,
        weights_path=str(args.weights_path) if args.weights_path else None,
        image_size=args.image_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        expert_hidden_dim=args.expert_hidden,
        balance_loss_weight=args.balance_weight,
        mixstyle_prob=mixstyle_prob,
        mixstyle_alpha=args.mixstyle_alpha,
        num_domains=max(1, len(source_domains)),
    )
    superpixel_config = SuperpixelConfig(levels=cfg.superpixel_levels, cache_dir=args.superpixel_cache_dir) if cfg.use_superpixels else None
    model = FASModel(cfg).to(device)
    print(f"[setup] phase={args.phase} protocol={manifest['protocol_name']} cross_domain={is_cross_domain}")
    print(f"[setup] params total={total_parameters(model):,} trainable={trainable_parameters(model):,} domains={source_domains}")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, source_domains = build_loaders(
        manifest,
        args.image_size,
        args.batch_size,
        args.seed,
        args.num_workers,
        args.augment,
        superpixel_config=superpixel_config,
        domain_balanced=bool(args.domain_balanced_batches),
    )
    domain_index = {name: index for index, name in enumerate(source_domains)}
    print(f"[setup] train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    optimizer = _optimizer(model, args)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_auc, best_hter, best_threshold, best_epoch = -1.0, float("inf"), None, -1
    start = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        if isinstance(train_loader.batch_sampler, BalancedDomainClassBatchSampler):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            progress = global_step / max(1, total_steps - 1)
            grl = 0.0
            if domain_loss_weight > 0.0 and epoch >= args.domain_warmup_epochs:
                local_progress = (epoch - args.domain_warmup_epochs) / max(1, args.epochs - args.domain_warmup_epochs)
                grl = 2.0 / (1.0 + math.exp(-10.0 * local_progress)) - 1.0
            optimizer.zero_grad(set_to_none=True)
            targets, logits, details = _forward_batch(model, batch, device, domain_adversarial_coefficient=grl)
            loss = criterion(logits, targets) + args.balance_weight * details["balance_loss"]
            domain_loss_value = 0.0
            if domain_loss_weight > 0.0 and "domain_logits" in details:
                domain_targets = torch.tensor([domain_index[str(name)] for name in batch["dataset"]], dtype=torch.long, device=device)
                domain_loss = nn.functional.cross_entropy(details["domain_logits"], domain_targets)
                loss = loss + domain_loss_weight * domain_loss
                domain_loss_value = float(domain_loss.detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))
            global_step += 1
        item: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "lr_backbone": float(scheduler.get_last_lr()[0]),
            "lr_modules": float(scheduler.get_last_lr()[1]),
            "domain_loss": domain_loss_value,
            "grl": grl,
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            val_labels, val_scores, val_domains, val_loss = collect_scores(model, val_loader, device)
            threshold_report = select_macro_hter_threshold(val_labels, val_scores, val_domains)
            val_metrics = evaluate_scores(val_labels, val_scores, val_domains, threshold=float(threshold_report["threshold"]))
            item.update(
                {
                    "val_loss": val_loss,
                    "val_macro_auc": val_metrics["macro"]["auc"],
                    "val_macro_hter": val_metrics["macro"]["hter"],
                    "val_threshold": float(threshold_report["threshold"]),
                }
            )
            current_auc, current_hter = float(val_metrics["macro"]["auc"]), float(val_metrics["macro"]["hter"])
            if current_auc > best_auc + 1e-12 or (abs(current_auc - best_auc) <= 1e-12 and current_hter < best_hter - 1e-12):
                best_auc, best_hter = current_auc, current_hter
                best_threshold, best_epoch = float(threshold_report["threshold"]), epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": asdict(cfg),
                        "epoch": epoch,
                        "best_auc": best_auc,
                        "best_hter": best_hter,
                        "threshold": best_threshold,
                        "protocol": manifest["protocol"],
                        "manifest_sha256": manifest_sha256(args.manifest),
                    },
                    args.output_dir / "best_auc.pt",
                )
                (args.output_dir / "metrics_val.json").write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        history.append(item)
        (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[epoch {epoch:02d}] train={item['train_loss']:.4f} val_auc={item.get('val_macro_auc', float('nan')):.4f} "
            f"val_hter={item.get('val_macro_hter', float('nan')):.4f} lr={item['lr_modules']:.2e} ({time.time()-start:.0f}s)",
            flush=True,
        )

    if best_threshold is None:
        raise RuntimeError("no validation checkpoint produced")
    load_checkpoint(model, args.output_dir / "best_auc.pt", map_location=device)
    test_labels, test_scores, test_domains, test_loss = collect_scores(model, test_loader, device)
    test_metrics = evaluate_scores(test_labels, test_scores, test_domains, threshold=best_threshold)
    test_metrics.update({"threshold_source": "source-validation minimum macro-HTER", "best_epoch": best_epoch, "loss": test_loss, "phase": args.phase})
    (args.output_dir / "metrics_test.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    macro = test_metrics["macro"]
    print(f"[test] phase={args.phase} AUC={macro['auc']:.4f} HTER={macro['hter']:.4f} APCER={macro['apcer']:.4f} BPCER={macro['bpcer']:.4f}", flush=True)
    config = vars(args)
    config.update(
        {
            "protocol": manifest["protocol"],
            "manifest_sha256": manifest_sha256(args.manifest),
            "model_config": asdict(cfg),
            "source_domains": source_domains,
            "mixstyle_prob": mixstyle_prob,
            "domain_loss_weight": domain_loss_weight,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "total_seconds": round(time.time() - start, 1),
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
