"""Unit tests for A--E models, data manifests, metrics and checkpoints."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fas_moe.checkpoint import load_checkpoint, validate_checkpoint
from fas_moe.data import (
    FASDataset,
    available_datasets,
    dataset_records,
    generate_manifests,
    load_manifest,
    subject_split,
)
from fas_moe.metrics import equal_error_rate, error_rates, evaluate_scores, roc_auc, select_macro_hter_threshold
from fas_moe.model import FASModel, FASModelConfig, NativeMoE, pool_superpixel_regions
from fas_moe.superpixels import SuperpixelConfig, cached_superpixels, segment_image


def make_synthetic_dataset(root: Path, dataset: str, live: int = 12, spoof: int = 24) -> None:
    folder_map = {"CASIA-FASD": "CASIA-FASD", "Idiap Replay-Attack": "Idiap Replay-Attack", "MSU-MFSD": "MSU-MFSD"}
    prefix_map = {"CASIA-FASD": "casia", "Idiap Replay-Attack": "replay", "MSU-MFSD": "MSU"}
    folder = root / "domain-generalization" / folder_map[dataset]
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    prefix = prefix_map[dataset]
    for class_name, count in (("live", live), ("spoof", spoof)):
        images = rng.integers(0, 256, size=(count, 32, 32, 3), dtype=np.uint8).astype(np.float32) / (255.0 * 255.0)
        subjects = np.arange(count, dtype=np.int64) % 6
        np.save(folder / f"{prefix}_images_{class_name}.npy", images)
        np.save(folder / f"{prefix}_subject_{class_name}.npy", subjects)


class ModelTests(unittest.TestCase):
    def _config(self, phase: str) -> FASModelConfig:
        return FASModelConfig(
            phase=phase,
            backbone="resnet34",
            pretrained=False,
            num_experts=2,
            top_k=1,
            expert_hidden_dim=32,
            superpixel_levels=(16, 9, 4),
            superpixel_token_dim=32,
            superpixel_hidden_dim=32,
            head_hidden_dim=64,
            head_dropout=0.0,
            expert_dropout=0.0,
            moe_dropout=0.0,
        )

    def test_native_moe_topk_and_balance(self) -> None:
        torch.manual_seed(0)
        moe = NativeMoE(8, num_experts=4, top_k=2, hidden_dim=16, dropout=0.0)
        output, balance = moe(torch.randn(6, 8))
        self.assertEqual(tuple(output.shape), (6, 8))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(float(balance.detach()), 0.0)

    def test_superpixel_segment_cache(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = (180, 80, 40)
        with tempfile.TemporaryDirectory() as tmp:
            config = SuperpixelConfig(levels=(16, 9, 4), cache_dir=tmp)
            first = cached_superpixels(image, source="synthetic.npy", index=0, config=config)
            second = cached_superpixels(image, source="synthetic.npy", index=0, config=config)
            for level in config.levels:
                self.assertEqual(first[f"labels_{level}"].shape, (64, 64))
                self.assertEqual(first[f"positions_{level}"].shape, (level, 5))
                np.testing.assert_array_equal(first[f"labels_{level}"], second[f"labels_{level}"])

    def test_all_model_phases_forward_backward(self) -> None:
        image = torch.rand(2, 3, 64, 64) * 255.0
        views = segment_image(image[0].permute(1, 2, 0).byte().numpy(), SuperpixelConfig())
        labels = {
            str(level): torch.from_numpy(
                np.stack([views[f"labels_{level}"] for _ in range(2)]).astype(np.int64)
            )
            for level in (16, 9, 4)
        }
        positions = {
            str(level): torch.from_numpy(
                np.stack([views[f"positions_{level}"] for _ in range(2)]).astype(np.float32)
            )
            for level in (16, 9, 4)
        }
        valid = {
            str(level): torch.from_numpy(
                np.stack([views[f"valid_{level}"] for _ in range(2)]).astype(np.bool_)
            )
            for level in (16, 9, 4)
        }
        for phase in ("A", "B", "C", "D", "E"):
            model = FASModel(self._config(phase))
            kwargs = (
                {
                    "superpixel_labels": labels,
                    "superpixel_positions": positions,
                    "superpixel_valid": valid,
                }
                if phase in ("C", "D", "E")
                else {}
            )
            logits, details = model(image, **kwargs)
            self.assertEqual(tuple(logits.shape), (2, 2), phase)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertIn("balance_loss", details)
            logits.sum().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()), phase)

    def test_region_pooling_shape(self) -> None:
        feature_map = torch.randn(2, 8, 7, 7)
        labels = torch.arange(16).reshape(1, 4, 4).repeat(2, 1, 1)
        pooled = pool_superpixel_regions(feature_map, labels, 16)
        self.assertEqual(tuple(pooled.shape), (2, 16, 8))

    def test_phase_e_checkpoint_round_trip(self) -> None:
        config = self._config("E")
        model = FASModel(config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save({"model_state": model.state_dict(), "model_config": config.__dict__}, path)
            fresh = FASModel(config)
            self.assertTrue(load_checkpoint(fresh, path)["validated_strict"])
            with self.assertRaises(ValueError):
                validate_checkpoint(FASModel(self._config("C")), torch.load(path, weights_only=False))


class MetricTests(unittest.TestCase):
    def test_auc_rates_and_threshold(self) -> None:
        labels = [1, 1, 1, 0, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]
        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)
        self.assertEqual(error_rates(labels, scores, 0.5)["hter"], 0.0)
        report = select_macro_hter_threshold(labels, scores, ["C"] * 6)
        metrics = evaluate_scores(labels, scores, ["C"] * 6, threshold=report["threshold"])
        self.assertEqual(metrics["macro"]["hter"], 0.0)
        self.assertGreaterEqual(equal_error_rate(labels, scores), 0.0)


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fas_test_"))
        make_synthetic_dataset(self.tmp, "CASIA-FASD")
        make_synthetic_dataset(self.tmp, "Idiap Replay-Attack", live=6, spoof=9)
        make_synthetic_dataset(self.tmp, "MSU-MFSD", live=6, spoof=9)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_available_domains_and_placeholders(self) -> None:
        self.assertEqual(available_datasets(self.tmp), ["CASIA-FASD", "Idiap Replay-Attack", "MSU-MFSD"])
        paths = generate_manifests(self.tmp, self.tmp / "splits", seed=1)
        self.assertIn("intra_O", paths)
        self.assertEqual(load_manifest(paths["intra_O"])["status"], "pending_dataset")
        self.assertEqual(load_manifest(paths["lodo_O"])["status"], "pending_dataset")
        self.assertEqual(load_manifest(paths["four_domain_protocol"])["status"], "pending_dataset")

    def test_manifest_splits_and_dataset_item(self) -> None:
        paths = generate_manifests(self.tmp, self.tmp / "splits", seed=1)
        payload = load_manifest(paths["intra_C"])
        self.assertTrue(all(payload["counts"][split] > 0 for split in ("train", "val", "test")))
        records = dataset_records(self.tmp, "CASIA-FASD")
        train, val, test = subject_split(records, seed=2)
        self.assertFalse(train & val or train & test or val & test)
        dataset = FASDataset(payload["records"], split="train", transform=lambda image: torch.from_numpy(image).permute(2, 0, 1).float())
        self.assertEqual(tuple(dataset[0]["image"].shape), (3, 32, 32))


if __name__ == "__main__":
    unittest.main()
