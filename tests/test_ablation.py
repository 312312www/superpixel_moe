"""Ablation-mode, manifest, metric, BN, and strict-cache acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest

import numpy as np
import torch
from torch import nn

from fas_moe import SuperpixelMoE, SuperpixelMoEConfig, segment_views
from fas_moe.datasets import (
    FixedDomainClassBatchSampler,
    generate_lodo_manifests,
    load_manifest,
)
from fas_moe.metrics import error_rates, evaluate_scores, roc_auc, select_macro_hter_threshold
from fas_moe.segmentation import SuperpixelConfig


def _image(size: int = 64) -> np.ndarray:
    yy, xx = np.indices((size, size))
    return np.stack(
        ((xx * 3 + 20) % 255, (yy * 4 + 30) % 255, (xx + yy + 50) % 255), axis=2
    ).astype(np.uint8)


class AblationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = _image()
        cls.views = segment_views(
            cls.image,
            SuperpixelConfig(use_landmarks=False, image_size=(64, 64), slic_cache_dir=None),
        )
        cls.tensor = torch.from_numpy(cls.image).permute(2, 0, 1).unsqueeze(0).float()

    def test_experiment_modules_are_conditional(self) -> None:
        expected = {
            "A": (False, False, "none"),
            "B": (False, False, "single"),
            "C": (True, False, "single"),
            "D": (True, True, "single"),
            "E": (True, True, "multiple"),
        }
        for experiment, values in expected.items():
            model = SuperpixelMoE(
                SuperpixelMoEConfig(experiment=experiment, pretrained_backbone=False)
            )
            self.assertEqual(
                (model.config.use_superpixel, model.config.use_landmarks, model.config.moe_mode),
                values,
            )
            self.assertEqual(model.position is not None, experiment in "CDE")
            self.assertEqual(model.part_embedding is not None, experiment in "DE")
            self.assertEqual(model.global_moe is not None, experiment == "B")
            self.assertEqual(model.shared_moe is not None, experiment in "CD")
            self.assertEqual(model.moes is not None, experiment == "E")
            if experiment == "E":
                assert model.moes is not None
                first_parameters = [next(model.moes[level].parameters()) for level in ("128", "64", "16")]
                self.assertEqual(len({parameter.data_ptr() for parameter in first_parameters}), 3)

    def test_all_modes_forward_backward_and_backbone_gradients(self) -> None:
        for experiment in "ABCDE":
            model = SuperpixelMoE(
                SuperpixelMoEConfig(experiment=experiment, pretrained_backbone=False)
            )
            logits, details = model(
                self.tensor, views=self.views if experiment in "CDE" else None
            )
            self.assertEqual(tuple(logits.shape), (1, 2))
            if experiment in "CDE":
                self.assertEqual(tuple(details["tokens_128"].shape), (1, 128, 512))
                self.assertEqual(tuple(details["tokens_64"].shape), (1, 64, 512))
                self.assertEqual(tuple(details["tokens_16"].shape), (1, 16, 512))
            nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
            convolution_grads = [
                parameter.grad
                for name, parameter in model.backbone.named_parameters()
                if "conv" in name or parameter.ndim == 4
            ]
            self.assertTrue(any(gradient is not None for gradient in convolution_grads))

    def test_shared_moe_is_called_three_times(self) -> None:
        model = SuperpixelMoE(
            SuperpixelMoEConfig(experiment="D", pretrained_backbone=False)
        )
        assert model.shared_moe is not None
        calls: list[int] = []
        handle = model.shared_moe.register_forward_hook(lambda *_: calls.append(1))
        try:
            model(self.tensor, views=self.views)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 3)

    def test_batch_norm_is_eval_and_frozen_while_convolutions_train(self) -> None:
        model = SuperpixelMoE(
            SuperpixelMoEConfig(
                experiment="A",
                pretrained_backbone=False,
                freeze_backbone=False,
                freeze_batch_norm=True,
            )
        )
        model.train()
        batch_norms = [
            module for module in model.backbone.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        self.assertTrue(batch_norms)
        self.assertTrue(all(not module.training for module in batch_norms))
        self.assertTrue(
            all(not parameter.requires_grad for module in batch_norms for parameter in module.parameters())
        )
        self.assertTrue(
            any(parameter.requires_grad for parameter in model.backbone.parameters() if parameter.ndim == 4)
        )

    def test_required_cache_misses_fail_and_hits_pass(self) -> None:
        root = Path("outputs") / "strict_cache_test"
        shutil.rmtree(root, ignore_errors=True)
        try:
            strict = SuperpixelConfig(
                use_landmarks=False,
                image_size=(64, 64),
                slic_cache_dir=root,
                require_slic_cache=True,
            )
            with self.assertRaisesRegex(RuntimeError, "required SLIC cache miss"):
                segment_views(self.image, strict)
            segment_views(
                self.image,
                SuperpixelConfig(
                    use_landmarks=False, image_size=(64, 64), slic_cache_dir=root
                ),
            )
            hit = segment_views(self.image, strict)
            self.assertTrue(hit.metadata["slic_cache_hit"])
            with self.assertRaisesRegex(RuntimeError, "required Landmark cache miss"):
                segment_views(
                    self.image,
                    SuperpixelConfig(
                        use_landmarks=True,
                        image_size=(64, 64),
                        slic_cache_dir=root,
                        landmark_cache_dir=root / "landmarks",
                        require_slic_cache=True,
                        require_landmark_cache=True,
                    ),
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)


class ManifestAndMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("outputs") / "ablation_manifest_fixture"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        specs = {
            "CASIA-FASD": "casia",
            "Idiap Replay-Attack": "replay",
            "MSU-MFSD": "MSU",
            "OULU-NPU": "Oulu",
        }
        image = np.zeros((4, 4, 3), dtype=np.float32)
        for folder_name, prefix in specs.items():
            folder = self.root / "domain-generalization" / folder_name
            folder.mkdir(parents=True)
            live = np.stack([image + index / 1000 for index in range(5)])
            spoof = np.stack([image + (index + 10) / 1000 for index in range(5)])
            np.save(folder / f"{prefix}_images_live.npy", live)
            np.save(folder / f"{prefix}_images_spoof.npy", spoof)
            np.save(folder / f"{prefix}_subject_live.npy", np.arange(5, dtype=np.int64))
            np.save(folder / f"{prefix}_subject_spoof.npy", np.arange(5, dtype=np.int64))

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_lodo_manifest_is_subject_disjoint_and_has_expected_counts(self) -> None:
        paths = generate_lodo_manifests(
            self.root, self.root / "splits", split_seed=19, val_fraction=0.2
        )
        payload = load_manifest(paths["OCI_M"])
        self.assertEqual(payload["counts"], {"train": 24, "val": 6, "test": 10})
        train = {
            (item["domain"], item["subject_id"])
            for item in payload["records"] if item["split"] == "train"
        }
        val = {
            (item["domain"], item["subject_id"])
            for item in payload["records"] if item["split"] == "val"
        }
        self.assertFalse(train & val)
        self.assertEqual(
            {item["domain"] for item in payload["records"] if item["split"] == "test"},
            {"M"},
        )

    def test_balanced_batches_are_model_rng_independent(self) -> None:
        records = [
            {"domain": domain, "label": label, "sample_id": f"{domain}:{label}:{index}"}
            for domain in "OCI" for label in (0, 1) for index in range(4)
        ]
        first = FixedDomainClassBatchSampler(records, batch_size=6, seed=7, num_batches=3)
        torch.manual_seed(1)
        _ = torch.randn(100)
        a = first.batches()
        second = FixedDomainClassBatchSampler(records, batch_size=6, seed=7, num_batches=3)
        torch.manual_seed(999)
        _ = torch.randn(1000)
        b = second.batches()
        self.assertEqual(a, b)
        for batch in a:
            groups = {(records[index]["domain"], records[index]["label"]) for index in batch}
            self.assertEqual(len(groups), 6)

    def test_metric_directions_macro_threshold_and_auc(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.8, 0.4, 0.9]
        rates = error_rates(labels, scores, 0.5)
        self.assertEqual(rates["apcer"], 0.5)
        self.assertEqual(rates["bpcer"], 0.5)
        self.assertAlmostEqual(roc_auc(labels, scores), 0.75)
        domains = ["C"] * 4 + ["I"] * 4
        doubled_labels = labels + labels
        doubled_scores = scores + scores
        selected = select_macro_hter_threshold(doubled_labels, doubled_scores, domains)
        report = evaluate_scores(
            doubled_labels, doubled_scores, domains, threshold=selected["threshold"]
        )
        self.assertEqual(set(report["per_domain"]), {"C", "I"})
        self.assertTrue(0.0 <= report["macro"]["auc"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
