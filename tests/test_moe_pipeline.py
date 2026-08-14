"""Tests for the minimal independent-view Superpixel-MoE baseline."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import unittest

import numpy as np
import torch
from skimage.measure import label as connected_components

from fas_moe import (
    EqualWeightMoE,
    SuperpixelMoE,
    SuperpixelMoEConfig,
    load_input,
    pool_regions,
    prepare_image,
    part_distribution_for_labels,
    segment_views,
    unknown_part_distributions,
)
from fas_moe.segmentation import SuperpixelConfig


def synthetic_image(size: int = 64) -> np.ndarray:
    yy, xx = np.indices((size, size))
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = np.clip(30 + xx * 3, 0, 255)
    image[..., 1] = np.clip(20 + yy * 3, 0, 255)
    image[..., 2] = 70
    face = ((xx - size / 2) / (size * 0.32)) ** 2 + ((yy - size / 2) / (size * 0.42)) ** 2 <= 1
    image[face] = np.stack(
        [130 + xx[face], 85 + yy[face] // 2, 70 + (xx[face] + yy[face]) // 4], axis=1
    )
    image[size // 3 : size // 3 + 3, size // 4 : 3 * size // 4] = (10, 10, 10)
    return image


class SuperpixelMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = synthetic_image()
        cls.no_landmarks = SuperpixelConfig(use_landmarks=False)
        cls.views = segment_views(cls.image, cls.no_landmarks)

    def test_exact_counts_connectivity_and_features(self) -> None:
        for level in (128, 64, 16):
            labels = self.views.labels[level]
            self.assertEqual(np.unique(labels).size, level)
            self.assertEqual(self.views.features[level].shape, (level, 19))
            self.assertEqual(self.views.positions[level].shape, (level, 5))
            self.assertEqual(self.views.part_distributions[level].shape, (level, 11))
            self.assertTrue(np.isfinite(self.views.features[level]).all())
            self.assertTrue(np.isfinite(self.views.positions[level]).all())
            self.assertTrue(np.isfinite(self.views.part_distributions[level]).all())
            np.testing.assert_allclose(self.views.part_distributions[level].sum(axis=1), 1.0)
            for region in range(level):
                self.assertEqual(connected_components(labels == region, connectivity=1).max(), 1)

    def test_segmentation_is_deterministic(self) -> None:
        repeated = segment_views(self.image, self.no_landmarks)
        for level in (128, 64, 16):
            np.testing.assert_array_equal(self.views.labels[level], repeated.labels[level])
            np.testing.assert_allclose(self.views.features[level], repeated.features[level])

    def test_slic_cache_round_trip_and_corruption_recovery(self) -> None:
        cache_dir = Path("outputs") / f"_unit_slic_cache_{os.getpid()}"
        shutil.rmtree(cache_dir, ignore_errors=True)
        try:
            config = SuperpixelConfig(
                use_landmarks=False,
                image_size=(64, 64),
                slic_cache_dir=cache_dir,
            )
            first = segment_views(self.image, config)
            self.assertFalse(first.metadata["slic_cache_hit"])
            cache_path = Path(str(first.metadata["slic_cache_path"]))
            self.assertTrue(cache_path.is_file())
            self.assertEqual(cache_path.parent, cache_dir.resolve())
            second = segment_views(self.image, config)
            self.assertTrue(second.metadata["slic_cache_hit"])
            for level in config.levels:
                np.testing.assert_array_equal(first.labels[level], second.labels[level])
                np.testing.assert_allclose(first.features[level], second.features[level])
                np.testing.assert_array_equal(first.edges[level], second.edges[level])
                np.testing.assert_allclose(first.positions[level], second.positions[level])

            # A truncated/invalid entry is a miss and is atomically repaired.
            cache_path.write_bytes(b"not a valid npz")
            repaired = segment_views(self.image, config)
            self.assertFalse(repaired.metadata["slic_cache_hit"])
            self.assertTrue(cache_path.is_file())
            with np.load(cache_path, allow_pickle=False) as archive:
                self.assertTrue(archive.files)
            self.assertEqual(list(cache_path.parent.glob("*.tmp.npz")), [])
            restored = segment_views(self.image, config)
            self.assertTrue(restored.metadata["slic_cache_hit"])
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_slic_cache_can_be_disabled(self) -> None:
        config = SuperpixelConfig(use_landmarks=False, slic_cache_dir=None)
        views = segment_views(self.image, config)
        self.assertFalse(views.metadata["slic_cache_enabled"])
        self.assertFalse(views.metadata["slic_cache_hit"])
        self.assertIsNone(views.metadata["slic_cache_path"])

    def test_input_loading_and_range_restoration(self) -> None:
        damaged = self.image.astype(np.float32) / (255.0 * 255.0)
        prepared, metadata = prepare_image(damaged, (64, 64))
        self.assertEqual(metadata["normalization"], "float_[0,1/255]_restored_by_255_squared")
        np.testing.assert_allclose(prepared, self.image, atol=1)
        fixture_dir = Path("outputs") / f"_unit_input_fixture_{os.getpid()}"
        shutil.rmtree(fixture_dir, ignore_errors=True)
        fixture_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = fixture_dir / "batch.npy"
            np.save(path, np.stack([damaged, damaged]))
            loaded, input_metadata = load_input(path, index=1)
            self.assertEqual(input_metadata["input_kind"], "npy")
            np.testing.assert_array_equal(loaded, damaged)
        finally:
            shutil.rmtree(fixture_dir, ignore_errors=True)

    def test_region_pooling_shape(self) -> None:
        feature_map = torch.arange(512 * 32 * 32, dtype=torch.float32).reshape(512, 32, 32)
        pooled = pool_regions(feature_map, self.views.labels[128], 128)
        self.assertEqual(tuple(pooled.shape), (128, 512))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_part_overlap_is_a_soft_distribution(self) -> None:
        labels = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
        masks = np.zeros((11, 2, 4), dtype=bool)
        masks[0] = True
        masks[0, :, 1:3] = False
        masks[3, :, 1:3] = True
        distribution = part_distribution_for_labels(labels, masks)
        self.assertEqual(distribution.shape, (2, 11))
        np.testing.assert_allclose(distribution.sum(axis=1), 1.0)
        np.testing.assert_allclose(distribution[:, 0], 0.5)
        np.testing.assert_allclose(distribution[:, 3], 0.5)

    def test_landmark_failure_uses_unknown(self) -> None:
        distributions = unknown_part_distributions(self.views.labels)
        for level in (128, 64, 16):
            self.assertEqual(distributions[level].shape, (level, 11))
            np.testing.assert_array_equal(distributions[level][:, 0], np.ones(level))
            np.testing.assert_array_equal(distributions[level][:, 1:], np.zeros((level, 10)))

    def test_missing_landmarker_model_falls_back_without_interrupting(self) -> None:
        views = segment_views(
            self.image,
            SuperpixelConfig(
                use_landmarks=True,
                landmark_model_path=Path("definitely_missing_face_landmarker.task"),
                landmark_cache_dir=None,
            ),
        )
        self.assertFalse(views.metadata["landmarks_detected"])
        for level in (128, 64, 16):
            np.testing.assert_array_equal(views.part_distributions[level][:, 0], np.ones(level))
            np.testing.assert_array_equal(views.part_distributions[level][:, 1:], np.zeros((level, 10)))

    def test_equal_weight_moe_is_an_average(self) -> None:
        torch.manual_seed(3)
        moe = EqualWeightMoE(channels=8, hidden_dim=12, num_experts=4)
        tokens = torch.randn(2, 5, 8)
        output, expert_outputs = moe(tokens)
        torch.testing.assert_close(output, expert_outputs.mean(dim=0))

    def test_model_forward_and_backward(self) -> None:
        torch.manual_seed(7)
        model = SuperpixelMoE(SuperpixelMoEConfig(pretrained_backbone=False))
        images = torch.from_numpy(self.image).permute(2, 0, 1).unsqueeze(0).float()
        logits, details = model(images, views=self.views)
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertEqual(tuple(details["tokens_128"].shape), (1, 128, 512))
        self.assertEqual(tuple(details["tokens_64"].shape), (1, 64, 512))
        self.assertEqual(tuple(details["tokens_16"].shape), (1, 16, 512))
        self.assertEqual(tuple(details["input_tokens_128"].shape), (1, 128, 512))
        loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([1]))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.classifier.parameters()))
        self.assertIsNotNone(model.part_embedding.weight.grad)
        self.assertTrue(torch.isfinite(model.part_embedding.weight.grad).all())
        self.assertFalse(any(parameter.grad is not None for parameter in model.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
