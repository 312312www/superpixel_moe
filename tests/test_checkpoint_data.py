"""Focused tests for checkpoint validation and dataset range handling."""

from __future__ import annotations

from pathlib import Path
import os
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from fas_moe import infer_image_range, prepare_image, restore_image_range, validate_checkpoint
from fas_moe.segmentation import (
    NUM_FACE_PARTS,
    SuperpixelConfig,
    _landmark_cache_path,
    _load_part_cache,
    _save_part_cache,
)
from train_moe import NpyBinaryFASDataset


class CheckpointAndDataTests(unittest.TestCase):
    def test_restore_image_range_supports_all_documented_float_ranges(self) -> None:
        source = np.asarray([[[0, 64, 255]]], dtype=np.uint8)
        for source_range, encoded in (
            ("0-1/255", source.astype(np.float32) / (255.0 * 255.0)),
            ("0-1", source.astype(np.float32) / 255.0),
            ("0-255", source.astype(np.float32)),
        ):
            restored, metadata = restore_image_range(encoded, source_range)
            np.testing.assert_allclose(restored, source, atol=1e-4)
            self.assertEqual(metadata["source_range"], source_range)
            self.assertEqual(metadata["range_detection"], "explicit")
        prepared, metadata = prepare_image(source.astype(np.float32) / 255.0, (1, 1), "0-1")
        self.assertEqual(metadata["normalization"], "float_[0,1]_scaled_by_255")
        self.assertEqual(metadata["source_range"], "0-1")
        self.assertEqual(prepared.dtype, np.uint8)

    def test_auto_range_scales_wide_integer_arrays(self) -> None:
        source = np.asarray([[[0, 32768, 65535]]], dtype=np.uint16)
        inferred, inference_metadata = infer_image_range(source)
        self.assertEqual(inferred, "0-255")
        self.assertTrue(inference_metadata["integer_scaled_to_255"])
        restored, metadata = restore_image_range(source)
        np.testing.assert_allclose(restored, source.astype(np.float32) * (255.0 / 65535.0), atol=1e-4)
        self.assertEqual(metadata["normalization"], "integer_[0,65535]_scaled_to_255")

    def test_landmark_cache_path_normalizes_windows_separators(self) -> None:
        config = SuperpixelConfig(
            use_landmarks=True,
            landmark_cache_dir=Path("outputs") / "windows\\landmark_cache",
        )
        path = _landmark_cache_path(np.zeros((4, 4, 3), dtype=np.uint8), config)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.parent.name, "landmark_cache")
        self.assertEqual(path.parent.parent.name, "windows")

    def test_landmark_cache_atomic_round_trip_and_validation(self) -> None:
        # Keep the fixture under the workspace.  Some managed Windows runners
        # deny or stall recursive operations in the process-wide TEMP folder.
        directory = Path("outputs") / f"_unit_landmark_cache_{os.getpid()}"
        import shutil

        shutil.rmtree(directory, ignore_errors=True)
        try:
            directory.mkdir(parents=True)
            path = directory / "landmarks.npz"
            levels = (4, 2)
            distributions = {
                level: np.zeros((level, NUM_FACE_PARTS), dtype=np.float32) for level in levels
            }
            for distribution in distributions.values():
                distribution[:, 0] = 1.0
            self.assertTrue(
                _save_part_cache(path, distributions, np.zeros((478, 2), dtype=np.float32), False, "no face detected")
            )
            self.assertTrue(path.is_file())
            self.assertEqual(list(path.parent.glob("*.tmp.npz")), [])
            loaded = _load_part_cache(path, levels)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            loaded_distributions, detected, reason = loaded
            self.assertFalse(detected)
            self.assertEqual(reason, "no face detected")
            np.testing.assert_array_equal(loaded_distributions[4], distributions[4])

            # A malformed scalar and a non-normalized distribution are both
            # treated as disposable cache misses.
            np.savez_compressed(
                path,
                landmarks=np.zeros((478, 2), dtype=np.float32),
                detected=np.asarray(7, dtype=np.uint8),
                reason=np.asarray("bad"),
                **{f"part_{level}": value for level, value in distributions.items()},
            )
            self.assertIsNone(_load_part_cache(path, levels))

            invalid_distributions = dict(distributions)
            invalid_distributions[4] = invalid_distributions[4].copy()
            invalid_distributions[4][0, 0] = 0.25
            np.savez_compressed(
                path,
                landmarks=np.zeros((478, 2), dtype=np.float32),
                detected=np.asarray(0, dtype=np.uint8),
                reason=np.asarray("bad"),
                **{f"part_{level}": value for level, value in invalid_distributions.items()},
            )
            self.assertIsNone(_load_part_cache(path, levels))
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_invalid_levels_and_negative_dataset_limit_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            SuperpixelConfig(levels=(128, 64, 64))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            NpyBinaryFASDataset("missing", "CASIA-FASD", limit_samples=-1)

    def test_npy_binary_dataset_honors_explicit_ranges(self) -> None:
        source = np.asarray(
            [[[0, 64, 255], [255, 32, 1]], [[10, 20, 30], [40, 50, 60]]], dtype=np.uint8
        )
        # Use a stable, workspace-local fixture root; some managed Windows
        # runners deny recursive creation/removal under their global TEMP.
        temp = Path("outputs") / "checkpoint_data_fixture"
        if temp.exists():
            import shutil

            shutil.rmtree(temp, ignore_errors=True)
        try:
            temp.mkdir(parents=True)
            folder = Path(temp) / "domain-generalization" / "CASIA-FASD"
            folder.mkdir(parents=True)
            for source_range, divisor in (("0-1/255", 255.0 * 255.0), ("0-1", 255.0), ("0-255", 1.0)):
                encoded = source.astype(np.float32) / divisor
                np.save(folder / "casia_images_live.npy", encoded[None])
                np.save(folder / "casia_images_spoof.npy", encoded[None])
                dataset = NpyBinaryFASDataset(temp, "CASIA-FASD", image_range=source_range)
                item = dataset[0]
                np.testing.assert_allclose(item["image"].permute(1, 2, 0).numpy(), source, atol=1e-4)
        finally:
            import shutil

            shutil.rmtree(temp, ignore_errors=True)

    def test_checkpoint_validation_is_strict_and_checks_config(self) -> None:
        model = nn.Linear(3, 2)
        model.config = SimpleNamespace(
            levels=(128, 64, 16),
            feature_channels=512,
            position_dim=5,
            expert_hidden_dim=256,
            num_experts=4,
            num_classes=2,
            pretrained_backbone=False,
            freeze_backbone=True,
            use_landmarks=True,
            image_range="auto",
        )
        payload = {
            "model_state": model.state_dict(),
            "model_config": {
                "levels": [128, 64, 16],
                "feature_channels": 512,
                "position_dim": 5,
                "expert_hidden_dim": 256,
                "num_experts": 4,
                "num_classes": 2,
                "pretrained_backbone": False,
                "freeze_backbone": True,
                "use_landmarks": True,
                "image_range": "auto",
            },
        }
        report = validate_checkpoint(model, payload)
        self.assertTrue(report["validated_strict"])
        broken = dict(payload)
        broken["model_config"] = dict(payload["model_config"], use_landmarks=False)
        with self.assertRaisesRegex(ValueError, "model_config mismatches"):
            validate_checkpoint(model, broken)
        broken_range = dict(payload)
        broken_range["model_config"] = dict(payload["model_config"], image_range="0-255")
        with self.assertRaisesRegex(ValueError, "image_range"):
            validate_checkpoint(model, broken_range)
        broken_state = dict(payload, model_state={"weight": torch.zeros(1)})
        with self.assertRaisesRegex(ValueError, "missing keys"):
            validate_checkpoint(model, broken_state)


if __name__ == "__main__":
    unittest.main()
