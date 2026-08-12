"""Deterministic handcrafted descriptors for each SLIC region."""

from __future__ import annotations

import math

import numpy as np
from skimage.color import rgb2lab


FEATURE_NAMES = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "lab_mean_l",
    "lab_mean_a",
    "lab_mean_b",
    "lab_std_l",
    "lab_std_a",
    "lab_std_b",
    "gradient_mean",
    "gradient_std",
    "area_ratio",
    "centroid_x",
    "centroid_y",
    "perimeter_ratio",
    "compactness",
)


def _gradient_map(image: np.ndarray) -> np.ndarray:
    gray = np.dot(
        image.astype(np.float32) / 255.0,
        np.array([0.299, 0.587, 0.114], dtype=np.float32),
    )
    dy, dx = np.gradient(gray)
    gradient = np.hypot(dx, dy).astype(np.float32)
    maximum = float(gradient.max())
    return gradient / maximum if maximum > 0 else gradient


def _perimeter_by_region(labels: np.ndarray) -> np.ndarray:
    region_count = int(labels.max()) + 1
    perimeter = np.zeros(region_count, dtype=np.float64)

    def add(values: np.ndarray) -> None:
        nonlocal perimeter
        perimeter += np.bincount(values.ravel(), minlength=region_count)

    horizontal = labels[:, :-1] != labels[:, 1:]
    add(labels[:, :-1][horizontal])
    add(labels[:, 1:][horizontal])
    vertical = labels[:-1, :] != labels[1:, :]
    add(labels[:-1, :][vertical])
    add(labels[1:, :][vertical])
    add(labels[0, :])
    add(labels[-1, :])
    add(labels[:, 0])
    add(labels[:, -1])
    return perimeter


def extract_region_features(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute the documented 19-dimensional feature vector per region."""

    rgb = image.astype(np.float32) / 255.0
    lab = rgb2lab(rgb).astype(np.float32)
    gradient = _gradient_map(image)
    region_count = int(labels.max()) + 1
    flat_labels = labels.ravel()
    area_pixels = np.bincount(flat_labels, minlength=region_count).astype(np.float64)
    height, width = labels.shape
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    perimeter = _perimeter_by_region(labels)
    features = np.empty((region_count, len(FEATURE_NAMES)), dtype=np.float32)
    for region in range(region_count):
        mask = labels == region
        rgb_values = rgb[mask]
        lab_values = lab[mask]
        gradient_values = gradient[mask]
        area = float(area_pixels[region])
        perimeter_ratio = float(perimeter[region] / (2.0 * (height + width)))
        compactness = float(4.0 * math.pi * area / (perimeter[region] ** 2 + 1e-8))
        features[region] = np.concatenate(
            [
                rgb_values.mean(axis=0),
                rgb_values.std(axis=0),
                lab_values.mean(axis=0),
                lab_values.std(axis=0),
                [gradient_values.mean(), gradient_values.std()],
                [
                    area / float(height * width),
                    float(xx[mask].mean()) / max(width - 1, 1),
                    float(yy[mask].mean()) / max(height - 1, 1),
                    perimeter_ratio,
                    compactness,
                ],
            ]
        ).astype(np.float32)
    return features


__all__ = ["FEATURE_NAMES", "extract_region_features"]
