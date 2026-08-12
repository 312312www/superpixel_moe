"""Convert MediaPipe face landmarks into 11 exclusive semantic part masks."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from skimage.draw import polygon


PART_NAMES: tuple[str, ...] = (
    "unknown",
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
    "left_cheek",
    "right_cheek",
    "forehead",
    "chin",
)
NUM_FACE_PARTS = len(PART_NAMES)

# MediaPipe Face Mesh landmark groups. Left/right use the subject's anatomy.
PART_LANDMARK_INDICES: dict[int, tuple[int, ...]] = {
    1: (336, 296, 334, 293, 300, 285, 295, 282, 283, 276),
    2: (70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
    3: (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
    4: (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    5: (168, 193, 122, 196, 3, 51, 45, 4, 275, 281, 248, 419, 197, 195, 5, 1, 19, 94, 2),
    6: (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95),
    7: (280, 330, 347, 346, 352, 376, 433, 416, 434, 432, 422, 424),
    8: (50, 101, 118, 117, 123, 147, 213, 192, 214, 212, 202, 204),
    9: (54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 301, 300, 293, 334, 296, 336, 285, 8, 55, 107, 66, 105, 63, 70),
    10: (172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 435, 367, 364, 394, 395, 369, 396, 175),
}

FACE_OVAL_INDICES = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
)

# Local parts overwrite the complete forehead/cheek/chin skin partition.
_LOCAL_PAINT_ORDER = (1, 2, 3, 4, 5, 6)


def unknown_part_masks(image_shape: tuple[int, int]) -> np.ndarray:
    """Return masks in which every pixel belongs to ``unknown``."""

    height, width = image_shape
    masks = np.zeros((NUM_FACE_PARTS, height, width), dtype=bool)
    masks[0] = True
    return masks


def landmarks_to_part_masks(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Rasterize landmarks into mutually exclusive ``[11,H,W]`` masks."""

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 468:
        return unknown_part_masks(image_shape)
    if not np.isfinite(points).all():
        return unknown_part_masks(image_shape)
    height, width = image_shape
    label_map = np.zeros((height, width), dtype=np.uint8)
    oval = np.clip(points[np.asarray(FACE_OVAL_INDICES)], 0.0, 1.0)
    face_rr, face_cc = polygon(
        oval[:, 1] * (height - 1), oval[:, 0] * (width - 1), shape=(height, width)
    )
    face_mask = np.zeros((height, width), dtype=bool)
    face_mask[face_rr, face_cc] = True
    yy, xx = np.indices((height, width))
    center_x = float(points[1, 0]) * (width - 1)
    brow_y = float(points[np.asarray(PART_LANDMARK_INDICES[1] + PART_LANDMARK_INDICES[2]), 1].mean()) * (height - 1)
    mouth_y = float(points[np.asarray(PART_LANDMARK_INDICES[6]), 1].mean()) * (height - 1)
    chin_y = float(points[152, 1]) * (height - 1)
    forehead_limit = brow_y + 0.04 * height
    chin_limit = mouth_y + 0.28 * max(chin_y - mouth_y, 1.0)
    label_map[face_mask & (yy <= forehead_limit)] = 9
    label_map[face_mask & (yy >= chin_limit)] = 10
    middle_face = face_mask & (yy > forehead_limit) & (yy < chin_limit)
    # Subject-left appears on the right side of a frontal image.
    label_map[middle_face & (xx >= center_x)] = 7
    label_map[middle_face & (xx < center_x)] = 8
    for part_id in _LOCAL_PAINT_ORDER:
        indices = np.asarray(PART_LANDMARK_INDICES[part_id], dtype=np.int64)
        vertices = np.clip(points[indices], 0.0, 1.0)
        center = vertices.mean(axis=0)
        angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
        vertices = vertices[np.argsort(angles)]
        rr, cc = polygon(vertices[:, 1] * (height - 1), vertices[:, 0] * (width - 1), shape=(height, width))
        label_map[rr, cc] = part_id
    return np.stack([label_map == part_id for part_id in range(NUM_FACE_PARTS)], axis=0)


def part_distribution_for_labels(labels: np.ndarray, part_masks: np.ndarray) -> np.ndarray:
    """Compute normalized soft part overlap for every contiguous superpixel."""

    labels = np.asarray(labels)
    masks = np.asarray(part_masks, dtype=bool)
    if labels.ndim != 2 or masks.shape != (NUM_FACE_PARTS, *labels.shape):
        raise ValueError("part masks must have shape [11,H,W] matching labels")
    region_count = int(labels.max()) + 1
    flat_labels = labels.reshape(-1)
    areas = np.bincount(flat_labels, minlength=region_count).astype(np.float64)
    if np.any(areas == 0):
        raise ValueError("labels must be contiguous and every region must contain pixels")
    distribution = np.empty((region_count, NUM_FACE_PARTS), dtype=np.float32)
    for part_id in range(NUM_FACE_PARTS):
        overlap = np.bincount(
            flat_labels,
            weights=masks[part_id].reshape(-1).astype(np.float32),
            minlength=region_count,
        )
        distribution[:, part_id] = overlap / areas
    row_sums = distribution.sum(axis=1, keepdims=True)
    invalid = row_sums[:, 0] <= 0
    distribution[invalid] = 0.0
    distribution[invalid, 0] = 1.0
    distribution[~invalid] /= row_sums[~invalid]
    return distribution


def part_distributions_for_levels(
    labels: Mapping[int, np.ndarray], part_masks: np.ndarray
) -> dict[int, np.ndarray]:
    return {level: part_distribution_for_labels(level_labels, part_masks) for level, level_labels in labels.items()}


def unknown_part_distributions(labels: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Return one-hot unknown distributions for every scale."""

    result: dict[int, np.ndarray] = {}
    for level, level_labels in labels.items():
        count = int(np.asarray(level_labels).max()) + 1
        distribution = np.zeros((count, NUM_FACE_PARTS), dtype=np.float32)
        distribution[:, 0] = 1.0
        result[level] = distribution
    return result


__all__ = [
    "FACE_OVAL_INDICES",
    "NUM_FACE_PARTS",
    "PART_LANDMARK_INDICES",
    "PART_NAMES",
    "landmarks_to_part_masks",
    "part_distribution_for_labels",
    "part_distributions_for_levels",
    "unknown_part_distributions",
    "unknown_part_masks",
]
