"""Google MediaPipe Face Landmarker adapter with a safe failure fallback."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np


LANDMARK_MODEL_ENV = "FAS_FACE_LANDMARKER_MODEL"


@dataclass(frozen=True)
class FaceLandmarkResult:
    """Normalized face landmarks and detection status for one image."""

    points: np.ndarray
    detected: bool
    reason: str
    model_id: str


_LANDMARKERS: dict[tuple[str, float, float], Any] = {}


def resolve_model_path(model_path: str | Path | None) -> Path | None:
    """Resolve an explicit model path or ``FAS_FACE_LANDMARKER_MODEL``."""

    value = model_path or os.environ.get(LANDMARK_MODEL_ENV)
    return Path(value).expanduser().resolve() if value else None


def model_identity(model_path: str | Path | None) -> str:
    """Return a cache-safe identifier that changes when the model changes."""

    resolved = resolve_model_path(model_path)
    if resolved is None:
        return "model_not_configured"
    try:
        stat = resolved.stat()
    except OSError:
        return f"missing:{resolved}"
    return f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}"


def _failure(reason: str, model_path: str | Path | None) -> FaceLandmarkResult:
    return FaceLandmarkResult(
        points=np.empty((0, 2), dtype=np.float32),
        detected=False,
        reason=reason,
        model_id=model_identity(model_path),
    )


def _get_landmarker(model_path: Path, detection_confidence: float, presence_confidence: float) -> Any:
    key = (str(model_path), float(detection_confidence), float(presence_confidence))
    if key in _LANDMARKERS:
        return _LANDMARKERS[key]
    import mediapipe as mp

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=float(detection_confidence),
        min_face_presence_confidence=float(presence_confidence),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    _LANDMARKERS[key] = landmarker
    return landmarker


def detect_face_landmarks(
    image: np.ndarray,
    model_path: str | Path | None = None,
    *,
    detection_confidence: float = 0.5,
    presence_confidence: float = 0.5,
) -> FaceLandmarkResult:
    """Detect one face and return normalized ``[N,2]`` landmarks.

    MediaPipe, model, initialization, and detection failures are deliberately
    converted to a result object so the caller can continue with ``unknown``.
    """

    resolved = resolve_model_path(model_path)
    if resolved is None:
        return _failure("face landmarker model is not configured", model_path)
    if not resolved.is_file():
        return _failure(f"face landmarker model does not exist: {resolved}", resolved)
    try:
        import mediapipe as mp
    except (ImportError, ModuleNotFoundError) as error:
        return _failure(f"mediapipe is unavailable: {error}", resolved)
    try:
        rgb = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return _failure(f"expected HWC RGB image, got {rgb.shape}", resolved)
        landmarker = _get_landmarker(resolved, detection_confidence, presence_confidence)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return _failure("no face detected", resolved)
        points = np.asarray(
            [(landmark.x, landmark.y) for landmark in result.face_landmarks[0]],
            dtype=np.float32,
        )
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            return _failure("invalid landmarks returned by MediaPipe", resolved)
        points = np.clip(points, 0.0, 1.0)
        return FaceLandmarkResult(points, True, "ok", model_identity(resolved))
    except Exception as error:  # MediaPipe raises several backend-specific exception types.
        return _failure(f"face landmark detection failed: {type(error).__name__}: {error}", resolved)


__all__ = [
    "FaceLandmarkResult",
    "LANDMARK_MODEL_ENV",
    "detect_face_landmarks",
    "model_identity",
    "resolve_model_path",
]
