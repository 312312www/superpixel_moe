"""Image and NumPy input adapters for the Superpixel-MoE baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_RANGE_CHOICES = ("auto", "0-1/255", "0-1", "0-255")


def infer_image_range(image: np.ndarray) -> tuple[str, dict[str, object]]:
    """Infer a canonical float/integer image range from an array's extrema.

    Call this once on a complete NPY source (or on representative data), then
    pass the returned range explicitly to :func:`restore_image_range` for each
    sample.  Resolving the range at dataset scope avoids a dark ``[0,1]`` frame
    being mistaken for the legacy ``[0,1/255]`` encoding.
    """

    array = np.asarray(image)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"image dtype must be numeric, got {array.dtype}")
    if array.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0:
        raise ValueError(f"image contains negative values: min={minimum}")
    integer_metadata: dict[str, object] = {}
    if np.issubdtype(array.dtype, np.integer):
        # The public range vocabulary describes the representation expected by
        # the rest of the pipeline, which is always canonical ``[0,255]``.
        # Keep wide-integer provenance in metadata instead of returning a
        # dynamically named value (for example ``0-65535``) that callers cannot
        # pass back to ``restore_image_range``.
        dtype_maximum = int(np.iinfo(array.dtype).max)
        selected = "0-255"
        integer_metadata = {
            "source_dtype": str(array.dtype),
            "source_dtype_max": dtype_maximum,
            "integer_scaled_to_255": bool(maximum > 255.0),
        }
    elif maximum <= (1.0 / 255.0) + 1e-6:
        selected = "0-1/255"
    elif maximum <= 1.0 + 1e-6:
        selected = "0-1"
    elif maximum <= 255.0 + 1e-6:
        selected = "0-255"
    else:
        raise ValueError(f"unsupported image range: min={minimum}, max={maximum}")
    return selected, {
        "source_range": selected,
        "source_min": minimum,
        "source_max": maximum,
        **integer_metadata,
    }


def restore_image_range(
    image: np.ndarray, source_range: str = "auto"
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a numeric image as float32 in ``[0, 255]``.

    ``auto`` distinguishes the three float encodings used by the project from
    their observed maximum.  An all-black image has the same result under every
    encoding; for very dark non-black ``[0, 1]`` data, callers can pass an
    explicit range to remove the inherent ambiguity.
    """

    if source_range not in IMAGE_RANGE_CHOICES:
        raise ValueError(
            f"unsupported source_range {source_range!r}; choose from {IMAGE_RANGE_CHOICES}"
        )
    array = np.asarray(image)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"image dtype must be numeric, got {array.dtype}")
    original_dtype = array.dtype
    array = array.astype(np.float32, copy=False)
    if array.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0:
        raise ValueError(f"image contains negative values: min={minimum}")

    selected_range = source_range
    integer_scaled = False
    if np.issubdtype(original_dtype, np.integer) and maximum > 255.0:
        if source_range not in ("auto", "0-255"):
            raise ValueError(
                f"wide integer image cannot use declared {source_range!r} range; use 'auto' or '0-255'"
            )
        # Wide integer arrays are canonicalized to [0,255] for both ``auto``
        # and the canonical range returned by ``infer_image_range``.  Scaling
        # by the dtype maximum preserves the established uint16/uint32
        # semantics and makes ``restore_image_range(array, inferred_range)``
        # round-trip without requiring a non-standard range string.
        dtype_maximum = float(np.iinfo(original_dtype).max)
        array = array * (255.0 / dtype_maximum)
        selected_range = "0-255"
        normalization = f"integer_[0,{int(dtype_maximum)}]_scaled_to_255"
        integer_scaled = True
    elif source_range == "auto" and np.issubdtype(original_dtype, np.integer):
        selected_range = "0-255"
    elif source_range == "auto":
        selected_range, _ = infer_image_range(array)

    if selected_range == "0-1/255":
        if maximum > (1.0 / 255.0) + 1e-6:
            raise ValueError(
                f"image exceeds declared [0,1/255] range: min={minimum}, max={maximum}"
            )
        array = array * (255.0 * 255.0)
        normalization = "float_[0,1/255]_restored_by_255_squared"
    elif selected_range == "0-1":
        if maximum > 1.0 + 1e-6:
            raise ValueError(f"image exceeds declared [0,1] range: min={minimum}, max={maximum}")
        array = array * 255.0
        normalization = "float_[0,1]_scaled_by_255"
    elif selected_range == "0-255":
        if maximum > 255.0 + 1e-6 and not integer_scaled:
            raise ValueError(
                f"image exceeds declared [0,255] range: min={minimum}, max={maximum}"
            )
        if not integer_scaled:
            normalization = "already_[0,255]"

    return np.clip(array, 0.0, 255.0), {
        "normalization": normalization,
        "source_range": selected_range,
        "range_detection": "auto" if source_range == "auto" else "explicit",
        "source_min": minimum,
        "source_max": maximum,
    }


def load_input(path: str | Path, index: int = 0) -> tuple[np.ndarray, dict[str, object]]:
    """Load one image from an image file or an HWC/CHW/NHWC/NCHW NPY."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"input does not exist: {source}")
    metadata: dict[str, object] = {
        "input_path": str(source.resolve()),
        "input_index": int(index),
    }
    if source.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(source) as image:
            array = np.asarray(image.convert("RGB"))
        metadata["input_kind"] = "image_file"
        return array, metadata
    if source.suffix.lower() == ".npy":
        mapped = np.load(source, mmap_mode="r", allow_pickle=False)
        try:
            if mapped.ndim == 4:
                if index < 0 or index >= mapped.shape[0]:
                    raise IndexError(f"index {index} is outside the batch of {mapped.shape[0]} samples")
                selected = mapped[index]
            elif mapped.ndim == 3:
                selected = mapped
            else:
                raise ValueError(f"NPY must contain HWC/CHW or NHWC/NCHW data, got {mapped.shape}")
            array = np.array(selected, copy=True)
        finally:
            if isinstance(mapped, np.memmap):
                mapped._mmap.close()
        metadata["input_kind"] = "npy"
        metadata["npy_shape"] = [int(value) for value in array.shape]
        return array, metadata
    raise ValueError(
        f"unsupported input suffix {source.suffix!r}; use {sorted(IMAGE_SUFFIXES)} or '.npy'"
    )


def prepare_image(
    image: np.ndarray,
    image_size: tuple[int, int] = (256, 256),
    source_range: str = "auto",
) -> tuple[np.ndarray, dict[str, object]]:
    """Convert one HWC/CHW image to resized RGB uint8."""

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"expected one HWC/CHW image, got shape {array.shape}")
    original_shape = tuple(int(value) for value in array.shape)
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected RGB-like HWC/CHW input, got shape {array.shape}")
    array, range_metadata = restore_image_range(array, source_range)
    array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    target_height, target_width = image_size
    if array.shape[:2] != (target_height, target_width):
        resized = resize(
            array,
            (target_height, target_width, 3),
            order=1,
            mode="reflect",
            preserve_range=True,
            anti_aliasing=True,
        )
        array = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
    return array, {
        "original_shape": list(original_shape),
        "prepared_shape": list(array.shape),
        **range_metadata,
    }


__all__ = [
    "IMAGE_RANGE_CHOICES",
    "IMAGE_SUFFIXES",
    "load_input",
    "prepare_image",
    "infer_image_range",
    "restore_image_range",
]
