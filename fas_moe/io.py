"""Image and NumPy input adapters for the Superpixel-MoE baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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
    image: np.ndarray, image_size: tuple[int, int] = (256, 256)
) -> tuple[np.ndarray, dict[str, object]]:
    """Convert one HWC/CHW image to resized RGB uint8."""

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"expected one HWC/CHW image, got shape {array.shape}")
    original_shape = tuple(int(value) for value in array.shape)
    original_dtype = array.dtype
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected RGB-like HWC/CHW input, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"image dtype must be numeric, got {array.dtype}")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0:
        raise ValueError(f"image contains negative values: min={minimum}")
    if np.issubdtype(original_dtype, np.integer) and maximum <= 255.0:
        normalization = "already_[0,255]"
    elif np.issubdtype(original_dtype, np.integer):
        dtype_maximum = float(np.iinfo(original_dtype).max)
        array = array * (255.0 / dtype_maximum)
        normalization = f"integer_[0,{int(dtype_maximum)}]_scaled_to_255"
    elif maximum <= (1.0 / 255.0) + 1e-6:
        array = array * 255.0 * 255.0
        normalization = "float_[0,1/255]_restored_by_255_squared"
    elif maximum <= 1.0 + 1e-6:
        array = array * 255.0
        normalization = "float_[0,1]_scaled_by_255"
    elif maximum <= 255.0 + 1e-6:
        normalization = "already_[0,255]"
    else:
        raise ValueError(f"unsupported image range: min={minimum}, max={maximum}")
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
        "normalization": normalization,
    }


__all__ = ["IMAGE_SUFFIXES", "load_input", "prepare_image"]
