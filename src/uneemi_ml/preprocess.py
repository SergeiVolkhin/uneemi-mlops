"""Препроцессинг изображений под SigLIP 2 vision-энкодер.

Схема: RGB → resize bilinear до IMAGE_SIZE×IMAGE_SIZE → /255 → (x - 0.5) / 0.5 → CHW.
Параметры (mean=0.5, std=0.5, resample=BILINEAR) совпадают с официальным
`SiglipImageProcessor` (конфиг google/siglip2-base-patch16-224: resample=2 → BILINEAR);
parity проверяется в `tests/test_smoke.py::test_preprocess_matches_hf_processor`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image

from uneemi_ml.config import IMAGE_SIZE

_MEAN: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
_STD: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Преобразовать одно PIL-изображение в тензор `(1, 3, IMAGE_SIZE, IMAGE_SIZE)` float32."""
    rgb = image.convert("RGB")
    resized = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)
    normalized = (chw - _MEAN) / _STD
    return normalized[np.newaxis, ...].astype(np.float32, copy=False)


def preprocess_batch(images: Iterable[Image.Image]) -> np.ndarray:
    """Преобразовать набор PIL-изображений в тензор `(B, 3, IMAGE_SIZE, IMAGE_SIZE)` float32."""
    tensors = [preprocess_image(img) for img in images]
    if not tensors:
        raise ValueError("preprocess_batch: список изображений пуст.")
    return np.concatenate(tensors, axis=0)
