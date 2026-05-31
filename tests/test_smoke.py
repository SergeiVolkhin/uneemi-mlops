"""Smoke-тесты ONNX-пайплайна SigLIP 2.

Если ONNX-файл ещё не экспортирован - все тесты skip с подсказкой запустить
`scripts/export_onnx.py`. Это удобно для CI: тесты не падают до экспорта.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from transformers import AutoProcessor

from uneemi_ml import EMBED_DIM, MODEL_ID, ONNX_VISION_PATH, Siglip2Encoder, preprocess_image

FIXTURE_DIR: Path = Path(__file__).parent / "fixtures"
_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _find_fixture_image() -> Path | None:
    if not FIXTURE_DIR.exists():
        return None
    candidates = sorted(p for p in FIXTURE_DIR.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    return candidates[0] if candidates else None


@pytest.fixture(scope="module")
def encoder() -> Siglip2Encoder:
    if not ONNX_VISION_PATH.exists():
        pytest.skip(
            f"ONNX-модель не найдена ({ONNX_VISION_PATH}). "
            "Запустите: `uv run python scripts/export_onnx.py`"
        )
    return Siglip2Encoder()


@pytest.fixture(scope="module")
def real_image() -> Image.Image:
    path = _find_fixture_image()
    if path is None:
        pytest.skip(
            f"В {FIXTURE_DIR} не найдено ни одного файла {_IMAGE_EXTS}. "
            "Положите туда любую реальную картинку."
        )
    return Image.open(path)


def _random_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def test_encoder_loads(encoder: Siglip2Encoder) -> None:
    """Класс Siglip2Encoder создаётся и держит активную ONNX-сессию."""
    assert encoder.onnx_path.exists()


def test_encode_random_image_shape(encoder: Siglip2Encoder) -> None:
    """encode на случайной картинке 224×224 → (1, 768) float32, ненулевой и без NaN."""
    img = _random_image(seed=0)
    vec = encoder.encode(img)

    assert vec.shape == (1, EMBED_DIM), f"shape {vec.shape}, ожидалось (1, {EMBED_DIM})"
    assert vec.dtype == np.float32, f"dtype {vec.dtype}, ожидался float32"
    assert np.isfinite(vec).all(), "в векторе встречаются NaN/Inf"
    assert float(np.linalg.norm(vec)) > 0.0, "L2-норма вектора равна нулю"


def test_preprocess_matches_hf_processor(real_image: Image.Image) -> None:
    """Наш preprocess_image бит-в-бит совпадает с официальным SiglipImageProcessor."""
    custom = preprocess_image(real_image)

    # Эталонный процессор тянется с HF Hub. В офлайн-CI без кэша - skip (как и
    # остальные ресурсо-зависимые тесты здесь): сверка идёт там, где модель закэширована.
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"Процессор {MODEL_ID} недоступен офлайн ({exc.__class__.__name__}); "
            "сверка препроцессинга требует локального кэша модели"
        )
    official = processor(images=real_image, return_tensors="np")["pixel_values"]

    assert custom.shape == official.shape, (
        f"shape расходится: custom={custom.shape}, official={official.shape}"
    )
    assert np.allclose(custom, official, atol=1e-5), (
        "Препроцессинг расходится с официальным процессором. "
        "Проверь mean/std/resize-алгоритм в src/uneemi_ml/preprocess.py. "
        f"max|Δ|={float(np.max(np.abs(custom - official))):.2e}"
    )


def test_real_image_vs_noise(encoder: Siglip2Encoder, real_image: Image.Image) -> None:
    """Эмбеддинг реальной картинки заметно отличается от эмбеддинга случайного шума."""
    v_real = encoder.encode(real_image)
    v_noise = encoder.encode(_random_image(seed=0))

    cos = float((v_real @ v_noise.T).item() / (np.linalg.norm(v_real) * np.linalg.norm(v_noise)))
    assert cos < 0.95, (
        f"cos(real, noise) = {cos:.4f} ≥ 0.95 - модель не различает контент и шум. "
        "Проверь препроцессинг и корректность ONNX-экспорта (delta-сверку в export_onnx.py)."
    )
