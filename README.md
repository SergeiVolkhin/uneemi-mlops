# Uneemi ML

ML-пайплайн для «вайб-матчинга» через визуальные доски. На первом этапе - ONNX-инференс энкодера `google/siglip2-base-patch16-224` на CPU.

Подробный план оценки качества см. [`docs/metrics_and_benchmarks.md`](docs/metrics_and_benchmarks.md).

## Окружение

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) как пакетный менеджер

```powershell
uv sync
```

## Экспорт модели

```powershell
uv run python scripts/export_onnx.py
```

Скрипт скачает `google/siglip2-base-patch16-224`, экспортирует только `vision_model` в `models/siglip2_vision.onnx` (~370 MB), выполнит delta-сверку PyTorch ↔ ONNX (порог `max|Δ| < 1e-4`).

## Тесты

```powershell
uv run pytest tests/ -v
```

Если ONNX-файла ещё нет - все тесты `skip` с подсказкой запустить экспорт.

## Использование

```python
from PIL import Image
from uneemi_ml import Siglip2Encoder

encoder = Siglip2Encoder()
vec = encoder.encode(Image.open("path/to/image.jpg"))  # (1, 768) float32
```
