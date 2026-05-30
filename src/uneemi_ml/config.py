"""Конфигурация ML-пайплайна Uneemi: пути и константы."""

from __future__ import annotations

from pathlib import Path

import psutil

MODEL_ID: str = "google/siglip2-base-patch16-224"
IMAGE_SIZE: int = 224
EMBED_DIM: int = 768

# Дефолт - число физических ядер с cap=16. Эксперимент threads=4/8/12/16 показал,
# что выше 12 нет выгоды (HT-contention на compute-bound трансформере); cap защищает
# huge-CPU (32+ ядер) от деградации. См. docs/benchmark_results.md.
ORT_INTRA_OP_THREADS: int = min(psutil.cpu_count(logical=False) or 4, 16)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"
ONNX_VISION_PATH: Path = MODELS_DIR / "siglip2_vision.onnx"
