"""Константы и пути демо-данных пайплайна (синтетика для демонстрации).

Здесь только лёгкие константы и хелперы путей - без тяжёлых импортов (модель,
torch), чтобы и host-скрипт скачивания картинок, и DAG могли импортировать одно
и то же определение кластеров без побочных эффектов.

Кластеры названы по эстетическим направлениям из docs/metrics_and_benchmarks.md
(раздел 2.2) - так демо-данные согласованы с уже описанной методикой оценки.
"""

from __future__ import annotations

from pathlib import Path

from uneemi_ml.config import DATA_DIR

# Эстетические кластеры = «вайбы». Доска профиля набирается из картинок одного
# кластера, поэтому профили внутри кластера получают близкие board-эмбеддинги -
# это и даёт классификатору честный сигнал близости.
CLUSTERS: tuple[str, ...] = (
    "cottagecore",
    "dark_academia",
    "minimalism",
    "cyberpunk",
    "y2k",
    "weirdcore",
)
NUM_CLUSTERS: int = len(CLUSTERS)

IMAGES_DIR: Path = DATA_DIR / "images"
IMAGES_DONE_MARKER: Path = IMAGES_DIR / ".done"
IMAGES_MANIFEST: Path = IMAGES_DIR / "manifest.json"

RAW_DIR: Path = DATA_DIR / "raw"
PAIRS_DIR: Path = DATA_DIR / "pairs"


def cluster_dir(cluster: str) -> Path:
    """Каталог картинок конкретного кластера."""
    return IMAGES_DIR / cluster
