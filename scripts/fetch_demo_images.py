"""Скачивание банка демо-изображений для feature_pipeline (синтетика для демонстрации).

Зачем: по требованию проекта board-эмбеддинги извлекаются РЕАЛЬНЫМ SigLIP, а не
синтезируются. Значит нужны реальные картинки. Берём их детерминированно с
picsum.photos (по seed - воспроизводимо), раскладывая по эстетическим кластерам.
Изображения внутри кластера - общий пул, из которого профили набирают доски;
пересечение картинок даёт близость board-эмбеддингов внутри кластера.

Идемпотентность: уже скачанные файлы пропускаются; при наличии маркера .done и
полного набора скрипт завершается мгновенно. Это позволяет `make up` не качать
банк заново.

Офлайн-фолбэк: если picsum недоступен, генерируем детерминированную процедурную
картинку (кластеро-зависимый цвет/текстура). SigLIP по ней всё равно реальный -
требование «реальный SigLIP» сохраняется, меняется лишь источник пикселей.

Запуск: uv run python scripts/fetch_demo_images.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# Скрипт лежит вне пакета - добавляем src/ в путь, чтобы импортировать константы.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")  # консоль Windows иначе бьётся об кириллицу

from uneemi_ml.config import IMAGE_SIZE  # noqa: E402
from uneemi_ml.demo import (  # noqa: E402
    CLUSTERS,
    IMAGES_DIR,
    IMAGES_DONE_MARKER,
    IMAGES_MANIFEST,
    cluster_dir,
)

IMAGES_PER_CLUSTER: int = int(os.environ.get("DEMO_IMAGES_PER_CLUSTER", "18"))
PICSUM_TEMPLATE = "https://picsum.photos/seed/{seed}/{size}/{size}"
_HTTP_TIMEOUT = 15


def _procedural_image(cluster_idx: int, i: int) -> Image.Image:
    """Детерминированная процедурная картинка как офлайн-фолбэк.

    Цвет/частоты зависят от кластера - чтобы кластеры различались и в офлайне.
    """
    n_clusters = len(CLUSTERS)
    rng = np.random.default_rng(1000 * cluster_idx + i)
    base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float32)
    freq = 0.02 + 0.01 * cluster_idx
    for ch in range(3):
        phase = rng.uniform(0, np.pi)
        tint = 0.4 + 0.6 * ((cluster_idx + ch) % n_clusters) / n_clusters
        base[:, :, ch] = tint * (0.5 + 0.5 * np.sin(freq * xx + phase) * np.cos(freq * yy + phase))
    base += rng.normal(0, 0.03, base.shape)
    arr = np.clip(base * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _download(seed: str, target: Path) -> str:
    """Скачать картинку с picsum. Вернуть источник: 'picsum' или 'procedural'."""
    url = PICSUM_TEMPLATE.format(seed=seed, size=IMAGE_SIZE)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "uneemi-demo/1.0"})
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        img.save(target, format="JPEG", quality=88)
        return "picsum"
    except Exception as exc:  # noqa: BLE001 - сеть капризна, спокойно уходим в фолбэк
        print(f"      picsum недоступен для {seed} ({exc}); процедурный фолбэк")
        return "procedural"


def main() -> int:
    # Быстрый выход, если банк уже полон (идемпотентность для make up).
    if IMAGES_DONE_MARKER.exists() and IMAGES_MANIFEST.exists():
        print(f"Банк демо-изображений уже готов: {IMAGES_DIR} (пропуск)")
        return 0

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {}
    sources: dict[str, int] = {"picsum": 0, "procedural": 0}

    for c_idx, cluster in enumerate(CLUSTERS):
        cdir = cluster_dir(cluster)
        cdir.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        print(f"[{c_idx + 1}/{len(CLUSTERS)}] кластер {cluster}: {IMAGES_PER_CLUSTER} картинок")
        for i in range(IMAGES_PER_CLUSTER):
            target = cdir / f"{i:03d}.jpg"
            if target.exists():  # частично скачанный банк - дозаливаем недостающее
                files.append(str(target.relative_to(IMAGES_DIR)))
                continue
            seed = f"uneemi-{cluster}-{i:03d}"
            src = _download(seed, target)
            if src == "procedural":
                _procedural_image(c_idx, i).save(target, format="JPEG", quality=88)
            sources[src] += 1
            files.append(str(target.relative_to(IMAGES_DIR)))
        manifest[cluster] = files

    IMAGES_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    IMAGES_DONE_MARKER.write_text("ok\n", encoding="utf-8")
    total = sum(len(v) for v in manifest.values())
    print(
        f"\nГотово: {total} изображений в {len(CLUSTERS)} кластерах "
        f"(picsum={sources['picsum']}, procedural={sources['procedural']}). "
        f"Каталог: {IMAGES_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
