"""Reference experiment: SigLIP 2 B/16@256 zero-shot для resolution-gap hypothesis.

Этап 3.5 Шага 2. Цель: проверить, насколько разница resolution (paper @256 vs наш
ONNX-экспорт @224) объясняет gap -9 п.п. между нашим pipeline (top-1 70.02%) и paper
(79.1%).

Один-off reference: НЕ экспортируем @256 в ONNX (не засоряем models/), используем
PyTorch напрямую. AutoProcessor берёт canonical preprocess для @256 (size=256,
BILINEAR, mean/std=0.5).

Прогон на том же 5000-stratified subset (seed=42), что и основной @224-прогон -
честное сравнение.

Запуск:
    uv run python scripts/sanity_imagenet_256_ref.py

Время: ~5 мин download + ~25 мин class embeddings + ~5-10 мин image inference = ~35-40 мин.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Windows console падает на Unicode.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


# --- Импорт helpers из sanity_imagenet.py (без проблем relative-import) ---


def _load_sanity_module():
    spec = importlib.util.spec_from_file_location(
        "sanity_imagenet", _PROJECT_ROOT / "scripts" / "sanity_imagenet.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SI = _load_sanity_module()
IMAGENET_TEMPLATES: tuple[str, ...] = _SI.IMAGENET_TEMPLATES
load_imagenet_val_subset = _SI.load_imagenet_val_subset
_load_openai_classnames = _SI._load_openai_classnames
_normalize_rows = _SI._normalize_rows

# --- Constants ---

MODEL_ID_256: str = "google/siglip2-base-patch16-256"
N_CLASSES: int = 1000
TEXT_BATCH: int = 128
IMAGE_BATCH: int = 32
SUBSET_SIZE: int = 5000
SEED: int = 42

DATA_DIR: Path = _PROJECT_ROOT / "data"
CLASS_EMBEDDINGS_PATH_256: Path = DATA_DIR / "imagenet_class_embeddings_256.npy"
RESULTS_PATH: Path = _PROJECT_ROOT / "docs" / "sanity_results.md"


# --- Build class embeddings @256 ---


def build_class_embeddings_256(class_names: list[str]) -> np.ndarray:
    """1000 × 768 class embeddings @256, кеш в data/imagenet_class_embeddings_256.npy."""
    if CLASS_EMBEDDINGS_PATH_256.exists():
        print(f"Загружаю class embeddings @256 из кеша: {CLASS_EMBEDDINGS_PATH_256}")
        emb = np.load(CLASS_EMBEDDINGS_PATH_256)
        if emb.shape != (N_CLASSES, 768):
            raise RuntimeError(
                f"Кеш имеет shape {emb.shape}, ожидалось ({N_CLASSES}, 768). "
                f"Удалите {CLASS_EMBEDDINGS_PATH_256} и пересчитайте."
            )
        return emb

    print(f"Загружаю PyTorch модель {MODEL_ID_256} (text encoder)...")
    model = AutoModel.from_pretrained(MODEL_ID_256).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_256)

    print(f"Строю class embeddings @256: {N_CLASSES} × {len(IMAGENET_TEMPLATES)} promtps")
    embeddings = np.zeros((N_CLASSES, 768), dtype=np.float32)
    t_start = time.perf_counter()

    with torch.no_grad():
        for class_idx, name in enumerate(class_names):
            prompts = [tpl.format(name) for tpl in IMAGENET_TEMPLATES]
            vecs: list[np.ndarray] = []
            for i in range(0, len(prompts), TEXT_BATCH):
                batch = prompts[i : i + TEXT_BATCH]
                inputs = tokenizer(
                    batch,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                )
                features = model.text_model(**inputs).pooler_output
                vecs.append(features.cpu().numpy())
            stacked = _normalize_rows(np.concatenate(vecs, axis=0))
            mean_emb = stacked.mean(axis=0)
            embeddings[class_idx] = mean_emb / max(np.linalg.norm(mean_emb), 1e-12)

            if (class_idx + 1) % 100 == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (class_idx + 1) * (N_CLASSES - class_idx - 1)
                print(f"  {class_idx + 1}/{N_CLASSES}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)")

    np.save(CLASS_EMBEDDINGS_PATH_256, embeddings)
    print(f"  Сохранено в {CLASS_EMBEDDINGS_PATH_256}")
    return embeddings


# --- Image inference @256 via PyTorch + AutoProcessor ---


def compute_image_embeddings_256(images: list[Image.Image]) -> np.ndarray:
    print(f"Загружаю PyTorch модель {MODEL_ID_256} (vision_model)...")
    model = AutoModel.from_pretrained(MODEL_ID_256).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID_256)

    n = len(images)
    print(f"Image inference @256: {n} картинок батчами по {IMAGE_BATCH}...")
    image_embs = np.zeros((n, 768), dtype=np.float32)
    t_start = time.perf_counter()

    with torch.no_grad():
        for start in range(0, n, IMAGE_BATCH):
            batch = images[start : start + IMAGE_BATCH]
            inputs = processor(images=batch, return_tensors="pt")
            features = model.vision_model(**inputs).pooler_output
            image_embs[start : start + len(batch)] = features.cpu().numpy()

            if (start // IMAGE_BATCH + 1) % 10 == 0:
                elapsed = time.perf_counter() - t_start
                done = start + len(batch)
                eta = elapsed / done * (n - done)
                print(f"  {done}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)")

    return _normalize_rows(image_embs)


# --- Append section to md ---


def append_section(
    top1: float,
    top5: float,
    n_samples: int,
    total_time_s: float,
    commit: str,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    delta_paper = (top1 - 0.791) * 100
    delta_224 = (top1 - 0.7002) * 100

    # Вывод по порогам
    if top1 >= 0.76:
        conclusion = (
            "**Вывод:** resolution полностью объясняет gap. Наш ONNX @224 pipeline - "
            "на пределе разрешения. Paper-числа @256 воспроизводятся при подключении @256-весов."
        )
    elif top1 >= 0.72:
        residual = (0.78 - top1) * 100  # 78% - paper minus -1 за нашу @224-extrapolation
        conclusion = (
            f"**Вывод:** resolution **частично** объясняет gap "
            f"(+{delta_224:.2f} п.п. от @224 = {top1 * 100:.2f}% @256, paper 79.1%). "
            f"Остаток ~{residual:.1f} п.п. от других факторов (evaluation script, dataset variant)."
        )
    else:
        conclusion = (
            "**Вывод:** resolution **не объясняет** gap. @256 даёт "
            f"{top1 * 100:.2f}%, что недостаточно для воспроизведения paper 79.1%. "
            "Нужно дальнейшее расследование: evaluation script, dataset variant, версия модели."
        )

    section = "\n".join(
        [
            "## Resolution hypothesis check (B/16@256 reference)",
            "",
            f"**Прогон:** {timestamp}",
            f"**Commit:** `{commit}`",
            f"**Модель:** `{MODEL_ID_256}` через PyTorch reference (без ONNX-экспорта)",
            "**Subset:** 5000 картинок (stratified, тот же seed=42 что в @224-прогоне)",
            "**Classnames:** `openai` | **Templates:** 80 OpenAI ImageNet",
            f"**Время прогона:** {total_time_s:.0f}s",
            "",
            "| Resolution | Backend | top-1 | top-5 | Δ от paper 79.1% |",
            "|---|---|---|---|---|",
            "| @224 | наш ONNX (PyTorch parity 7.6e-06) | 70.02% | 86.32% | -9.08 |",
            (
                f"| @256 | HF PyTorch reference | "
                f"{top1 * 100:.2f}% | {top5 * 100:.2f}% | {delta_paper:+.2f} |"
            ),
            "",
            f"**Δ resolution effect (@256 - @224):** {delta_224:+.2f} п.п.",
            "",
            conclusion,
        ]
    )

    existing = RESULTS_PATH.read_text(encoding="utf-8").rstrip()
    RESULTS_PATH.write_text(f"{existing}\n\n---\n\n{section}\n", encoding="utf-8")
    print(f"\nСекция дописана в {RESULTS_PATH}")


def _git_commit_hash() -> str:
    import subprocess

    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return r.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


def main() -> int:
    print("=" * 70)
    print("Resolution hypothesis check - SigLIP 2 B/16@256 reference")
    print("=" * 70)
    print(f"Model:       {MODEL_ID_256}")
    print(f"Subset size: {SUBSET_SIZE} (seed={SEED})")
    print()

    t_start = time.perf_counter()

    class_names = _load_openai_classnames()
    sample_line = (
        f"OpenAI classnames: {len(class_names)} (sample: {class_names[0]!r}, {class_names[999]!r})"
    )
    print(sample_line + "\n")

    images, labels = load_imagenet_val_subset(SUBSET_SIZE, seed=SEED)
    print(f"Subset: {len(images)} картинок, {len(set(labels))} уникальных классов\n")

    class_embs = build_class_embeddings_256(class_names)
    print(f"Class embeddings @256: {class_embs.shape}\n")

    image_embs = compute_image_embeddings_256(images)

    # cosine similarity (both L2-normalized)
    sims = image_embs @ class_embs.T
    labels_arr = np.asarray(labels)
    top1 = float((sims.argmax(axis=1) == labels_arr).mean())
    top5_idx = np.argpartition(-sims, kth=5, axis=1)[:, :5]
    top5 = float(np.mean([labels_arr[i] in top5_idx[i] for i in range(len(labels))]))

    total_time = time.perf_counter() - t_start
    print()
    print("=" * 70)
    print("Результаты @256 (PyTorch reference):")
    print(f"  top-1: {top1 * 100:.2f}%")
    print(f"  top-5: {top5 * 100:.2f}%")
    print(f"  N samples: {len(images)}")
    print(f"  Время прогона: {total_time:.0f}s")
    print()
    print("  Сравнение с @224 (наш ONNX): 70.02% top-1")
    print(f"  Δ resolution effect (@256 - @224): {(top1 - 0.7002) * 100:+.2f} п.п.")
    print(f"  Δ от paper 79.1%: {(top1 - 0.791) * 100:+.2f} п.п.")
    print("=" * 70)

    commit = _git_commit_hash()
    append_section(top1, top5, len(images), total_time, commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
