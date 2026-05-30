"""Sanity benchmark: XM3600 image-text retrieval, только русский язык.

Логика zero-shot retrieval:
1. Скачать XM3600 (captions.jsonl + images.tgz) с google.github.io/crossmodal-3600/
   (одноразово, ~17 MB + ~302 MB, кеш в data/xm3600_cache/).
2. Парсить captions.jsonl, отфильтровать RU captions
   (по 2 captions на изображение → 3600 × 2 = 7200 пар).
3. Image embeddings через наш ONNX `Siglip2Encoder`.
4. Text embeddings через PyTorch text encoder SigLIP 2 с **тем же препроцессингом**,
   что закрыл ImageNet sanity: `Siglip2Tokenizer` (auto-lowercase) +
   `normalize_for_siglip2` (manual remove punctuation). См. issue #43054.
5. cosine similarity → метрики:
   - Image→Text R@K: для каждой картинки top-K похожих captions; правильный, если
     хотя бы один из 2 GT-captions в top-K.
   - Text→Image R@K: для каждого caption top-K похожих картинок; правильный, если
     GT image в top-K.

Запуск:
    uv run python scripts/sanity_xm3600_ru.py --subset-size 100   # smoke (~3 мин)
    uv run python scripts/sanity_xm3600_ru.py                     # full ~3600 imgs (~15-20 мин)

Гарантии:
- **Catastrophic guard:** R@1 < 15% по любой из двух метрик (i→t, t→i) → exit 2.
  На N≥100 это сигнал поломки pipeline (не та модальность текста, не тот язык
  captions, перепутаны эмбеддинги).
- **Pass gate** (только full ≥ 3000): avg R@1 ≥ 35%.

Parity-таргет для XM3600 RU намеренно не задаётся: paper SigLIP 2 публикует
только avg R@1 по 36 языкам (40.7% для siglip2-base-patch16-224), без per-language
breakdown. Crossmodal-3600 original paper formal evaluation protocol явно не
фиксирует, community-репортов с RU-specific цифрами не найдено. См.
`docs/sanity_results.md` секцию «Caveat по parity».
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tarfile
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from uneemi_ml import Siglip2Encoder  # noqa: E402
from uneemi_ml.config import MODEL_ID  # noqa: E402


def _load_sanity_imagenet():
    spec = importlib.util.spec_from_file_location(
        "sanity_imagenet", _PROJECT_ROOT / "scripts" / "sanity_imagenet.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SI = _load_sanity_imagenet()
normalize_for_siglip2 = _SI.normalize_for_siglip2
_normalize_rows = _SI._normalize_rows


# --- Константы ---

DATA_DIR: Path = _PROJECT_ROOT / "data"
XM_CACHE: Path = DATA_DIR / "xm3600_cache"
CAPTIONS_ZIP: Path = XM_CACHE / "captions.zip"
CAPTIONS_JSONL: Path = XM_CACHE / "captions.jsonl"
IMAGES_TGZ: Path = XM_CACHE / "images.tgz"
IMAGES_DIR: Path = XM_CACHE / "images"
RESULTS_PATH: Path = _PROJECT_ROOT / "docs" / "sanity_results.md"

CAPTIONS_URL: str = "https://google.github.io/crossmodal-3600/web-data/captions.zip"
IMAGES_URL: str = "https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz"

LANG: str = "ru"
TEXT_BATCH: int = 128
IMAGE_BATCH: int = 32

PASS_THRESHOLD: float = 0.35  # avg R@1
CATASTROPHIC_THRESHOLD: float = 0.15


# --- Download / unpack ---


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю {url} → {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  готово: {dest.stat().st_size / 1024 / 1024:.1f} MB")


def _unpack_captions() -> None:
    if CAPTIONS_JSONL.exists():
        return
    _download(CAPTIONS_URL, CAPTIONS_ZIP)
    print(f"Распаковываю captions.zip → {CAPTIONS_JSONL}...")
    with zipfile.ZipFile(CAPTIONS_ZIP) as zf:
        zf.extractall(XM_CACHE)


def _unpack_images() -> None:
    if IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()):
        return
    _download(IMAGES_URL, IMAGES_TGZ)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Распаковываю images.tgz → {IMAGES_DIR}...")
    with tarfile.open(IMAGES_TGZ, "r:gz") as tf:
        tf.extractall(IMAGES_DIR)


def _find_image_file(image_key: str) -> Path | None:
    """Найти картинку по image_key (формат key обычно совпадает с именем файла)."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = IMAGES_DIR / f"{image_key}{ext}"
        if p.exists():
            return p
    # Может быть в подпапке
    candidates = list(IMAGES_DIR.rglob(f"{image_key}.*"))
    return candidates[0] if candidates else None


# --- Загрузка данных ---


def load_ru_dataset(subset_size: int | None = None) -> tuple[list[Path], list[list[str]]]:
    """Загрузить пары (image_path, [caption1, caption2]) для RU.

    Returns: (image_paths, captions_per_image), len = N изображений.
    Если subset_size задан и < общего числа - берёт первые N (для smoke).
    """
    _unpack_captions()
    _unpack_images()

    print(f"Читаю {CAPTIONS_JSONL}...")
    pairs: list[tuple[Path, list[str]]] = []
    skipped_missing = 0
    skipped_no_lang = 0
    with CAPTIONS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            key = entry["image/key"]
            lang_data = entry.get(LANG)
            if lang_data is None:
                skipped_no_lang += 1
                continue
            captions = lang_data.get("caption") or []
            if not captions:
                skipped_no_lang += 1
                continue
            img_path = _find_image_file(key)
            if img_path is None:
                skipped_missing += 1
                continue
            pairs.append((img_path, list(captions)))

    print(
        f"  Total entries: {len(pairs)}; missing images: {skipped_missing}; "
        f"missing {LANG} captions: {skipped_no_lang}"
    )

    if subset_size is not None and subset_size < len(pairs):
        pairs = pairs[:subset_size]
        print(f"  Subset: первые {len(pairs)} изображений")

    image_paths = [p for p, _ in pairs]
    captions_per_image = [caps for _, caps in pairs]
    return image_paths, captions_per_image


# --- Embeddings ---


def compute_image_embeddings(encoder: Siglip2Encoder, image_paths: list[Path]) -> np.ndarray:
    n = len(image_paths)
    print(f"Image embeddings: {n} картинок батчами по {IMAGE_BATCH}...")
    out = np.zeros((n, 768), dtype=np.float32)
    t0 = time.perf_counter()
    for start in range(0, n, IMAGE_BATCH):
        batch_paths = image_paths[start : start + IMAGE_BATCH]
        batch = [Image.open(p).convert("RGB") for p in batch_paths]
        vecs = encoder.encode_batch(batch)
        out[start : start + len(batch)] = vecs
        if (start // IMAGE_BATCH + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            done = start + len(batch)
            eta = elapsed / done * (n - done)
            print(f"  {done}/{n}  ({elapsed:.0f}s, ~{eta:.0f}s ETA)")
    return _normalize_rows(out)


def compute_text_embeddings(captions_flat: list[str]) -> np.ndarray:
    """SigLIP 2 text encoder с правильным препроцессингом.

    Использует Siglip2Tokenizer (auto-lowercase) + normalize_for_siglip2
    (remove punctuation) - то же, что выровняло ImageNet sanity на parity.
    """
    from transformers import AutoModel, Siglip2Tokenizer

    print(f"Загружаю PyTorch text encoder {MODEL_ID} (Siglip2Tokenizer)...")
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    tokenizer = Siglip2Tokenizer.from_pretrained(MODEL_ID)

    n = len(captions_flat)
    print(f"Text embeddings: {n} captions батчами по {TEXT_BATCH}...")
    normalized = [normalize_for_siglip2(c) for c in captions_flat]

    out = np.zeros((n, 768), dtype=np.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n, TEXT_BATCH):
            batch = normalized[start : start + TEXT_BATCH]
            inputs = tokenizer(
                batch,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            features = model.text_model(**inputs).pooler_output
            out[start : start + len(batch)] = features.cpu().numpy()
            done = start + len(batch)
            if done % (TEXT_BATCH * 5) == 0 or done == n:
                elapsed = time.perf_counter() - t0
                eta = elapsed / done * (n - done) if done < n else 0
                print(f"  {done}/{n}  ({elapsed:.0f}s, ~{eta:.0f}s ETA)")
    return _normalize_rows(out)


# --- Retrieval metrics ---


def evaluate_retrieval(
    img_embs: np.ndarray,
    txt_embs: np.ndarray,
    captions_per_image: list[list[str]],
) -> dict[str, float]:
    """Image↔Text R@1, R@5.

    captions_per_image[i] - captions для image i; в нашем плоском txt_embs они идут
    подряд (caption_to_image: i-я строка = (caption-position // captions_per_img)).
    """
    n_img = img_embs.shape[0]
    n_txt = txt_embs.shape[0]
    # Map: idx_in_flat_txt -> idx_in_img
    txt_to_img = []
    for i, caps in enumerate(captions_per_image):
        txt_to_img.extend([i] * len(caps))
    txt_to_img_arr = np.asarray(txt_to_img)
    assert len(txt_to_img_arr) == n_txt, f"{len(txt_to_img_arr)} vs {n_txt}"

    # similarity matrix (n_img, n_txt)
    sims = img_embs @ txt_embs.T

    # Image→Text: для каждой картинки берём top-K captions, считаем hit если
    # ЛЮБОЙ caption этой картинки попал в top-K.
    top5_i2t = np.argpartition(-sims, kth=5, axis=1)[:, :5]
    top1_i2t = sims.argmax(axis=1)

    i2t_r1 = 0
    i2t_r5 = 0
    for i in range(n_img):
        gt_caption_indices = np.where(txt_to_img_arr == i)[0]
        if top1_i2t[i] in gt_caption_indices:
            i2t_r1 += 1
        if any(c in top5_i2t[i] for c in gt_caption_indices):
            i2t_r5 += 1
    i2t_r1 /= n_img
    i2t_r5 /= n_img

    # Text→Image: для каждого caption берём top-K картинок, hit если GT image в top-K.
    sims_t = sims.T  # (n_txt, n_img)
    top1_t2i = sims_t.argmax(axis=1)
    top5_t2i = np.argpartition(-sims_t, kth=5, axis=1)[:, :5]

    t2i_r1 = float((top1_t2i == txt_to_img_arr).mean())
    t2i_r5 = float(np.mean([txt_to_img_arr[j] in top5_t2i[j] for j in range(n_txt)]))

    avg_r1 = (i2t_r1 + t2i_r1) / 2.0
    avg_r5 = (i2t_r5 + t2i_r5) / 2.0

    return {
        "i2t_r1": float(i2t_r1),
        "i2t_r5": float(i2t_r5),
        "t2i_r1": float(t2i_r1),
        "t2i_r5": float(t2i_r5),
        "avg_r1": avg_r1,
        "avg_r5": avg_r5,
        "n_images": float(n_img),
        "n_captions": float(n_txt),
    }


def evaluate_targets(avg_r1: float, n_images: int) -> dict:
    """XM3600 RU: paper не публикует per-language breakdown.

    Используется только Pass-gate как индикатор корректности pipeline; parity
    intentionally omitted - для RU specifically нет reference number (paper
    avg 40.7% усреднён по 36 языкам). См. docs/sanity_results.md → «Caveat по parity».
    """
    pass_gate = (n_images >= 3000) and (avg_r1 >= PASS_THRESHOLD)
    pass_status = "n/a (smoke run)" if n_images < 3000 else ("PASS" if pass_gate else "FAIL")
    return {
        "pass": pass_gate,
        "pass_status": pass_status,
        "catastrophic_avg_r1": avg_r1 < CATASTROPHIC_THRESHOLD,
    }


def _format_section(metrics: dict, targets: dict, commit: str, n_images: int) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cata_str = "FAIL ⚠️" if targets["catastrophic_avg_r1"] else "PASS"

    return "\n".join(
        [
            f"## XM3600 RU · Прогон {timestamp}",
            "",
            f"**Commit:** `{commit}`",
            f"**Subset:** {int(metrics['n_images'])} изображений, "
            f"{int(metrics['n_captions'])} RU captions",
            "**Text preprocess:** `Siglip2Tokenizer + normalize_for_siglip2` "
            "(тот же фикс, что закрыл ImageNet sanity)",
            "",
            "### Метрики (zero-shot retrieval)",
            "",
            f"- Image→Text R@1: **{metrics['i2t_r1'] * 100:.2f}%** | "
            f"R@5: {metrics['i2t_r5'] * 100:.2f}%",
            f"- Text→Image R@1: **{metrics['t2i_r1'] * 100:.2f}%** | "
            f"R@5: {metrics['t2i_r5'] * 100:.2f}%",
            (
                f"- **avg R@1: {metrics['avg_r1'] * 100:.2f}%** | "
                f"avg R@5: {metrics['avg_r5'] * 100:.2f}%"
            ),
            "",
            "### Таргеты",
            "",
            f"- Pass (avg R@1 ≥ {PASS_THRESHOLD * 100:.0f}%): **{targets['pass_status']}**",
            f"- Catastrophic guard (avg R@1 ≥ {CATASTROPHIC_THRESHOLD * 100:.0f}%): **{cata_str}**",
            "- Parity: intentionally omitted (см. «Caveat по parity» в этом документе)",
        ]
    )


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


def _append_results(section: str) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = RESULTS_PATH.read_text(encoding="utf-8").rstrip() if RESULTS_PATH.exists() else ""
    RESULTS_PATH.write_text(f"{existing}\n\n---\n\n{section}\n", encoding="utf-8")


# --- Main ---


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity benchmark XM3600 RU retrieval (SigLIP 2).")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Сколько первых изображений взять (None = все ~3600). <500 = smoke.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Не аппендить в docs/sanity_results.md.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("XM3600 RU Retrieval Sanity - Siglip2Encoder (ONNX)")
    print("=" * 70)
    print(f"Subset size: {args.subset_size if args.subset_size else 'ALL (~3600)'}")
    print()

    t_start = time.perf_counter()

    image_paths, captions_per_image = load_ru_dataset(args.subset_size)
    captions_flat = [c for caps in captions_per_image for c in caps]
    n_img = len(image_paths)
    n_txt = len(captions_flat)
    print(f"\nИтого: {n_img} изображений, {n_txt} RU captions (avg {n_txt / n_img:.1f}/img)")
    print(f"Пример caption (raw): {captions_per_image[0][0]!r}")
    print(f"  → normalized:        {normalize_for_siglip2(captions_per_image[0][0])!r}\n")

    encoder = Siglip2Encoder()
    img_embs = compute_image_embeddings(encoder, image_paths)
    txt_embs = compute_text_embeddings(captions_flat)

    print("\nСчитаю retrieval-метрики...")
    metrics = evaluate_retrieval(img_embs, txt_embs, captions_per_image)
    total_time = time.perf_counter() - t_start

    print()
    print("=" * 70)
    print("Результаты:")
    print(f"  Image→Text:  R@1={metrics['i2t_r1'] * 100:.2f}%  R@5={metrics['i2t_r5'] * 100:.2f}%")
    print(f"  Text→Image:  R@1={metrics['t2i_r1'] * 100:.2f}%  R@5={metrics['t2i_r5'] * 100:.2f}%")
    print(f"  avg R@1: {metrics['avg_r1'] * 100:.2f}%   avg R@5: {metrics['avg_r5'] * 100:.2f}%")
    print(f"  N images: {n_img}, N captions: {n_txt}, time: {total_time:.0f}s")

    targets = evaluate_targets(metrics["avg_r1"], n_img)
    # Catastrophic guard ALSO on each direction R@1 (per user)
    catastrophic_dir = (
        metrics["i2t_r1"] < CATASTROPHIC_THRESHOLD or metrics["t2i_r1"] < CATASTROPHIC_THRESHOLD
    )
    catastrophic = targets["catastrophic_avg_r1"] or catastrophic_dir

    print()
    print(
        f"  Catastrophic guard (любой R@1 ≥ {CATASTROPHIC_THRESHOLD * 100:.0f}%): "
        f"{'FAIL ⚠️' if catastrophic else 'PASS'}"
    )
    print(f"  Pass gate (avg R@1 ≥ {PASS_THRESHOLD * 100:.0f}%): {targets['pass_status']}")
    print("  Parity: intentionally omitted (no RU reference in paper)")
    print("=" * 70)

    if not args.no_write:
        commit = _git_commit_hash()
        section = _format_section(metrics, targets, commit, n_img)
        _append_results(section)
        print(f"\nСекция дописана в {RESULTS_PATH}")

    if catastrophic:
        print(
            "\n⚠️  CATASTROPHIC FAILURE: R@1 ниже 15%. НЕ запускайте полный прогон. "
            "Вероятные причины: не тот язык captions (фильтрация по 'ru'), не та "
            "модальность text embeddings, перепутаны i2t/t2i направления.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
