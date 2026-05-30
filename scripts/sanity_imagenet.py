"""Sanity benchmark: ImageNet zero-shot top-1 для нашего ONNX SigLIP 2.

Не используется clip_benchmark, потому что:
1. Нет extension hook для ONNX - `--model_type` ограничен open_clip/ja_clip.
2. open_clip ViT-B-16-SigLIP2/webli - это timm-checkpoint с bicubic resampling,
   отличающийся от google/siglip2-base-patch16-224 + bilinear (нашего pipeline).

Здесь мы валидируем именно наш ONNX pipeline.

Логика zero-shot:
1. Загрузить stratified subset из HF `mrm8488/ImageNet1K-val` (split="train",
   там лежит ImageNet val: 39.3K rows, ~39+ строк/класс).
2. 1000 классов × 80 промптов → text embeddings через PyTorch text encoder
   SigLIP 2 (однократно, кеш в data/imagenet_class_embeddings.npy).
3. Картинки через наш ONNX `Siglip2Encoder` → image embeddings.
4. cosine similarity (после L2-norm) → argmax → top-1 / top-5.

Запуск:
    uv run python scripts/sanity_imagenet.py --subset-size 100    # smoke (~1 мин)
    uv run python scripts/sanity_imagenet.py --subset-size 5000   # full (~40 мин)

Гарантии:
- **Catastrophic guard**: top-1 < 50% → exit code 2. На N≥100 такое значение
  не объяснить статистикой - это сигнал поломки pipeline (mapping classnames,
  L2-norm, prompt-шаблоны, выбор text-encoder API).
- **Pass gate** (только при subset_size ≥ 5000): top-1 ≥ 75%.
- **Parity** (желаемое): top-1 ∈ [76.7, 78.7] (paper 79.1% @ B/16@256,
  у нас B/16@224 → возможно -1 п.п.).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

# Windows console падает на Unicode (м/с, символы перцентилей).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# Скрипт лежит вне пакета.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from uneemi_ml import Siglip2Encoder  # noqa: E402
from uneemi_ml.config import MODEL_ID  # noqa: E402

# --- Пути и константы ---

DATA_DIR: Path = _PROJECT_ROOT / "data"
HF_CACHE: Path = DATA_DIR / "hf_datasets"
KERAS_CLASS_INDEX_PATH: Path = DATA_DIR / "imagenet_class_index.json"
OPENAI_CLASSES_PATH: Path = DATA_DIR / "openai_imagenet_classes.json"
RESULTS_PATH: Path = _PROJECT_ROOT / "docs" / "sanity_results.md"

KERAS_CLASS_INDEX_URL: str = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)
OPENAI_CLASSES_URL: str = (
    "https://raw.githubusercontent.com/mlfoundations/open_clip/"
    "main/src/open_clip/zero_shot_metadata.py"
)

DATASET_ID: str = "mrm8488/ImageNet1K-val"
DATASET_SPLIT: str = "train"  # это ImageNet val, но залит под именем split="train"

# Источники имён классов:
# - keras: simple Keras-port (great_white_shark → "great white shark"). Используется
#   в torchvision/datasets, но даёт ~-3..-5 п.п. на CLIP-style моделях.
# - openai: curated имена из open_clip's IMAGENET_CLASSNAMES (= оригинал OpenAI CLIP).
CLASSNAMES_KERAS: str = "keras"
CLASSNAMES_OPENAI: str = "openai"
CLASSNAMES_CHOICES: tuple[str, ...] = (CLASSNAMES_KERAS, CLASSNAMES_OPENAI)

# Text-preprocessing варианты для tokenization промптов:
# - auto: AutoTokenizer (= GemmaTokenizer), без преобразований
# - siglip2-normalized: Siglip2Tokenizer (auto-lowercase) + manual remove punctuation
#   (per HF community recommendation для воспроизведения paper-чисел, см.
#    https://github.com/huggingface/transformers/issues/43054)
TEXT_PP_AUTO: str = "auto"
TEXT_PP_SIGLIP2_NORM: str = "siglip2-normalized"
TEXT_PP_CHOICES: tuple[str, ...] = (TEXT_PP_AUTO, TEXT_PP_SIGLIP2_NORM)


def class_embeddings_path(source: str, text_preprocess: str = TEXT_PP_AUTO) -> Path:
    """Per-source + per-text-preprocess cache путь."""
    if text_preprocess == TEXT_PP_AUTO:
        return DATA_DIR / f"imagenet_class_embeddings_{source}.npy"
    return DATA_DIR / f"imagenet_class_embeddings_{source}_{text_preprocess.replace('-', '_')}.npy"


_PUNCT_REMOVAL_TABLE = str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def normalize_for_siglip2(text: str) -> str:
    """Lowercase + remove standard ASCII punctuation + collapse whitespace.

    Per HF community recommendation для воспроизведения paper SigLIP2 ImageNet zero-shot.
    Источник: HF forum «SigLIP-2 models show lower zero-shot accuracy than reported»,
    issue transformers#43054. `Siglip2Tokenizer` сам делает lowercase, но
    punctuation removal - нет, делаем здесь.
    """
    text = text.lower()
    text = text.translate(_PUNCT_REMOVAL_TABLE)
    return " ".join(text.split())


N_CLASSES: int = 1000
TEXT_BATCH: int = 128
IMAGE_BATCH: int = 32

PASS_THRESHOLD: float = 0.75
PARITY_RANGE: tuple[float, float] = (0.767, 0.787)
CATASTROPHIC_THRESHOLD: float = 0.50

# 80 стандартных OpenAI CLIP ImageNet prompt-шаблонов.
# Источник: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
IMAGENET_TEMPLATES: tuple[str, ...] = (
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
)


# --- Загрузка class index (synset → human name) ---


def _load_keras_classnames() -> list[str]:
    """Keras imagenet_class_index.json → 1000 строк (underscore→space)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not KERAS_CLASS_INDEX_PATH.exists():
        print(f"Скачиваю Keras imagenet_class_index.json → {KERAS_CLASS_INDEX_PATH}...")
        urllib.request.urlretrieve(KERAS_CLASS_INDEX_URL, KERAS_CLASS_INDEX_PATH)
    with KERAS_CLASS_INDEX_PATH.open("r", encoding="utf-8") as f:
        idx_data = json.load(f)
    if len(idx_data) != N_CLASSES:
        raise RuntimeError(f"Keras: ожидалось {N_CLASSES} классов, получено {len(idx_data)}")
    return [idx_data[str(i)][1].replace("_", " ") for i in range(N_CLASSES)]


def _load_openai_classnames() -> list[str]:
    """OpenAI curated IMAGENET_CLASSNAMES из open_clip/zero_shot_metadata.py.

    Парсит Python-исходник через ast и сохраняет как JSON-массив.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not OPENAI_CLASSES_PATH.exists():
        import ast

        print(f"Скачиваю OpenAI classnames из {OPENAI_CLASSES_URL}...")
        src = urllib.request.urlopen(OPENAI_CLASSES_URL).read().decode("utf-8")
        tree = ast.parse(src)
        classnames: tuple[str, ...] | None = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "IMAGENET_CLASSNAMES":
                        classnames = ast.literal_eval(node.value)
                        break
        if classnames is None:
            raise RuntimeError("IMAGENET_CLASSNAMES не найден в открытом исходнике open_clip.")
        with OPENAI_CLASSES_PATH.open("w", encoding="utf-8") as f:
            json.dump(list(classnames), f, ensure_ascii=False, indent=2)
        print(f"  Сохранено в {OPENAI_CLASSES_PATH}")

    with OPENAI_CLASSES_PATH.open("r", encoding="utf-8") as f:
        names = json.load(f)
    if len(names) != N_CLASSES:
        raise RuntimeError(f"OpenAI: ожидалось {N_CLASSES} классов, получено {len(names)}")
    return list(names)


def load_class_names(source: str) -> list[str]:
    """1000 имён классов от выбранного источника. Индекс совпадает с label_idx HF датасета."""
    if source == CLASSNAMES_KERAS:
        return _load_keras_classnames()
    if source == CLASSNAMES_OPENAI:
        return _load_openai_classnames()
    raise ValueError(
        f"Неизвестный источник classnames: {source!r} (ожидалось одно из {CLASSNAMES_CHOICES})"
    )


# --- Загрузка subset из HF датасета ---


def load_imagenet_val_subset(
    subset_size: int, seed: int = 42
) -> tuple[list[Image.Image], list[int]]:
    """Загрузить subset картинок + labels.

    - subset_size ≥ 1000: stratified (subset_size // 1000 на класс).
    - subset_size < 1000: случайные subset_size картинок (для smoke).
    """
    from datasets import load_dataset

    print(f"Загружаю датасет {DATASET_ID} (split={DATASET_SPLIT})...")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT, cache_dir=str(HF_CACHE))
    print(f"  Всего строк: {len(ds)}")

    print("  Чтение labels-колонки (без декодирования картинок)...")
    labels_all: list[int] = ds["label"]

    rng = np.random.default_rng(seed)

    if subset_size >= N_CLASSES:
        n_per_class = subset_size // N_CLASSES
        print(f"  Stratified: {n_per_class} на класс × {N_CLASSES} = {n_per_class * N_CLASSES}")
        by_label: defaultdict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels_all):
            by_label[lbl].append(i)
        selected: list[int] = []
        for lbl in range(N_CLASSES):
            pool = by_label.get(lbl, [])
            if not pool:
                print(f"  ВНИМАНИЕ: класс {lbl} отсутствует в датасете")
                continue
            take = min(n_per_class, len(pool))
            chosen = rng.choice(pool, size=take, replace=False)
            selected.extend(chosen.tolist())
    else:
        print(f"  Random subset (subset_size={subset_size} < {N_CLASSES} классов)")
        selected = rng.choice(len(ds), size=subset_size, replace=False).tolist()

    selected.sort()  # для последовательного чтения parquet
    print(f"  Выбрано индексов: {len(selected)}")

    print("  Декодирую выбранные картинки...")
    samples = ds.select(selected)
    images: list[Image.Image] = []
    labels: list[int] = []
    for row in samples:
        img = row["image"]
        if not isinstance(img, Image.Image):
            raise RuntimeError(f"Ожидался PIL.Image, получен {type(img)}")
        images.append(img.convert("RGB"))
        labels.append(int(row["label"]))

    return images, labels


# --- Построение class embeddings через PyTorch text encoder ---


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def build_class_embeddings(
    class_names: list[str], source: str, text_preprocess: str = TEXT_PP_AUTO
) -> np.ndarray:
    """1000 × 768 class embeddings (L2-normalized), кеш в data/ per-(source,preprocess).

    Для каждого класса: 80 промптов → text encoder → L2-norm каждого → mean → L2-norm mean.

    `text_preprocess`:
    - auto: AutoTokenizer (= GemmaTokenizer), промпты как есть
    - siglip2-normalized: Siglip2Tokenizer (auto-lowercase) + ручной remove punctuation
    """
    cache_path = class_embeddings_path(source, text_preprocess)
    if cache_path.exists():
        print(f"Загружаю class embeddings из кеша: {cache_path}")
        embeddings = np.load(cache_path)
        if embeddings.shape != (N_CLASSES, 768):
            raise RuntimeError(
                f"Кеш имеет shape {embeddings.shape}, ожидалось ({N_CLASSES}, 768). "
                f"Удалите {cache_path} и пересчитайте."
            )
        return embeddings

    print(
        f"Строю class embeddings: {N_CLASSES} классов × {len(IMAGENET_TEMPLATES)} промптов "
        f"(text_preprocess={text_preprocess})"
    )
    print(f"  Загружаю PyTorch text encoder {MODEL_ID} (однократно)...")
    from transformers import AutoModel

    full_model = AutoModel.from_pretrained(MODEL_ID).eval()
    if text_preprocess == TEXT_PP_SIGLIP2_NORM:
        from transformers import Siglip2Tokenizer

        tokenizer = Siglip2Tokenizer.from_pretrained(MODEL_ID)
        print("  Используется Siglip2Tokenizer (auto-lowercase) + manual punctuation removal.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # В transformers 5.x model.get_text_features() возвращает обёртку-объект,
    # не тензор; зовём text_model напрямую и берём pooler_output - для SigLIP
    # это и есть готовый text-feature (с финальной проекцией внутри pooler).
    if not hasattr(full_model, "text_model"):
        raise RuntimeError(
            "Загруженная модель не имеет .text_model. Проверьте transformers версию и тип модели."
        )

    embeddings = np.zeros((N_CLASSES, 768), dtype=np.float32)
    t_start = time.perf_counter()

    with torch.no_grad():
        for class_idx, name in enumerate(class_names):
            prompts = [tpl.format(name) for tpl in IMAGENET_TEMPLATES]
            if text_preprocess == TEXT_PP_SIGLIP2_NORM:
                prompts = [normalize_for_siglip2(p) for p in prompts]
            class_vecs: list[np.ndarray] = []
            for i in range(0, len(prompts), TEXT_BATCH):
                batch = prompts[i : i + TEXT_BATCH]
                inputs = tokenizer(
                    batch,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                )
                features = full_model.text_model(**inputs).pooler_output
                class_vecs.append(features.cpu().numpy())

            stacked = np.concatenate(class_vecs, axis=0)
            stacked = _normalize_rows(stacked)
            mean_emb = stacked.mean(axis=0)
            embeddings[class_idx] = mean_emb / max(np.linalg.norm(mean_emb), 1e-12)

            if (class_idx + 1) % 100 == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (class_idx + 1) * (N_CLASSES - class_idx - 1)
                print(f"  {class_idx + 1}/{N_CLASSES}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)")

    np.save(cache_path, embeddings)
    print(f"  Сохранено в {cache_path}")
    return embeddings


# --- Evaluation ---


def evaluate_zero_shot(
    encoder: Siglip2Encoder,
    images: list[Image.Image],
    labels: list[int],
    class_embeddings: np.ndarray,
) -> dict[str, float]:
    """Прогоняет картинки батчами, считает top-1 / top-5."""
    n = len(images)
    print(f"Inference: {n} картинок батчами по {IMAGE_BATCH}...")

    image_embs = np.zeros((n, 768), dtype=np.float32)
    t_start = time.perf_counter()

    for start in range(0, n, IMAGE_BATCH):
        batch = images[start : start + IMAGE_BATCH]
        vecs = encoder.encode_batch(batch)
        image_embs[start : start + len(batch)] = vecs

        if (start // IMAGE_BATCH + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            done = start + len(batch)
            eta = elapsed / done * (n - done)
            print(f"  {done}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)")

    image_embs = _normalize_rows(image_embs)
    # class_embeddings уже L2-normalized; cosine = dot product.
    sims = image_embs @ class_embeddings.T  # (n, 1000)

    top5_idx = np.argpartition(-sims, kth=5, axis=1)[:, :5]
    labels_arr = np.asarray(labels)
    top1 = float((sims.argmax(axis=1) == labels_arr).mean())
    top5 = float(np.mean([labels_arr[i] in top5_idx[i] for i in range(n)]))

    elapsed = time.perf_counter() - t_start
    return {
        "top1": top1,
        "top5": top5,
        "n_samples": n,
        "time_per_image_ms": (elapsed / n) * 1000.0,
        "total_time_s": elapsed,
    }


def evaluate_targets(top1: float, subset_size: int) -> dict[str, bool | str]:
    """Pass / Parity / Catastrophic checks."""
    catastrophic = top1 < CATASTROPHIC_THRESHOLD
    pass_gate = (subset_size >= 5000) and (top1 >= PASS_THRESHOLD)
    parity = PARITY_RANGE[0] <= top1 <= PARITY_RANGE[1]

    pass_status = "n/a (smoke run)" if subset_size < 5000 else ("PASS" if pass_gate else "FAIL")

    return {
        "catastrophic": catastrophic,
        "pass": pass_gate,
        "pass_status": pass_status,
        "parity": parity,
    }


def _format_section(
    subset_size: int,
    results: dict,
    targets: dict,
    commit: str,
    classnames_source: str,
    text_preprocess: str,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parity_str = "PASS" if targets["parity"] else "FAIL"
    catastrophic_marker = " CATASTROPHIC" if targets["catastrophic"] else ""
    subset_kind = "stratified" if subset_size >= N_CLASSES else "random"
    time_line = f"{results['total_time_s']:.0f}s ({results['time_per_image_ms']:.1f} мс/img)"
    parity_line = (
        f"- Parity (top-1 ∈ "
        f"[{PARITY_RANGE[0] * 100:.1f}, {PARITY_RANGE[1] * 100:.1f}]): "
        f"**{parity_str}**"
    )

    return "\n".join(
        [
            f"## ImageNet zs · Прогон {timestamp}",
            "",
            f"**Commit:** `{commit}`",
            f"**Subset:** {subset_size} картинок ({subset_kind})",
            f"**Classnames:** `{classnames_source}`",
            f"**Text preprocess:** `{text_preprocess}`",
            f"**Время:** {time_line}",
            "",
            f"- **top-1: {results['top1'] * 100:.2f}%**{catastrophic_marker}",
            f"- top-5: {results['top5'] * 100:.2f}%",
            f"- N samples: {results['n_samples']}",
            "",
            "### Таргеты",
            "",
            f"- Pass (top-1 ≥ {PASS_THRESHOLD * 100:.0f}%): **{targets['pass_status']}**",
            parity_line,
            f"- Catastrophic guard (top-1 ≥ {CATASTROPHIC_THRESHOLD * 100:.0f}%): "
            f"**{'PASS' if not targets['catastrophic'] else 'FAIL'}**",
        ]
    )


def _git_commit_hash() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


def append_to_results(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8").rstrip() if path.exists() else ""
    path.write_text(f"{existing}\n\n---\n\n{section}\n", encoding="utf-8")


# --- Main ---


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity benchmark: ImageNet zero-shot top-1 для нашего SigLIP 2 ONNX."
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=5000,
        help="Размер subset (default 5000 = 5/класс stratified). <1000 = random subset для smoke.",
    )
    parser.add_argument(
        "--classnames",
        choices=CLASSNAMES_CHOICES,
        default=CLASSNAMES_OPENAI,
        help="Источник имён классов (default: openai - curated CLIP-style names).",
    )
    parser.add_argument(
        "--text-preprocess",
        choices=TEXT_PP_CHOICES,
        default=TEXT_PP_AUTO,
        help=(
            "Text preprocessing для промптов и classnames: "
            "auto = AutoTokenizer (GemmaTokenizer) as-is; "
            "siglip2-normalized = Siglip2Tokenizer + manual remove punctuation "
            "(per HF community recommendation для воспроизведения paper-чисел)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Не аппендить секцию в docs/sanity_results.md (для отладочных прогонов).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ImageNet Zero-Shot Sanity - Siglip2Encoder (ONNX)")
    print("=" * 70)
    print(f"Subset size:     {args.subset_size}")
    print(f"Classnames:      {args.classnames}")
    print(f"Text preprocess: {args.text_preprocess}")
    print(f"Seed:            {args.seed}")
    print()

    class_names = load_class_names(args.classnames)
    print(f"Класс-имена загружены ({args.classnames}): {len(class_names)}")
    print(f"  [0]={class_names[0]}  [1]={class_names[1]}  [999]={class_names[999]}")
    print()

    images, labels = load_imagenet_val_subset(args.subset_size, seed=args.seed)
    print(f"Subset: {len(images)} картинок, {len(set(labels))} уникальных классов\n")

    class_embeddings = build_class_embeddings(class_names, args.classnames, args.text_preprocess)
    print(f"Class embeddings: {class_embeddings.shape}, dtype={class_embeddings.dtype}\n")

    encoder = Siglip2Encoder()
    results = evaluate_zero_shot(encoder, images, labels, class_embeddings)

    print()
    print("=" * 70)
    print("Результаты:")
    print(f"  top-1: {results['top1'] * 100:.2f}%")
    print(f"  top-5: {results['top5'] * 100:.2f}%")
    print(f"  N samples: {results['n_samples']}")
    print(f"  Время: {results['total_time_s']:.0f}s ({results['time_per_image_ms']:.1f} мс/img)")

    targets = evaluate_targets(results["top1"], args.subset_size)
    print()
    print(
        f"  Catastrophic guard (≥ {CATASTROPHIC_THRESHOLD * 100:.0f}%): "
        f"{'PASS' if not targets['catastrophic'] else 'FAIL'}"
    )
    print(f"  Pass gate (≥ {PASS_THRESHOLD * 100:.0f}%): {targets['pass_status']}")
    print(
        f"  Parity ([{PARITY_RANGE[0] * 100:.1f}, {PARITY_RANGE[1] * 100:.1f}]): "
        f"{'PASS' if targets['parity'] else 'FAIL'}"
    )
    print("=" * 70)

    if not args.no_write:
        commit = _git_commit_hash()
        section = _format_section(
            args.subset_size, results, targets, commit, args.classnames, args.text_preprocess
        )
        append_to_results(RESULTS_PATH, section)
        print(f"\nРезультаты дописаны в {RESULTS_PATH}")

    if targets["catastrophic"]:
        print(
            "\n  CATASTROPHIC FAILURE: top-1 ниже 50%. "
            "НЕ запускайте полный прогон. "
            "Вероятные причины: неправильный mapping classnames, не та нормализация "
            "эмбеддингов, отсутствует L2-norm перед cosine, перепутаны промпт-шаблоны, "
            "не тот text-encoder API.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
