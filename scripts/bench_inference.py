"""Performance benchmark: Siglip2Encoder на CPU.

Замеряет латентность (p50/p95/p99) и пиковый RSS для:
- preprocess (PIL resize + normalize)
- encode(single image) - прод-API
- encode_batch для batch ∈ {1, 8, 16, 32}
- full board (40 изображений, чанки 32+8): preprocess + encode

Результаты пишутся в stdout и аппендятся новой датированной секцией в
docs/benchmark_results.md (история прогонов сохраняется).

Использует прод-API Siglip2Encoder, не сырой ONNX Runtime - реалистичные числа
с учётом препроцессинга.

Запуск:
    uv run python scripts/bench_inference.py
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

# Windows console падает на Unicode (м/с, символы перцентилей).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# Скрипт лежит вне пакета.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from uneemi_ml import Siglip2Encoder, preprocess_batch, preprocess_image  # noqa: E402
from uneemi_ml.config import ORT_INTRA_OP_THREADS  # noqa: E402

WARMUP_ITERS: int = 20
BENCH_ITERS_SINGLE: int = 200
IMAGE_GEN_SIZE: int = 800
BATCH_SIZES: tuple[int, ...] = (1, 8, 16, 32)
BOARD_SIZE: int = 40
BOARD_CHUNKS: tuple[int, ...] = (32, 8)
FULL_BOARD_REPEATS: int = 3

TARGET_P99_SINGLE_MS: float = 30.0
TARGET_P99_BS32_MS: float = 500.0
TARGET_RAM_BS32_MB: float = 2048.0

RESULTS_PATH: Path = _PROJECT_ROOT / "docs" / "benchmark_results.md"

_PROCESS = psutil.Process(os.getpid())


def _batch_iters(batch_size: int) -> int:
    return max(BENCH_ITERS_SINGLE // batch_size, 30)


def _rss_mb() -> float:
    return _PROCESS.memory_info().rss / (1024 * 1024)


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    return {
        "p50": float(np.percentile(samples_ms, 50)),
        "p95": float(np.percentile(samples_ms, 95)),
        "p99": float(np.percentile(samples_ms, 99)),
    }


def get_system_info() -> dict[str, str]:
    """Сводка по системе для секции отчёта."""
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "") or "unknown"
    cores_logical = psutil.cpu_count(logical=True) or 0
    cores_physical = psutil.cpu_count(logical=False) or 0
    ram_gb = psutil.virtual_memory().total / (1024**3)
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu_name.strip(),
        "cores_logical": str(cores_logical),
        "cores_physical": str(cores_physical),
        "ram_gb": f"{ram_gb:.1f}",
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "ort_threads": str(ORT_INTRA_OP_THREADS),
    }


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown"


def _gen_random_images(n: int, size: int, seed: int) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    images: list[Image.Image] = []
    for _ in range(n):
        pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        images.append(Image.fromarray(pixels, mode="RGB"))
    return images


def measure_preprocess_latency(image: Image.Image, iters: int) -> dict[str, float]:
    """p50/p95/p99 для preprocess_image на одном изображении (мс)."""
    for _ in range(WARMUP_ITERS):
        preprocess_image(image)

    gc.collect()
    gc.disable()
    samples_ms: list[float] = []
    try:
        for _ in range(iters):
            t0 = time.perf_counter()
            preprocess_image(image)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
    finally:
        gc.enable()

    return _percentiles(samples_ms)


def measure_encode_latency(
    encoder: Siglip2Encoder,
    images: list[Image.Image],
    batch_size: int,
    warmup: int,
    iters: int,
    use_single_api: bool = False,
) -> dict[str, float]:
    """p50/p95/p99 + per_img + ram_peak_mb для encode/encode_batch (мс)."""
    if use_single_api:
        assert batch_size == 1, "use_single_api поддерживается только при batch_size=1"
        single = images[0]

        def call() -> None:
            encoder.encode(single)
    else:
        chunk = images[:batch_size]
        if len(chunk) < batch_size:
            raise ValueError(f"Недостаточно картинок: нужно {batch_size}, есть {len(chunk)}")

        def call() -> None:
            encoder.encode_batch(chunk)

    for _ in range(warmup):
        call()

    gc.collect()
    gc.disable()
    samples_ms: list[float] = []
    ram_peak_mb = _rss_mb()
    try:
        for _ in range(iters):
            t0 = time.perf_counter()
            call()
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
            ram_peak_mb = max(ram_peak_mb, _rss_mb())
    finally:
        gc.enable()

    result = _percentiles(samples_ms)
    result["per_img"] = result["p50"] / batch_size
    result["ram_peak_mb"] = ram_peak_mb
    return result


def measure_full_board(
    encoder: Siglip2Encoder,
    images: list[Image.Image],
) -> dict[str, float]:
    """Полный pipeline для BOARD_SIZE картинок (preprocess+encode чанками)."""
    if len(images) < BOARD_SIZE:
        raise ValueError(f"Нужно {BOARD_SIZE} картинок, есть {len(images)}")
    board = images[:BOARD_SIZE]

    # Warmup общим вызовом.
    for _ in range(3):
        _ = preprocess_batch(board)
    chunk_warmup = encoder.encode_batch(board[: BOARD_CHUNKS[0]])
    assert chunk_warmup.shape[0] == BOARD_CHUNKS[0]

    pre_samples_ms: list[float] = []
    enc_samples_ms: list[float] = []
    total_samples_ms: list[float] = []

    gc.collect()
    gc.disable()
    try:
        for _ in range(FULL_BOARD_REPEATS):
            t_total_start = time.perf_counter()

            t0 = time.perf_counter()
            preprocessed = preprocess_batch(board)
            t_pre = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            offset = 0
            for chunk_size in BOARD_CHUNKS:
                chunk_pixels = preprocessed[offset : offset + chunk_size]
                encoder._run(chunk_pixels)  # noqa: SLF001
                offset += chunk_size
            t_enc = (time.perf_counter() - t0) * 1000.0

            t_total = (time.perf_counter() - t_total_start) * 1000.0

            pre_samples_ms.append(t_pre)
            enc_samples_ms.append(t_enc)
            total_samples_ms.append(t_total)
    finally:
        gc.enable()

    return {
        "t_preprocess_ms": statistics.median(pre_samples_ms),
        "t_encode_ms": statistics.median(enc_samples_ms),
        "t_total_ms": statistics.median(total_samples_ms),
    }


def _format_row(name: str, m: dict[str, float], with_per_img: bool, ram_col: str | float) -> str:
    p50 = f"{m['p50']:.2f}"
    p95 = f"{m['p95']:.2f}"
    p99 = f"{m['p99']:.2f}"
    per_img = f"{m['per_img']:.2f}" if with_per_img else "-"
    ram = f"{ram_col:.1f}" if isinstance(ram_col, (int, float)) else ram_col
    return f"| {name} | {p50} | {p95} | {p99} | {per_img} | {ram} |"


def format_markdown(
    system_info: dict[str, str],
    commit: str,
    ram_after_init_mb: float,
    preprocess: dict[str, float],
    encode_single: dict[str, float],
    encode_batches: dict[int, dict[str, float]],
    board: dict[str, float],
    effective_threads: int,
    threads_source: str,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    host = (
        f"{system_info['os']}, {system_info['cpu']} "
        f"({system_info['cores_physical']}C/{system_info['cores_logical']}T), "
        f"{system_info['ram_gb']} GB RAM"
    )

    rows: list[str] = [
        "| Операция | p50 (мс) | p95 (мс) | p99 (мс) | per_img (мс) | ram_peak_mb |",
        "|---|---|---|---|---|---|",
        _format_row(
            "preprocess (PIL+normalize)",
            {**preprocess, "per_img": 0.0},
            with_per_img=False,
            ram_col="-",
        ),
        _format_row(
            "encode(single)",
            {**encode_single, "per_img": 0.0},
            with_per_img=False,
            ram_col=encode_single["ram_peak_mb"],
        ),
    ]
    for bs in BATCH_SIZES:
        m = encode_batches[bs]
        rows.append(
            _format_row(
                f"encode_batch({bs})",
                m,
                with_per_img=True,
                ram_col=m["ram_peak_mb"],
            )
        )

    p99_single = encode_single["p99"]
    p99_bs32 = encode_batches[32]["p99"]
    ram_bs32 = encode_batches[32]["ram_peak_mb"]

    def _check(ok: bool) -> str:
        return "[x]" if ok else "[❌]"

    targets_lines = [
        f"- {_check(p99_single <= TARGET_P99_SINGLE_MS)} p99 encode(single) ≤ "
        f"{TARGET_P99_SINGLE_MS:.0f} мс (фактическое: {p99_single:.2f} мс)",
        f"- {_check(p99_bs32 <= TARGET_P99_BS32_MS)} p99 encode_batch(32) ≤ "
        f"{TARGET_P99_BS32_MS:.0f} мс (фактическое: {p99_bs32:.2f} мс)",
        f"- {_check(ram_bs32 <= TARGET_RAM_BS32_MB)} ram_peak_bs32 ≤ "
        f"{TARGET_RAM_BS32_MB:.0f} MB (фактическое: {ram_bs32:.1f} MB)",
    ]

    section = [
        f"## Прогон {timestamp}",
        "",
        f"**Commit:** `{commit}`",
        f"**Host:** {host}",
        f"**Python:** {system_info['python']} | "
        f"**onnxruntime:** {system_info['onnxruntime']} | "
        f"**ORT threads:** {effective_threads} ({threads_source})",
        f"**RAM после init Siglip2Encoder:** {ram_after_init_mb:.1f} MB",
        "",
        "### Латентность",
        "",
        *rows,
        "",
        (
            f"### Full board pipeline ({BOARD_SIZE} изображений, "
            f"чанки {'+'.join(map(str, BOARD_CHUNKS))})"
        ),
        "",
        f"- t_preprocess: {board['t_preprocess_ms']:.2f} мс",
        f"- t_encode: {board['t_encode_ms']:.2f} мс",
        f"- **t_total: {board['t_total_ms']:.2f} мс**",
        "",
        "### Таргеты",
        "",
        *targets_lines,
    ]
    return "\n".join(section)


def append_to_results_file(path: Path, section_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Performance Benchmark - Siglip2Encoder\n\n"
        "История прогонов performance-бенчмарка. Каждая секция - отдельный запуск "
        "`scripts/bench_inference.py`. Новые секции аппендятся в конец.\n\n"
    )
    if path.exists():
        existing = path.read_text(encoding="utf-8").rstrip()
        new_content = f"{existing}\n\n---\n\n{section_text}\n"
    else:
        new_content = f"{header}---\n\n{section_text}\n"
    path.write_text(new_content, encoding="utf-8")


def _print_kv(key: str, value: str) -> None:
    print(f"  {key:<16} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Performance benchmark Siglip2Encoder")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override ORT intra_op_num_threads. Default: config.ORT_INTRA_OP_THREADS",
    )
    args = parser.parse_args()

    system_info = get_system_info()
    commit = _git_commit_hash()

    print("=" * 70)
    print("Performance Benchmark - Siglip2Encoder")
    print("=" * 70)
    print("Система:")
    for k, v in system_info.items():
        _print_kv(k, v)
    _print_kv("commit", commit)
    print()

    print("Инициализация Siglip2Encoder...")
    gc.collect()
    rss_before = _rss_mb()
    encoder = Siglip2Encoder(intra_op_threads=args.threads)
    gc.collect()
    rss_after = _rss_mb()
    ram_after_init = rss_after - rss_before
    effective_threads = encoder.intra_op_threads
    threads_source = "cli" if args.threads is not None else "config"
    print(f"  RAM после init: {rss_after:.1f} MB (delta {ram_after_init:+.1f} MB)")
    print(f"  ORT intra_op_num_threads = {effective_threads} ({threads_source})")
    print()

    print(
        f"Генерация {BOARD_SIZE} случайных PIL.Image {IMAGE_GEN_SIZE}x{IMAGE_GEN_SIZE} (seed=0)..."
    )
    images = _gen_random_images(n=BOARD_SIZE, size=IMAGE_GEN_SIZE, seed=0)
    print()

    print(f"[1] preprocess (single, {BENCH_ITERS_SINGLE} iters)...")
    preprocess_m = measure_preprocess_latency(images[0], iters=BENCH_ITERS_SINGLE)
    print(
        f"    p50={preprocess_m['p50']:.2f}мс  "
        f"p95={preprocess_m['p95']:.2f}мс  "
        f"p99={preprocess_m['p99']:.2f}мс"
    )

    print(f"[2] encode(single), прод-API, {BENCH_ITERS_SINGLE} iters...")
    encode_single = measure_encode_latency(
        encoder,
        images,
        batch_size=1,
        warmup=WARMUP_ITERS,
        iters=BENCH_ITERS_SINGLE,
        use_single_api=True,
    )
    print(
        f"    p50={encode_single['p50']:.2f}мс  "
        f"p95={encode_single['p95']:.2f}мс  "
        f"p99={encode_single['p99']:.2f}мс  "
        f"ram_peak={encode_single['ram_peak_mb']:.1f}MB"
    )

    encode_batches: dict[int, dict[str, float]] = {}
    for bs in BATCH_SIZES:
        iters = _batch_iters(bs)
        print(f"[3] encode_batch({bs}), {iters} iters...")
        encode_batches[bs] = measure_encode_latency(
            encoder,
            images,
            batch_size=bs,
            warmup=WARMUP_ITERS,
            iters=iters,
            use_single_api=False,
        )
        m = encode_batches[bs]
        print(
            f"    p50={m['p50']:.2f}мс  p99={m['p99']:.2f}мс  "
            f"per_img={m['per_img']:.2f}мс  ram_peak={m['ram_peak_mb']:.1f}MB"
        )
        gc.collect()

    print(
        f"[4] full board ({BOARD_SIZE} imgs, чанки {'+'.join(map(str, BOARD_CHUNKS))}), "
        f"{FULL_BOARD_REPEATS} repeats..."
    )
    board = measure_full_board(encoder, images)
    print(
        f"    t_preprocess={board['t_preprocess_ms']:.2f}мс  "
        f"t_encode={board['t_encode_ms']:.2f}мс  "
        f"t_total={board['t_total_ms']:.2f}мс"
    )

    print()
    print("=" * 70)
    print("Таргеты:")
    p99_single = encode_single["p99"]
    p99_bs32 = encode_batches[32]["p99"]
    ram_bs32 = encode_batches[32]["ram_peak_mb"]
    print(
        f"  [{'OK' if p99_single <= TARGET_P99_SINGLE_MS else 'FAIL'}] "
        f"p99 encode(single) = {p99_single:.2f}мс ≤ {TARGET_P99_SINGLE_MS:.0f}мс"
    )
    print(
        f"  [{'OK' if p99_bs32 <= TARGET_P99_BS32_MS else 'FAIL'}] "
        f"p99 encode_batch(32) = {p99_bs32:.2f}мс ≤ {TARGET_P99_BS32_MS:.0f}мс"
    )
    print(
        f"  [{'OK' if ram_bs32 <= TARGET_RAM_BS32_MB else 'FAIL'}] "
        f"ram_peak_bs32 = {ram_bs32:.1f}MB ≤ {TARGET_RAM_BS32_MB:.0f}MB"
    )
    print("=" * 70)

    section = format_markdown(
        system_info=system_info,
        commit=commit,
        ram_after_init_mb=rss_after,
        preprocess=preprocess_m,
        encode_single=encode_single,
        encode_batches=encode_batches,
        board=board,
        effective_threads=effective_threads,
        threads_source=threads_source,
    )
    append_to_results_file(RESULTS_PATH, section)
    print(f"\nРезультаты дописаны в {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
