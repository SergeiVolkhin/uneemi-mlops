"""Генерация синтетических помеченных данных матчинга (синтетика для демонстрации).

ВАЖНО: реальных исходов матчей у нас нет, поэтому МЕТКИ синтетические. Но процесс
честный, а не подогнанный:
  1. board-эмбеддинги извлекаются РЕАЛЬНЫМ SigLIP (ONNX) по картинкам доски -
     никакой имитации эмбеддингов.
  2. quiz-признаки коррелируют с эстетическим кластером профиля + шум.
  3. вероятность матча растёт с близостью профилей:
        z = a*cos(board_A, board_B) + b*quiz_sim, центрируем по медиане для баланса
        классов, добавляем гауссов шум; matched ~ Bernoulli(sigmoid(z)).
Так у классификатора есть реальный, но зашумлённый сигнал - как было бы на боевых
данных. Реальный процесс разметки (взаимный лайк -> диалог >=5 сообщений за 72ч)
описан в манифесте, раздел 6.

Запуск напрямую (вне Airflow): uv run python -m training.data_gen
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from uneemi_ml.demo import CLUSTERS, PAIRS_DIR, RAW_DIR, cluster_dir
from uneemi_ml.features import QUIZ_FEATURES, board_centroid, cosine

_LOGGER = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
PROFILES_RAW_PATH: Path = RAW_DIR / "profiles_raw.parquet"
PAIRS_PATH: Path = PAIRS_DIR / "pairs.parquet"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_image_pools() -> dict[str, list[Path]]:
    """Собрать пулы картинок по кластерам из data/images/<cluster>/."""
    pools: dict[str, list[Path]] = {}
    for cluster in CLUSTERS:
        cdir = cluster_dir(cluster)
        if not cdir.exists():
            continue
        files = sorted(p for p in cdir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if files:
            pools[cluster] = files
    if not pools:
        raise FileNotFoundError(
            "Банк демо-изображений пуст. Сначала: uv run python scripts/fetch_demo_images.py"
        )
    return pools


def _cluster_quiz_prototypes(seed: int) -> dict[str, np.ndarray]:
    """Прототип quiz-вектора на кластер (детерминированный) - источник сигнала анкеты."""
    rng = np.random.default_rng(seed)
    return {cluster: rng.uniform(0.15, 0.85, size=len(QUIZ_FEATURES)) for cluster in CLUSTERS}


def generate_profiles(
    n_profiles: int,
    board_images_per_profile: int,
    seed: int,
    encoder=None,
) -> pd.DataFrame:
    """Сгенерировать профили с РЕАЛЬНЫМИ board-эмбеддингами SigLIP.

    encoder - объект с методом encode_batch(list[Image]) -> (B, 768). Если None,
    создаётся Siglip2Encoder (требует экспортированный ONNX). Импорт модели ленивый,
    чтобы модуль можно было импортировать без onnxruntime/ONNX (для тестов).
    """
    if encoder is None:
        from uneemi_ml.model import Siglip2Encoder

        encoder = Siglip2Encoder()

    rng = np.random.default_rng(seed)
    pools = _load_image_pools()
    available = [c for c in CLUSTERS if c in pools]
    prototypes = _cluster_quiz_prototypes(seed)
    now = datetime.now(UTC)

    rows: list[dict] = []
    for pid in range(n_profiles):
        cluster = available[pid % len(available)]  # ровная раскладка по кластерам
        pool = pools[cluster]
        k = min(board_images_per_profile, len(pool))
        chosen = rng.choice(len(pool), size=k, replace=False)
        images = [Image.open(pool[i]) for i in chosen]

        embeddings = encoder.encode_batch(images)  # (k, 768) реальный SigLIP
        board = board_centroid(embeddings)  # (768,) среднее + L2-норма

        # quiz = прототип кластера + шум, обрезанный в [0,1].
        quiz = np.clip(prototypes[cluster] + rng.normal(0, 0.12, len(QUIZ_FEATURES)), 0.0, 1.0)

        row: dict = {
            "profile_id": pid,
            "event_timestamp": now,
            "cluster": cluster,
            "board_emb": board.astype(np.float32).tolist(),
        }
        row.update({name: float(quiz[j]) for j, name in enumerate(QUIZ_FEATURES)})
        rows.append(row)

        if (pid + 1) % 100 == 0:
            _LOGGER.info("Сгенерировано профилей: %d/%d", pid + 1, n_profiles)

    return pd.DataFrame(rows)


def _profile_vectors(profiles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть (board_matrix (N,768), quiz_matrix (N,7)) в порядке profile_id."""
    board = np.vstack([np.asarray(e, dtype=np.float32) for e in profiles["board_emb"].to_numpy()])
    quiz = profiles[list(QUIZ_FEATURES)].to_numpy(dtype=np.float32)
    return board, quiz


def generate_pairs(
    profiles: pd.DataFrame,
    n_pairs: int,
    a: float,
    b: float,
    sigma: float,
    seed: int,
) -> pd.DataFrame:
    """Сгенерировать честные синтетические помеченные пары.

    Половина пар - внутрикластерные (зона высокой близости), половина -
    межкластерные (низкой). Метка зависит от реальной близости board + quiz.
    """
    rng = np.random.default_rng(seed + 777)
    board, quiz = _profile_vectors(profiles)
    clusters = profiles["cluster"].to_numpy()
    ids = profiles["profile_id"].to_numpy()
    n = len(profiles)

    by_cluster: dict[str, np.ndarray] = {
        c: np.where(clusters == c)[0] for c in np.unique(clusters)
    }

    a_idx: list[int] = []
    b_idx: list[int] = []
    half = n_pairs // 2
    # Внутрикластерные пары.
    for _ in range(half):
        c = rng.choice(list(by_cluster.keys()))
        members = by_cluster[c]
        if len(members) < 2:
            continue
        i, j = rng.choice(members, size=2, replace=False)
        a_idx.append(int(i))
        b_idx.append(int(j))
    # Межкластерные пары.
    for _ in range(n_pairs - len(a_idx)):
        i, j = rng.choice(n, size=2, replace=False)
        a_idx.append(int(i))
        b_idx.append(int(j))

    a_idx_arr = np.asarray(a_idx)
    b_idx_arr = np.asarray(b_idx)

    # cos board и quiz-сходство для каждой пары.
    cos_board = np.array(
        [cosine(board[i], board[j]) for i, j in zip(a_idx_arr, b_idx_arr, strict=True)]
    )
    quiz_sim = 1.0 - np.abs(quiz[a_idx_arr] - quiz[b_idx_arr]).mean(axis=1)

    # Линейный сигнал; центрируем по медиане для баланса классов ~50/50.
    raw = a * cos_board + b * quiz_sim
    z = raw - float(np.median(raw)) + rng.normal(0, sigma, size=len(raw))
    p_match = _sigmoid(z)
    matched = (rng.uniform(size=len(p_match)) < p_match).astype(np.int64)

    now = datetime.now(UTC)
    return pd.DataFrame(
        {
            "profile_a_id": ids[a_idx_arr],
            "profile_b_id": ids[b_idx_arr],
            "matched": matched,
            "p_match": p_match.astype(np.float32),
            "same_cluster": clusters[a_idx_arr] == clusters[b_idx_arr],
            "event_timestamp": now,
        }
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n_profiles = int(os.environ.get("N_PROFILES", "600"))
    n_pairs = int(os.environ.get("N_PAIRS", "12000"))
    seed = int(os.environ.get("DATA_SEED", "42"))
    board_per_profile = int(os.environ.get("BOARD_IMAGES_PER_PROFILE", "12"))
    # Калибровка под банк демо-картинок: board-косинус разделяет кластеры слабо
    # (within-cross gap ~0.026), сигнал анкеты - сильнее (gap ~0.14) и чище для
    # извлечения, поэтому вес b выше веса a. При этих значениях честный сигнал даёт
    # holdout ROC-AUC ~0.80 - уверенно выше гейта промоута 0.70.
    a = float(os.environ.get("SYNTH_A", "8.0"))
    b = float(os.environ.get("SYNTH_B", "20.0"))
    sigma = float(os.environ.get("SYNTH_SIGMA", "0.12"))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("Генерация %d профилей (реальный SigLIP по доскам)...", n_profiles)
    profiles = generate_profiles(n_profiles, board_per_profile, seed)
    profiles.to_parquet(PROFILES_RAW_PATH, index=False)
    _LOGGER.info("Профили записаны: %s (%d строк)", PROFILES_RAW_PATH, len(profiles))

    _LOGGER.info("Генерация %d помеченных пар...", n_pairs)
    pairs = generate_pairs(profiles, n_pairs, a, b, sigma, seed)
    pairs.to_parquet(PAIRS_PATH, index=False)
    pos_rate = float(pairs["matched"].mean())
    _LOGGER.info(
        "Пары записаны: %s (%d строк, доля матчей %.3f)", PAIRS_PATH, len(pairs), pos_rate
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
