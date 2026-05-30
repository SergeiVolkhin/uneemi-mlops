"""Юнит-тесты тренировочного ядра (clean/validate/prepare/train/evaluate).

Без MLflow и без SigLIP: синтетические профили с управляемым сигналом, чтобы
проверить, что пайплайн собирает признаки пар, обучает модели и считает метрики.
Пропускаются, если не установлены pandas/sklearn (ставятся в образ Airflow).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import pandas as pd  # noqa: E402

from training.clean import clean_profiles  # noqa: E402
from training.evaluate import evaluate_model  # noqa: E402
from training.prepare import build_pair_dataset, split_dataset  # noqa: E402
from training.train import train_challenger, train_champion  # noqa: E402
from training.validate import (  # noqa: E402
    DataValidationError,
    psi,
    validate_pairs,
    validate_profile_schema,
    validate_value_skew,
)
from uneemi_ml.config import EMBED_DIM  # noqa: E402
from uneemi_ml.features import QUIZ_FEATURES  # noqa: E402

_N_CLUSTERS = 4


def _synth_profiles(n: int, noise_seed: int = 0) -> pd.DataFrame:
    """Профили с кластерной структурой: board-центроид зависит от кластера + шум.

    Распределение (центроиды/прототипы) фиксировано (dist_rng), варьируется только
    шум выборки (noise_seed) - так разные вызовы дают один и тот же закон с разными
    реализациями, что и нужно для проверки value-skew.
    """
    dist_rng = np.random.default_rng(100)
    centroids = dist_rng.normal(size=(_N_CLUSTERS, EMBED_DIM)).astype(np.float32)
    quiz_proto = dist_rng.uniform(0.2, 0.8, size=(_N_CLUSTERS, len(QUIZ_FEATURES)))
    rng = np.random.default_rng(noise_seed)
    rows = []
    for pid in range(n):
        c = pid % _N_CLUSTERS
        board = centroids[c] + rng.normal(0, 0.3, EMBED_DIM).astype(np.float32)
        board = board / np.linalg.norm(board)
        quiz = np.clip(quiz_proto[c] + rng.normal(0, 0.05, len(QUIZ_FEATURES)), 0, 1)
        row = {
            "profile_id": pid,
            "event_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
            "cluster": f"c{c}",
            "board_emb": board.astype(np.float32).tolist(),
        }
        row.update({name: float(quiz[j]) for j, name in enumerate(QUIZ_FEATURES)})
        rows.append(row)
    return pd.DataFrame(rows)


def _synth_pairs(profiles: pd.DataFrame, n: int, seed: int = 1) -> pd.DataFrame:
    """Пары: внутри кластера чаще матч, между - реже (управляемый сигнал)."""
    rng = np.random.default_rng(seed)
    ids = profiles["profile_id"].to_numpy()
    clusters = profiles["cluster"].to_numpy()
    a = rng.choice(len(ids), size=n)
    b = rng.choice(len(ids), size=n)
    # Исключаем self-пары (A == B): их отвергает гейт validate_pairs.
    collide = a == b
    b[collide] = (b[collide] + 1) % len(ids)
    same = clusters[a] == clusters[b]
    p = np.where(same, 0.8, 0.2)
    matched = (rng.uniform(size=n) < p).astype(np.int64)
    return pd.DataFrame(
        {"profile_a_id": ids[a], "profile_b_id": ids[b], "matched": matched}
    )


def test_clean_and_schema_ok() -> None:
    df = _synth_profiles(120)
    cleaned = clean_profiles(df)
    report = validate_profile_schema(cleaned, min_rows=50)
    assert report["status"] == "ok"
    assert report["n_rows"] == 120


def test_schema_rejects_out_of_range_quiz() -> None:
    """Демонстрация защиты: значение quiz вне [0,1] валит схему."""
    df = _synth_profiles(80)
    df.loc[0, "extraversion"] = 5.0  # впрыск битого значения
    with pytest.raises(DataValidationError):
        validate_profile_schema(df)


def test_schema_rejects_bad_board_width() -> None:
    df = _synth_profiles(80)
    df.at[0, "board_emb"] = [0.0] * (EMBED_DIM - 1)  # неверная ширина
    with pytest.raises(DataValidationError):
        validate_profile_schema(df)


def test_end_to_end_training_has_signal() -> None:
    profiles = _synth_profiles(200)
    pairs = _synth_pairs(profiles, 3000)
    validate_pairs(pairs)
    x, y = build_pair_dataset(pairs, profiles)
    assert x.shape[1] == EMBED_DIM + len(QUIZ_FEATURES) == 775
    splits = split_dataset(x, y, seed=42)
    champion = train_champion(splits["x_train"], splits["y_train"])
    challenger = train_challenger(splits["x_train"], splits["y_train"])
    m_champ = evaluate_model(champion, splits["x_test"], splits["y_test"])
    m_chal = evaluate_model(challenger, splits["x_test"], splits["y_test"])
    # Сигнал есть - обе модели заметно лучше случайного угадывания.
    assert m_champ["roc_auc"] > 0.65
    assert m_chal["roc_auc"] > 0.65


def test_psi_zero_on_same_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) < 0.01


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    shifted = rng.normal(2, 1, 5000)
    assert psi(base, shifted) > 0.25


def test_value_skew_reference_then_drift(tmp_path) -> None:
    ref = tmp_path / "reference_stats.json"
    df0 = _synth_profiles(150, noise_seed=0)
    r0 = validate_value_skew(df0, ref, psi_threshold=0.25)
    assert r0["status"] == "reference_created"
    # Тот же распределённый набор проходит проверку.
    r1 = validate_value_skew(_synth_profiles(150, noise_seed=1), ref, psi_threshold=0.25)
    assert r1["status"] == "ok"
