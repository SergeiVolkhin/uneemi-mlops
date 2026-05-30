"""Подготовка обучающей выборки: 775d признаки пар + стратифицированный сплит.

Признаки пары собираются из профильных векторов (board+quiz) по контракту
features.build_pair_matrix. Сплит train/val/test стратифицирован по метке и
детерминирован (seed) - чтобы сравнение champion/challenger было честным и
воспроизводимым на одном и том же holdout.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from uneemi_ml.features import QUIZ_FEATURES, build_pair_matrix, build_profile_vector

_LOGGER = logging.getLogger(__name__)


def _profile_lookup(profiles: pd.DataFrame) -> dict[int, np.ndarray]:
    """Отображение profile_id -> profile_vec (775,)."""
    lookup: dict[int, np.ndarray] = {}
    quiz_cols = list(QUIZ_FEATURES)
    for row in profiles.itertuples(index=False):
        board = np.asarray(row.board_emb, dtype=np.float32)
        quiz = np.asarray([getattr(row, c) for c in quiz_cols], dtype=np.float32)
        lookup[int(row.profile_id)] = build_profile_vector(board, quiz)
    return lookup


def build_pair_dataset(
    pairs: pd.DataFrame,
    profiles: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Построить (X (N,775), y (N,)) из пар и профильных признаков."""
    lookup = _profile_lookup(profiles)
    valid = pairs[
        pairs["profile_a_id"].isin(lookup) & pairs["profile_b_id"].isin(lookup)
    ]
    if len(valid) < len(pairs):
        _LOGGER.warning("Отброшено %d пар без признаков профиля", len(pairs) - len(valid))

    profiles_a = np.vstack([lookup[int(i)] for i in valid["profile_a_id"]])
    profiles_b = np.vstack([lookup[int(i)] for i in valid["profile_b_id"]])
    x = build_pair_matrix(profiles_a, profiles_b)
    y = valid["matched"].to_numpy(dtype=np.int64)
    _LOGGER.info("Собрана выборка пар: X=%s, доля матчей=%.3f", x.shape, float(y.mean()))
    return x, y


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> dict[str, np.ndarray]:
    """Стратифицированный train/val/test сплит. val_size берётся от остатка после test."""
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=seed
    )
    splits = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }
    _LOGGER.info(
        "Сплит: train=%d, val=%d, test=%d", len(y_train), len(y_val), len(y_test)
    )
    return splits
