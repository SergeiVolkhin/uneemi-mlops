"""Очистка сырых профилей перед валидацией и записью в фичестор.

Очистка консервативна: убираем дубли по profile_id, строки с пропусками и
битыми board-эмбеддингами, подрезаем quiz в допустимый диапазон [0,1]. Цель -
не «подкрутить» данные, а гарантировать инварианты схемы для downstream-шагов.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from uneemi_ml.config import EMBED_DIM
from uneemi_ml.features import QUIZ_FEATURES

_LOGGER = logging.getLogger(__name__)


def _valid_board(value) -> bool:
    """board_emb валиден, если это последовательность из EMBED_DIM конечных чисел."""
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (ValueError, TypeError):
        return False
    return arr.shape == (EMBED_DIM,) and bool(np.isfinite(arr).all())


def clean_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Очистить датафрейм профилей. Возвращает новый df, исходный не мутируется."""
    n0 = len(df)
    df = df.copy()

    # Дубликаты профиля - оставляем последний (самый свежий event_timestamp).
    df = df.sort_values("event_timestamp").drop_duplicates("profile_id", keep="last")

    # Битые board-эмбеддинги выкидываем целиком (чинить нечем).
    mask_board = df["board_emb"].apply(_valid_board)
    if (~mask_board).any():
        _LOGGER.warning("Удаляю %d профилей с битым board_emb", int((~mask_board).sum()))
    df = df[mask_board]

    # quiz: пропуски запрещены, значения подрезаем в [0,1] (анкета нормирована).
    quiz_cols = list(QUIZ_FEATURES)
    df = df.dropna(subset=quiz_cols)
    df[quiz_cols] = df[quiz_cols].clip(lower=0.0, upper=1.0)

    df = df.reset_index(drop=True)
    _LOGGER.info("Очистка профилей: %d -> %d строк", n0, len(df))
    return df
