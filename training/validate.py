"""Валидация данных: схема + value-skew (PSI). Это гейты пайплайнов.

Два уровня:
1. Схема (validate_profile_schema) - жёсткие инварианты: нужные колонки, типы,
   ширина board_emb=768, quiz в [0,1], отсутствие пропусков, минимум строк.
   Нарушение - исключение (задача DAG падает, плохие данные не идут дальше).
2. Value-skew (validate_value_skew) - сравнение распределений признаков с
   эталонным снапшотом по PSI. На первом прогоне эталона нет - он создаётся,
   проверка проходит. Дальше любой признак с PSI выше порога валит гейт.

PSI и KS считаем на numpy, без scipy (минимизируем зависимости).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from uneemi_ml.config import EMBED_DIM
from uneemi_ml.features import QUIZ_FEATURES

_LOGGER = logging.getLogger(__name__)

REQUIRED_PROFILE_COLUMNS: tuple[str, ...] = (
    "profile_id",
    "event_timestamp",
    "board_emb",
    *QUIZ_FEATURES,
)


class DataValidationError(ValueError):
    """Нарушение инвариантов данных - сигнал остановить пайплайн."""


def validate_profile_schema(df: pd.DataFrame, min_rows: int = 50) -> dict:
    """Жёсткая проверка схемы профилей. Бросает DataValidationError при нарушении."""
    if len(df) < min_rows:
        raise DataValidationError(f"Слишком мало профилей: {len(df)} < {min_rows}")

    missing = [c for c in REQUIRED_PROFILE_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Отсутствуют колонки: {missing}")

    # board_emb: каждая строка - вектор ровно 768d из конечных чисел.
    bad_board = 0
    for value in df["board_emb"]:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (EMBED_DIM,) or not np.isfinite(arr).all():
            bad_board += 1
    if bad_board:
        raise DataValidationError(f"Битых board_emb: {bad_board} (ожидалась ширина {EMBED_DIM})")

    quiz_cols = list(QUIZ_FEATURES)
    quiz = df[quiz_cols]
    if quiz.isnull().any().any():
        raise DataValidationError("В quiz-признаках есть пропуски")
    if float(quiz.min().min()) < -1e-6 or float(quiz.max().max()) > 1.0 + 1e-6:
        raise DataValidationError("quiz-признаки вне диапазона [0,1] (value-skew схемы)")

    if df["profile_id"].duplicated().any():
        raise DataValidationError("Дубликаты profile_id")

    report = {"n_rows": len(df), "n_features": len(quiz_cols), "status": "ok"}
    _LOGGER.info("Схема профилей валидна: %s", report)
    return report


def validate_pairs(df: pd.DataFrame, min_rows: int = 100) -> dict:
    """Проверка датасета пар: размер, баланс классов, отсутствие self-пар."""
    if len(df) < min_rows:
        raise DataValidationError(f"Слишком мало пар: {len(df)} < {min_rows}")
    for col in ("profile_a_id", "profile_b_id", "matched"):
        if col not in df.columns:
            raise DataValidationError(f"В парах нет колонки {col}")
    if (df["profile_a_id"] == df["profile_b_id"]).any():
        raise DataValidationError("Найдены self-пары (A == B)")
    pos_rate = float(df["matched"].mean())
    if not (0.05 < pos_rate < 0.95):
        raise DataValidationError(f"Вырожденный баланс классов: доля матчей {pos_rate:.3f}")
    report = {"n_pairs": len(df), "pos_rate": pos_rate, "status": "ok"}
    _LOGGER.info("Датасет пар валиден: %s", report)
    return report


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index по квантильным бинам эталона.

    PSI < 0.1 - стабильно; 0.1-0.25 - умеренный сдвиг; > 0.25 - сильный сдвиг.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    eps = 1e-6
    e_frac = e_hist / max(e_hist.sum(), 1) + eps
    a_frac = a_hist / max(a_hist.sum(), 1) + eps
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    """Двухвыборочная статистика Колмогорова-Смирнова D (максимум |CDF1-CDF2|)."""
    expected = np.sort(np.asarray(expected, dtype=np.float64))
    actual = np.sort(np.asarray(actual, dtype=np.float64))
    grid = np.concatenate([expected, actual])
    cdf_e = np.searchsorted(expected, grid, side="right") / len(expected)
    cdf_a = np.searchsorted(actual, grid, side="right") / len(actual)
    return float(np.max(np.abs(cdf_e - cdf_a)))


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Свести профиль к числовым колонкам для дрифта: 7 quiz + 2 сводные по board."""
    board = np.vstack([np.asarray(e, dtype=np.float32) for e in df["board_emb"]])
    out = df[list(QUIZ_FEATURES)].copy()
    # 768d board не дрифт-тестируем поколоночно (шумно/широко) - берём две сводные.
    out["board_emb_mean"] = board.mean(axis=1)
    out["board_emb_norm"] = np.linalg.norm(board, axis=1)
    return out


def validate_value_skew(
    df: pd.DataFrame,
    reference_path: str | Path,
    psi_threshold: float = 0.25,
) -> dict:
    """Сравнить распределения признаков с эталоном по PSI. Первый прогон - создаёт эталон."""
    reference_path = Path(reference_path)
    feats = _feature_frame(df)

    if not reference_path.exists():
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {col: feats[col].to_numpy().tolist() for col in feats.columns}
        reference_path.write_text(json.dumps(snapshot), encoding="utf-8")
        _LOGGER.info("Эталон value-skew создан: %s (первый прогон)", reference_path)
        return {"status": "reference_created", "max_psi": 0.0, "per_feature": {}}

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    per_feature: dict[str, float] = {}
    for col in feats.columns:
        if col in reference:
            per_feature[col] = psi(np.asarray(reference[col]), feats[col].to_numpy())
    max_psi = max(per_feature.values(), default=0.0)
    drifted = {k: v for k, v in per_feature.items() if v > psi_threshold}
    if drifted:
        raise DataValidationError(
            f"Value-skew: признаки с PSI > {psi_threshold}: "
            + ", ".join(f"{k}={v:.3f}" for k, v in drifted.items())
        )
    report = {"status": "ok", "max_psi": max_psi, "per_feature": per_feature}
    _LOGGER.info("Value-skew в норме: max PSI=%.3f", max_psi)
    return report
