"""Оценка качества классификатора матча на holdout.

Метрики переводят бизнес-задачу в измеримое: ROC-AUC (ранжирующая способность -
ключевая для подбора кандидатов), Precision/Recall/F1 при пороге 0.5 (баланс
ложных матчей и пропущенных хороших пар), accuracy для полноты картины. ROC-AUC -
главный гейт промоута, т.к. не зависит от выбранного порога.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_LOGGER = logging.getLogger(__name__)


def evaluate_model(model, x: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Посчитать метрики качества на (x, y). Возвращает плоский dict для логирования."""
    proba = model.predict_proba(x)[:, 1]
    preds = (proba >= threshold).astype(np.int64)
    metrics = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y, preds)),
        "n_samples": int(len(y)),
    }
    _LOGGER.info(
        "Оценка: AUC=%.4f P=%.4f R=%.4f F1=%.4f (n=%d)",
        metrics["roc_auc"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["n_samples"],
    )
    return metrics
