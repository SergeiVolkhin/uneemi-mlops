"""Обучение моделей матча: champion (логистическая регрессия) и challenger (MLP).

Обе модели - sklearn-пайплайны со StandardScaler впереди: признаки пары имеют
разный масштаб (произведение board-компонент мелкое, |разность quiz| в [0,1]),
нормировка помогает и логрегрессии, и MLP. sklearn выбран намеренно - serving и
training остаются лёгкими (без torch в рантайме стека).

champion - простая, быстрая, интерпретируемая базовая модель. challenger - более
ёмкий MLP, который должен побить champion на holdout, чтобы попасть в Production.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_LOGGER = logging.getLogger(__name__)

CHAMPION_PARAMS: dict = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}
CHALLENGER_PARAMS: dict = {
    "hidden_layer_sizes": (128, 32),
    "alpha": 1e-3,
    "max_iter": 300,
    "early_stopping": True,
    "n_iter_no_change": 10,
}


def train_champion(x: np.ndarray, y: np.ndarray, seed: int = 42) -> Pipeline:
    """Логистическая регрессия - базовая модель (champion baseline)."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, **CHAMPION_PARAMS)),
        ]
    )
    model.fit(x, y)
    _LOGGER.info("champion (LogReg) обучен на %d примерах", len(y))
    return model


def train_challenger(x: np.ndarray, y: np.ndarray, seed: int = 42) -> Pipeline:
    """MLP - претендент (challenger), должен побить champion на holdout."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(random_state=seed, **CHALLENGER_PARAMS)),
        ]
    )
    model.fit(x, y)
    _LOGGER.info("challenger (MLP) обучен на %d примерах", len(y))
    return model
