"""Bootstrap-модель: завести начальный Production, чтобы serving был здоров сразу.

Проблема курицы и яйца: serving отдаёт /health=200 только при наличии Production-
модели, но первой реальной модели ещё нет до прогона training_pipeline. Поэтому на
старте стека регистрируем заведомо СЛАБУЮ базовую модель (bootstrap baseline) и
кладём её в Production. Первый же реальный challenger её уверенно побьёт - это и
демонстрирует промоут.

Данные здесь - быстрая случайная синтетика (без SigLIP/картинок), чтобы bootstrap
поднимался за секунды. Помечаем версию тегом bootstrap=true и data_note.

Идемпотентность: если Production уже есть - выходим без изменений (для повторного up).

Запуск: python scripts/bootstrap_model.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from mlflow.tracking import MlflowClient  # noqa: E402

from training import promote as P  # noqa: E402
from training.evaluate import evaluate_model  # noqa: E402
from training.train import train_champion  # noqa: E402
from uneemi_ml.features import EMBED_DIM, PAIR_DIM, QUIZ_DIM  # noqa: E402

_LOGGER = logging.getLogger(__name__)


def _weak_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Случайная, но слегка обучаемая выборка 775d - чтобы bootstrap был слабым baseline."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, PAIR_DIM)).astype(np.float32)
    # Слабый сигнал: метка зависит от небольшой суммы признаков + сильный шум.
    signal = x[:, :8].sum(axis=1) + 0.5 * x[:, EMBED_DIM : EMBED_DIM + QUIZ_DIM].sum(axis=1)
    z = 0.6 * signal + rng.normal(0, 3.0, size=n)
    y = (z > np.median(z)).astype(np.int64)
    return x, y


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name = os.environ.get("MODEL_NAME", "uneemi_match")
    P.setup_mlflow()
    client = MlflowClient()

    # Идемпотентность: если Production уже есть - ничего не делаем.
    try:
        existing = client.get_latest_versions(name, stages=[P.STAGE_PRODUCTION])
    except Exception:  # noqa: BLE001 - модель ещё не зарегистрирована
        existing = []
    if existing:
        _LOGGER.info("Production есть (версия %s), bootstrap пропущен", existing[0].version)
        return 0

    _LOGGER.info("Регистрирую bootstrap baseline в Production...")
    x_tr, y_tr = _weak_dataset(2000, seed=0)
    x_te, y_te = _weak_dataset(800, seed=1)
    model = train_champion(x_tr, y_tr, seed=0)
    val_metrics = evaluate_model(model, x_tr, y_tr)
    test_metrics = evaluate_model(model, x_te, y_te)

    version = P.register_candidate(
        model=model,
        model_type="bootstrap_logreg",
        params={"note": "bootstrap baseline"},
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        name=name,
        extra_tags={"bootstrap": "true", "data_note": "синтетика для демонстрации"},
    )
    P.promote_to_production(name, version)
    auc = test_metrics["roc_auc"]
    _LOGGER.info("Bootstrap версии %s в Production (test AUC=%.3f)", version, auc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
