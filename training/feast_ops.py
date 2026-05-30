"""Операции с Feast: запись offline, apply, материализация, исторические выборки.

Вынесено отдельно, чтобы DAG-и оставались тонкими, а тяжёлый импорт feast
происходил внутри функций (не на парсинге DAG). feast apply делаем через CLI -
это канонично и не требует ручного импорта определений фич.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from uneemi_ml.features import QUIZ_FEATURES

_LOGGER = logging.getLogger(__name__)

_DEFAULT_REPO = "/opt/airflow/feature_repo"


def repo_path() -> str:
    return os.environ.get("FEAST_REPO_PATH", _DEFAULT_REPO)


def offline_parquet_path(repo: str | None = None) -> Path:
    return Path(repo or repo_path()) / "data" / "profile_features.parquet"


def write_offline(profiles: pd.DataFrame, repo: str | None = None) -> Path:
    """Записать offline-паркет профилей (источник FileSource для Feast)."""
    path = offline_parquet_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(path, index=False)
    _LOGGER.info("Offline-паркет записан: %s (%d профилей)", path, len(profiles))
    return path


def feast_apply(repo: str | None = None) -> None:
    """feast apply - регистрация сущностей и FeatureView в реестре."""
    repo = repo or repo_path()
    _LOGGER.info("feast apply в %s", repo)
    subprocess.run(["feast", "apply"], cwd=repo, check=True)


def materialize(repo: str | None = None, end: datetime | None = None) -> None:
    """Материализовать последние значения в online-store (Redis)."""
    from feast import FeatureStore

    repo = repo or repo_path()
    store = FeatureStore(repo_path=repo)
    store.materialize_incremental(end_date=end or datetime.now(UTC))
    _LOGGER.info("Материализация в online-store завершена")


def get_training_features(
    profile_ids: list[int],
    repo: str | None = None,
    ts: datetime | None = None,
) -> pd.DataFrame:
    """Историческая выборка признаков профилей из offline-store (point-in-time join)."""
    from feast import FeatureStore

    repo = repo or repo_path()
    store = FeatureStore(repo_path=repo)
    entity_df = pd.DataFrame(
        {"profile_id": list(profile_ids), "event_timestamp": ts or datetime.now(UTC)}
    )
    features = ["profile_features:board_emb"] + [f"profile_features:{q}" for q in QUIZ_FEATURES]
    df = store.get_historical_features(entity_df=entity_df, features=features).to_df()
    _LOGGER.info("Историческая выборка: %d профилей, %d колонок", len(df), df.shape[1])
    return df
