"""Определения фич Feast для профиля Uneemi (775d = 768 board + 7 quiz).

board_emb хранится одним полем Array(Float32) длиной 768 - это компактнее и
честнее, чем 768 скалярных колонок. Семь quiz-признаков - отдельные Float32 поля,
чтобы по ним можно было независимо считать дрифт (Evidently) и валидировать схему.

Источник offline - паркет, который пишет feature_pipeline. Путь берём от каталога
этого файла, чтобы `feast apply` работал из любого cwd.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Array, Float32

from uneemi_ml.features import QUIZ_FEATURES

_DATA = Path(__file__).resolve().parent / "data"
_PROFILE_PARQUET = str(_DATA / "profile_features.parquet")

# Сущность - профиль пользователя; ключ соединения profile_id.
profile = Entity(name="profile", join_keys=["profile_id"])

# Offline-источник: паркет с колонками profile_id, event_timestamp, board_emb, quiz.
profile_source = FileSource(
    name="profile_source",
    path=_PROFILE_PARQUET,
    timestamp_field="event_timestamp",
)

# Представление фич профиля. ttl большой - демо-данные не должны «протухать»
# между прогонами пайплайнов.
profile_features = FeatureView(
    name="profile_features",
    entities=[profile],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="board_emb", dtype=Array(Float32)),  # 768d board-эмбеддинг SigLIP
        *[Field(name=q, dtype=Float32) for q in QUIZ_FEATURES],  # 7 quiz-признаков
    ],
    online=True,
    source=profile_source,
)
