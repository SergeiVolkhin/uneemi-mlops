"""Регистрация, гейт качества и промоут модели в MLflow Model Registry (C6/C7).

Это ядро требования ТЗ про полный жизненный цикл и ВЫВОД устаревшей модели из
эксплуатации. Используем MLflow 2.x, где есть И стадии (Production/Archived), И
алиасы (champion/challenger) - в 3.x стадии удалены, поэтому пин 2.22.1.

Семантика:
- champion - текущая боевая версия (стадия Production, алиас champion).
- challenger - последний претендент (алиас challenger), пока не прошёл гейт.
- Гейт промоута: test-AUC претендента >= порога И строго выше AUC текущего champion.
- Промоут: claimer -> Production с archive_existing_versions=True, то есть старый
  champion автоматически уходит в Archived (это и есть вывод из эксплуатации).
- Откат: latest Archived -> Production (используется serving при срыве guardrails).
"""

from __future__ import annotations

import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

_LOGGER = logging.getLogger(__name__)

EXPERIMENT_NAME = "uneemi_match_training"
ALIAS_CHAMPION = "champion"
ALIAS_CHALLENGER = "challenger"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"
TAG_HOLDOUT_AUC = "holdout_auc"
TAG_MODEL_TYPE = "model_type"


def setup_mlflow() -> str:
    """Настроить трекинг по env и вернуть имя эксперимента (создать при отсутствии)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def register_candidate(
    model,
    model_type: str,
    params: dict,
    val_metrics: dict,
    test_metrics: dict,
    name: str,
    extra_tags: dict | None = None,
) -> str:
    """Залогировать претендента в run, зарегистрировать версию, повесить алиас challenger.

    Возвращает номер созданной версии (строкой). test-AUC сохраняем тегом версии -
    по нему позже сравниваем с champion в гейте.
    """
    client = MlflowClient()
    with mlflow.start_run(run_name=f"{name}-{model_type}") as run:
        mlflow.log_params({f"{model_type}__{k}": v for k, v in params.items()})
        mlflow.log_param(TAG_MODEL_TYPE, model_type)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.set_tag("data_note", "синтетика для демонстрации")
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    version = mlflow.register_model(f"runs:/{run_id}/model", name).version
    client.set_model_version_tag(name, version, TAG_HOLDOUT_AUC, f"{test_metrics['roc_auc']:.6f}")
    client.set_model_version_tag(name, version, TAG_MODEL_TYPE, model_type)
    for k, v in (extra_tags or {}).items():
        client.set_model_version_tag(name, version, k, str(v))
    client.set_registered_model_alias(name, ALIAS_CHALLENGER, version)
    _LOGGER.info("Зарегистрирована версия %s (%s), алиас challenger", version, model_type)
    return version


def get_champion_auc(name: str) -> float | None:
    """Holdout-AUC текущего champion (Production). None, если Production пуст."""
    client = MlflowClient()
    versions = client.get_latest_versions(name, stages=[STAGE_PRODUCTION])
    if not versions:
        return None
    auc = versions[0].tags.get(TAG_HOLDOUT_AUC)
    return float(auc) if auc is not None else None


def decide_promotion(
    challenger_auc: float,
    champion_auc: float | None,
    auc_threshold: float,
) -> tuple[bool, str]:
    """Гейт: претендент проходит порог И бьёт текущего champion (если он есть)."""
    if challenger_auc < auc_threshold:
        return False, f"AUC {challenger_auc:.4f} ниже порога {auc_threshold:.4f}"
    if champion_auc is not None and challenger_auc <= champion_auc:
        return False, (
            f"AUC {challenger_auc:.4f} не выше текущего champion {champion_auc:.4f}"
        )
    base = "первый champion" if champion_auc is None else f"бьёт champion {champion_auc:.4f}"
    return True, f"AUC {challenger_auc:.4f} проходит порог и {base}"


def promote_to_production(name: str, version: str) -> None:
    """Перевести версию в Production с архивацией прежней (вывод из эксплуатации)."""
    client = MlflowClient()
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=STAGE_PRODUCTION,
        archive_existing_versions=True,  # прежний Production -> Archived
    )
    client.set_registered_model_alias(name, ALIAS_CHAMPION, version)
    _LOGGER.info("Версия %s -> Production (champion); прежняя -> Archived", version)


def rollback_to_archived(name: str) -> str | None:
    """Откатить Production на самую свежую Archived-версию. Возвращает её номер."""
    client = MlflowClient()
    archived = client.get_latest_versions(name, stages=[STAGE_ARCHIVED])
    if not archived:
        _LOGGER.warning("Нет Archived-версии для отката %s", name)
        return None
    target = max(archived, key=lambda mv: int(mv.version))
    client.transition_model_version_stage(
        name=name, version=target.version, stage=STAGE_PRODUCTION, archive_existing_versions=True
    )
    client.set_registered_model_alias(name, ALIAS_CHAMPION, target.version)
    _LOGGER.warning("Откат: версия %s возвращена в Production", target.version)
    return target.version
