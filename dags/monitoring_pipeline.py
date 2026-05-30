"""DAG monitoring_pipeline: дрифт + качество -> continuous training (CT).

Граф: collect (снимок текущих признаков; при сценарии дрифта - сдвиг батча) ->
drift_report (Evidently PSI/KS) -> quality_check (live ROC-AUC текущей Production на
свежей разметке) -> push_metrics (в Pushgateway) -> decision_gate (Branch) ->
{trigger_training | no_action}. Триггер обучения при дрифте выше порога ИЛИ падении
метрики ниже SLO - это и есть петля непрерывного переобучения (CT).

Сценарий демонстрации дрифта: запустить DAG с conf {"drift_scenario": true} -
текущий батч сдвигается, Evidently фиксирует drift, запускается training_pipeline.

Расписание по умолчанию - каждые 30 минут (на демо удобнее запускать вручную).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DEFAULT_ARGS = {"owner": "uneemi", "retries": 0}
MON = Path("/opt/airflow/runs/monitoring")
MODEL_NAME = os.environ.get("MODEL_NAME", "uneemi_match")


def collect(**context) -> None:
    """Снять текущий батч признаков; при drift_scenario - сдвинуть распределение."""
    import numpy as np
    import pandas as pd

    from monitoring.evidently_job import feature_frame
    from training.feast_ops import offline_parquet_path

    MON.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_parquet(offline_parquet_path())
    reference = feature_frame(profiles)

    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    drift_scenario = bool(conf.get("drift_scenario", False))
    current_profiles = profiles.copy()
    if drift_scenario:
        # Сдвиг батча: quiz +0.3 (клип [0,1]) и масштаб board - искусственный дрифт.
        from uneemi_ml.features import QUIZ_FEATURES

        current_profiles[list(QUIZ_FEATURES)] = (
            current_profiles[list(QUIZ_FEATURES)] + 0.3
        ).clip(0, 1)
        current_profiles["board_emb"] = current_profiles["board_emb"].apply(
            lambda v: (np.asarray(v, dtype=np.float32) * 1.15).tolist()
        )
    current = feature_frame(current_profiles)

    reference.to_parquet(MON / "reference.parquet", index=False)
    current.to_parquet(MON / "current.parquet", index=False)
    (MON / "scenario.json").write_text(json.dumps({"drift_scenario": drift_scenario}))


def drift_report() -> None:
    import pandas as pd

    from monitoring.evidently_job import compute_drift

    reference = pd.read_parquet(MON / "reference.parquet")
    current = pd.read_parquet(MON / "current.parquet")
    result = compute_drift(reference, current)
    (MON / "drift.json").write_text(json.dumps(result, ensure_ascii=False))


def quality_check() -> None:
    """Измерить live ROC-AUC текущей Production-модели на свежей разметке."""
    import mlflow
    import pandas as pd

    from training.evaluate import evaluate_model
    from training.feast_ops import offline_parquet_path
    from training.prepare import build_pair_dataset
    from training.promote import setup_mlflow
    from uneemi_ml.demo import PAIRS_DIR

    setup_mlflow()
    live_auc = None
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
        profiles = pd.read_parquet(offline_parquet_path())
        all_pairs = pd.read_parquet(PAIRS_DIR / "pairs.parquet")
        pairs = all_pairs.sample(n=min(2000, len(all_pairs)), random_state=0)
        x, y = build_pair_dataset(pairs, profiles)
        live_auc = evaluate_model(model, x, y)["roc_auc"]
    except Exception as exc:  # noqa: BLE001 - модель может быть ещё не готова
        print(f"quality_check: не удалось измерить live-качество ({exc})")
    (MON / "quality.json").write_text(json.dumps({"live_auc": live_auc}))


def push_metrics() -> None:
    from monitoring.evidently_job import push_metrics as _push

    drift = json.loads((MON / "drift.json").read_text())
    quality = json.loads((MON / "quality.json").read_text())
    _push(drift["drift_share"], drift["dataset_drift"], quality["live_auc"])


def decision_gate() -> str:
    """Решение о CT: дрифт выше порога ИЛИ live-AUC ниже SLO -> переобучение."""
    drift = json.loads((MON / "drift.json").read_text())
    quality = json.loads((MON / "quality.json").read_text())
    drift_threshold = float(os.environ.get("DRIFT_THRESHOLD", "0.5"))
    auc_slo = float(os.environ.get("LIVE_AUC_SLO", "0.65"))

    drift_trip = drift["drift_share"] > drift_threshold
    quality_trip = quality["live_auc"] is not None and quality["live_auc"] < auc_slo
    if drift_trip or quality_trip:
        reason = []
        if drift_trip:
            reason.append(f"drift_share={drift['drift_share']:.3f}>{drift_threshold}")
        if quality_trip:
            reason.append(f"live_auc={quality['live_auc']:.3f}<{auc_slo}")
        print(f"Триггер CT: {', '.join(reason)}")
        return "trigger_training"
    print("Дрифта/деградации нет - переобучение не требуется")
    return "no_action"


with DAG(
    dag_id="monitoring_pipeline",
    description="Мониторинг дрифта и качества с триггером непрерывного переобучения",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="*/30 * * * *",
    catchup=False,
    tags=["uneemi", "monitoring", "C9"],
) as dag:
    t_collect = PythonOperator(task_id="collect", python_callable=collect)
    t_drift = PythonOperator(task_id="drift_report", python_callable=drift_report)
    t_quality = PythonOperator(task_id="quality_check", python_callable=quality_check)
    t_push = PythonOperator(task_id="push_metrics", python_callable=push_metrics)
    t_gate = BranchPythonOperator(task_id="decision_gate", python_callable=decision_gate)
    t_trigger = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training_pipeline",
        wait_for_completion=False,
        reset_dag_run=True,
    )
    t_noop = EmptyOperator(task_id="no_action")

    t_collect >> t_drift >> t_quality >> t_push >> t_gate >> [t_trigger, t_noop]
