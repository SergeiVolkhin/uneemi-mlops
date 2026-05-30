"""Дрифт-отчёт Evidently (PSI/KS) и экспорт метрик в Prometheus Pushgateway.

Дрифт считаем не по всем 768 компонентам board (шумно и широко), а по 7 quiz-
признакам + двум сводным по board (среднее и норма) - это устойчивые сигналы
сдвига распределения. Метрики пушим в Pushgateway, т.к. задачи Airflow эфемерны и
их нельзя скрейпить напрямую.

Pin Evidently 0.4.40: используем Report + DataDriftPreset, результат берём из
report.as_dict()["metrics"][0]["result"].
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from training.validate import _feature_frame

_LOGGER = logging.getLogger(__name__)

REPORTS_DIR = Path("/opt/airflow/monitoring_reports")


def feature_frame(profiles: pd.DataFrame) -> pd.DataFrame:
    """Числовой кадр для дрифта: 7 quiz + 2 сводные по board."""
    return _feature_frame(profiles)


def compute_drift(
    reference: pd.DataFrame, current: pd.DataFrame, psi_threshold: float = 0.2
) -> dict:
    """Посчитать DataDrift (PSI) по числовым признакам. Сохраняет HTML-отчёт."""
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    columns = list(current.columns)
    mapping = ColumnMapping(numerical_features=columns)
    report = Report(metrics=[DataDriftPreset(stattest="psi", stattest_threshold=psi_threshold)])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / "drift_report.html"
    report.save_html(str(html_path))

    result = report.as_dict()["metrics"][0]["result"]
    out = {
        "dataset_drift": bool(result["dataset_drift"]),
        "drift_share": float(result["share_of_drifted_columns"]),
        "n_drifted": int(result["number_of_drifted_columns"]),
        "n_columns": int(result["number_of_columns"]),
        "html": str(html_path),
    }
    _LOGGER.info(
        "Дрифт: dataset_drift=%s, share=%.3f (%d/%d колонок)",
        out["dataset_drift"], out["drift_share"], out["n_drifted"], out["n_columns"],
    )
    return out


def push_metrics(drift_share: float, dataset_drift: bool, live_auc: float | None) -> None:
    """Отправить метрики мониторинга в Pushgateway (их затем скрейпит Prometheus)."""
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    host = os.environ.get("PUSHGATEWAY_HOST", "pushgateway")
    port = os.environ.get("PUSHGATEWAY_PORT", "9091")
    registry = CollectorRegistry()
    Gauge("uneemi_drift_share", "Доля задрейфивших признаков", registry=registry).set(drift_share)
    Gauge("uneemi_dataset_drift", "1 если обнаружен дрифт набора", registry=registry).set(
        1 if dataset_drift else 0
    )
    if live_auc is not None:
        Gauge("uneemi_live_auc", "ROC-AUC на свежей разметке", registry=registry).set(live_auc)
    push_to_gateway(f"{host}:{port}", job="uneemi_monitoring", registry=registry)
    _LOGGER.info("Метрики мониторинга отправлены в Pushgateway %s:%s", host, port)
