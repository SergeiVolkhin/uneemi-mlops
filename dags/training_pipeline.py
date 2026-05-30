"""DAG training_pipeline: обучение, гейт качества, промоут с выводом старой модели.

Граф: extract_features (из Feast offline) -> data_validation -> prepare (775d пары +
сплит) -> [train_logreg | train_mlp] -> evaluate (выбор претендента по val-AUC) ->
register (в MLflow, алиас challenger) -> gate (BranchPythonOperator) ->
{promote | skip_promote}. При промоуте претендент уходит в Production, прежний
champion архивируется (вывод из эксплуатации).

Сценарий gate-fail для демонстрации защиты от деградации: запустить DAG с conf
{"auc_threshold": 0.999} - претендент не пройдёт порог, Production останется прежним.

Промежуточные артефакты - в общем каталоге RUNS (LocalExecutor, общая ФС).

Замечание по joblib: модели сериализуются/читаются joblib только в пределах одного
доверенного прогона DAG (наши же, только что обученные артефакты в приватном
RUNS-каталоге) - это не загрузка pickle из недоверенного источника.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

DEFAULT_ARGS = {"owner": "uneemi", "retries": 0}
RUNS = Path("/opt/airflow/runs/training")
MODEL_NAME = os.environ.get("MODEL_NAME", "uneemi_match")


def _pairs_path():
    from uneemi_ml.demo import PAIRS_DIR

    return PAIRS_DIR / "pairs.parquet"


def extract_features() -> None:
    """Достать признаки профилей из Feast offline (исторический join)."""
    import pandas as pd

    from training.feast_ops import get_training_features

    RUNS.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(_pairs_path())
    ids = sorted(set(pairs["profile_a_id"]).union(pairs["profile_b_id"]))
    profiles = get_training_features([int(i) for i in ids])
    profiles.to_parquet(RUNS / "profiles.parquet", index=False)


def data_validation() -> None:
    import pandas as pd

    from training.validate import validate_pairs, validate_profile_schema

    profiles = pd.read_parquet(RUNS / "profiles.parquet")
    pairs = pd.read_parquet(_pairs_path())
    validate_profile_schema(profiles)
    validate_pairs(pairs)


def prepare() -> None:
    import numpy as np
    import pandas as pd

    from training.prepare import build_pair_dataset, split_dataset

    profiles = pd.read_parquet(RUNS / "profiles.parquet")
    pairs = pd.read_parquet(_pairs_path())
    x, y = build_pair_dataset(pairs, profiles)
    splits = split_dataset(x, y, seed=int(os.environ.get("DATA_SEED", "42")))
    np.savez(RUNS / "splits.npz", **splits)


def train_logreg() -> None:
    import joblib
    import numpy as np

    from training.train import train_champion

    s = np.load(RUNS / "splits.npz")
    model = train_champion(s["x_train"], s["y_train"])
    joblib.dump(model, RUNS / "model_logreg.pkl")


def train_mlp() -> None:
    import joblib
    import numpy as np

    from training.train import train_challenger

    s = np.load(RUNS / "splits.npz")
    model = train_challenger(s["x_train"], s["y_train"])
    joblib.dump(model, RUNS / "model_mlp.pkl")


def evaluate() -> None:
    """Оценить обе модели; претендент - с лучшим val-AUC."""
    import joblib
    import numpy as np

    from training.evaluate import evaluate_model

    s = np.load(RUNS / "splits.npz")
    models = {
        "logreg": joblib.load(RUNS / "model_logreg.pkl"),
        "mlp": joblib.load(RUNS / "model_mlp.pkl"),
    }
    report: dict = {"models": {}}
    for name, model in models.items():
        report["models"][name] = {
            "val": evaluate_model(model, s["x_val"], s["y_val"]),
            "test": evaluate_model(model, s["x_test"], s["y_test"]),
        }
    contender = max(report["models"], key=lambda m: report["models"][m]["val"]["roc_auc"])
    report["contender"] = contender
    (RUNS / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def register() -> None:
    """Зарегистрировать претендента в MLflow (алиас challenger), сохранить версию."""
    import joblib

    from training.promote import register_candidate, setup_mlflow
    from training.train import CHALLENGER_PARAMS, CHAMPION_PARAMS

    setup_mlflow()
    report = json.loads((RUNS / "eval.json").read_text())
    contender = report["contender"]
    metrics = report["models"][contender]
    params = CHAMPION_PARAMS if contender == "logreg" else CHALLENGER_PARAMS
    model = joblib.load(RUNS / f"model_{contender}.pkl")
    version = register_candidate(
        model=model,
        model_type=contender,
        params=params,
        val_metrics=metrics["val"],
        test_metrics=metrics["test"],
        name=MODEL_NAME,
        extra_tags={"data_note": "синтетика для демонстрации"},
    )
    (RUNS / "registered.json").write_text(
        json.dumps({"version": version, "test_auc": metrics["test"]["roc_auc"]})
    )


def gate(**context) -> str:
    """Гейт промоута. Возвращает task_id следующей ветки."""
    from training.promote import decide_promotion, get_champion_auc, setup_mlflow

    setup_mlflow()
    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    threshold = float(conf.get("auc_threshold", os.environ.get("AUC_THRESHOLD", "0.70")))
    reg = json.loads((RUNS / "registered.json").read_text())
    champion_auc = get_champion_auc(MODEL_NAME)
    passed, reason = decide_promotion(reg["test_auc"], champion_auc, threshold)
    print(f"Гейт промоута: {'PASS' if passed else 'FAIL'} - {reason}")
    return "promote" if passed else "skip_promote"


def promote() -> None:
    from training.promote import promote_to_production, setup_mlflow

    setup_mlflow()
    reg = json.loads((RUNS / "registered.json").read_text())
    promote_to_production(MODEL_NAME, reg["version"])


with DAG(
    dag_id="training_pipeline",
    description="Обучение champion/challenger, гейт качества, промоут и вывод старой модели",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["uneemi", "training", "C5", "C6"],
) as dag:
    t_extract = PythonOperator(task_id="extract_features", python_callable=extract_features)
    t_validate = PythonOperator(task_id="data_validation", python_callable=data_validation)
    t_prepare = PythonOperator(task_id="prepare", python_callable=prepare)
    t_logreg = PythonOperator(task_id="train_logreg", python_callable=train_logreg)
    t_mlp = PythonOperator(task_id="train_mlp", python_callable=train_mlp)
    t_eval = PythonOperator(task_id="evaluate", python_callable=evaluate)
    t_register = PythonOperator(task_id="register", python_callable=register)
    t_gate = BranchPythonOperator(task_id="gate", python_callable=gate)
    t_promote = PythonOperator(task_id="promote", python_callable=promote)
    t_skip = EmptyOperator(task_id="skip_promote")

    t_extract >> t_validate >> t_prepare >> [t_logreg, t_mlp] >> t_eval
    t_eval >> t_register >> t_gate >> [t_promote, t_skip]
