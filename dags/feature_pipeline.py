"""DAG feature_pipeline: извлечение -> очистка -> валидация -> запись в Feast.

Граф: extract_raw (РЕАЛЬНЫЙ SigLIP по доскам -> профили + пары) -> clean ->
validate_schema -> validate_skew (PSI vs эталон) -> write_offline -> feast_apply ->
materialize (online Redis). Если валидация падает - плохие данные не доезжают до
фичестора (демонстрация защиты данных).

Тяжёлые импорты - внутри задач, чтобы парсинг DAG оставался лёгким.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {"owner": "uneemi", "retries": 0}


def _profiles_clean_path():
    from uneemi_ml.demo import RAW_DIR

    return RAW_DIR / "profiles_clean.parquet"


def extract_raw() -> None:
    """Сгенерировать профили (реальный SigLIP) и помеченные пары."""
    from training.data_gen import main as data_gen_main

    rc = data_gen_main()
    if rc != 0:
        raise RuntimeError("Генерация данных завершилась с ошибкой")


def clean() -> None:
    import pandas as pd

    from training.clean import clean_profiles
    from training.data_gen import PROFILES_RAW_PATH

    df = pd.read_parquet(PROFILES_RAW_PATH)
    cleaned = clean_profiles(df)
    cleaned.to_parquet(_profiles_clean_path(), index=False)


def validate_schema() -> None:
    import pandas as pd

    from training.validate import validate_profile_schema

    df = pd.read_parquet(_profiles_clean_path())
    validate_profile_schema(df)


def validate_skew() -> None:
    import pandas as pd

    from training.feast_ops import repo_path
    from training.validate import validate_value_skew

    df = pd.read_parquet(_profiles_clean_path())
    reference = f"{repo_path()}/data/reference_stats.json"
    validate_value_skew(df, reference)


def write_offline() -> None:
    import pandas as pd

    from training.feast_ops import write_offline as _write

    df = pd.read_parquet(_profiles_clean_path())
    _write(df)


def feast_apply() -> None:
    from training.feast_ops import feast_apply as _apply

    _apply()


def materialize() -> None:
    from training.feast_ops import materialize as _materialize

    _materialize()


with DAG(
    dag_id="feature_pipeline",
    description="Извлечение SigLIP-признаков, валидация и запись в Feast",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["uneemi", "features", "C4"],
) as dag:
    t_extract = PythonOperator(task_id="extract_raw", python_callable=extract_raw)
    t_clean = PythonOperator(task_id="clean", python_callable=clean)
    t_schema = PythonOperator(task_id="validate_schema", python_callable=validate_schema)
    t_skew = PythonOperator(task_id="validate_skew", python_callable=validate_skew)
    t_write = PythonOperator(task_id="write_offline", python_callable=write_offline)
    t_apply = PythonOperator(task_id="feast_apply", python_callable=feast_apply)
    t_mat = PythonOperator(task_id="materialize", python_callable=materialize)

    t_extract >> t_clean >> t_schema >> t_skew >> t_write >> t_apply >> t_mat
