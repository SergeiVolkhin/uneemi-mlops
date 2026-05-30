-- Инициализация Postgres: основная БД airflow создаётся образом из POSTGRES_DB,
-- здесь доводим вторую БД под MLflow. Скрипт монтируется в
-- /docker-entrypoint-initdb.d/ и выполняется однократно при первой инициализации
-- кластера (когда том pgdata пуст). Идемпотентность через guard ниже.
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
