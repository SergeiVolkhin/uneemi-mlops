#!/usr/bin/env bash
# Сквозной сценарий: прогон трёх DAG + проверка промоута, дрифта и гейта.
# Запуск из infra/:  bash ./e2e.sh   (стек уже должен быть поднят)
set -u

DC="docker compose --env-file .env -f docker-compose.yml"
SCHED="$DC exec -T airflow-scheduler"

run_dag() {
  # Запустить DAG и дождаться финального состояния. $1 - dag_id, $2 - conf (json|"").
  local dag="$1" conf="${2:-}" args=""
  [ -n "$conf" ] && args="--conf $conf"
  echo ">> trigger $dag $args"
  $SCHED airflow dags unpause "$dag" >/dev/null 2>&1 || true
  # shellcheck disable=SC2086
  $SCHED airflow dags trigger "$dag" $args >/dev/null
  for _ in $(seq 1 60); do
    sleep 10
    local state
    state=$($SCHED airflow dags list-runs -d "$dag" -o plain 2>/dev/null | awk 'NR==2{print $3}')
    echo "   $dag: $state"
    case "$state" in
      success) return 0 ;;
      failed) return 1 ;;
    esac
  done
  return 2
}

echo "== 1) feature_pipeline =="
run_dag feature_pipeline || { echo "feature_pipeline не прошёл"; exit 1; }

echo "== 2) training_pipeline (промоут лучшей модели) =="
run_dag training_pipeline || { echo "training_pipeline не прошёл"; exit 1; }

echo "== Production-версия в MLflow =="
$SCHED python -c "
import os; from mlflow.tracking import MlflowClient
os.environ.setdefault('MLFLOW_TRACKING_URI','http://mlflow:5000')
c=MlflowClient()
p=c.get_latest_versions('uneemi_match', stages=['Production'])
a=c.get_latest_versions('uneemi_match', stages=['Archived'])
print('Production:', [v.version for v in p], 'Archived:', [v.version for v in a])
"

echo "== 3) monitoring_pipeline со сдвигом (дрифт -> CT) =="
run_dag monitoring_pipeline '{\"drift_scenario\": true}' || echo "monitoring завершился не success (см. логи)"

echo "== 4) training_pipeline с завышенным порогом (gate-fail) =="
run_dag training_pipeline '{\"auc_threshold\": 0.999}' || echo "training (gate-fail) завершился (ожидаемо без промоута)"

echo "E2E завершён. Проверьте Production/Archived выше и Grafana."
