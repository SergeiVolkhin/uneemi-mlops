#!/usr/bin/env bash
# Дымовой тест: проверяем, что все сервисы стека отвечают здоровьем.
# Запуск из infra/:  bash ./smoke.sh
set -u

# Порты берём из .env (host-порты в обход занятых 6379/8000).
set -a
# shellcheck disable=SC1091
. ./.env
set +a

DC="docker compose --env-file .env -f docker-compose.yml"
fail=0

check() {
  local name="$1" url="$2" expect="${3:-200}"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo 000)
  if [ "$code" = "$expect" ] || { [ "$expect" = "2xx" ] && [ "${code:0:1}" = "2" ]; }; then
    echo "  OK   $name ($url) -> $code"
  else
    echo "  FAIL $name ($url) -> $code (ожидалось $expect)"
    fail=1
  fi
}

echo "== Статус контейнеров =="
$DC ps

echo "== Health эндпоинты =="
check "serving"     "http://localhost:${SERVING_HOST_PORT}/health"
check "mlflow"      "http://localhost:${MLFLOW_HOST_PORT}/health"
check "airflow"     "http://localhost:${AIRFLOW_WEB_HOST_PORT}/health"
check "prometheus"  "http://localhost:${PROMETHEUS_HOST_PORT}/-/healthy"
check "pushgateway" "http://localhost:${PUSHGATEWAY_HOST_PORT}/-/healthy"
check "grafana"     "http://localhost:${GRAFANA_HOST_PORT}/api/health"
check "minio"       "http://localhost:${MINIO_API_HOST_PORT}/minio/health/live"

echo "== Redis ping =="
if $DC exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
  echo "  OK   redis -> PONG"
else
  echo "  FAIL redis ping"; fail=1
fi

if [ "$fail" = 0 ]; then
  echo "SMOKE: всё зелёное"
else
  echo "SMOKE: есть проблемы (см. выше)"
fi
exit $fail
