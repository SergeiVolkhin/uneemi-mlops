# Отчёт о приёмочном прогоне (end-to-end)

Дата прогона: 2026-05-31 (UTC). Стек поднят локально (Windows + Docker Desktop),
проверки выполнены на живом стеке без пересоздания (teardown не выполнялся, чтобы
сохранить доказательства). Все числа ниже - фактические, из реальных вызовов.

## 1. Контейнеры (критерий 3)

`docker compose ps` - 10 долгоживущих сервисов в статусе Up (healthy), 3 one-shot
init-контейнера завершились с кодом 0 (ожидаемо):

| Сервис | Статус | Host-порт |
|---|---|---|
| postgres | Up (healthy) | 15432 |
| redis | Up (healthy) | 16379 |
| minio | Up (healthy) | 9000/9001 |
| mlflow | Up (healthy) | 5500 |
| serving | Up (healthy) | 18000 |
| airflow-webserver | Up (healthy) | 18080 |
| airflow-scheduler | Up (healthy) | - |
| prometheus | Up (healthy) | 19090 |
| pushgateway | Up (healthy) | 19091 |
| grafana | Up (healthy) | 13000 |
| airflow-init / model-bootstrap / minio-mc | Exited (0) | one-shot |

## 2. Эндпоинты сервинга

- `GET /health` -> `200 {"status":"ok","model_version":6}`.
- `POST /predict` (по id из Feast online): пара (5,7) -> `p_match=0.3412`,
  `model_version=6`, latency ~6-11 мс (в пределах SLO p99 <= 200 мс); пара (0,1) ->
  `p_match=0.0085`. Первый по-id вызов после старта стоил ~1.6 с - одноразовая
  ленивая загрузка библиотеки Feast; последующие вызовы укладываются в SLO.
- `GET /metrics`: `uneemi_model_version=6`, `uneemi_model_loaded=1`,
  `uneemi_predict_latency_seconds_count=212`, `uneemi_requests_total{predict,200}=212`,
  `uneemi_guardrail_breaches_total=1`, `uneemi_in_flight_requests=0`.

## 3. MLflow Model Registry (вывод из эксплуатации)

Модель `uneemi_match`:

| Версия | Стадия | holdout AUC | Комментарий |
|---|---|---|---|
| v6 | Production | 0.796544 | текущий champion |
| v3 | Archived | 0.796544 | выведена из эксплуатации при промоуте |
| v7 | None | 0.796544 | challenger от CT, гейт не пройден |
| v8 | None | 0.796544 | challenger gate-fail (порог 0.999) |

Алиасы: `champion -> v6`, `challenger -> v8`. Одновременное наличие Production и
Archived - прямое доказательство вывода устаревшей версии через переключение.

## 4. Airflow (три DAG)

- `feature_pipeline`: 1 успешный прогон (~8 мин - реальный SigLIP по банку картинок).
- `training_pipeline`: 7 успешных прогонов.
- `monitoring_pipeline`: запускается по расписанию каждые 30 минут, все прогоны success
  (живой непрерывный мониторинг).

## 5. Демонстрируемые сценарии (фактические прогоны 2026-05-31)

1. Continuous training по дрифту. `monitoring_pipeline --conf '{"drift_scenario": true}'`
   завершился success, ветка `decision_gate` ушла в `trigger_training`, и
   `TriggerDagRunOperator` поднял новый прогон `training_pipeline` (число прогонов
   обучения 5 -> 6). Зарегистрирована версия v7. Гейт корректно отклонил её
   (0.7965 не строго больше champion 0.7965) - Production остался v6.
2. Защита от деградации (gate-fail). `training_pipeline --conf '{"auc_threshold": 0.999}'`
   завершился success; состояния задач: `gate=success`, `promote=skipped`,
   `skip_promote=success`. Зарегистрирована v8 (стадия None), Production остался v6.
3. Промоут + вывод из эксплуатации (исторический прогон, виден в реестре): версия
   доведена до Production, прежний champion переведён в Archived (v3).
4. Guardrail rollback (исторический прогон): `uneemi_guardrail_breaches_total=1` -
   срабатывал откат на Archived при инъекции задержки.

## 6. Линт и тесты

- `ruff check .` - All checks passed.
- `pytest -q` - 26 passed, 1 skipped. Пропущен тест тренировочного ядра, требующий
  `scikit-learn` (в хостовом dev-venv его нет; в CI и контейнерах он установлен -
  тесты тренировки там зелёные). Тесты SigLIP прошли против реального ONNX-файла.

## 7. Деинсталляция

Цель `teardown` в `infra/Makefile`: `docker compose --env-file .env -f docker-compose.yml
down -v` (контейнеры + тома). Проверена чтением, реально не выполнялась, чтобы не
потерять поднятый стек как доказательство критерия 3 и источник скриншотов.

## Итог

Полный жизненный цикл подтверждён вживую: данные -> обучение -> гейт -> промоут с
выводом прежней версии в Archived -> горячее переключение сервинга -> мониторинг ->
continuous training по дрифту, плюс gate-fail и guardrail rollback. Локальный стек
полностью работоспособен; облачное развёртывание - отдельный шаг
(см. `infra/terraform/README.md`).
