# Скриншоты (доказательства критерия 3)

Сюда кладутся скриншоты с локального запуска стека. Имена файлов используются
в ссылках README (раздел "Скриншоты").

| Файл | Что должно быть видно |
|---|---|
| `docker-ps.png` | Вывод `docker compose ps`: все сервисы STATUS Up (healthy) |
| `health.png` | `GET http://localhost:18000/health` -> 200 и `model_version` |
| `mlflow-registry.png` | MLflow Model Registry: версия в Production, прежняя в Archived, алиас champion |
| `airflow-dags.png` | Airflow: успешные прогоны feature_pipeline / training_pipeline / monitoring_pipeline |
| `grafana.png` | Grafana: дашборд с тремя уровнями SLI (latency, drift, live AUC, версия модели) |

После добавления файлов раскомментировать блок с картинками в корневом `README.md`.
