# Uneemi MLOps

ML-система вайб-матчинга (RU/CIS) уровня зрелости 2. Обслуживает полный жизненный
цикл модели: от очистки данных и обучения до вывода устаревшей модели из эксплуатации
через переключение трафика на новую. Обучаемая модель - классификатор матча (вход
775d признаков пары, выход вероятность матча); SigLIP 2 (ONNX) - слой извлечения
768d board-эмбеддингов (фичестор).

Артефакты по критериям оценки:
- Постановка цели и метрики: [`docs/goal_and_metrics.md`](docs/goal_and_metrics.md)
- Манифест (12 разделов, уровень 2): [`docs/manifest.md`](docs/manifest.md)
- Архитектура + диаграмма + маппинг C1-C9: [`docs/architecture.md`](docs/architecture.md)
- SLI/SLO на 3 уровнях: [`docs/sli_slo.md`](docs/sli_slo.md)
- MDD-анализ и ADR: [`docs/mdd/`](docs/mdd/)
- План оценки качества: [`docs/metrics_and_benchmarks.md`](docs/metrics_and_benchmarks.md)

## Компоненты (C1-C9)
CI/CD (GitHub Actions), репозиторий (git), оркестратор (Airflow), фичестор (Feast:
offline parquet/MinIO, online Redis), тренировочная инфра (Airflow), Model Registry
(MLflow, стадии + алиасы), ML metadata (MLflow Tracking: Postgres + MinIO), сервинг
(FastAPI с горячим переключением Production), мониторинг (Prometheus + Grafana +
Evidently).

## Требования
- Windows 10/11 + Docker Desktop (Linux-контейнеры), ~8 ГБ свободной RAM под стек
  (рекомендуется `.wslconfig` с `memory=8GB`).
- `uv` (для одноразового экспорта ONNX на хосте). Python 3.11.
- Заняты host-порты 6379 и 8000? Не страшно - стек использует другие (см. ниже).

## Запуск одной командой
```powershell
cd infra
copy .env.example .env   # 12-factor конфиг (dev-дефолты; для прода перегенерировать секреты)
.\make.ps1 up            # Windows без make. Linux/CI: make -C infra up
```
`up` идемпотентно: экспортирует ONNX (если ещё нет), скачивает банк демо-картинок
(если ещё нет), собирает образы и поднимает стек.

Оценка времени: **первый запуск ~10-20 минут** (сборка образов с ML-стеком + экспорт
ONNX ~370 МБ через torch + загрузка картинок). **Повторный `up` - 1-2 минуты**
(образы и артефакты закешированы, экспорт/загрузка пропускаются по идемпотентности).

Дождаться готовности:
```powershell
.\make.ps1 ps      # ждём, пока все сервисы = Up (healthy)
.\make.ps1 smoke   # дымовой тест эндпоинтов
```

## URL и доступы (после up)
| Сервис | URL | Доступ |
|---|---|---|
| Serving /health | http://localhost:18000/health | - |
| Serving /predict, /metrics | http://localhost:18000/docs | - |
| Airflow | http://localhost:18080 | admin / admin |
| MLflow | http://localhost:5500 | - |
| Grafana | http://localhost:13000 | admin / admin |
| Prometheus | http://localhost:19090 | - |
| MinIO консоль | http://localhost:9001 | minioadmin / minioadmin_dev |

## Демонстрируемые сценарии
DAG-и создаются на паузе. Запуск из Airflow UI (unpause + Trigger с config) или из CLI:

```powershell
# 1) Подготовка фич (реальный SigLIP -> Feast)
docker compose -f docker-compose.yml --env-file .env exec airflow-scheduler `
  airflow dags trigger feature_pipeline

# 2) Обучение + промоут лучшей модели -> serving переключится без рестарта
docker compose -f docker-compose.yml --env-file .env exec airflow-scheduler `
  airflow dags trigger training_pipeline

# 3) Дрифт -> continuous training (сдвинутый батч детектится Evidently, триггерит обучение)
docker compose -f docker-compose.yml --env-file .env exec airflow-scheduler `
  airflow dags trigger monitoring_pipeline --conf '{\"drift_scenario\": true}'

# 4) Защита от деградации: гейт не пройден -> Production не меняется
docker compose -f docker-compose.yml --env-file .env exec airflow-scheduler `
  airflow dags trigger training_pipeline --conf '{\"auc_threshold\": 0.999}'
```

Откат по guardrails (rollback на Archived): включить демо-хук (ENABLE_DEMO_HOOKS=true
в .env уже выставлен), затем впрыснуть задержку и нагрузить /predict:
```powershell
curl -X POST "http://localhost:18000/admin/inject_latency?ms=700&count=400"
# затем поток запросов к /predict -> p99 пробивает 500мс -> serving откатывается на Archived
```

## Скриншоты (для критерия 3)
Прикрепить в этот раздел после запуска:
- `docker compose ps` со STATUS Up (healthy) у всех сервисов.
- `GET /health` -> 200 с `model_version`.
- Grafana-дашборд "Uneemi MLOps - обзор" (latency, drift, AUC, версия модели).
- MLflow Model Registry: версия в Production, прежняя в Archived, алиас champion.
- Airflow: успешные прогоны трёх DAG.

## Smoke / e2e / teardown
```powershell
.\make.ps1 smoke      # health всех сервисов
.\make.ps1 e2e        # сквозной прогон пайплайнов + проверки
.\make.ps1 down       # остановить (данные сохранены)
.\make.ps1 teardown   # снести с томами (docker compose down -v)
```

## Тесты ядра
```powershell
uv sync
uv run ruff check .
uv run pytest -q   # тесты SigLIP skip без ONNX; контракт признаков и тренировка - зелёные
```

## Ядро SigLIP 2 (фичестор)
ONNX-инференс энкодера `google/siglip2-base-patch16-224`. Экспорт:
`uv run python scripts/export_onnx.py` (vision_model -> `models/siglip2_vision.onnx`,
delta-сверка PyTorch<->ONNX). Sanity-результаты (не выдуманы): ImageNet zs top-1
78.48%, XM3600 RU avg R@1 73.82% (`docs/sanity_results.md`). Использование:
```python
from PIL import Image
from uneemi_ml import Siglip2Encoder
vec = Siglip2Encoder().encode(Image.open("path.jpg"))  # (1, 768)
```

## Облачный деплой (открытый гейт критерия 3)
Скелет в [`infra/terraform/`](infra/terraform/) (Yandex Cloud): VM + cloud-init
(Docker + `docker compose up`) + security group. Реальный деплой (`terraform apply`
+ внешняя ссылка до устной защиты) - отдельный шаг; пароли/CIDR/секреты для прода
перегенерировать (см. `infra/terraform/README.md`).
