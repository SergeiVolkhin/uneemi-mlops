# SLI и SLO Uneemi MLOps (критерий 4)

SLI (что измеряем) и SLO (целевой/критический уровень) на ТРЁХ уровнях -
техническом, модельном и бизнес. Пороги согласованы с `docs/metrics_and_benchmarks.md`,
`docs/benchmark_results.md`, `docs/sanity_results.md` и конфигурацией (`infra/.env.example`).
Технические latency-числа относятся к эталону 12 физических ядер, fp32 ONNX.

## 1. Технический уровень (инфраструктура и сервинг)

| Компонент | SLI | SLO (норма) | Критический уровень (SLO breach) | Где наблюдаем |
|---|---|---|---|---|
| Serving | p99 latency /predict | <= 200 мс | > 500 мс -> rollback | Prometheus histogram, Grafana |
| Serving | p50 / p95 latency /predict | p50 <= 50 мс, p95 <= 120 мс | p95 > 300 мс | Prometheus |
| Serving | Доля ошибок (4xx/5xx) | <= 0.5% | > 0.5% -> rollback | `uneemi_requests_total`, guardrail |
| Serving | Доступность /health | 200 при загруженной Production | 503 (модель не загружена) | healthcheck контейнера |
| SigLIP-инференс | encode(single) p99 | <= 62 мс (факт 54.7) | > 90 мс | `docs/benchmark_results.md` |
| SigLIP-инференс | encode_batch(32) p99 | <= 1850 мс (факт 1664) | > 2300 мс | `docs/benchmark_results.md` |
| Инфраструктура | RAM пик инференса bs=32 | <= 2048 МБ (факт 1256) | > 2048 МБ | `docs/benchmark_results.md` |
| Инфраструктура | CPU воркера сервинга | <= 80% устойчиво | > 95% устойчиво | хост/Prometheus |
| Все сервисы | Статус контейнера | Up (healthy) | unhealthy / restart loop | `docker compose ps`, healthcheck |

## 2. Модельный уровень (качество и дрифт)

| Компонент | SLI | SLO (норма) | Критический уровень | Где наблюдаем |
|---|---|---|---|---|
| Классификатор матча | ROC-AUC на holdout | >= 0.70 (гейт промоута) | < 0.70 -> не промоутить | MLflow, гейт training_pipeline |
| Классификатор матча | live ROC-AUC на свежей разметке | >= 0.65 (SLO) | < 0.65 -> триггер CT | monitoring_pipeline, Pushgateway |
| Классификатор матча | Precision / Recall (порог 0.5) | P и R >= 0.60 | любой < 0.50 | MLflow |
| Фичи | Дрифт распределений (PSI) | доля задрейфивших < 0.5 | >= 0.5 -> триггер CT | Evidently, `uneemi_drift_share` |
| Фичи | PSI отдельного признака (value-skew) | < 0.25 | >= 0.25 -> гейт фич падает | validate_value_skew |
| Фичи | Дрифт по KS (доп. контроль) | p-value стабилен | резкое смещение CDF | Evidently |
| SigLIP-фичестор (якоря) | ImageNet zs top-1 | ~78.48% (паспорт) | < 70% (катастрофа) | `docs/sanity_results.md` |
| SigLIP-фичестор (якоря) | XM3600 RU avg R@1 | ~73.82% (паспорт) | < 30% (катастрофа) | `docs/sanity_results.md` |

## 3. Бизнес-уровень (ценность продукта)

| SLI | SLO (норма) | Критический уровень | Где наблюдаем |
|---|---|---|---|
| m2c_rate (match-to-chat) - главная | >= 0.25 | lift < +40% к tag-based -> пересмотр | event-логи, A/B |
| Swipe-right rate (CTR ленты) | 0.15-0.25 | < 0.10 (плохие реки) или > 0.35 (нет селекции) | event-логи |
| Mutual like rate | >= 0.08 | < 0.05 | event-логи |
| D7 retention | >= 0.25 | падение > 10% от baseline | аналитика |
| D30 retention | >= 0.12 | падение > 10% от baseline | аналитика |
| Report rate (guardrail) | на уровне baseline | рост > 20% -> откат раскатки | модерация/логи |
| Block rate (guardrail) | на уровне baseline | рост > 15% | модерация/логи |
| Time-to-first-match (guardrail) | на уровне baseline | рост > 30% | event-логи |

## Связь уровней и реакции
- Технический breach (p99 > 500 мс или error rate > 0.5%) -> сервинг откатывается на
  Archived (rollback), инцидент виден в Grafana (`uneemi_guardrail_breaches_total`).
- Модельный breach (live AUC < 0.65 или доля дрифта >= 0.5) -> monitoring_pipeline
  триггерит training_pipeline (continuous training).
- Бизнес breach (guardrails: report/block/latency) -> блокировка раскатки модели
  независимо от роста primary-метрики; решение по A/B (см. `docs/metrics_and_benchmarks.md`).
