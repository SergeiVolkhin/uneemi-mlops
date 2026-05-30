# Makefile оркестрации стека Uneemi MLOps.
# Запускать из каталога infra/ (make -C infra <target>) или из CI.
# На Windows без установленного make используйте infra/make.ps1 (те же цели).
#
# Используем .RECIPEPREFIX = > вместо TAB - устойчивее к редактированию на Windows.
.RECIPEPREFIX = >
.DEFAULT_GOAL := help

# Все docker-compose вызовы идут с единым env-файлом и compose-файлом.
DC := docker compose --env-file .env -f docker-compose.yml
ROOT := ..
ONNX := ../models/siglip2_vision.onnx
IMAGES_MARKER := ../data/images/.done

.PHONY: help up down teardown build logs ps smoke e2e test export-onnx fetch-images env

help: ## Показать список целей
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

env: ## Создать infra/.env из .env.example, если ещё нет
> @test -f .env || (cp .env.example .env && echo "infra/.env создан из .env.example")

export-onnx: ## Экспортировать SigLIP 2 vision в ONNX (идемпотентно: пропуск если файл есть)
> @test -f $(ONNX) && echo "ONNX уже на месте: $(ONNX) (пропуск)" || (cd $(ROOT) && uv run python scripts/export_onnx.py)

fetch-images: ## Скачать банк демо-изображений (идемпотентно: пропуск если уже скачаны)
> @test -f $(IMAGES_MARKER) && echo "Банк картинок уже на месте (пропуск)" || (cd $(ROOT) && uv run python scripts/fetch_demo_images.py)

build: ## Собрать кастомные образы (airflow, mlflow, serving)
> $(DC) build

up: env export-onnx fetch-images build ## Поднять весь стек (идемпотентная подготовка артефактов + compose up -d)
> $(DC) up -d
> @echo "Стек поднимается. Дождитесь healthy: make ps. Первый запуск может занять несколько минут."

down: ## Остановить стек (тома сохраняются)
> $(DC) down

teardown: ## Полный снос: контейнеры + тома (данные потеряются)
> $(DC) down -v

logs: ## Хвост логов всех сервисов
> $(DC) logs -f --tail=100

ps: ## Статус сервисов (ищем Up healthy)
> $(DC) ps

smoke: ## Дымовой тест: health всех сервисов
> $(DC) exec -T serving python /app/smoke.py || bash ./smoke.sh

e2e: ## End-to-end: прогон пайплайнов + проверка промоута/дрифта/гейта
> bash ./e2e.sh

test: ## Линт и юнит-тесты ядра (на хосте)
> cd $(ROOT) && uv run ruff check . && uv run pytest -q
