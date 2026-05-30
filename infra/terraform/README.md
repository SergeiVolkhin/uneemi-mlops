# Terraform-скелет облачного деплоя (Yandex Cloud)

СТАТУС: скелет. Это открытый гейт 2-го балла критерия 3 - реальный облачный
деплой выполняется отдельным шагом перед устной защитой.

## Что делает
Поднимает один VPS в Yandex Cloud, ставит Docker и через cloud-init запускает весь
стек `docker compose up -d`. Группа безопасности открывает порты сервисов (serving
18000, airflow 18080, grafana 13000, mlflow 5500, prometheus 19090) и SSH.

## Применение
1. Установить Terraform и YC CLI, получить токен/cloud_id/folder_id.
2. `cp terraform.tfvars.example terraform.tfvars` и заполнить (токен, ключ, repo_url).
3. `terraform init && terraform apply`.
4. Внешний IP и URL сервисов - в `terraform output`.

## Донастройка ONNX
SigLIP ONNX (~370MB) на VM нужно подготовить отдельно (он gitignored):
- вариант А: на VM `cd /opt/uneemi && uv sync && uv run python scripts/export_onnx.py`
  до `docker compose up` (тянет torch, одноразово);
- вариант Б: положить готовый `models/siglip2_vision.onnx` в объектное хранилище
  и скачать в cloud-init перед подъёмом стека.

## Безопасность (важно для прод/облака)
Локальный стек использует dev-дефолты (admin/admin, секреты из .env.example) -
это осознанно для подъёма одной командой. Для облака:
- `ssh_allowed_cidr` без дефолта: оператор обязан задать свой IP/CIDR (fail-safe).
- cloud-init НЕ копирует .env.example: реальный .env с уникальными секретами
  оператор подкладывает сам (секрет-стор / file-provisioner), иначе стек не
  поднимется (fail-closed). Перегенерировать FERNET_KEY, пароли Airflow/Grafana.
- Админ-сервисы (MLflow, Airflow, Prometheus, Grafana) НЕ выставлять в 0.0.0.0/0:
  держать за reverse-proxy с аутентификацией или bastion; для MLflow включить
  mlflow basic-auth. Группа безопасности по умолчанию ограничена `ssh_allowed_cidr`.
- Клон репозитория пинуется на ref (`repo_ref`), а не на плавающий main.

## Снос
`terraform destroy` (соответствует требованию деинсталляции инфраструктуры).
