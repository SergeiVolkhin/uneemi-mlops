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

## Снос
`terraform destroy` (соответствует требованию деинсталляции инфраструктуры).
