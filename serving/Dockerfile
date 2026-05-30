# Образ сервинга матч-классификатора. Контекст сборки - корень репозитория.
# Лёгкий: sklearn-модель из MLflow, без torch/onnxruntime.
FROM python:3.11-slim

# curl - для HEALTHCHECK контейнера (slim-образ его не содержит).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install --no-cache-dir -r /app/serving/requirements.txt

# Код: ядро признаков, шаги промоута (для отката) и сам сервинг + Feast-репозиторий.
# scripts нужен для bootstrap-модели (тот же образ запускает scripts/bootstrap_model.py).
COPY src/uneemi_ml /app/src/uneemi_ml
COPY training /app/training
COPY serving /app/serving
COPY feature_repo /app/feature_repo
COPY scripts/bootstrap_model.py /app/scripts/bootstrap_model.py

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
