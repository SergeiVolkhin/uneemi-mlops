"""FastAPI-сервинг матч-классификатора (C8): /health, /predict, /metrics.

Ключевые свойства по ТЗ:
- Горячее переключение трафика БЕЗ рестарта: фоновый поток раз в POLL_INTERVAL_SEC
  опрашивает MLflow Registry на текущую Production-версию и при смене атомарно
  подменяет модель под блокировкой. Запросы в полёте дорабатывают на старой версии.
- /health отдаёт 200 только когда Production-модель загружена (иначе 503) - это и
  есть цель healthcheck контейнера.
- Guardrails: при срыве скользящего p99 latency или доли ошибок инициируется откат
  на последнюю Archived-версию (rollback), который тут же подхватывает поллер.

Модель - sklearn-пайплайн из MLflow, поэтому serving лёгкий (без torch/onnxruntime).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from mlflow.tracking import MlflowClient
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, model_validator

from training.promote import STAGE_PRODUCTION, rollback_to_archived
from uneemi_ml.features import EMBED_DIM, QUIZ_DIM, build_pair_features, build_profile_vector

_LOGGER = logging.getLogger("uneemi.serving")

MODEL_NAME = os.environ.get("MODEL_NAME", "uneemi_match")
POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "15"))
GUARDRAIL_P99_MS = float(os.environ.get("GUARDRAIL_P99_MS", "500"))
GUARDRAIL_ERROR_RATE = float(os.environ.get("GUARDRAIL_ERROR_RATE", "0.005"))
GUARDRAIL_WINDOW = int(os.environ.get("GUARDRAIL_WINDOW", "200"))
# Демо-хуки (инъекция неисправности) по умолчанию ВЫКЛЮЧЕНЫ. Включаются только для
# показа guardrail-отката флагом ENABLE_DEMO_HOOKS=true в .env - в проде их нет.
DEMO_HOOKS_ENABLED = os.environ.get("ENABLE_DEMO_HOOKS", "false").lower() in ("1", "true", "yes")

# --- Метрики Prometheus -----------------------------------------------------
PREDICT_LATENCY = Histogram(
    "uneemi_predict_latency_seconds",
    "Латентность /predict",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)
REQUESTS = Counter("uneemi_requests_total", "Запросы по эндпоинтам", ["endpoint", "status"])
IN_FLIGHT = Gauge("uneemi_in_flight_requests", "Запросы в обработке")
MODEL_VERSION_GAUGE = Gauge("uneemi_model_version", "Номер загруженной Production-версии")
MODEL_LOADED = Gauge("uneemi_model_loaded", "1 если Production-модель загружена, иначе 0")
GUARDRAIL_BREACHES = Counter("uneemi_guardrail_breaches_total", "Срывы guardrails (откаты)")


class _ModelState:
    """Потокобезопасный держатель текущей модели и её версии."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.model = None
        self.version: int | None = None

    def get(self):
        with self._lock:
            return self.model, self.version

    def set(self, model, version: int) -> None:
        with self._lock:
            self.model = model
            self.version = version


STATE = _ModelState()
_STOP = threading.Event()
# Скользящее окно (latency_ms, is_error) для guardrails.
_WINDOW: deque[tuple[float, bool]] = deque(maxlen=GUARDRAIL_WINDOW)
_WINDOW_LOCK = threading.Lock()
_rollback_in_progress = threading.Event()
# Демо-инъекция неисправности (только для показа guardrail-отката).
_FAULT = {"latency_ms": 0.0, "count": 0}


def _production_version(client: MlflowClient) -> int | None:
    versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE_PRODUCTION])
    return int(versions[0].version) if versions else None


def _load_version(version: int):
    """Загрузить модель конкретной версии из реестра (артефакт тянется из MinIO)."""
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


def _refresh_model(client: MlflowClient) -> None:
    """Сверить Production-версию с загруженной и при расхождении горячо подменить."""
    target = _production_version(client)
    if target is None:
        return
    _, current = STATE.get()
    if target == current:
        return
    _LOGGER.info("Обнаружена новая Production-версия %s (была %s), загружаю", target, current)
    model = _load_version(target)
    STATE.set(model, target)
    MODEL_VERSION_GAUGE.set(target)
    MODEL_LOADED.set(1)
    _LOGGER.info("Модель версии %s загружена и активна", target)


def _poll_loop() -> None:
    """Фоновый поллер: периодически подтягивает текущую Production-версию."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()
    while not _STOP.is_set():
        try:
            _refresh_model(client)
        except Exception:  # noqa: BLE001 - сеть/реестр капризны, держим старую модель
            _LOGGER.exception("Поллер: не удалось обновить модель (держу текущую)")
        _STOP.wait(POLL_INTERVAL_SEC)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def _check_guardrails() -> None:
    """Проверить скользящие p99 latency и долю ошибок; при срыве - откат на Archived."""
    if _rollback_in_progress.is_set():
        return
    with _WINDOW_LOCK:
        if len(_WINDOW) < _WINDOW.maxlen:
            return
        latencies = [lm for lm, _ in _WINDOW]
        error_rate = sum(1 for _, e in _WINDOW if e) / len(_WINDOW)
    p99 = _percentile(latencies, 99)
    if p99 <= GUARDRAIL_P99_MS and error_rate <= GUARDRAIL_ERROR_RATE:
        return
    _LOGGER.warning(
        "Срыв guardrails: p99=%.1fмс (порог %.0f), error_rate=%.4f (порог %.4f) - откат",
        p99, GUARDRAIL_P99_MS, error_rate, GUARDRAIL_ERROR_RATE,
    )
    GUARDRAIL_BREACHES.inc()
    _rollback_in_progress.set()
    threading.Thread(target=_do_rollback, daemon=True).start()


def _do_rollback() -> None:
    """Выполнить откат на Archived и сразу подтянуть новую Production-версию."""
    try:
        target = rollback_to_archived(MODEL_NAME)
        if target is not None:
            _refresh_model(MlflowClient())
        with _WINDOW_LOCK:
            _WINDOW.clear()  # после отката окно стартует заново
    except Exception:  # noqa: BLE001
        _LOGGER.exception("Откат не удался")
    finally:
        _rollback_in_progress.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    MODEL_LOADED.set(0)
    poller = threading.Thread(target=_poll_loop, name="mlflow-poller", daemon=True)
    poller.start()
    _LOGGER.info("Сервинг запущен, поллер Production активен (интервал %dс)", POLL_INTERVAL_SEC)
    yield
    _STOP.set()


app = FastAPI(title="Uneemi match serving", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Пара профилей: либо id (достаём из Feast online), либо явные векторы (демо)."""

    profile_a_id: int | None = None
    profile_b_id: int | None = None
    board_emb_a: list[float] | None = None
    quiz_a: list[float] | None = None
    board_emb_b: list[float] | None = None
    quiz_b: list[float] | None = None

    @model_validator(mode="after")
    def _check_inputs(self):
        has_ids = self.profile_a_id is not None and self.profile_b_id is not None
        has_vecs = all(
            v is not None
            for v in (self.board_emb_a, self.quiz_a, self.board_emb_b, self.quiz_b)
        )
        if not (has_ids or has_vecs):
            raise ValueError("Нужны либо profile_a_id+profile_b_id, либо все четыре вектора")
        return self


class PredictResponse(BaseModel):
    p_match: float
    model_version: int
    latency_ms: float


def _profile_from_vectors(board: list[float], quiz: list[float]) -> np.ndarray:
    board_arr = np.asarray(board, dtype=np.float32)
    quiz_arr = np.asarray(quiz, dtype=np.float32)
    if board_arr.shape[0] != EMBED_DIM or quiz_arr.shape[0] != QUIZ_DIM:
        raise HTTPException(422, f"Ожидались board {EMBED_DIM}d и quiz {QUIZ_DIM}d")
    return build_profile_vector(board_arr, quiz_arr)


def _profiles_from_feast(ids: list[int]) -> dict[int, np.ndarray]:
    """Достать профильные векторы из Feast online по id (ленивая инициализация store)."""
    from feast import FeatureStore

    from uneemi_ml.features import QUIZ_FEATURES

    repo = os.environ.get("FEAST_REPO_PATH", "/opt/feature_repo")
    store = FeatureStore(repo_path=repo)
    feats = ["profile_features:board_emb"] + [f"profile_features:{q}" for q in QUIZ_FEATURES]
    resp = store.get_online_features(
        features=feats, entity_rows=[{"profile_id": i} for i in ids]
    ).to_dict()
    out: dict[int, np.ndarray] = {}
    for k, pid in enumerate(resp["profile_id"]):
        raw_board = resp["board_emb"][k]
        if raw_board is None:  # промах online-store по этому профилю
            raise HTTPException(404, f"Профиль {pid} не найден в online-store")
        board = np.asarray(raw_board, dtype=np.float32)
        quiz = np.asarray([resp[q][k] for q in QUIZ_FEATURES], dtype=np.float32)
        out[int(pid)] = build_profile_vector(board, quiz)
    return out


@app.get("/health")
def health() -> JSONResponse:
    """Готовность: 200 только если Production-модель загружена."""
    model, version = STATE.get()
    if model is None:
        REQUESTS.labels("health", "503").inc()
        return JSONResponse(status_code=503, content={"status": "no_model"})
    REQUESTS.labels("health", "200").inc()
    return JSONResponse(status_code=200, content={"status": "ok", "model_version": version})


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model, version = STATE.get()
    if model is None:
        REQUESTS.labels("predict", "503").inc()
        raise HTTPException(503, "Production-модель ещё не загружена")

    start = time.perf_counter()
    is_error = False
    p_match: float | None = None
    IN_FLIGHT.inc()
    try:
        if req.board_emb_a is not None:
            prof_a = _profile_from_vectors(req.board_emb_a, req.quiz_a)
            prof_b = _profile_from_vectors(req.board_emb_b, req.quiz_b)
        else:
            profiles = _profiles_from_feast([req.profile_a_id, req.profile_b_id])
            prof_a = profiles[req.profile_a_id]
            prof_b = profiles[req.profile_b_id]

        pair = build_pair_features(prof_a, prof_b).reshape(1, -1)

        # Демо-инъекция задержки для показа guardrail-отката (только если включена).
        if _FAULT["count"] > 0:
            _FAULT["count"] -= 1
            time.sleep(_FAULT["latency_ms"] / 1000.0)

        p_match = float(model.predict_proba(pair)[0, 1])
        REQUESTS.labels("predict", "200").inc()
    except HTTPException:
        is_error = True
        REQUESTS.labels("predict", "4xx").inc()
        raise
    except Exception as exc:  # noqa: BLE001
        is_error = True
        REQUESTS.labels("predict", "500").inc()
        _LOGGER.exception("Ошибка инференса")
        raise HTTPException(500, f"Ошибка инференса: {exc}") from exc
    finally:
        latency_ms = (time.perf_counter() - start) * 1000.0
        PREDICT_LATENCY.observe(latency_ms / 1000.0)
        IN_FLIGHT.dec()
        with _WINDOW_LOCK:
            _WINDOW.append((latency_ms, is_error))
        _check_guardrails()

    return PredictResponse(p_match=p_match, model_version=int(version), latency_ms=latency_ms)


@app.post("/admin/inject_latency")
def inject_latency(ms: float = 600.0, count: int = 300) -> dict:
    """Демо-хук: заставить следующие count запросов /predict спать ms миллисекунд.

    Нужен только чтобы воспроизводимо показать срабатывание guardrails и откат.
    Доступен лишь при ENABLE_DEMO_HOOKS=true; в проде эндпоинт отвечает 403 -
    это не точка управления трафиком, а тестовый рубильник.
    """
    if not DEMO_HOOKS_ENABLED:
        raise HTTPException(403, "Демо-хуки выключены (ENABLE_DEMO_HOOKS != true)")
    _FAULT["latency_ms"] = ms
    _FAULT["count"] = count
    _LOGGER.warning("Инъекция задержки %.0fмс на %d запросов (демо guardrails)", ms, count)
    return {"injected_latency_ms": ms, "for_requests": count}
