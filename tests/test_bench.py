"""Smoke-тест для bench_inference: импорт и сбор system info.

Полный прогон бенча (5-10 минут) НЕ запускается в pytest - он отдельным
запуском `uv run python scripts/bench_inference.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_bench_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "bench_inference.py"
    spec = importlib.util.spec_from_file_location("bench_inference", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_imports() -> None:
    """Скрипт импортируется без ошибок (нет недостающих deps, синтаксис ок)."""
    _load_bench_module()


def test_get_system_info_keys() -> None:
    """get_system_info() возвращает dict с ожидаемым набором строковых полей."""
    module = _load_bench_module()
    info = module.get_system_info()

    expected = {
        "os",
        "cpu",
        "cores_logical",
        "cores_physical",
        "ram_gb",
        "python",
        "onnxruntime",
        "ort_threads",
    }
    assert expected.issubset(info.keys()), f"отсутствуют ключи: {expected - info.keys()}"
    assert all(isinstance(v, str) for v in info.values()), "все значения должны быть str"
