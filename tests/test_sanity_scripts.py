"""Smoke-тесты для скриптов sanity-бенчмарков.

Полные бенчмарки занимают 70+ минут и не подходят для pytest. Здесь - только:
1. Импорт модулей (защита от import-time SyntaxError / NameError на module-load).
2. Unit на функции `evaluate_targets` - границы порогов Pass / Parity / Catastrophic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR: Path = _PROJECT_ROOT / "scripts"


def _import_script(name: str):
    """Импортировать `scripts/<name>.py` как модуль без модификации sys.path проекта."""
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sanity_imagenet():
    return _import_script("sanity_imagenet")


@pytest.fixture(scope="module")
def sanity_xm3600_ru():
    return _import_script("sanity_xm3600_ru")


# --- Импорт-smoke ---


def test_sanity_imagenet_imports(sanity_imagenet) -> None:
    """Модуль импортируется + ключевые символы экспортируются."""
    assert callable(sanity_imagenet.main)
    assert callable(sanity_imagenet.evaluate_targets)
    assert callable(sanity_imagenet.normalize_for_siglip2)
    assert isinstance(sanity_imagenet.IMAGENET_TEMPLATES, tuple)
    assert len(sanity_imagenet.IMAGENET_TEMPLATES) == 80


def test_sanity_xm3600_ru_imports(sanity_xm3600_ru) -> None:
    """Модуль импортируется + ключевые символы экспортируются."""
    assert callable(sanity_xm3600_ru.main)
    assert callable(sanity_xm3600_ru.evaluate_targets)
    assert callable(sanity_xm3600_ru.evaluate_retrieval)


# --- evaluate_targets: ImageNet ---


def test_imagenet_targets_pass_and_parity(sanity_imagenet) -> None:
    """top-1 0.78, N=5000 → pass=T, parity=T (в окне [0.767, 0.787])."""
    r = sanity_imagenet.evaluate_targets(top1=0.78, subset_size=5000)
    assert r["pass"] is True
    assert r["pass_status"] == "PASS"
    assert r["parity"] is True
    assert r["catastrophic"] is False


def test_imagenet_targets_pass_no_parity(sanity_imagenet) -> None:
    """top-1 0.76, N=5000 → pass=T (≥0.75), parity=F (вне окна)."""
    r = sanity_imagenet.evaluate_targets(top1=0.76, subset_size=5000)
    assert r["pass"] is True
    assert r["parity"] is False


def test_imagenet_targets_below_pass(sanity_imagenet) -> None:
    """top-1 0.70, N=5000 → pass=F, parity=F."""
    r = sanity_imagenet.evaluate_targets(top1=0.70, subset_size=5000)
    assert r["pass"] is False
    assert r["pass_status"] == "FAIL"
    assert r["parity"] is False
    assert r["catastrophic"] is False


def test_imagenet_targets_catastrophic(sanity_imagenet) -> None:
    """top-1 0.30 → catastrophic=T."""
    r = sanity_imagenet.evaluate_targets(top1=0.30, subset_size=5000)
    assert r["catastrophic"] is True
    assert r["pass"] is False


def test_imagenet_targets_smoke_gate(sanity_imagenet) -> None:
    """N<5000 → pass=F принудительно (smoke run не зачитывается)."""
    r = sanity_imagenet.evaluate_targets(top1=0.99, subset_size=100)
    assert r["pass"] is False
    assert r["pass_status"] == "n/a (smoke run)"


# --- evaluate_targets: XM3600 RU (parity intentionally omitted) ---


def test_xm3600_targets_pass(sanity_xm3600_ru) -> None:
    """avg R@1 0.74, N=3600 → pass=T."""
    r = sanity_xm3600_ru.evaluate_targets(avg_r1=0.74, n_images=3600)
    assert r["pass"] is True
    assert r["pass_status"] == "PASS"
    assert r["catastrophic_avg_r1"] is False


def test_xm3600_targets_below_pass(sanity_xm3600_ru) -> None:
    """avg R@1 0.30, N=3600 → pass=F (ниже 0.35)."""
    r = sanity_xm3600_ru.evaluate_targets(avg_r1=0.30, n_images=3600)
    assert r["pass"] is False
    assert r["pass_status"] == "FAIL"


def test_xm3600_targets_catastrophic(sanity_xm3600_ru) -> None:
    """avg R@1 0.10 → catastrophic=T."""
    r = sanity_xm3600_ru.evaluate_targets(avg_r1=0.10, n_images=3600)
    assert r["catastrophic_avg_r1"] is True
    assert r["pass"] is False


def test_xm3600_targets_smoke_gate(sanity_xm3600_ru) -> None:
    """N<3000 → pass=F принудительно, status=n/a (smoke)."""
    r = sanity_xm3600_ru.evaluate_targets(avg_r1=0.95, n_images=100)
    assert r["pass"] is False
    assert r["pass_status"] == "n/a (smoke run)"


def test_xm3600_targets_no_parity_key(sanity_xm3600_ru) -> None:
    """Parity intentionally omitted - ключа в результате быть не должно.

    Гарант, что мы не пропустили parity-логику в evaluate_targets обратно.
    """
    r = sanity_xm3600_ru.evaluate_targets(avg_r1=0.74, n_images=3600)
    assert "parity" not in r
