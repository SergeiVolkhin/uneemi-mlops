"""Экспорт SigLIP 2 vision-энкодера в ONNX.

Прямой `torch.onnx.export` только `vision_model` (без текстовой башни).
Выход графа: единственный тензор `pooler_output` формы (B, 768) в float32.

После экспорта выполняется delta-сверка PyTorch ↔ ONNX на случайном входе:
   max(|torch_out - onnx_out|) должен быть < 1e-4. Нулевой вход не используется
сознательно - он может пройти проверку при поломанной нормализации/слое.

Запуск:
   uv run python scripts/export_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from transformers import AutoModel

# Windows console по умолчанию использует ANSI codepage и падает на Unicode-символах.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# Скрипт лежит вне пакета; добавляем src/ в sys.path, чтобы импортировать config.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from uneemi_ml.config import (  # noqa: E402
    EMBED_DIM,
    IMAGE_SIZE,
    MODEL_ID,
    MODELS_DIR,
    ONNX_VISION_PATH,
)

DELTA_THRESHOLD: float = 1e-4
OPSET_VERSION: int = 17


class VisionPoolerWrapper(nn.Module):
    """Обёртка над siglip vision_model, возвращающая ровно pooler_output (1 выход)."""

    def __init__(self, vision_model: nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values=pixel_values).pooler_output


def export() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Загружаю модель {MODEL_ID} (только vision_model)...")
    full_model = AutoModel.from_pretrained(MODEL_ID)
    wrapper = VisionPoolerWrapper(full_model.vision_model).eval()

    print(f"[2/4] Экспортирую в ONNX: {ONNX_VISION_PATH} (opset={OPSET_VERSION})...")
    torch.manual_seed(0)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(ONNX_VISION_PATH),
            opset_version=OPSET_VERSION,
            input_names=["pixel_values"],
            output_names=["pooler_output"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "pooler_output": {0: "batch"},
            },
            do_constant_folding=True,
            # dynamo=False - используем легаси TorchScript-экспортер, чтобы не тянуть
            # onnxscript в runtime-зависимости (политика минимальных deps).
            dynamo=False,
        )

    size_mb = ONNX_VISION_PATH.stat().st_size / (1024 * 1024)
    print(f"      Файл записан, размер: {size_mb:.1f} MB")

    print("[3/4] Загружаю ONNX-сессию и проверяю форму выхода...")
    session = ort.InferenceSession(
        str(ONNX_VISION_PATH),
        providers=["CPUExecutionProvider"],
    )
    sanity_input = (
        np.random.default_rng(123)
        .standard_normal((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        .astype(np.float32)
    )
    sanity_out = session.run(None, {"pixel_values": sanity_input})[0]
    if sanity_out.shape != (1, EMBED_DIM):
        raise RuntimeError(
            f"Неверная форма выхода ONNX: {sanity_out.shape}, ожидалось (1, {EMBED_DIM})."
        )
    print(f"      shape OK: {sanity_out.shape}")

    print("[4/4] Delta-сверка PyTorch ↔ ONNX на случайном входе (seed=42)...")
    torch.manual_seed(42)
    sample = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        torch_out = wrapper(sample).cpu().numpy()
    onnx_out = session.run(None, {"pixel_values": sample.numpy()})[0]
    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"      max|Δ| = {max_abs_diff:.2e} (порог < {DELTA_THRESHOLD:.0e})")

    if not (max_abs_diff < DELTA_THRESHOLD):
        raise AssertionError(
            f"ONNX-экспорт расходится с PyTorch: max|Δ|={max_abs_diff:.2e} "
            f"(порог {DELTA_THRESHOLD:.0e}). "
            "Возможные причины: opset, dtype, неверная обёртка forward, "
            "несовпадение режима eval/train."
        )

    print(f"\nГотово. Модель: {ONNX_VISION_PATH} ({size_mb:.1f} MB), delta OK.")
    return ONNX_VISION_PATH


def main() -> int:
    if ONNX_VISION_PATH.exists():
        print(f"ONNX-файл уже существует: {ONNX_VISION_PATH}")
        print("Перезаписываю (для повторной delta-сверки на актуальной версии torch)...")
    try:
        export()
    except Exception as exc:
        print(f"\nОШИБКА экспорта: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
