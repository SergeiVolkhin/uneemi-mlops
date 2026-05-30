"""ONNX-инференс SigLIP 2 vision-энкодера на CPU."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from uneemi_ml.config import EMBED_DIM, ONNX_VISION_PATH, ORT_INTRA_OP_THREADS
from uneemi_ml.preprocess import preprocess_batch, preprocess_image

_INPUT_NAME: str = "pixel_values"
_LOGGER = logging.getLogger(__name__)


class Siglip2Encoder:
    """Класс-обёртка над ONNX Runtime сессией SigLIP 2 vision-энкодера.

    Возвращает `pooler_output` формы (B, EMBED_DIM) в float32.
    """

    def __init__(
        self,
        onnx_path: Path | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        # Ленивый импорт onnxruntime: тяжёлый рантайм инференса нужен только при
        # создании сессии. Так лёгкие потребители контракта признаков
        # (uneemi_ml.features, serving на sklearn) импортируют пакет без onnxruntime.
        import onnxruntime as ort

        path = Path(onnx_path) if onnx_path is not None else ONNX_VISION_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX-модель не найдена: {path}. "
                "Сначала запустите экспорт: `uv run python scripts/export_onnx.py`."
            )

        threads = intra_op_threads if intra_op_threads is not None else ORT_INTRA_OP_THREADS
        _LOGGER.info("Siglip2Encoder: intra_op_num_threads=%d", threads)

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._output_name = self._session.get_outputs()[0].name
        self._path = path
        self._intra_op_threads = threads

    @property
    def onnx_path(self) -> Path:
        return self._path

    @property
    def intra_op_threads(self) -> int:
        return self._intra_op_threads

    def encode(self, image: Image.Image) -> np.ndarray:
        """Закодировать одно изображение → вектор формы (1, EMBED_DIM) float32."""
        pixel_values = preprocess_image(image)
        return self._run(pixel_values)

    def encode_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Закодировать список изображений → массив формы (B, EMBED_DIM) float32."""
        pixel_values = preprocess_batch(images)
        return self._run(pixel_values)

    def _run(self, pixel_values: np.ndarray) -> np.ndarray:
        output = self._session.run([self._output_name], {_INPUT_NAME: pixel_values})[0]
        if output.shape[-1] != EMBED_DIM:
            raise RuntimeError(
                f"Неожиданная размерность выхода ONNX: {output.shape}, "
                f"ожидалось (..., {EMBED_DIM})."
            )
        return output.astype(np.float32, copy=False)
