"""Thin inference wrapper — keeps model I/O out of the policy library.

Architecture
------------
A backend is anything with the shape::

    class Backend(Protocol):
        def predict(self, frame_bgr: np.ndarray) -> Detections: ...

Experiment runners construct one backend per model in a pair (fast and
accurate) and route per-frame calls to whichever the active policy
selected. The ``Detections`` data class is a plain numpy carrier — no
torch, no ultralytics, no opencv on the public surface — so downstream
code in ``dms.matching`` / ``dms.metrics`` can consume it directly.

The :class:`UltralyticsYOLOBackend` adapter lazy-imports ultralytics so
unit tests can import :mod:`dms.inference` even on hosts that have no
``ultralytics`` package installed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np


# ===========================================================================
# Detection carrier
# ===========================================================================
@dataclass
class Detections:
    """Detection batch in xyxy pixel coordinates.

    Attributes:
        boxes:   ``(N, 4)`` float, ``[x1, y1, x2, y2]``.
        scores:  ``(N,)`` float in ``[0, 1]``.
        classes: ``(N,)`` int.
    """

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    def __post_init__(self) -> None:
        b = np.asarray(self.boxes, dtype=float).reshape(-1, 4)
        s = np.asarray(self.scores, dtype=float).reshape(-1)
        c = np.asarray(self.classes).reshape(-1).astype(int)
        if not (b.shape[0] == s.shape[0] == c.shape[0]):
            raise ValueError(
                f"length mismatch: boxes={b.shape[0]}, scores={s.shape[0]}, "
                f"classes={c.shape[0]}"
            )
        self.boxes = b
        self.scores = s
        self.classes = c

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @classmethod
    def empty(cls) -> "Detections":
        return cls(
            boxes=np.zeros((0, 4), dtype=float),
            scores=np.zeros((0,), dtype=float),
            classes=np.zeros((0,), dtype=int),
        )

    def filter_score(self, conf_floor: float) -> "Detections":
        """Return a new Detections containing only entries at or above conf_floor."""
        keep = self.scores >= float(conf_floor)
        return Detections(
            boxes=self.boxes[keep],
            scores=self.scores[keep],
            classes=self.classes[keep],
        )


# ===========================================================================
# Backend protocol
# ===========================================================================
class InferenceBackend(Protocol):
    """Anything with a ``predict(frame_bgr) -> Detections`` method."""

    def predict(self, frame_bgr: np.ndarray) -> Detections: ...  # pragma: no cover


# ===========================================================================
# Timing helper
# ===========================================================================
def time_inference(
    backend: InferenceBackend,
    frame_bgr: np.ndarray,
) -> Tuple[Detections, float]:
    """Run ``backend.predict(frame_bgr)`` and return ``(detections, time_ms)``.

    Wall-clock measurement only. Subtracting input transfer time or the
    model's internal pre/post-processing is the backend's responsibility.
    """
    t0 = time.perf_counter()
    dets = backend.predict(frame_bgr)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return dets, float(elapsed_ms)


# ===========================================================================
# Concrete backend: ultralytics YOLO
# ===========================================================================
class UltralyticsYOLOBackend:
    """Thin adapter around ``ultralytics.YOLO``.

    Lazy-imports ``ultralytics`` so that importing
    :mod:`dms.inference` itself does not require the package. The
    constructor will raise ``ImportError`` if ultralytics is missing.

    The same class loads ``.pt`` (PyTorch), ``.onnx``, and ``.engine``
    (TensorRT) files — ultralytics dispatches by extension.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou_nms: float = 0.7,
    ):
        from ultralytics import YOLO  # local import for testability
        self._YOLO = YOLO
        self._model = YOLO(weights_path)
        self.weights_path = str(weights_path)
        self.device = str(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou_nms = float(iou_nms)

    def predict(self, frame_bgr: np.ndarray) -> Detections:
        results = self._model.predict(
            source=frame_bgr,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou_nms,
            verbose=False,
        )
        if not results:
            return Detections.empty()
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return Detections.empty()
        boxes = result.boxes.xyxy.cpu().numpy().astype(float)
        scores = result.boxes.conf.cpu().numpy().astype(float)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        return Detections(boxes=boxes, scores=scores, classes=classes)


__all__ = [
    "Detections",
    "InferenceBackend",
    "time_inference",
    "UltralyticsYOLOBackend",
]
