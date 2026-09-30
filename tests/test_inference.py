"""Unit tests for dms.inference — Detections carrier + backend timing.

Audit checklist:
  - inference.py is thin: no policy logic, no proxy logic
  - Detections validates shapes; .empty() produces correctly-shaped arrays
  - time_inference returns (Detections, time_ms) with positive time
  - filter_score drops sub-threshold detections without mutating original
  - UltralyticsYOLOBackend lazy-imports ultralytics (module imports OK
    even when ultralytics is absent on the host)
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from dms.inference import Detections, time_inference


# ===========================================================================
# Detections carrier
# ===========================================================================
class TestDetections:
    def test_empty_factory(self):
        d = Detections.empty()
        assert len(d) == 0
        assert d.boxes.shape == (0, 4)
        assert d.scores.shape == (0,)
        assert d.classes.shape == (0,)

    def test_basic_construction(self):
        d = Detections(
            boxes=np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float),
            scores=np.array([0.9, 0.7]),
            classes=np.array([0, 0]),
        )
        assert len(d) == 2
        assert d.boxes.shape == (2, 4)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            Detections(
                boxes=np.zeros((3, 4)),
                scores=np.zeros((2,)),
                classes=np.zeros((3,), dtype=int),
            )

    def test_classes_coerced_to_int(self):
        d = Detections(
            boxes=np.zeros((1, 4)),
            scores=np.array([0.5]),
            classes=np.array([1.0]),  # float input
        )
        assert d.classes.dtype.kind == "i"

    def test_filter_score_drops_below_floor(self):
        d = Detections(
            boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]],
                           dtype=float),
            scores=np.array([0.9, 0.1, 0.5]),
            classes=np.array([0, 0, 0]),
        )
        out = d.filter_score(0.3)
        assert len(out) == 2
        assert out.scores.tolist() == [0.9, 0.5]

    def test_filter_score_does_not_mutate_original(self):
        d = Detections(
            boxes=np.zeros((2, 4)),
            scores=np.array([0.1, 0.9]),
            classes=np.array([0, 0]),
        )
        _ = d.filter_score(0.5)
        # Original retains both entries.
        assert len(d) == 2

    def test_filter_score_inclusive_at_threshold(self):
        d = Detections(
            boxes=np.zeros((2, 4)),
            scores=np.array([0.25, 0.24]),
            classes=np.array([0, 0]),
        )
        out = d.filter_score(0.25)
        assert len(out) == 1
        assert out.scores[0] == pytest.approx(0.25)


# ===========================================================================
# Time inference
# ===========================================================================
class FakeBackend:
    """Minimal backend for unit testing — no GPU, no model weights."""

    def __init__(self, detections: Detections, sleep_s: float = 0.001):
        self._dets = detections
        self._sleep = sleep_s
        self.call_count = 0
        self.last_frame_shape = None

    def predict(self, frame_bgr):
        self.call_count += 1
        self.last_frame_shape = frame_bgr.shape
        if self._sleep:
            time.sleep(self._sleep)
        return self._dets


class TestTimeInference:
    def test_returns_detections_and_positive_time(self):
        dets = Detections(
            boxes=np.array([[0, 0, 10, 10]], dtype=float),
            scores=np.array([0.8]),
            classes=np.array([0]),
        )
        backend = FakeBackend(dets, sleep_s=0.002)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out, ms = time_inference(backend, frame)
        assert out is dets
        assert ms > 0
        # The sleep was 2 ms; the measurement should be at least that.
        assert ms >= 1.5  # allow some slack for clock granularity

    def test_calls_backend_exactly_once(self):
        backend = FakeBackend(Detections.empty(), sleep_s=0)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        time_inference(backend, frame)
        assert backend.call_count == 1
        assert backend.last_frame_shape == (50, 50, 3)

    def test_handles_empty_detections(self):
        backend = FakeBackend(Detections.empty(), sleep_s=0)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        out, ms = time_inference(backend, frame)
        assert len(out) == 0
        assert ms >= 0


# ===========================================================================
# UltralyticsYOLOBackend — adapter shape only (no real model load)
# ===========================================================================
class TestUltralyticsYOLOBackend:
    def test_module_imports_without_ultralytics_installed(self):
        # Importing dms.inference must not require ultralytics — the
        # YOLO import is local to the constructor. The test passes by
        # virtue of the file being importable at collection time.
        import dms.inference as inf  # noqa: F401
        assert hasattr(inf, "UltralyticsYOLOBackend")

    def test_constructor_lazy_imports_ultralytics(self):
        # If ultralytics is not installed, constructing the backend
        # should raise ImportError. If it IS installed, constructing
        # with a bogus weights path should raise some other error
        # (FileNotFoundError or ultralytics-internal). Either is fine —
        # the test just confirms the import path is exercised at
        # construction time, not at module import time.
        from dms.inference import UltralyticsYOLOBackend
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError):
                UltralyticsYOLOBackend("/nonexistent/path.pt")
        else:
            with pytest.raises(Exception):
                UltralyticsYOLOBackend("/nonexistent/path.pt")
