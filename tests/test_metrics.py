"""Unit tests for dms.metrics — per-frame metrics and run aggregation.

Audit checklist:
  - iou_match: strict perfect-1:1 at threshold (precision AND recall)
  - count_agree: cheap count-only secondary
  - benefit_positive: locked definition (accurate has detection that
    cannot be matched to any fast detection at IoU >= threshold)
  - latency_summary handles empty gracefully
  - summarise_run rolls per-frame records to a single row
"""
from __future__ import annotations

import numpy as np
import pytest

from dms.metrics import (
    benefit_positive_frame,
    count_agree_frame,
    det_coverage_frame,
    det_recall_frame,
    iou_match_frame,
    latency_summary,
    summarise_run,
)


def _box(*coords):
    return np.array([coords], dtype=float)


def _boxes(*rows):
    return np.array(rows, dtype=float)


# ===========================================================================
# iou_match_frame
# ===========================================================================
class TestIoUMatchFrame:
    def test_empty_both_sides(self):
        empty = np.zeros((0, 4))
        assert iou_match_frame(empty, empty) is True

    def test_identical_one_box(self):
        a = _box(0, 0, 10, 10)
        assert iou_match_frame(a, a) is True

    def test_off_by_missing_detection(self):
        # Policy missed one of two reference detections.
        policy = _boxes([0, 0, 10, 10])
        ref = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        assert iou_match_frame(policy, ref) is False

    def test_off_by_extra_detection(self):
        policy = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        ref = _boxes([0, 0, 10, 10])
        assert iou_match_frame(policy, ref) is False

    def test_low_iou_does_not_match(self):
        # IoU 1/3 < default 0.5
        policy = _boxes([0, 0, 10, 10])
        ref = _boxes([5, 0, 15, 10])
        assert iou_match_frame(policy, ref) is False

    def test_low_iou_matches_with_loose_threshold(self):
        policy = _boxes([0, 0, 10, 10])
        ref = _boxes([5, 0, 15, 10])
        assert iou_match_frame(policy, ref, iou_threshold=0.3) is True

    def test_two_boxes_perfect_match(self):
        boxes = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        assert iou_match_frame(boxes, boxes) is True

    def test_class_mismatch_does_not_match(self):
        policy = _boxes([0, 0, 10, 10])
        ref = _boxes([0, 0, 10, 10])
        out = iou_match_frame(
            policy, ref,
            policy_classes=np.array([0]),
            ref_classes=np.array([1]),
        )
        assert out is False


# ===========================================================================
# det_recall_frame  /  det_coverage_frame  (relaxed, recall-only)
# ===========================================================================
class TestDetRecallFrame:
    def test_empty_both_full_recall(self):
        empty = np.zeros((0, 4))
        assert det_recall_frame(empty, empty) == pytest.approx(1.0)

    def test_no_policy_with_ref_zero_recall(self):
        empty = np.zeros((0, 4))
        ref = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        assert det_recall_frame(empty, ref) == pytest.approx(0.0)

    def test_no_ref_with_policy_full_recall(self):
        # No reference -> nothing to recall, treat as vacuously full.
        policy = _boxes([0, 0, 10, 10])
        empty = np.zeros((0, 4))
        assert det_recall_frame(policy, empty) == pytest.approx(1.0)

    def test_partial_recall(self):
        # Policy covers one of two reference detections.
        policy = _boxes([0, 0, 10, 10])
        ref = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        assert det_recall_frame(policy, ref) == pytest.approx(0.5)

    def test_extra_policy_boxes_not_penalised(self):
        # Recall-only: extra policy boxes are tolerated (unlike iou_match).
        policy = _boxes([0, 0, 10, 10], [100, 100, 110, 110])
        ref = _boxes([0, 0, 10, 10])
        assert det_recall_frame(policy, ref) == pytest.approx(1.0)
        assert iou_match_frame(policy, ref) is False  # strict still fails

    def test_full_recall(self):
        boxes = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        assert det_recall_frame(boxes, boxes) == pytest.approx(1.0)


class TestDetCoverageFrame:
    def test_full_recall_is_coverage(self):
        boxes = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        assert det_coverage_frame(boxes, boxes) is True

    def test_partial_recall_is_not_coverage(self):
        policy = _boxes([0, 0, 10, 10])
        ref = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        assert det_coverage_frame(policy, ref) is False

    def test_extra_policy_boxes_still_coverage(self):
        policy = _boxes([0, 0, 10, 10], [100, 100, 110, 110])
        ref = _boxes([0, 0, 10, 10])
        assert det_coverage_frame(policy, ref) is True

    def test_coverage_is_relaxation_of_iou_match(self):
        # iou_match=True implies coverage=True; the converse is not true.
        # Construct a frame where coverage holds but iou_match does not.
        policy = _boxes([0, 0, 10, 10], [100, 100, 110, 110])
        ref = _boxes([0, 0, 10, 10])
        assert iou_match_frame(policy, ref) is False
        assert det_coverage_frame(policy, ref) is True


# ===========================================================================
# count_agree_frame
# ===========================================================================
class TestCountAgreeFrame:
    def test_same_count_true(self):
        # Boxes do NOT need to overlap; count_agree is count-only.
        a = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        b = _boxes([100, 100, 110, 110], [200, 200, 210, 210])
        assert count_agree_frame(a, b) is True

    def test_diff_count_false(self):
        a = _boxes([0, 0, 10, 10])
        b = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        assert count_agree_frame(a, b) is False

    def test_empty_both_true(self):
        empty = np.zeros((0, 4))
        assert count_agree_frame(empty, empty) is True

    def test_count_agree_is_strictly_weaker_than_iou_match(self):
        # Same count, but boxes are spatially disjoint -> count_agree
        # passes, iou_match fails. This documents the relationship.
        a = _boxes([0, 0, 10, 10])
        b = _boxes([100, 100, 110, 110])
        assert count_agree_frame(a, b) is True
        assert iou_match_frame(a, b) is False


# ===========================================================================
# benefit_positive_frame — LOCKED definition
# ===========================================================================
class TestBenefitPositiveFrame:
    """The locked thesis definition of benefit-positive.

    'A frame is benefit-positive iff the accurate model produces at
    least one valid detection that cannot be matched to any fast-model
    detection at IoU >= 0.5.'
    """

    def test_empty_accurate_no_benefit(self):
        empty = np.zeros((0, 4))
        fast = _boxes([0, 0, 10, 10])
        # Accurate found nothing; no recall benefit by definition.
        assert benefit_positive_frame(fast, empty) is False

    def test_accurate_finds_extra_is_benefit(self):
        fast = _boxes([0, 0, 10, 10])
        accurate = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        # The (50,50,60,60) accurate detection cannot be matched to any
        # fast detection at IoU >= 0.5.
        assert benefit_positive_frame(fast, accurate) is True

    def test_accurate_subset_of_fast_no_benefit(self):
        # Fast finds everything accurate finds (and more) -> no benefit.
        fast = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        accurate = _boxes([0, 0, 10, 10])
        assert benefit_positive_frame(fast, accurate) is False

    def test_no_fast_detections_full_benefit(self):
        empty = np.zeros((0, 4))
        accurate = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        assert benefit_positive_frame(empty, accurate) is True

    def test_low_iou_counts_as_unmatched(self):
        # Fast and accurate overlap at IoU 1/3, below default 0.5.
        fast = _boxes([0, 0, 10, 10])
        accurate = _boxes([5, 0, 15, 10])
        assert benefit_positive_frame(fast, accurate, iou_threshold=0.5) is True
        assert benefit_positive_frame(fast, accurate, iou_threshold=0.3) is False

    def test_class_filter_independence(self):
        # Same boxes, different classes -> accurate cannot match fast.
        fast = _boxes([0, 0, 10, 10])
        accurate = _boxes([0, 0, 10, 10])
        out = benefit_positive_frame(
            fast, accurate,
            fast_classes=np.array([0]),
            accurate_classes=np.array([1]),
        )
        assert out is True


# ===========================================================================
# latency_summary
# ===========================================================================
class TestLatencySummary:
    def test_empty(self):
        s = latency_summary([])
        assert s["n"] == 0
        assert s["mean"] == 0.0
        assert s["fps"] == 0.0
        assert s["p50"] == 0.0

    def test_constant(self):
        s = latency_summary([10.0, 10.0, 10.0, 10.0])
        assert s["n"] == 4
        assert s["mean"] == pytest.approx(10.0)
        assert s["p50"] == pytest.approx(10.0)
        assert s["p95"] == pytest.approx(10.0)
        assert s["p99"] == pytest.approx(10.0)
        assert s["fps"] == pytest.approx(100.0)

    def test_percentiles_match_numpy(self):
        times = list(range(1, 101))
        s = latency_summary(times)
        assert s["mean"] == pytest.approx(50.5)
        np.testing.assert_allclose(s["p50"], np.percentile(times, 50))
        np.testing.assert_allclose(s["p95"], np.percentile(times, 95))
        np.testing.assert_allclose(s["p99"], np.percentile(times, 99))
        assert s["fps"] == pytest.approx(1000.0 / 50.5)

    def test_single_sample(self):
        s = latency_summary([12.5])
        assert s["mean"] == pytest.approx(12.5)
        assert s["p50"] == pytest.approx(12.5)
        assert s["p99"] == pytest.approx(12.5)


# ===========================================================================
# summarise_run
# ===========================================================================
class TestSummariseRun:
    def test_empty_returns_zero_frames(self):
        s = summarise_run([])
        assert s["n_frames"] == 0

    def test_basic_aggregation(self):
        records = [
            {"iou_match": True,  "det_coverage": True,  "det_recall": 1.0, "count_agree": True,  "benefit": False, "choice": "n", "time_ms": 5.0},
            {"iou_match": True,  "det_coverage": True,  "det_recall": 1.0, "count_agree": True,  "benefit": False, "choice": "n", "time_ms": 5.0},
            {"iou_match": False, "det_coverage": True,  "det_recall": 1.0, "count_agree": True,  "benefit": True,  "choice": "s", "time_ms": 15.0},
            {"iou_match": False, "det_coverage": False, "det_recall": 0.5, "count_agree": False, "benefit": True,  "choice": "s", "time_ms": 15.0},
        ]
        s = summarise_run(records)
        assert s["n_frames"] == 4
        assert s["iou_match_rate"] == pytest.approx(0.5)
        assert s["det_coverage_rate"] == pytest.approx(0.75)
        assert s["det_recall_mean"] == pytest.approx((1.0 + 1.0 + 1.0 + 0.5) / 4)
        assert s["count_agree_rate"] == pytest.approx(0.75)
        assert s["benefit_rate"] == pytest.approx(0.5)
        assert s["s_choice_rate"] == pytest.approx(0.5)
        assert s["latency"]["mean"] == pytest.approx(10.0)
        assert s["latency"]["n"] == 4

    def test_missing_keys_default_to_zero(self):
        records = [
            {"choice": "n", "time_ms": 5.0},
            {"choice": "s", "time_ms": 5.0},
        ]
        s = summarise_run(records)
        assert s["iou_match_rate"] == 0.0
        assert s["count_agree_rate"] == 0.0
        assert s["benefit_rate"] == 0.0
        assert s["s_choice_rate"] == pytest.approx(0.5)

    def test_no_time_ms_yields_empty_latency(self):
        records = [
            {"iou_match": True, "choice": "n"},
            {"iou_match": False, "choice": "s"},
        ]
        s = summarise_run(records)
        assert s["latency"]["n"] == 0
        assert s["iou_match_rate"] == pytest.approx(0.5)
