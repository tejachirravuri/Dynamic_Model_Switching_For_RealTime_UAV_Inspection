"""Unit tests for dms.matching — pure box geometry and greedy assignment.

Audit checklist (supervisor review):
  - matching.py defines scientific correctness; test edge cases on
    synthetic boxes
  - matching must NOT depend on YOLO, OpenCV, or experiment paths
  - threshold boundary matters: >= is the rule
  - tie-breaking is deterministic
  - class filtering excludes cross-class pairs
"""
from __future__ import annotations

import numpy as np
import pytest

from dms.matching import (
    Match,
    MatchResult,
    box_iou,
    filter_by_score,
    greedy_match,
    iou_matrix,
)


def _box(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=float)


def _boxes(*rows):
    return np.array(rows, dtype=float)


# ===========================================================================
# IoU primitives
# ===========================================================================
class TestIoU:
    def test_identical_boxes_iou_one(self):
        a = _box(0, 0, 10, 10)
        assert box_iou(a, a) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self):
        a = _box(0, 0, 10, 10)
        b = _box(20, 20, 30, 30)
        assert box_iou(a, b) == pytest.approx(0.0)

    def test_touching_boxes_iou_zero(self):
        # Boxes that share only an edge have zero intersection area.
        a = _box(0, 0, 10, 10)
        b = _box(10, 0, 20, 10)
        assert box_iou(a, b) == pytest.approx(0.0)

    def test_half_overlap_horizontal(self):
        # area_a=area_b=100, inter=5*10=50, union=200-50=150 -> IoU=1/3.
        a = _box(0, 0, 10, 10)
        b = _box(5, 0, 15, 10)
        assert box_iou(a, b) == pytest.approx(50 / 150)

    def test_containment(self):
        # b fully inside a: inter=area(b)=16, union=area(a)=100.
        a = _box(0, 0, 10, 10)
        b = _box(2, 2, 6, 6)
        assert box_iou(a, b) == pytest.approx(16 / 100)

    def test_iou_matrix_shape(self):
        a = _boxes([0, 0, 10, 10], [5, 5, 15, 15])
        b = _boxes([0, 0, 10, 10])
        M = iou_matrix(a, b)
        assert M.shape == (2, 1)
        assert M[0, 0] == pytest.approx(1.0)

    def test_iou_matrix_symmetry(self):
        a = _boxes([0, 0, 10, 10], [5, 5, 15, 15])
        b = _boxes([2, 2, 8, 8], [10, 10, 20, 20])
        M_ab = iou_matrix(a, b)
        M_ba = iou_matrix(b, a)
        np.testing.assert_allclose(M_ab, M_ba.T)

    def test_empty_inputs(self):
        empty = np.zeros((0, 4))
        one = _boxes([0, 0, 10, 10])
        assert iou_matrix(empty, one).shape == (0, 1)
        assert iou_matrix(one, empty).shape == (1, 0)
        assert iou_matrix(empty, empty).shape == (0, 0)

    def test_zero_area_box_iou_zero(self):
        # Degenerate box: zero width. IoU should be 0, not NaN.
        degenerate = _box(5, 0, 5, 10)
        normal = _box(0, 0, 10, 10)
        out = box_iou(degenerate, normal)
        assert out == pytest.approx(0.0)
        assert np.isfinite(out)


# ===========================================================================
# Greedy matching
# ===========================================================================
class TestGreedyMatch:
    def test_empty_inputs(self):
        empty = np.zeros((0, 4))
        one = _boxes([0, 0, 10, 10])

        r = greedy_match(empty, empty)
        assert r.pairs == [] and r.unmatched_a == [] and r.unmatched_b == []

        r = greedy_match(one, empty)
        assert r.pairs == [] and r.unmatched_a == [0] and r.unmatched_b == []

        r = greedy_match(empty, one)
        assert r.pairs == [] and r.unmatched_a == [] and r.unmatched_b == [0]

    def test_one_to_one_identical(self):
        a = _boxes([0, 0, 10, 10])
        b = a.copy()
        r = greedy_match(a, b, iou_threshold=0.5)
        assert len(r) == 1
        assert r.pairs[0] == Match(a_idx=0, b_idx=0, iou=1.0)
        assert r.unmatched_a == [] and r.unmatched_b == []

    def test_one_to_one_disjoint(self):
        a = _boxes([0, 0, 10, 10])
        b = _boxes([100, 100, 110, 110])
        r = greedy_match(a, b, iou_threshold=0.5)
        assert r.pairs == []
        assert r.unmatched_a == [0] and r.unmatched_b == [0]

    def test_threshold_boundary_inclusive(self):
        # IoU = 1/3 ~ 0.333. The >= rule must admit a threshold equal to
        # the actual IoU.
        a = _boxes([0, 0, 10, 10])
        b = _boxes([5, 0, 15, 10])
        r = greedy_match(a, b, iou_threshold=1 / 3)
        assert len(r) == 1

    def test_threshold_above_iou_rejects(self):
        a = _boxes([0, 0, 10, 10])
        b = _boxes([5, 0, 15, 10])
        r = greedy_match(a, b, iou_threshold=0.5)
        assert r.pairs == []

    def test_greedy_picks_best_when_multiple_candidates(self):
        # B has two boxes overlapping the single A box; greedy must pick
        # the perfect-overlap one and leave the partial one unmatched.
        a = _boxes([0, 0, 10, 10])
        b = _boxes([5, 0, 15, 10], [0, 0, 10, 10])  # idx 1 = perfect
        r = greedy_match(a, b, iou_threshold=0.3)
        assert len(r) == 1
        assert r.pairs[0].b_idx == 1
        assert r.pairs[0].iou == pytest.approx(1.0)
        assert r.unmatched_b == [0]

    def test_class_filter_blocks_cross_class(self):
        a = _boxes([0, 0, 10, 10])
        b = a.copy()
        r = greedy_match(
            a, b, iou_threshold=0.5,
            classes_a=np.array([0]), classes_b=np.array([1]),
        )
        assert r.pairs == []
        assert r.unmatched_a == [0] and r.unmatched_b == [0]

    def test_class_filter_allows_same_class(self):
        a = _boxes([0, 0, 10, 10])
        b = a.copy()
        r = greedy_match(
            a, b, iou_threshold=0.5,
            classes_a=np.array([3]), classes_b=np.array([3]),
        )
        assert len(r) == 1

    def test_class_shape_mismatch_raises(self):
        a = _boxes([0, 0, 10, 10], [10, 10, 20, 20])
        b = _boxes([0, 0, 10, 10])
        with pytest.raises(ValueError):
            greedy_match(
                a, b,
                classes_a=np.array([0]),  # length 1 vs Na=2
                classes_b=np.array([0]),
            )

    def test_input_order_does_not_drop_pairs(self):
        # Two pairs of identical boxes. Reordering B should still match
        # both pairs (though indices may swap).
        a = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        b1 = _boxes([0, 0, 10, 10], [50, 50, 60, 60])
        b2 = _boxes([50, 50, 60, 60], [0, 0, 10, 10])
        r1 = greedy_match(a, b1)
        r2 = greedy_match(a, b2)
        assert len(r1) == 2 and len(r2) == 2
        assert r1.unmatched_a == [] and r2.unmatched_a == []
        assert r1.unmatched_b == [] and r2.unmatched_b == []

    def test_deterministic_tie_breaking(self):
        # Two A boxes both equally overlap one B box at IoU=1.
        # Greedy must pick A[0] first, leaving A[1] unmatched (deterministic).
        a = _boxes([0, 0, 10, 10], [0, 0, 10, 10])
        b = _boxes([0, 0, 10, 10])
        r = greedy_match(a, b, iou_threshold=0.5)
        assert len(r) == 1
        assert r.pairs[0].a_idx == 0
        assert r.unmatched_a == [1]


# ===========================================================================
# filter_by_score
# ===========================================================================
class TestFilterByScore:
    def test_filters_below_floor(self):
        boxes = _boxes(
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [20, 20, 30, 30],
        )
        scores = np.array([0.9, 0.1, 0.5])
        b, s, c = filter_by_score(boxes, scores, conf_floor=0.25)
        assert b.shape == (2, 4)
        assert (s >= 0.25).all()
        assert c is None

    def test_keeps_boundary_score(self):
        boxes = _boxes([0, 0, 10, 10], [10, 10, 20, 20])
        scores = np.array([0.25, 0.24])
        b, s, _ = filter_by_score(boxes, scores, conf_floor=0.25)
        assert b.shape[0] == 1
        assert s[0] == pytest.approx(0.25)

    def test_passes_classes_through(self):
        boxes = np.zeros((3, 4))
        scores = np.array([0.9, 0.1, 0.5])
        classes = np.array([0, 1, 2])
        _, _, c = filter_by_score(boxes, scores, classes=classes, conf_floor=0.25)
        assert c.tolist() == [0, 2]

    def test_no_floor_keeps_all(self):
        boxes = np.zeros((3, 4))
        scores = np.array([0.0, 0.5, 1.0])
        b, _, _ = filter_by_score(boxes, scores, conf_floor=0.0)
        assert b.shape[0] == 3

    def test_length_mismatch_raises(self):
        boxes = np.zeros((3, 4))
        scores = np.array([0.5, 0.5])  # length 2 vs 3 boxes
        with pytest.raises(ValueError):
            filter_by_score(boxes, scores)
