"""Pure box-matching utilities — no YOLO, no I/O dependencies.

All functions accept and return plain numpy arrays. This module is the
scientific-correctness backbone for thesis metrics: it defines what
"two detections refer to the same object" means.

Conventions
-----------
* Boxes are ``(N, 4)`` numpy arrays in xyxy format ``[x1, y1, x2, y2]``.
* Classes (optional) are ``(N,)`` integer arrays. When provided to
  :func:`greedy_match`, only same-class boxes can be paired.
* Scores (optional) are ``(N,)`` float arrays. Used by
  :func:`filter_by_score`; matching itself is greedy on IoU descending
  and ignores scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ===========================================================================
# IoU primitives
# ===========================================================================
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU matrix.

    Args:
        boxes_a: ``(Na, 4)`` xyxy.
        boxes_b: ``(Nb, 4)`` xyxy.

    Returns:
        ``(Na, Nb)`` IoU. Empty input dims propagate to empty output.
        Degenerate (zero-area) boxes contribute IoU 0 rather than NaN.
    """
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)

    a_x1 = a[:, None, 0]
    a_y1 = a[:, None, 1]
    a_x2 = a[:, None, 2]
    a_y2 = a[:, None, 3]
    b_x1 = b[None, :, 0]
    b_y1 = b[None, :, 1]
    b_x2 = b[None, :, 2]
    b_y2 = b[None, :, 3]

    inter_w = np.clip(np.minimum(a_x2, b_x2) - np.maximum(a_x1, b_x1), 0, None)
    inter_h = np.clip(np.minimum(a_y2, b_y2) - np.maximum(a_y1, b_y1), 0, None)
    inter = inter_w * inter_h

    area_a = np.clip(a_x2 - a_x1, 0, None) * np.clip(a_y2 - a_y1, 0, None)
    area_b = np.clip(b_x2 - b_x1, 0, None) * np.clip(b_y2 - b_y1, 0, None)

    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two single boxes (xyxy)."""
    a = np.asarray(box_a, dtype=float).reshape(1, 4)
    b = np.asarray(box_b, dtype=float).reshape(1, 4)
    return float(iou_matrix(a, b)[0, 0])


# ===========================================================================
# Match data structures
# ===========================================================================
@dataclass(frozen=True)
class Match:
    """One matched pair (index into A, index into B, recorded IoU)."""
    a_idx: int
    b_idx: int
    iou: float


@dataclass
class MatchResult:
    """Result of greedy matching: pairs + the indices left over on each side."""
    pairs: List[Match]
    unmatched_a: List[int]
    unmatched_b: List[int]

    def __len__(self) -> int:
        return len(self.pairs)


# ===========================================================================
# Greedy matching
# ===========================================================================
def greedy_match(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    iou_threshold: float = 0.5,
    classes_a: Optional[np.ndarray] = None,
    classes_b: Optional[np.ndarray] = None,
) -> MatchResult:
    """One-to-one greedy matching by descending IoU.

    Algorithm: enumerate all (i, j) with ``IoU(a_i, b_j) >= iou_threshold``,
    sort by IoU descending (ties broken by ``(i, j)`` ascending for
    determinism), then walk the list claiming pairs whose row and column
    are both unused. Cross-class pairs are excluded when both class
    arrays are provided.
    """
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    Na, Nb = a.shape[0], b.shape[0]

    if Na == 0 or Nb == 0:
        return MatchResult(
            pairs=[],
            unmatched_a=list(range(Na)),
            unmatched_b=list(range(Nb)),
        )

    M = iou_matrix(a, b)

    if classes_a is not None and classes_b is not None:
        ca = np.asarray(classes_a).reshape(-1)
        cb = np.asarray(classes_b).reshape(-1)
        if ca.shape[0] != Na or cb.shape[0] != Nb:
            raise ValueError("classes_* shape mismatch with boxes_*")
        cross = (ca[:, None] != cb[None, :])
        M = np.where(cross, -1.0, M)

    candidates: List[Tuple[float, int, int]] = []
    for i in range(Na):
        for j in range(Nb):
            iou = float(M[i, j])
            if iou >= iou_threshold:
                candidates.append((iou, i, j))
    # Sort by IoU descending; for ties, smaller (i, j) wins -> deterministic.
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_a, used_b = set(), set()
    pairs: List[Match] = []
    for iou, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        pairs.append(Match(a_idx=i, b_idx=j, iou=iou))
        used_a.add(i)
        used_b.add(j)

    unmatched_a = [i for i in range(Na) if i not in used_a]
    unmatched_b = [j for j in range(Nb) if j not in used_b]
    return MatchResult(pairs=pairs, unmatched_a=unmatched_a, unmatched_b=unmatched_b)


# ===========================================================================
# Confidence filtering
# ===========================================================================
def filter_by_score(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: Optional[np.ndarray] = None,
    conf_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Drop detections whose score is below ``conf_floor``.

    Returns filtered ``(boxes, scores, classes)``. ``classes`` is None
    in the output iff it was None on input.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.shape[0] != boxes.shape[0]:
        raise ValueError("boxes and scores length mismatch")
    keep = scores >= conf_floor
    out_boxes = boxes[keep]
    out_scores = scores[keep]
    out_classes = None
    if classes is not None:
        classes = np.asarray(classes).reshape(-1)
        if classes.shape[0] != boxes.shape[0]:
            raise ValueError("boxes and classes length mismatch")
        out_classes = classes[keep]
    return out_boxes, out_scores, out_classes


__all__ = [
    "iou_matrix",
    "box_iou",
    "Match",
    "MatchResult",
    "greedy_match",
    "filter_by_score",
]
