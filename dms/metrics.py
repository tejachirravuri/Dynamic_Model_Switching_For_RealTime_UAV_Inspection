"""Per-frame and per-run thesis metrics — pure numpy, no I/O.

Primary correctness metric (strict)
-----------------------------------
:func:`iou_match_frame` (per frame): 1 iff every policy detection matches
a unique reference detection at IoU >= threshold AND every reference
detection is matched. Strict precision-and-recall test at one threshold.

Relaxed correctness metrics (recall-only)
-----------------------------------------
:func:`det_coverage_frame` (per frame): 1 iff every reference detection
is matched at IoU >= threshold; tolerates extra policy boxes.
:func:`det_recall_frame` (per frame): fraction of reference detections
matched, in [0, 1]. ``det_coverage_frame == True`` iff
``det_recall_frame == 1.0``. Both are kept alongside the strict
``iou_match_frame`` so the thesis can report **detection preservation
under switching** without the precision penalty of the strict metric.

Secondary count metric
----------------------
:func:`count_agree_frame` (per frame): 1 iff ``len(policy) == len(ref)``.
Cheap, ignores spatial agreement, kept for backward comparison.

Frame label (used for trigger validity)
---------------------------------------
:func:`benefit_positive_frame` (per frame, model-pair property): 1 iff
the accurate model produces at least one valid detection that cannot be
matched to any fast-model detection at IoU >= threshold. **Locked**
definition; do not relax without a thesis-text update.

Run summary
-----------
:func:`latency_summary` rolls a list of per-frame times into mean / p50 /
p95 / p99 / fps. :func:`summarise_run` rolls per-frame records into one
dict per (video, policy) — the unit of a master_table row.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from .matching import greedy_match


# ===========================================================================
# Per-frame binary metrics
# ===========================================================================
def iou_match_frame(
    policy_boxes: np.ndarray,
    ref_boxes: np.ndarray,
    iou_threshold: float = 0.5,
    policy_classes: Optional[np.ndarray] = None,
    ref_classes: Optional[np.ndarray] = None,
) -> bool:
    """1 iff perfect 1:1 match of policy and reference at the threshold.

    Both empty -> True. Different counts -> False. Equal counts but any
    policy or reference box left unpaired by greedy matching -> False.
    """
    p = np.asarray(policy_boxes, dtype=float).reshape(-1, 4)
    r = np.asarray(ref_boxes, dtype=float).reshape(-1, 4)
    if p.shape[0] == 0 and r.shape[0] == 0:
        return True
    if p.shape[0] != r.shape[0]:
        return False
    res = greedy_match(
        p, r, iou_threshold,
        classes_a=policy_classes, classes_b=ref_classes,
    )
    return len(res.unmatched_a) == 0 and len(res.unmatched_b) == 0


def det_recall_frame(
    policy_boxes: np.ndarray,
    ref_boxes: np.ndarray,
    iou_threshold: float = 0.5,
    policy_classes: Optional[np.ndarray] = None,
    ref_classes: Optional[np.ndarray] = None,
) -> float:
    """Fraction of reference detections matched by policy at the threshold.

    Recall-only relaxation of :func:`iou_match_frame`. Tolerates extra
    policy boxes (no precision penalty). Returns ``len(matched) /
    len(ref)`` in ``[0, 1]``. Both empty -> 1.0 (vacuous match).
    """
    p = np.asarray(policy_boxes, dtype=float).reshape(-1, 4)
    r = np.asarray(ref_boxes, dtype=float).reshape(-1, 4)
    if r.shape[0] == 0:
        # Convention: no reference -> nothing to recall, treat as full recall.
        return 1.0
    if p.shape[0] == 0:
        return 0.0
    # greedy_match: A=policy, B=ref. unmatched_b are reference boxes the
    # policy failed to cover.
    res = greedy_match(
        p, r, iou_threshold,
        classes_a=policy_classes, classes_b=ref_classes,
    )
    matched = int(r.shape[0]) - len(res.unmatched_b)
    return float(matched) / float(r.shape[0])


def det_coverage_frame(
    policy_boxes: np.ndarray,
    ref_boxes: np.ndarray,
    iou_threshold: float = 0.5,
    policy_classes: Optional[np.ndarray] = None,
    ref_classes: Optional[np.ndarray] = None,
) -> bool:
    """1 iff every reference detection is matched by some policy detection.

    Recall-only binary relaxation of :func:`iou_match_frame`: tolerates
    extra policy boxes. Equivalent to ``det_recall_frame(...) == 1.0``.
    """
    return det_recall_frame(
        policy_boxes, ref_boxes, iou_threshold,
        policy_classes=policy_classes, ref_classes=ref_classes,
    ) >= 1.0


def count_agree_frame(
    policy_boxes: np.ndarray,
    ref_boxes: np.ndarray,
) -> bool:
    """1 iff ``|policy| == |reference|``. Spatial agreement ignored."""
    p = np.asarray(policy_boxes, dtype=float).reshape(-1, 4)
    r = np.asarray(ref_boxes, dtype=float).reshape(-1, 4)
    return int(p.shape[0]) == int(r.shape[0])


def benefit_positive_frame(
    fast_boxes: np.ndarray,
    accurate_boxes: np.ndarray,
    iou_threshold: float = 0.5,
    fast_classes: Optional[np.ndarray] = None,
    accurate_classes: Optional[np.ndarray] = None,
) -> bool:
    """LOCKED benefit-positive definition.

    A frame is benefit-positive iff the accurate model produces at least
    one valid detection that cannot be matched to any fast-model detection
    at IoU >= ``iou_threshold``.

    Empty accurate -> False (no benefit possible: nothing was found).
    Empty fast with non-empty accurate -> True (everything is unmatched).

    Caller is responsible for confidence-filtering both inputs before
    invocation (see :func:`dms.matching.filter_by_score`).
    """
    acc = np.asarray(accurate_boxes, dtype=float).reshape(-1, 4)
    if acc.shape[0] == 0:
        return False
    fast = np.asarray(fast_boxes, dtype=float).reshape(-1, 4)
    res = greedy_match(
        acc, fast, iou_threshold,
        classes_a=accurate_classes, classes_b=fast_classes,
    )
    return len(res.unmatched_a) > 0


# ===========================================================================
# Latency summary
# ===========================================================================
def latency_summary(times_ms: Iterable[float]) -> Dict[str, float]:
    """Mean / p50 / p95 / p99 / fps over an array of per-frame times (ms).

    fps is computed as ``1000 / mean`` (steady-state per-stream
    throughput, not parallel throughput).
    """
    arr = np.asarray(list(times_ms), dtype=float)
    if arr.size == 0:
        return dict(n=0, mean=0.0, p50=0.0, p95=0.0, p99=0.0, fps=0.0)
    mean = float(arr.mean())
    return dict(
        n=int(arr.size),
        mean=mean,
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        fps=float(1000.0 / max(mean, 1e-9)),
    )


# ===========================================================================
# Run aggregation
# ===========================================================================
def summarise_run(per_frame: List[Dict]) -> Dict:
    """Aggregate a list of per-frame records to a single run summary.

    Each record may contain:

    * ``iou_match``     bool/0-1   (strict: precision AND recall)
    * ``det_coverage``  bool/0-1   (relaxed binary: recall == 1.0)
    * ``det_recall``    float in [0, 1]
    * ``count_agree``   bool/0-1
    * ``benefit``       bool/0-1   (the locked benefit_positive label)
    * ``choice``        ``'n'`` | ``'s'``
    * ``time_ms``       float (frame total time)

    Missing keys default to 0 / not counted. Returns a flat dict suitable
    for a master_table row, with the latency block nested under
    ``"latency"``.
    """
    n = len(per_frame)
    if n == 0:
        return dict(n_frames=0)
    iou_rate = sum(int(bool(r.get("iou_match", 0))) for r in per_frame) / n
    cov_rate = sum(int(bool(r.get("det_coverage", 0))) for r in per_frame) / n
    rec_mean = sum(float(r.get("det_recall", 0.0)) for r in per_frame) / n
    cnt_rate = sum(int(bool(r.get("count_agree", 0))) for r in per_frame) / n
    ben_rate = sum(int(bool(r.get("benefit", 0))) for r in per_frame) / n
    s_rate = sum(1 for r in per_frame if r.get("choice") == "s") / n
    times = [r["time_ms"] for r in per_frame if "time_ms" in r]
    lat = latency_summary(times)
    return dict(
        n_frames=int(n),
        iou_match_rate=float(iou_rate),       # strict
        det_coverage_rate=float(cov_rate),    # relaxed binary (recall=1)
        det_recall_mean=float(rec_mean),      # relaxed continuous
        count_agree_rate=float(cnt_rate),
        benefit_rate=float(ben_rate),
        s_choice_rate=float(s_rate),
        latency=lat,
    )


__all__ = [
    "iou_match_frame",
    "det_coverage_frame",
    "det_recall_frame",
    "count_agree_frame",
    "benefit_positive_frame",
    "latency_summary",
    "summarise_run",
]
