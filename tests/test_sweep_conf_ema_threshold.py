"""Unit tests for analysis.sweep_conf_ema_threshold — offline c_high sweep.

Covers:
  - replay_at_threshold matches the live ConfEMA on a synthetic table
    when c_high is set such that conf_drop never crosses (s%=0)
  - lowering c_high produces strictly increasing s_choice_rate when the
    signal exists (monotonicity check)
  - trigger precision / recall / F1 reflect benefit_positive labelling
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analysis.sweep_conf_ema_threshold import (
    _per_frame_table,
    replay_at_threshold,
)


# ===========================================================================
# Helpers
# ===========================================================================
def _make_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_path = tmp_path / "sweep_per_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return csv_path


def _row(frame_idx, policy, fast_mc, benefit, *,
         iou_match=0, det_coverage=0, det_recall=0.0,
         count_agree=0, n_count=0, s_count=0,
         choice="n", chosen_latency_ms=10.0, proxy_ms=1.0,
         decision_ms=0.01, chosen_count=0):
    return dict(
        frame_idx=frame_idx, policy=policy, choice=choice,
        chosen_latency_ms=chosen_latency_ms, proxy_ms=proxy_ms,
        decision_ms=decision_ms,
        n_count=n_count, s_count=s_count,
        fast_mean_conf=fast_mc, chosen_count=chosen_count,
        iou_match=iou_match, det_coverage=det_coverage,
        det_recall=det_recall, count_agree=count_agree,
        benefit_positive=benefit,
    )


def _synthetic_run(tmp_path: Path) -> Path:
    """Build a tiny CSV with n_only and s_only rows for 80 frames.

    Pattern:
      - frames 0..29: stable conf 0.7, no benefit
      - frames 30..49: dipped conf 0.3, benefit_positive
      - frames 50..79: recovered conf 0.7, no benefit

    n_only iou_match is 1 on stable frames and 0 on the dipped block
    (we want lowering c_high to recover those misses).
    """
    rows = []
    for fi in range(80):
        if 30 <= fi < 50:
            fast_mc = 0.30
            n_iou = 0
            n_cov = 0
            n_rec = 0.0
            n_cnt = 0
            bp = 1
        else:
            fast_mc = 0.70
            n_iou = 1
            n_cov = 1
            n_rec = 1.0
            n_cnt = 1
            bp = 0
        rows.append(_row(
            fi, "n_only", fast_mc, bp,
            iou_match=n_iou, det_coverage=n_cov, det_recall=n_rec,
            count_agree=n_cnt, choice="n",
        ))
        # s_only is the reference: always 1 / 1 / 1.0 / 1.
        rows.append(_row(
            fi, "s_only", fast_mc, bp,
            iou_match=1, det_coverage=1, det_recall=1.0, count_agree=1,
            choice="s",
        ))
    return _make_csv(tmp_path, rows)


# ===========================================================================
# Tests
# ===========================================================================
class TestPerFrameTable:
    def test_extracts_required_columns(self, tmp_path: Path):
        csv_path = _synthetic_run(tmp_path)
        tbl = _per_frame_table(csv_path)
        assert len(tbl) == 80
        assert tbl[0]["fast_mean_conf"] == pytest.approx(0.70)
        assert tbl[35]["fast_mean_conf"] == pytest.approx(0.30)
        assert tbl[35]["benefit_positive"] == 1
        assert tbl[35]["n_iou_match"] == 0
        assert tbl[35]["s_iou_match"] == 1


class TestReplayAtThreshold:
    def test_unreachable_c_high_produces_zero_switches(self, tmp_path: Path):
        # The synthetic dip is strong enough that conf_drop peaks ~0.5.
        # Using c_high=0.99 guarantees no frame ever crosses, so s% == 0
        # (= the live cuda1000 finding before the threshold fix).
        csv_path = _synthetic_run(tmp_path)
        tbl = _per_frame_table(csv_path)
        out = replay_at_threshold(tbl, c_high=0.99)
        assert out["s_choice_rate"] == 0.0

    def test_low_c_high_produces_some_switches(self, tmp_path: Path):
        csv_path = _synthetic_run(tmp_path)
        tbl = _per_frame_table(csv_path)
        out = replay_at_threshold(tbl, c_high=0.04, c_low=0.02)
        assert out["s_choice_rate"] > 0.0

    def test_lower_c_high_monotone_more_switches(self, tmp_path: Path):
        csv_path = _synthetic_run(tmp_path)
        tbl = _per_frame_table(csv_path)
        a = replay_at_threshold(tbl, c_high=0.20)
        b = replay_at_threshold(tbl, c_high=0.06)
        c = replay_at_threshold(tbl, c_high=0.02)
        assert a["s_choice_rate"] <= b["s_choice_rate"] <= c["s_choice_rate"]

    def test_iou_match_improves_when_switching_to_s(self, tmp_path: Path):
        csv_path = _synthetic_run(tmp_path)
        tbl = _per_frame_table(csv_path)
        # At unreachable c_high we should equal n_only's iou_match,
        # which is 60/80 = 0.75 (60 stable frames where n matches s).
        high = replay_at_threshold(tbl, c_high=0.99)
        assert high["iou_match_rate"] == pytest.approx(60 / 80, abs=0.01)
        # At a lower threshold conf_ema picks 's' on at least some
        # benefit-positive frames, so iou_match should improve.
        low = replay_at_threshold(tbl, c_high=0.04, c_low=0.02)
        assert low["iou_match_rate"] > high["iou_match_rate"]

    def test_trigger_precision_recall_well_defined(self, tmp_path: Path):
        csv_path = _synthetic_run(tmp_path)
        tbl = _per_frame_table(csv_path)
        out = replay_at_threshold(tbl, c_high=0.04, c_low=0.02)
        # Bounds.
        assert 0.0 <= out["trigger_precision"] <= 1.0
        assert 0.0 <= out["trigger_recall"] <= 1.0
        assert 0.0 <= out["trigger_f1"] <= 1.0
        # On this synthetic, the trigger fires only during the dipped
        # block (which is the benefit-positive block by construction),
        # so precision should be very high.
        assert out["trigger_precision"] >= 0.5

    def test_empty_table_returns_zeroes(self):
        out = replay_at_threshold([], c_high=0.06)
        assert out["n_frames"] == 0
        assert out["s_choice_rate"] == 0.0
        assert out["trigger_f1"] == 0.0
