"""Unit tests for experiments.validate_sweep — sanity-gate logic.

Covers the relaxed G1 (n_only-faster-than-s_only) gate, which must:
  - PASS the normal case (n < s)
  - PASS the saturated-platform tie (n ~ s within tolerance)
  - PASS sub-1ms-gap noise
  - FAIL the swap / throttle case (n much slower than s)

The other gates are exercised end-to-end by the smoke run; this file
unit-tests just the latency comparison because that one has the
nuanced tolerance logic.
"""
from __future__ import annotations

from experiments.validate_sweep import (
    GATE_FAIL,
    GATE_OK,
    GATE_SKIP,
    gate_n_faster_than_s,
)


def _summary(n_lat: float, s_lat: float) -> dict:
    return {
        "per_policy": {
            "n_only": {"latency": {"mean": n_lat}},
            "s_only": {"latency": {"mean": s_lat}},
        }
    }


class TestGateNFasterThanS:
    def test_normal_fast_faster_passes(self):
        status, _ = gate_n_faster_than_s(_summary(n_lat=10.0, s_lat=25.0))
        assert status == GATE_OK

    def test_equal_latencies_pass(self):
        # Exactly equal — boundary case.
        status, _ = gate_n_faster_than_s(_summary(n_lat=8.0, s_lat=8.0))
        assert status == GATE_OK

    def test_saturated_tie_within_one_ms_passes(self):
        # RTX 5090 + small YOLOs case. Both ~7ms, n is slightly slower
        # by 0.2ms (sub-clock-noise). Must pass.
        status, msg = gate_n_faster_than_s(_summary(n_lat=7.03, s_lat=6.81))
        assert status == GATE_OK
        assert "sub-clock-noise" in msg or "saturated" in msg

    def test_saturated_tie_within_five_percent_passes(self):
        # Slightly above 1ms but under 5% of s. Still a saturated tie.
        status, msg = gate_n_faster_than_s(_summary(n_lat=21.0, s_lat=20.0))
        assert status == GATE_OK
        assert "saturated" in msg or "tie within tolerance" in msg

    def test_swapped_weights_fails(self):
        # n >> s by way more than 5%: this is the bug we want to catch.
        status, msg = gate_n_faster_than_s(_summary(n_lat=22.0, s_lat=8.0))
        assert status == GATE_FAIL
        assert "swap" in msg or "throttle" in msg or "warmup" in msg

    def test_thermal_throttle_fails(self):
        # Even more extreme — clear failure.
        status, _ = gate_n_faster_than_s(_summary(n_lat=120.0, s_lat=80.0))
        assert status == GATE_FAIL

    def test_borderline_just_over_5pct_fails(self):
        # 5.1% over s, gap > 1ms: just outside both tolerances.
        # 100 vs 95: gap 5, rel 0.0526 > 0.05 -> FAIL.
        status, _ = gate_n_faster_than_s(_summary(n_lat=100.0, s_lat=95.0))
        assert status == GATE_FAIL

    def test_missing_n_only_skips(self):
        s = {"per_policy": {"s_only": {"latency": {"mean": 10.0}}}}
        status, _ = gate_n_faster_than_s(s)
        assert status == GATE_SKIP

    def test_missing_s_only_skips(self):
        s = {"per_policy": {"n_only": {"latency": {"mean": 10.0}}}}
        status, _ = gate_n_faster_than_s(s)
        assert status == GATE_SKIP
