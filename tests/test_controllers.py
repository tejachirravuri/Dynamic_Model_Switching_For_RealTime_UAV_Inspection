"""Unit tests for dms.controllers — RollingPercentile, Hysteresis, SwitchStabiliser.

Audit checklist (supervisor review):
  - hysteresis preserves state between thresholds
  - dwell prevents rapid switching
  - rolling percentile evicts oldest correctly
"""
from __future__ import annotations

import pytest

from dms.controllers import Hysteresis, RollingPercentile, SwitchStabiliser


# ===========================================================================
# RollingPercentile
# ===========================================================================
class TestRollingPercentile:
    def test_window_must_be_positive(self):
        with pytest.raises(ValueError):
            RollingPercentile(0)
        with pytest.raises(ValueError):
            RollingPercentile(-5)

    def test_empty_returns_zero(self):
        rp = RollingPercentile(10)
        assert rp.percentile(50) == 0.0

    def test_basic_quantiles(self):
        rp = RollingPercentile(10)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rp.add(v)
        # Median of 1..5 -> 3.0 (idx = int(0.5 * 4) = 2 -> sorted[2] = 3)
        assert rp.percentile(50) == 3.0
        assert rp.percentile(0) == 1.0
        assert rp.percentile(100) == 5.0

    def test_percentile_clamps_to_range(self):
        rp = RollingPercentile(10)
        for v in [10.0, 20.0, 30.0]:
            rp.add(v)
        # Out-of-range percentile values are clamped, not raised.
        assert rp.percentile(-50) == rp.percentile(0)
        assert rp.percentile(150) == rp.percentile(100)

    def test_eviction_keeps_window_consistent(self):
        rp = RollingPercentile(window=3)
        # Insert 5 values into a 3-wide window. Final state should be
        # the last 3 inserts only.
        for v in [100.0, 1.0, 2.0, 3.0, 4.0]:
            rp.add(v)
        # Internal sorted list reflects only the last 3 values: 2,3,4.
        # Median = sorted[1] = 3.0
        assert len(rp) == 3
        assert rp.percentile(0) == 2.0
        assert rp.percentile(50) == 3.0
        assert rp.percentile(100) == 4.0

    def test_duplicate_values_evict_correctly(self):
        rp = RollingPercentile(window=3)
        for v in [5.0, 5.0, 5.0, 5.0]:
            rp.add(v)
        # All identical — eviction should not destroy the sorted list.
        assert len(rp) == 3
        assert rp.percentile(50) == 5.0


# ===========================================================================
# Hysteresis
# ===========================================================================
class TestHysteresis:
    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError, match="c_low"):
            Hysteresis(c_low=0.6, c_high=0.4)

    def test_invalid_initial_raises(self):
        with pytest.raises(ValueError, match="initial"):
            Hysteresis(c_low=0.4, c_high=0.6, initial="x")

    def test_initial_state_is_n(self):
        h = Hysteresis(c_low=0.4, c_high=0.6, initial="n")
        assert h.state == "n"

    def test_initial_state_is_s(self):
        h = Hysteresis(c_low=0.4, c_high=0.6, initial="s")
        assert h.state == "s"

    def test_n_to_s_only_at_or_above_c_high(self):
        h = Hysteresis(c_low=0.4, c_high=0.6, initial="n")
        # Below c_high -> stay in n.
        assert h.step(0.50) == "n"
        assert h.step(0.59) == "n"
        # At c_high -> flip to s.
        assert h.step(0.60) == "s"

    def test_s_to_n_only_at_or_below_c_low(self):
        h = Hysteresis(c_low=0.4, c_high=0.6, initial="s")
        assert h.step(0.50) == "s"
        assert h.step(0.41) == "s"
        assert h.step(0.40) == "n"

    def test_state_preserved_inside_band(self):
        """hysteresis preserves state between thresholds."""
        h = Hysteresis(c_low=0.4, c_high=0.6, initial="n")
        # In the band [0.4, 0.6] from n: stay in n.
        for sig in [0.40, 0.45, 0.50, 0.55, 0.59]:
            assert h.step(sig) == "n"
        # Cross to s, then dip into the band — state must persist.
        assert h.step(0.65) == "s"
        for sig in [0.59, 0.55, 0.50, 0.45, 0.41]:
            assert h.step(sig) == "s"
        # Only when we drop to or below c_low does it flip back.
        assert h.step(0.40) == "n"

    def test_reset_restores_initial(self):
        h = Hysteresis(c_low=0.4, c_high=0.6, initial="n")
        h.step(0.7)
        assert h.state == "s"
        h.reset()
        assert h.state == "n"


# ===========================================================================
# SwitchStabiliser
# ===========================================================================
class TestSwitchStabiliser:
    def test_invalid_dwell_raises(self):
        with pytest.raises(ValueError):
            SwitchStabiliser(min_dwell_frames=0)

    def test_invalid_max_switches_raises(self):
        with pytest.raises(ValueError):
            SwitchStabiliser(max_switches_per_100=-1)

    def test_first_call_locks_choice_without_recording_switch(self):
        s = SwitchStabiliser(min_dwell_frames=10, max_switches_per_100=12)
        out = s.step(frame_idx=0, requested="s")
        assert out == "s"
        assert s.last_choice == "s"
        assert s.total_switches == 0      # first call is not a switch

    def test_dwell_blocks_rapid_switch(self):
        """dwell prevents rapid switching."""
        s = SwitchStabiliser(min_dwell_frames=10, max_switches_per_100=999)
        s.step(0, "n")                    # lock in 'n'
        # Frames 1..9: dwell < 10, requested switch must be suppressed.
        for k in range(1, 10):
            assert s.step(k, "s") == "n", f"dwell suppressed switch at frame {k}"
        # Frame 10: dwell == 10 (>= min_dwell), switch allowed.
        assert s.step(10, "s") == "s"
        assert s.total_switches == 1

    def test_max_switches_per_100_blocks(self):
        # Allow free dwell; cap rate at 2 per 100 frames.
        s = SwitchStabiliser(min_dwell_frames=1, max_switches_per_100=2)
        s.step(0, "n")
        # Switch at frame 1 (dwell satisfied, 0 prior switches).
        assert s.step(1, "s") == "s"
        # Switch at frame 2 (dwell satisfied, 1 prior switch).
        assert s.step(2, "n") == "n"
        # Frame 3: rate cap reached (2 switches in window) -> suppressed.
        assert s.step(3, "s") == "n"
        assert s.total_switches == 2

    def test_rate_cap_releases_after_window(self):
        s = SwitchStabiliser(min_dwell_frames=1, max_switches_per_100=1)
        s.step(0, "n")
        assert s.step(1, "s") == "s"      # 1st switch at frame 1
        # Frame 50: cap reached (1 switch within last 100), suppressed.
        assert s.step(50, "n") == "s"
        # Frame 102: switch event at frame 1 falls out of the window.
        # Cap re-opens; switch allowed.
        assert s.step(102, "n") == "n"

    def test_reset_clears_state(self):
        s = SwitchStabiliser(min_dwell_frames=1, max_switches_per_100=12)
        s.step(0, "n")
        s.step(1, "s")
        assert s.total_switches == 1
        s.reset()
        assert s.total_switches == 0
        assert s.last_choice is None
        assert s.dwell == 0
