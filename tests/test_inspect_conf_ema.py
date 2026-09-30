"""Unit tests for analysis.inspect_conf_ema — offline conf_ema replay + classification.

Covers:
  - replay matches dms.policies.ConfEMA on a synthetic trace (numerical
    parity is the headline correctness property)
  - classify() returns the right case label for the three regimes
  - read_fast_mean_conf raises a clear error on a CSV missing the column
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analysis.inspect_conf_ema import (
    DEFAULT_FAST_BETA,
    DEFAULT_SLOW_BETA,
    _load_active_params,
    classify,
    read_fast_mean_conf,
    replay_conf_ema,
)


# ===========================================================================
# Replay matches the live policy
# ===========================================================================
class TestReplayMatchesLivePolicy:
    def test_numerical_parity_with_conf_ema(self):
        # Synthetic trace: stable then a sustained dip then recover.
        trace = [0.7] * 30 + [0.3] * 20 + [0.7] * 30
        offline = replay_conf_ema(
            trace, fast_beta=DEFAULT_FAST_BETA, slow_beta=DEFAULT_SLOW_BETA,
        )

        from dms.policies import ConfEMA, FrameFeatures, PolicyConfig
        cfg = PolicyConfig(conf_ema_warmup=0)  # no warmup forcing for parity
        live = ConfEMA(cfg)

        # Drive ConfEMA with the same per-frame last_mean_conf signal
        # and capture its internal conf_drop on each step. The live
        # policy multiplexes through hysteresis + stabiliser so we can't
        # read conf_drop off the .decide() return; we call the internal
        # _conf_drop directly because that is the quantity under test.
        live_drops = []
        for x in trace:
            live_drops.append(live._conf_drop(x))

        for i, (a, b) in enumerate(zip(offline.conf_drop, live_drops)):
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), (
                f"conf_drop diverged at frame {i}: offline={a}, live={b}"
            )

    def test_first_frame_initialises_emas_to_input(self):
        out = replay_conf_ema([0.42, 0.42, 0.42])
        assert out.fast_ema[0] == pytest.approx(0.42)
        assert out.slow_ema[0] == pytest.approx(0.42)
        assert out.conf_drop[0] == pytest.approx(0.0)

    def test_empty_input_returns_empty(self):
        out = replay_conf_ema([])
        assert out.fast_mean_conf == []
        assert out.fast_ema == []
        assert out.slow_ema == []
        assert out.conf_drop == []


# ===========================================================================
# Classification
# ===========================================================================
class TestClassify:
    def test_case_1_true_negative(self):
        # All conf_drop values well below 0.05.
        label, _ = classify([0.0] * 100 + [0.04, 0.03, 0.02])
        assert label == "CASE_1_TRUE_NEGATIVE"

    def test_case_2_misconfigured(self):
        # Peak in [0.05, 0.12) but never reaches c_high.
        cd = [0.0] * 100 + [0.07, 0.08, 0.09, 0.10]
        label, msg = classify(cd, c_high=0.12)
        assert label == "CASE_2_MISCONFIGURED"
        assert "Lower c_high" in msg

    def test_case_3_rare_spikes(self):
        # Most values low but with several spikes above c_high.
        cd = [0.0] * 95 + [0.13, 0.14, 0.20, 0.18, 0.15]
        label, msg = classify(cd, c_high=0.12)
        assert label == "CASE_3_RARE_SPIKES"
        assert "5/100" in msg

    def test_empty_input_returns_empty(self):
        label, _ = classify([])
        assert label == "EMPTY"


# ===========================================================================
# CSV reader
# ===========================================================================
def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("policy,fast_mean_conf\n")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestReadFastMeanConf:
    def test_picks_n_only_rows_when_present(self, tmp_path: Path):
        rows = []
        for fi in range(3):
            for pn in ("n_only", "s_only", "combined"):
                rows.append({
                    "frame_idx": fi,
                    "policy": pn,
                    "fast_mean_conf": 0.5 + fi * 0.1,
                })
        csv_path = tmp_path / "sweep_per_frame.csv"
        _write_csv(csv_path, rows)
        trace = read_fast_mean_conf(csv_path)
        assert trace == pytest.approx([0.5, 0.6, 0.7])

    def test_missing_column_raises(self, tmp_path: Path):
        rows = [{"frame_idx": 0, "policy": "n_only"}]
        csv_path = tmp_path / "sweep_per_frame.csv"
        _write_csv(csv_path, rows)
        with pytest.raises(ValueError, match="fast_mean_conf"):
            read_fast_mean_conf(csv_path)


# ===========================================================================
# Active-params loader (verdict must compare against the run's actual c_high)
# ===========================================================================
class TestLoadActiveParams:
    def test_reads_conf_ema_params_from_sweep_summary(self, tmp_path: Path):
        import json
        summary = {
            "config": {
                "conf_ema_params": {
                    "fast_beta": 0.30, "slow_beta": 0.02,
                    "c_low": 0.007, "c_high": 0.02, "warmup": 5,
                }
            }
        }
        (tmp_path / "sweep_summary.json").write_text(json.dumps(summary))
        params = _load_active_params(tmp_path)
        assert params["c_high"] == pytest.approx(0.02)
        assert params["c_low"] == pytest.approx(0.007)

    def test_returns_empty_when_no_summary(self, tmp_path: Path):
        params = _load_active_params(tmp_path)
        assert params == {}

    def test_returns_empty_when_summary_lacks_section(self, tmp_path: Path):
        import json
        (tmp_path / "sweep_summary.json").write_text(
            json.dumps({"config": {"video": "x"}})
        )
        params = _load_active_params(tmp_path)
        assert params == {}
