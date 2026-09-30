"""DMS policies — decision functions mapping per-frame features to a model choice.

Design
------
Each policy is a stateful object with a clean interface::

    policy = make_policy("conf_ema", PolicyConfig(...))
    for frame_idx, features in stream:
        choice = policy.decide(features)   # returns 'n' or 's'

State (EMAs, rolling windows, dwell counters) lives inside the policy
object, not in module globals. The reusable state primitives —
`RollingPercentile`, `Hysteresis`, `SwitchStabiliser` — are defined in
`dms.controllers` and composed by the policies here.

Canonical policies (the 7 evaluated in the thesis sweep)
-------------------------------------------------------
    1. n_only         — always fast (latency baseline)
    2. s_only         — always accurate (always-accurate reference)
    3. entropy_only   — switch on H >= rolling-median(H)
    4. combined       — linear threshold on weighted (L, H), with
                        explicit per-feature direction signs
    5. combined_hyst  — combined with hysteresis
    6. conf_ema       — switch on fast-model confidence drop (EMA)
    7. multi_proxy    — weighted EMA-deviation composite over multiple
                        scene proxies (kept for thesis comparison;
                        documented as falsified-hypothesis policy)

Experimental policy candidates
------------------------------
    local_contrast_hyst — trigger-validity-informed single-feature policy
                          for glass-like structural scenes. It is not part
                          of the locked 7-policy baseline; scripts must opt
                          into it explicitly when evaluating exploratory
                          policies.

Audit fixes baked in
--------------------
F1: `combined` exposes `L_direction` and `H_direction` so the sign
    assumption is explicit and testable, not hidden.
F2: `multi_proxy` accepts `mp_directional=True` to use signed deviation
    instead of absolute deviation. Default remains absolute for
    backward compatibility with the thesis sweep.
F3: `entropy_only` is documented as a relative-difficulty policy
    (~50% s by construction) — its threshold is the rolling median.
F4: `conf_ema` warmup and slow-EMA settling time are first-class
    config parameters with sane defaults.

"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .controllers import Hysteresis, RollingPercentile, SwitchStabiliser


# Uniform diagnostics shape exposed by every policy after `decide()`.
# Missing values are NaN (not None) so the runner can serialise them
# directly into the CSV without per-policy branching.
_NAN = float("nan")
_DIAG_KEYS = (
    "policy_score",        # the scalar that the policy thresholds against
    "policy_thresh_low",   # off-threshold (or single threshold for non-hyst)
    "policy_thresh_high",  # on-threshold (or single threshold for non-hyst)
    "policy_thresh_mid",   # single midpoint threshold (NaN for dual-thresh)
    "fast_ema",            # conf_ema only
    "slow_ema",            # conf_ema only
    "conf_drop",           # conf_ema only (= policy_score for conf_ema)
)


def _empty_diag() -> Dict[str, float]:
    return {k: _NAN for k in _DIAG_KEYS}


# ===========================================================================
# Frame features — input to a policy on frame k
# ===========================================================================
@dataclass
class FrameFeatures:
    """Observable signals available to a policy on frame k."""

    frame_idx: int

    # Scene proxies. None = the policy is not configured to read this proxy.
    L: Optional[float] = None                  # Laplacian variance
    H: Optional[float] = None                  # Shannon entropy (gray)
    color_entropy: Optional[float] = None      # HSV joint entropy
    tenengrad: Optional[float] = None          # Sobel gradient magnitude
    edge_density: Optional[float] = None       # Canny edge density
    local_contrast: Optional[float] = None     # RMS pixel std
    bright_fraction: Optional[float] = None    # fraction above brightness thr
    hue_std: Optional[float] = None            # std of HSV hue channel

    # Post-inference signal from frame k-1, carried by the engine.
    # CONTRACT: must be the FAST-MODEL mean confidence on frame k-1,
    # not the chosen-model confidence. Mixing fast and accurate score
    # distributions across the EMA history corrupts conf_ema's trigger.
    # See ConfEMA docstring for the rationale and the runner's enforcement.
    last_mean_conf: float = 0.0


# ===========================================================================
# Policy configuration
# ===========================================================================
@dataclass
class PolicyConfig:
    """Shared configuration for all policies.

    Defaults match the thesis sweep so a re-run reproduces the documented
    numbers. Override per-experiment via YAML configs.
    """

    name: str = ""

    # ---- Combined / CombinedHyst ----
    alpha: float = 0.6
    # Sign of each scene feature's contribution to C.
    # +1: high feature value treated as "harder" -> push C up -> switch to s.
    # -1: low feature value treated as "harder" -> flip the contribution.
    L_direction: int = 1
    H_direction: int = 1

    # Rolling percentile-normalisation window.
    history_window_size: int = 200
    norm_lo_pct: float = 10.0
    norm_hi_pct: float = 90.0

    # Combined thresholds.
    c_mid: float = 0.50
    c_low: float = 0.45
    c_high: float = 0.55

    # ---- LocalContrastHyst (experimental) ----
    # Uses positive direction: higher rolling-normalised local contrast
    # favours the accurate model.
    local_contrast_low: float = 0.45
    local_contrast_high: float = 0.65

    # ---- ConfEMA ----
    conf_ema_fast_beta: float = 0.30
    conf_ema_slow_beta: float = 0.02
    conf_ema_c_low: float = 0.04
    conf_ema_c_high: float = 0.12
    conf_ema_warmup: int = 5

    # ---- MultiProxy ----
    # Default weights match the thesis sweep configuration. Set any to 0
    # to disable that proxy. Auto-normalised by the policy.
    mp_w: Dict[str, float] = field(default_factory=lambda: {
        "L": 0.37,
        "H": 0.29,
        "tenengrad": 0.38,
        "color_entropy": 0.28,
        "edge_density": 0.0,
        "local_contrast": 0.0,
    })
    mp_fast_beta: float = 0.30
    mp_slow_beta: float = 0.02
    mp_c_low: float = 0.03
    mp_c_high: float = 0.10
    # F2 fix: when True, deviation is signed (slow - fast) / slow rather
    # than absolute. Caller chooses per-feature direction via weight sign.
    mp_directional: bool = False

    # ---- Stabilisers (apply to all switching policies) ----
    min_dwell_frames: int = 10
    max_switches_per_100: int = 12


# ===========================================================================
# Policy base class
# ===========================================================================
class Policy(ABC):
    """Abstract base class for switching policies."""

    name: str = "base"

    def __init__(self, config: PolicyConfig):
        self.config = config
        # Populated by `decide()` so the runner can log per-frame
        # diagnostics (the scalar the policy thresholds against, the
        # active threshold(s), and any EMA state). Trivial policies
        # leave entries as NaN.
        self.last_diag: Dict[str, float] = _empty_diag()

    @abstractmethod
    def decide(self, features: FrameFeatures) -> str:
        """Return 'n' or 's' for this frame."""
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover (default no-op)
        """Clear internal state. Override in stateful policies."""


# ===========================================================================
# Trivial policies
# ===========================================================================
class NOnly(Policy):
    """Always fast. Latency baseline."""

    name = "n_only"

    def decide(self, features: FrameFeatures) -> str:
        return "n"


class SOnly(Policy):
    """Always accurate. Always-accurate reference for `count_agree`/`iou_match`."""

    name = "s_only"

    def decide(self, features: FrameFeatures) -> str:
        return "s"


# ===========================================================================
# Scene-feature policies
# ===========================================================================
class EntropyOnly(Policy):
    """Switch when current H >= rolling-median(H).

    NOTE: This is a RELATIVE-difficulty policy by construction. The
    threshold is the rolling median, so on average ~50% of frames go to
    `s` regardless of absolute scene difficulty. The policy tracks
    "harder than typical for this video," not "hard in absolute terms."
    """

    name = "entropy_only"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.rolling_H = RollingPercentile(config.history_window_size)
        self.stabiliser = SwitchStabiliser(
            config.min_dwell_frames, config.max_switches_per_100,
        )

    def decide(self, features: FrameFeatures) -> str:
        if features.H is None:
            raise ValueError("EntropyOnly requires features.H")
        self.rolling_H.add(features.H)
        H_med = self.rolling_H.percentile(50)
        requested = "s" if features.H >= H_med else "n"
        self.last_diag = {
            "policy_score": float(features.H),
            "policy_thresh_low": float(H_med),
            "policy_thresh_high": float(H_med),
            "policy_thresh_mid": float(H_med),
            "fast_ema": _NAN,
            "slow_ema": _NAN,
            "conf_drop": _NAN,
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        self.rolling_H = RollingPercentile(self.config.history_window_size)
        self.stabiliser.reset()


class Combined(Policy):
    """Linear threshold on weighted (L, H) with explicit direction signs.

    ::

        C = alpha * signed(L_normalised) + (1 - alpha) * signed(H_normalised)

    where ``signed(x) = x if direction >= 0 else 1 - x`` and ``L_normalised``,
    ``H_normalised`` are percentile-normalised to [0, 1] over the rolling
    window.

    Direction signs make the L/H assumption explicit and per-pair
    overrideable. The thesis default is ``L_direction=+1, H_direction=+1``,
    meaning higher values are treated as "harder" — this assumption is
    validated in the trigger-validity table.
    """

    name = "combined"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.rolling_L = RollingPercentile(config.history_window_size)
        self.rolling_H = RollingPercentile(config.history_window_size)
        self.stabiliser = SwitchStabiliser(
            config.min_dwell_frames, config.max_switches_per_100,
        )

    def _compute_C(self, features: FrameFeatures) -> float:
        if features.L is None or features.H is None:
            raise ValueError("Combined requires features.L and features.H")
        self.rolling_L.add(features.L)
        self.rolling_H.add(features.H)
        L_lo = self.rolling_L.percentile(self.config.norm_lo_pct)
        L_hi = self.rolling_L.percentile(self.config.norm_hi_pct)
        H_lo = self.rolling_H.percentile(self.config.norm_lo_pct)
        H_hi = self.rolling_H.percentile(self.config.norm_hi_pct)
        L_n = max(0.0, min(1.0,
                           (features.L - L_lo) / max(1e-9, L_hi - L_lo)))
        H_n = max(0.0, min(1.0,
                           (features.H - H_lo) / max(1e-9, H_hi - H_lo)))
        L_signed = L_n if self.config.L_direction >= 0 else 1.0 - L_n
        H_signed = H_n if self.config.H_direction >= 0 else 1.0 - H_n
        return (
            self.config.alpha * L_signed
            + (1.0 - self.config.alpha) * H_signed
        )

    def decide(self, features: FrameFeatures) -> str:
        C = self._compute_C(features)
        requested = "s" if C >= self.config.c_mid else "n"
        self.last_diag = {
            "policy_score": float(C),
            "policy_thresh_low": float(self.config.c_mid),
            "policy_thresh_high": float(self.config.c_mid),
            "policy_thresh_mid": float(self.config.c_mid),
            "fast_ema": _NAN,
            "slow_ema": _NAN,
            "conf_drop": _NAN,
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        self.rolling_L = RollingPercentile(self.config.history_window_size)
        self.rolling_H = RollingPercentile(self.config.history_window_size)
        self.stabiliser.reset()


class CombinedHyst(Combined):
    """Combined with hysteresis.

    From 'n' need ``C >= c_high``; from 's' need ``C <= c_low``.
    """

    name = "combined_hyst"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.hyst = Hysteresis(
            config.c_low, config.c_high, initial="n",
        )

    def decide(self, features: FrameFeatures) -> str:
        C = self._compute_C(features)
        requested = self.hyst.step(C)
        self.last_diag = {
            "policy_score": float(C),
            "policy_thresh_low": float(self.config.c_low),
            "policy_thresh_high": float(self.config.c_high),
            "policy_thresh_mid": _NAN,
            "fast_ema": _NAN,
            "slow_ema": _NAN,
            "conf_drop": _NAN,
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        super().reset()
        self.hyst.reset()


class LocalContrastHyst(Policy):
    """Experimental local-contrast hysteresis policy.

    This trigger-validity-informed candidate reads only
    ``features.local_contrast``. It uses the same rolling percentile
    normalisation style as ``combined`` rather than an absolute raw
    threshold, then applies hysteresis:

    * switch on when normalised local contrast >=
      ``PolicyConfig.local_contrast_high``
    * switch off when normalised local contrast <=
      ``PolicyConfig.local_contrast_low``

    Direction is positive by design: higher local contrast favours the
    accurate model. Treat this as exploratory evidence for glass-like
    structural scenes, not as a replacement for the locked 7-policy
    baseline.
    """

    name = "local_contrast_hyst"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.rolling_C = RollingPercentile(config.history_window_size)
        self.hyst = Hysteresis(
            config.local_contrast_low,
            config.local_contrast_high,
            initial="n",
        )
        self.stabiliser = SwitchStabiliser(
            config.min_dwell_frames, config.max_switches_per_100,
        )

    def _score(self, features: FrameFeatures) -> float:
        if features.local_contrast is None:
            raise ValueError("LocalContrastHyst requires features.local_contrast")
        value = float(features.local_contrast)
        self.rolling_C.add(value)
        lo = self.rolling_C.percentile(self.config.norm_lo_pct)
        hi = self.rolling_C.percentile(self.config.norm_hi_pct)
        return max(0.0, min(1.0, (value - lo) / max(1e-9, hi - lo)))

    def decide(self, features: FrameFeatures) -> str:
        score = self._score(features)
        requested = self.hyst.step(score)
        self.last_diag = {
            "policy_score": float(score),
            "policy_thresh_low": float(self.config.local_contrast_low),
            "policy_thresh_high": float(self.config.local_contrast_high),
            "policy_thresh_mid": _NAN,
            "fast_ema": _NAN,
            "slow_ema": _NAN,
            "conf_drop": _NAN,
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        self.rolling_C = RollingPercentile(self.config.history_window_size)
        self.hyst.reset()
        self.stabiliser.reset()


# ===========================================================================
# Confidence-driven policy
# ===========================================================================
class ConfEMA(Policy):
    """Switch when fast-model confidence drops, measured by fast vs slow EMA.

    The driving signal is::

        fast_ema = bf * last_mean_conf + (1 - bf) * fast_ema_prev
        slow_ema = bs * last_mean_conf + (1 - bs) * slow_ema_prev
        conf_drop = max(0, (slow_ema - fast_ema) / (slow_ema + eps))

    Switching uses hysteresis on ``conf_drop``. During the warmup window
    (first ``conf_ema_warmup`` frames) the policy is forced to ``'n'``
    so the EMAs can settle.

    Input contract (LOCKED, audit F4 resolved)
    ------------------------------------------
    ``features.last_mean_conf`` MUST be the **fast-model** mean
    confidence from the previous frame, regardless of which model the
    policy itself selected last frame. Reasoning: fast and accurate
    detectors calibrate to different score distributions, so feeding a
    mix of the two into the EMA would corrupt the deviation signal.
    Trigger semantics are 'fast detector got less confident than its
    own recent baseline,' not 'whichever model just ran got less
    confident.' The experiment runners (``experiments.run_sweep``)
    enforce this by computing ``mean(n_dets.scores)`` once per frame
    and passing the same value to every policy via
    :class:`FrameFeatures`.
    """

    name = "conf_ema"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        self.hyst = Hysteresis(
            config.conf_ema_c_low, config.conf_ema_c_high, initial="n",
        )
        self.stabiliser = SwitchStabiliser(
            config.min_dwell_frames, config.max_switches_per_100,
        )

    def _conf_drop(self, last_mean_conf: float) -> float:
        bf = self.config.conf_ema_fast_beta
        bs = self.config.conf_ema_slow_beta
        if self.fast_ema is None:
            self.fast_ema = float(last_mean_conf)
            self.slow_ema = float(last_mean_conf)
        else:
            self.fast_ema = (
                bf * float(last_mean_conf) + (1.0 - bf) * self.fast_ema
            )
            self.slow_ema = (
                bs * float(last_mean_conf) + (1.0 - bs) * self.slow_ema
            )
        denom = self.slow_ema + 1e-9
        return max(0.0, (self.slow_ema - self.fast_ema) / denom)

    def decide(self, features: FrameFeatures) -> str:
        # EMAs are updated every frame including warmup.
        c = self._conf_drop(features.last_mean_conf)
        if features.frame_idx < self.config.conf_ema_warmup:
            requested = "n"
        else:
            requested = self.hyst.step(c)
        self.last_diag = {
            "policy_score": float(c),
            "policy_thresh_low": float(self.config.conf_ema_c_low),
            "policy_thresh_high": float(self.config.conf_ema_c_high),
            "policy_thresh_mid": _NAN,
            "fast_ema": float(self.fast_ema)
            if self.fast_ema is not None else _NAN,
            "slow_ema": float(self.slow_ema)
            if self.slow_ema is not None else _NAN,
            "conf_drop": float(c),
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        self.fast_ema = None
        self.slow_ema = None
        self.hyst.reset()
        self.stabiliser.reset()


# ===========================================================================
# Composite policy
# ===========================================================================
class MultiProxy(Policy):
    """Weighted EMA-deviation composite over multiple scene proxies.

    For each enabled proxy::

        fast_ema_i = bf * x_i + (1 - bf) * fast_ema_i_prev
        slow_ema_i = bs * x_i + (1 - bs) * slow_ema_i_prev
        if mp_directional:
            drop_i = (slow_ema_i - fast_ema_i) / (slow_ema_i + eps)
        else:
            drop_i = abs(fast_ema_i - slow_ema_i) / (slow_ema_i + eps)

    Composite::

        C = sum(w_i * drop_i) / sum(|w_i|)

    Hysteresis on C. The thesis documents this policy as a falsified
    hypothesis: the absolute-deviation formulation conflates scene
    novelty with scene difficulty (audit F2). ``mp_directional=True``
    addresses this partially by using signed deviation; the per-feature
    direction is encoded in the sign of the weight.
    """

    name = "multi_proxy"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.ema: Dict[str, tuple] = {}
        self.hyst = Hysteresis(
            config.mp_c_low, config.mp_c_high, initial="n",
        )
        self.stabiliser = SwitchStabiliser(
            config.min_dwell_frames, config.max_switches_per_100,
        )

    def _composite(self, features: FrameFeatures) -> float:
        bf = self.config.mp_fast_beta
        bs = self.config.mp_slow_beta
        proxy_values = {
            "L": features.L,
            "H": features.H,
            "color_entropy": features.color_entropy,
            "tenengrad": features.tenengrad,
            "edge_density": features.edge_density,
            "local_contrast": features.local_contrast,
        }
        weighted = 0.0
        wsum = 0.0
        for pname, val in proxy_values.items():
            w = self.config.mp_w.get(pname, 0.0)
            if abs(w) < 1e-12 or val is None:
                continue
            v = float(val)
            if pname not in self.ema:
                self.ema[pname] = (v, v)
            else:
                f_prev, s_prev = self.ema[pname]
                self.ema[pname] = (
                    bf * v + (1.0 - bf) * f_prev,
                    bs * v + (1.0 - bs) * s_prev,
                )
            f_v, s_v = self.ema[pname]
            denom = s_v + 1e-9
            if self.config.mp_directional:
                drop = (s_v - f_v) / denom
            else:
                drop = abs(f_v - s_v) / denom
            weighted += w * drop
            wsum += abs(w)
        return weighted / wsum if wsum > 1e-12 else 0.0

    def decide(self, features: FrameFeatures) -> str:
        C = self._composite(features)
        requested = self.hyst.step(C)
        self.last_diag = {
            "policy_score": float(C),
            "policy_thresh_low": float(self.config.mp_c_low),
            "policy_thresh_high": float(self.config.mp_c_high),
            "policy_thresh_mid": _NAN,
            "fast_ema": _NAN,
            "slow_ema": _NAN,
            "conf_drop": _NAN,
        }
        return self.stabiliser.step(features.frame_idx, requested)

    def reset(self) -> None:
        self.ema.clear()
        self.hyst.reset()
        self.stabiliser.reset()


# ===========================================================================
# Registry + factory
# ===========================================================================
POLICY_REGISTRY: Dict[str, type] = {
    "n_only": NOnly,
    "s_only": SOnly,
    "entropy_only": EntropyOnly,
    "combined": Combined,
    "combined_hyst": CombinedHyst,
    "local_contrast_hyst": LocalContrastHyst,
    "conf_ema": ConfEMA,
    "multi_proxy": MultiProxy,
}

CANONICAL_POLICIES: List[str] = [
    "n_only",
    "s_only",
    "entropy_only",
    "combined",
    "combined_hyst",
    "conf_ema",
    "multi_proxy",
]
EXPERIMENTAL_POLICIES: List[str] = ["local_contrast_hyst"]


def make_policy(name: str, config: Optional[PolicyConfig] = None) -> Policy:
    """Instantiate a policy by name.

    >>> p = make_policy("conf_ema")
    >>> p.name
    'conf_ema'
    """
    if name not in POLICY_REGISTRY:
        raise KeyError(
            f"unknown policy: {name!r}. valid: {CANONICAL_POLICIES}"
        )
    cfg = config or PolicyConfig(name=name)
    return POLICY_REGISTRY[name](cfg)


__all__ = [
    "FrameFeatures",
    "PolicyConfig",
    "Policy",
    "NOnly",
    "SOnly",
    "EntropyOnly",
    "Combined",
    "CombinedHyst",
    "LocalContrastHyst",
    "ConfEMA",
    "MultiProxy",
    "POLICY_REGISTRY",
    "CANONICAL_POLICIES",
    "EXPERIMENTAL_POLICIES",
    "make_policy",
]
