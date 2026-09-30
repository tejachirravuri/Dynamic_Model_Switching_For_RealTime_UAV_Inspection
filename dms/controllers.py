"""Reusable per-run state primitives composed by policies.

A "controller" here is the runtime state a policy carries between frames:

- `RollingPercentile` — bounded-window quantile estimator used for
  percentile normalisation of scene proxies and for the relative-difficulty
  median used by `entropy_only`.
- `Hysteresis` — two-threshold state machine that stops a policy from
  oscillating in the band between `c_low` and `c_high`.
- `SwitchStabiliser` — actuator-side guard that enforces minimum dwell
  and a maximum number of switches per 100 frames, suppressing requested
  switches that would violate either constraint.

These objects are deliberately small, pure-Python, and stateful. Each
policy owns its own instances; the library never relies on module-level
mutable state.
"""
from __future__ import annotations

import bisect
from collections import deque
from typing import Deque, List, Optional


# ===========================================================================
# Rolling percentile — bounded-window quantile estimator
# ===========================================================================
class RollingPercentile:
    """Bounded-window rolling percentile.

    O(log n) insert via `bisect`; O(1) percentile lookup. On eviction the
    oldest value is removed from the sorted list using a binary-search
    locate (with a linear-search fallback for floating-point ties).
    """

    def __init__(self, window: int):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window: int = int(window)
        self._history: Deque[float] = deque(maxlen=self.window)
        self._sorted: List[float] = []

    def add(self, x: float) -> None:
        x = float(x)
        if len(self._history) == self.window:
            old = self._history[0]
            idx = bisect.bisect_left(self._sorted, old)
            if 0 <= idx < len(self._sorted) and self._sorted[idx] == old:
                self._sorted.pop(idx)
            else:
                # Floating-point comparison fallback (shouldn't normally hit).
                try:
                    self._sorted.remove(old)
                except ValueError:
                    pass
        self._history.append(x)
        bisect.insort(self._sorted, x)

    def percentile(self, p: float) -> float:
        if not self._sorted:
            return 0.0
        p = max(0.0, min(100.0, float(p)))
        idx = int((p / 100.0) * (len(self._sorted) - 1))
        return float(self._sorted[idx])

    def __len__(self) -> int:
        return len(self._history)


# ===========================================================================
# Hysteresis — two-threshold state machine
# ===========================================================================
class Hysteresis:
    """Two-threshold state machine for switching between 'n' and 's'.

    From 'n', switches to 's' only when ``signal >= c_high``.
    From 's', switches to 'n' only when ``signal <= c_low``.
    Inside the band ``c_low < signal < c_high`` the state is preserved.
    Requires ``c_low <= c_high``.
    """

    def __init__(self, c_low: float, c_high: float, initial: str = "n"):
        if c_low > c_high:
            raise ValueError(
                f"c_low ({c_low}) must be <= c_high ({c_high})"
            )
        if initial not in ("n", "s"):
            raise ValueError(f"initial must be 'n' or 's', got {initial!r}")
        self.c_low: float = float(c_low)
        self.c_high: float = float(c_high)
        self.state: str = initial
        self._initial: str = initial

    def step(self, signal: float) -> str:
        s = float(signal)
        if self.state == "n":
            if s >= self.c_high:
                self.state = "s"
        else:  # state == "s"
            if s <= self.c_low:
                self.state = "n"
        return self.state

    def reset(self) -> None:
        self.state = self._initial


# ===========================================================================
# Switch stabiliser — dwell + rate-limit actuator protection
# ===========================================================================
class SwitchStabiliser:
    """Suppresses requested switches that would violate dwell or rate limits.

    - ``min_dwell_frames``: a switch is suppressed if fewer than this many
      frames have elapsed since the last switch.
    - ``max_switches_per_100``: a switch is suppressed if there are already
      this many switches in the trailing 100-frame window.

    The first call locks in the requested choice without rate accounting,
    so a fresh stabiliser does not record a "switch" on frame 0.
    """

    def __init__(
        self,
        min_dwell_frames: int = 10,
        max_switches_per_100: int = 12,
    ):
        if min_dwell_frames < 1:
            raise ValueError("min_dwell_frames must be >= 1")
        if max_switches_per_100 < 0:
            raise ValueError("max_switches_per_100 must be >= 0")
        self.min_dwell: int = int(min_dwell_frames)
        self.max_sw_100: int = int(max_switches_per_100)
        self.dwell: int = 0
        self.last_choice: Optional[str] = None
        self.switch_events: Deque[int] = deque()
        self.total_switches: int = 0

    def step(self, frame_idx: int, requested: str) -> str:
        # First call locks in the requested choice without rate accounting.
        if self.last_choice is None:
            self.last_choice = requested
            self.dwell = 1
            return requested

        # Trim switch events older than 100 frames.
        while self.switch_events and self.switch_events[0] <= frame_idx - 100:
            self.switch_events.popleft()

        # Suppress switches that violate constraints.
        if requested != self.last_choice:
            if self.dwell < self.min_dwell:
                requested = self.last_choice
            elif len(self.switch_events) >= self.max_sw_100:
                requested = self.last_choice

        # Bookkeeping.
        if requested == self.last_choice:
            self.dwell += 1
        else:
            self.switch_events.append(frame_idx)
            self.total_switches += 1
            self.dwell = 1
            self.last_choice = requested
        return requested

    def reset(self) -> None:
        self.dwell = 0
        self.last_choice = None
        self.switch_events.clear()
        self.total_switches = 0


__all__ = [
    "RollingPercentile",
    "Hysteresis",
    "SwitchStabiliser",
]
