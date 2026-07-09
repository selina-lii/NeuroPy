"""Pure-data epoch interval filtering for brain-state–aware CCG computation."""
from __future__ import annotations

from typing import Literal

from neuropy.core.intervals import IntervalOp

Intervals = list[tuple[float, float]]


class EpochFilter:
    """Wraps epoch bounds and performs interval set filtering."""

    NONE_LABEL = 'NONE'

    def __init__(self, bounds: list[tuple[float, float, str]]):
        self._bounds = bounds

    @property
    def labels(self) -> set[str]:
        return {lbl for _, _, lbl in self._bounds}

    def filter(
        self,
        active_labels: set[str],
        t0: float,
        t1: float,
    ) -> tuple[Intervals | None, float] | Literal[False]:
        """Return active intervals for the selected time range.

        Returns:
            (None, duration)        — no restriction (all labels active or bounds empty)
            (intervals, active_sec) — filtered and clipped to [t0, t1]
            False                   — no intervals found (caller should warn user)
        """
        if not self._bounds or not active_labels:
            return (None, t1 - t0)

        available = self.labels
        if active_labels >= (available | {self.NONE_LABEL}):
            return (None, t1 - t0)

        none_active = self.NONE_LABEL in active_labels
        real_active = (active_labels - {self.NONE_LABEL}) & available

        intervals: Intervals = []

        for s, e, lbl in self._bounds:
            if lbl in real_active:
                sc, ec = max(s, t0), min(e, t1)
                if ec > sc:
                    intervals.append((sc, ec))

        if none_active:
            all_clipped = [
                (max(s, t0), min(e, t1))
                for s, e, _ in self._bounds
                if min(e, t1) > max(s, t0)
            ]
            intervals.extend(IntervalOp.complement(sorted(all_clipped), t0, t1))

        intervals = IntervalOp.merge(sorted(intervals))

        if not intervals:
            return False

        active_sec = sum(e - s for s, e in intervals)
        return (intervals, active_sec)
