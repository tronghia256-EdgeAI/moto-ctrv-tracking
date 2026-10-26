"""
context_estimator.py
────────────────────
C1 Component: Context Estimator

Reads 4 traffic context signals from the current frame and track history,
and returns a TrafficState dataclass used by AdaptiveController.

Signals:
    density    -- number of currently active tracks (EMA-smoothed)
    hour       -- hour of day (0–23), proxy for rush-hour vs off-peak
    brightness -- mean pixel intensity of the frame (EMA-smoothed)
    fp_rate    -- fraction of tracks with age < MIN_AGE (likely false positives)

Density and brightness are exponentially smoothed (alpha = 0.1) so single-
frame flicker — a headlight glare, one dropped detection burst — does not
thrash the AdaptiveController.

The hour signal is injectable (`hour_override`) because offline experiments
on recorded footage must not use wall-clock time.

Author  : Trọng Nghĩa
Paper   : Context-Aware Adaptive Perception for Mixed Traffic
Target  : ICRA 2027 (primary) / IEEE IV 2027 (backup)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import cv2
import numpy as np


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class TrafficState:
    """Snapshot of traffic context at a single timestep."""
    density: int        # number of active tracks (smoothed)
    hour: int           # 0–23
    brightness: float   # 0.0–255.0 (mean frame intensity, smoothed)
    fp_rate: float      # 0.0–1.0 (fraction of short-lived tracks)


# ── Constants ────────────────────────────────────────────────────────

# TODO: tune these thresholds empirically from VN footage
MIN_TRACK_AGE = 3        # tracks younger than this are likely FPs
EMA_ALPHA = 0.1          # smoothing factor for density / brightness


# ── Main class ───────────────────────────────────────────────────────

class ContextEstimator:
    """
    Estimates the current traffic context from one frame and track history.

    Usage:
        estimator = ContextEstimator()
        state = estimator.estimate(frame, tracks)
    """

    def __init__(self, hour_override: int | None = None,
                 ema_alpha: float = EMA_ALPHA) -> None:
        self.hour_override = hour_override
        self.ema_alpha = ema_alpha
        # EMA state (None until the first frame)
        self._density_ema: float | None = None
        self._brightness_ema: float | None = None

    # ── Public API ───────────────────────────────────────────────────

    def estimate(self, frame: np.ndarray, tracks: List, pixel_mask=None) -> TrafficState:
        """
        Args:
            frame  : BGR frame (H, W, 3) as np.ndarray
            tracks : list of active track objects from ByteTrack
                     Each track must expose an `.age` attribute
                     (ClassAwareSTrack provides it).

        Returns:
            TrafficState with all 4 context signals filled.
        """
        density    = self._compute_density(tracks)
        hour       = self._get_hour()
        brightness = self._compute_brightness(frame,pixel_mask)
        fp_rate    = self._compute_fp_rate(tracks)

        return TrafficState(
            density=density,
            hour=hour,
            brightness=brightness,
            fp_rate=fp_rate,
        )

    def reset(self) -> None:
        """Clear EMA state (between experiment runs)."""
        self._density_ema = None
        self._brightness_ema = None

    # ── Private sub-methods ──────────────────────────────────────────

    def _ema(self, current: float, previous: float | None) -> float:
        if previous is None:
            return current
        return self.ema_alpha * current + (1.0 - self.ema_alpha) * previous

    def _compute_density(self, tracks: List) -> int:
        """Number of currently active tracks, EMA-smoothed."""
        self._density_ema = self._ema(float(len(tracks)), self._density_ema)
        return int(round(self._density_ema))

    def _get_hour(self) -> int:
        """Current hour of day (0–23); injectable for offline runs."""
        if self.hour_override is not None:
            return self.hour_override
        return datetime.now().hour

    def _compute_brightness(self, frame: np.ndarray, pixel_mask=None) -> float:
        """Mean grayscale intensity as a lighting proxy, EMA-smoothed."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        if pixel_mask is not None:
            if pixel_mask.shape!=gray.shape or not np.any(pixel_mask):raise ValueError('Invalid context ROI mask')
            values=gray[pixel_mask>0]
        else:values=gray
        self._brightness_ema = self._ema(float(values.mean()), self._brightness_ema)
        return self._brightness_ema

    def _compute_fp_rate(self, tracks: List) -> float:
        """
        Estimate false-positive rate as fraction of tracks
        younger than MIN_TRACK_AGE.
        """
        if not tracks:
            return 0.0
        young = sum(1 for t in tracks if t.age < MIN_TRACK_AGE)
        return young / len(tracks)
