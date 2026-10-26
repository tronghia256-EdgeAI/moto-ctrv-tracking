"""
adaptive_controller.py
──────────────────────
C1 Component: Adaptive Parameter Controller

Reads a TrafficState and returns a Params dataclass that configures:
  - YOLO confidence threshold (applied as the tracker's track_thresh —
    the detector itself always runs at a low fixed conf, see
    code/tracker/adaptive_bytetrack.py)
  - ByteTrack lost-track buffer length
  - Motion model selector for ByteTrack

Traffic state categories, priority NIGHT > DENSE_PEAK > DEFAULT:
  NIGHT       : low brightness
  DENSE_PEAK  : high density + peak hour
  DEFAULT     : all other conditions

Hysteresis: a condition must persist HYSTERESIS_FRAMES consecutive frames
before the controller switches params — otherwise density/brightness noise
at a threshold boundary would make the tracker thresholds oscillate.

Note on "conservative": the NIGHT row keeps the standard linear Kalman
motion model; its conservatism is expressed through the higher confidence
threshold (0.55) and shorter buffer (20), NOT through a third motion model.
Documented so the paper does not overclaim.

Author  : Trọng Nghĩa
Paper   : Context-Aware Adaptive Perception for Mixed Traffic
Target  : ICRA 2027 (primary) / IEEE IV 2027 (backup)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

from .context_estimator import TrafficState


# ── Data structures ──────────────────────────────────────────────────

MotionModelType = Literal["ctrv", "standard", "conservative"]


@dataclass
class Params:
    """Runtime parameters for detector + tracker."""
    yolo_conf: float           # confidence threshold  (0.0–1.0)
    buffer_len: int            # ByteTrack lost-track buffer (frames)
    motion_model: MotionModelType  # which motion model to activate


class TrafficCondition(Enum):
    """Categorical traffic condition inferred from TrafficState."""
    DENSE_PEAK   = auto()
    NIGHT        = auto()
    DEFAULT      = auto()


# ── Thresholds (tune after baseline experiments) ─────────────────────

DENSITY_THRESHOLD  = 30    # TODO: determine from VN footage histogram
BRIGHTNESS_NIGHT   = 80.0  # TODO: verify against actual nighttime frames
PEAK_HOURS         = range(7, 21)  # TODO: narrow if data shows tighter peaks
HYSTERESIS_FRAMES  = 15    # consecutive frames before a condition switch


# ── Parameter table (one row per condition) ──────────────────────────
# DENSE_PEAK is tuned against human GT (experiments/c1_sweep/). The other
# two rows are still initial guesses, and deliberately stay that way:
#   DEFAULT holds only 21 of the 600 GT frames, so fitting it there would
#           be fitting noise
#   NIGHT   has never once executed — the only labelled footage is an
#           08:00 clip, and replay feeds a constant-brightness dummy frame
# Both need footage that exercises them before they mean anything.

PARAM_TABLE: dict[TrafficCondition, Params] = {
    # Tuned on the `dense` GT segment: the shipped guess (0.35 / 45) cost
    # MOTA 0.917 -> 0.731 by flooding the tracker with weak detections
    # (FP 1388 -> 6254). 0.50 / 30 is the grid optimum and is strictly
    # better than the guess at every point tested. Note it coincides with
    # the tracker's fixed default, i.e. the best this row can currently do
    # is decline to differentiate — see experiments/c1_sweep/evidence.md.
    TrafficCondition.DENSE_PEAK: Params(
        yolo_conf=0.50,
        buffer_len=30,
        motion_model="ctrv",
    ),
    TrafficCondition.NIGHT: Params(      # TODO: untuned — needs night footage
        yolo_conf=0.55,
        buffer_len=20,
        motion_model="conservative",
    ),
    TrafficCondition.DEFAULT: Params(    # TODO: untuned — needs `mixed` GT
        yolo_conf=0.45,
        buffer_len=30,
        motion_model="standard",
    ),
}


# ── Main class ───────────────────────────────────────────────────────

class AdaptiveController:
    """
    Maps a TrafficState to a Params object for the current frame.

    Usage:
        controller = AdaptiveController()
        params = controller.get_params(state)
    """

    def __init__(self,
                 density_threshold: float = DENSITY_THRESHOLD,
                 brightness_night: float = BRIGHTNESS_NIGHT,
                 peak_hours: range = PEAK_HOURS,
                 hysteresis_frames: int = HYSTERESIS_FRAMES) -> None:
        self.density_threshold = density_threshold
        self.brightness_night = brightness_night
        self.peak_hours = peak_hours
        self.hysteresis_frames = hysteresis_frames

        # Hysteresis state
        self._active: TrafficCondition = TrafficCondition.DEFAULT
        self._candidate: TrafficCondition | None = None
        self._candidate_count: int = 0

    def get_params(self, state: TrafficState) -> Params:
        """
        Args:
            state: TrafficState from ContextEstimator.

        Returns:
            Params for the current frame (hysteresis-stabilized).
        """
        raw = self._classify(state)

        if raw == self._active:
            # Condition matches what we're already doing — clear any
            # pending switch.
            self._candidate = None
            self._candidate_count = 0
        elif raw == self._candidate:
            self._candidate_count += 1
            if self._candidate_count >= self.hysteresis_frames:
                self._active = raw
                self._candidate = None
                self._candidate_count = 0
        else:
            self._candidate = raw
            self._candidate_count = 1

        return PARAM_TABLE[self._active]

    def reset(self) -> None:
        """Reset hysteresis state (between experiment runs)."""
        self._active = TrafficCondition.DEFAULT
        self._candidate = None
        self._candidate_count = 0

    # ── Private ──────────────────────────────────────────────────────

    def _classify(self, state: TrafficState) -> TrafficCondition:
        """
        Map a TrafficState to a TrafficCondition category.
        Priority: NIGHT > DENSE_PEAK > DEFAULT
        """
        if state.brightness < self.brightness_night:
            return TrafficCondition.NIGHT
        if (state.density >= self.density_threshold
                and state.hour in self.peak_hours):
            return TrafficCondition.DENSE_PEAK
        return TrafficCondition.DEFAULT
