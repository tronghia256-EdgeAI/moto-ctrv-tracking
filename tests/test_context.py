"""
tests/test_context.py
─────────────────────
Unit tests for C1: ContextEstimator + AdaptiveController.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from code.models.context_estimator import ContextEstimator, TrafficState, MIN_TRACK_AGE
from code.models.adaptive_controller import (
    AdaptiveController,
    Params,
    TrafficCondition,
    PARAM_TABLE,
)


def mock_tracks(ages):
    return [SimpleNamespace(age=a) for a in ages]


def frame_of(value, hw=(120, 160)):
    return np.full((*hw, 3), value, dtype=np.uint8)


# ── ContextEstimator ─────────────────────────────────────────────────

class TestContextEstimator:
    def test_density_first_frame_exact(self):
        est = ContextEstimator(hour_override=12)
        state = est.estimate(frame_of(128), mock_tracks([10] * 7))
        assert state.density == 7

    def test_brightness_dark_and_bright(self):
        est = ContextEstimator(hour_override=12)
        assert est.estimate(frame_of(30), []).brightness == pytest.approx(30, abs=1)
        est.reset()
        assert est.estimate(frame_of(200), []).brightness == pytest.approx(200, abs=1)

    def test_grayscale_frame_accepted(self):
        est = ContextEstimator(hour_override=12)
        gray = np.full((120, 160), 99, dtype=np.uint8)
        assert est.estimate(gray, []).brightness == pytest.approx(99, abs=1)

    def test_hour_override(self):
        est = ContextEstimator(hour_override=3)
        assert est.estimate(frame_of(128), []).hour == 3

    def test_fp_rate(self):
        est = ContextEstimator(hour_override=12)
        # 2 of 4 tracks younger than MIN_TRACK_AGE
        tracks = mock_tracks([0, 1, 10, 20])
        assert MIN_TRACK_AGE == 3
        assert est.estimate(frame_of(128), tracks).fp_rate == 0.5

    def test_fp_rate_empty_tracks(self):
        est = ContextEstimator(hour_override=12)
        assert est.estimate(frame_of(128), []).fp_rate == 0.0

    def test_ema_smooths_flicker(self):
        """A single bright frame in a dark stream must not drag the
        smoothed brightness across the night threshold (80)."""
        est = ContextEstimator(hour_override=22)
        for _ in range(30):
            est.estimate(frame_of(40), [])
        state = est.estimate(frame_of(255), [])  # one glare frame
        assert state.brightness < 80.0

    def test_ema_converges(self):
        est = ContextEstimator(hour_override=12)
        for _ in range(80):
            state = est.estimate(frame_of(150), [])
        assert state.brightness == pytest.approx(150, abs=2)


# ── AdaptiveController: classification truth table ───────────────────

class TestClassification:
    @pytest.mark.parametrize("density,hour,brightness,expected", [
        (40, 8, 120.0, TrafficCondition.DENSE_PEAK),   # dense + peak
        (40, 8, 79.9, TrafficCondition.NIGHT),          # night wins over dense
        (40, 3, 120.0, TrafficCondition.DEFAULT),       # dense but off-peak
        (10, 8, 120.0, TrafficCondition.DEFAULT),       # peak but sparse
        (10, 3, 50.0, TrafficCondition.NIGHT),          # plain night
        (10, 12, 120.0, TrafficCondition.DEFAULT),      # plain day
        # boundaries
        (30, 8, 120.0, TrafficCondition.DENSE_PEAK),    # density == threshold
        (29, 8, 120.0, TrafficCondition.DEFAULT),
        (40, 8, 80.0, TrafficCondition.DENSE_PEAK),     # brightness == threshold → not night
        (40, 7, 120.0, TrafficCondition.DENSE_PEAK),    # first peak hour
        (40, 20, 120.0, TrafficCondition.DENSE_PEAK),   # last peak hour
        (40, 21, 120.0, TrafficCondition.DEFAULT),      # just past peak
    ])
    def test_truth_table(self, density, hour, brightness, expected):
        ctrl = AdaptiveController()
        state = TrafficState(density=density, hour=hour,
                             brightness=brightness, fp_rate=0.0)
        assert ctrl._classify(state) == expected


# ── AdaptiveController: hysteresis ───────────────────────────────────

def night_state():
    return TrafficState(density=10, hour=22, brightness=40.0, fp_rate=0.0)


def default_state():
    return TrafficState(density=10, hour=12, brightness=150.0, fp_rate=0.0)


class TestHysteresis:
    def test_short_flicker_does_not_switch(self):
        ctrl = AdaptiveController()
        for _ in range(10):
            ctrl.get_params(default_state())
        # 5 frames of "night" — below the 15-frame hysteresis
        for _ in range(5):
            params = ctrl.get_params(night_state())
        assert params == PARAM_TABLE[TrafficCondition.DEFAULT]
        # back to default — still default
        params = ctrl.get_params(default_state())
        assert params == PARAM_TABLE[TrafficCondition.DEFAULT]

    def test_sustained_condition_switches(self):
        ctrl = AdaptiveController()
        for _ in range(10):
            ctrl.get_params(default_state())
        for _ in range(20):
            params = ctrl.get_params(night_state())
        assert params == PARAM_TABLE[TrafficCondition.NIGHT]

    def test_interrupted_candidate_resets_counter(self):
        ctrl = AdaptiveController()
        for _ in range(10):
            ctrl.get_params(default_state())
        # 14 night frames, 1 default frame, 14 night frames → never switches
        for _ in range(14):
            ctrl.get_params(night_state())
        ctrl.get_params(default_state())
        for _ in range(14):
            params = ctrl.get_params(night_state())
        assert params == PARAM_TABLE[TrafficCondition.DEFAULT]

    def test_reset(self):
        ctrl = AdaptiveController()
        for _ in range(20):
            ctrl.get_params(night_state())
        assert ctrl._active == TrafficCondition.NIGHT
        ctrl.reset()
        assert ctrl.get_params(default_state()) == PARAM_TABLE[TrafficCondition.DEFAULT]


# ── End-to-end: TrafficState → Params rows match PARAM_TABLE ─────────

class TestParamTable:
    def test_dense_peak_row(self):
        """Tuned against `dense` GT — see experiments/c1_sweep/evidence.md.
        The shipped guess was 0.35 / 45, which cost MOTA 0.917 -> 0.731."""
        p = PARAM_TABLE[TrafficCondition.DENSE_PEAK]
        assert p == Params(yolo_conf=0.50, buffer_len=30, motion_model="ctrv")

    def test_dense_peak_is_stricter_than_default(self):
        """Records a finding, not a design intent: C1 was built on the
        premise that dense traffic wants a *lower* threshold to survive
        occlusion. Measured against GT the opposite holds — lowering it
        floods false positives — so the tuned DENSE_PEAK row is stricter
        than DEFAULT. If `mixed` GT later overturns this, that is a real
        result and this assertion should flip with it."""
        assert (PARAM_TABLE[TrafficCondition.DENSE_PEAK].yolo_conf
                > PARAM_TABLE[TrafficCondition.DEFAULT].yolo_conf)

    def test_night_row(self):
        p = PARAM_TABLE[TrafficCondition.NIGHT]
        assert p == Params(yolo_conf=0.55, buffer_len=20, motion_model="conservative")

    def test_default_row(self):
        p = PARAM_TABLE[TrafficCondition.DEFAULT]
        assert p == Params(yolo_conf=0.45, buffer_len=30, motion_model="standard")

    def test_estimator_to_params_end_to_end(self):
        est = ContextEstimator(hour_override=8)
        ctrl = AdaptiveController(hysteresis_frames=1)
        tracks = mock_tracks([10] * 40)  # dense
        params = None
        for _ in range(3):
            state = est.estimate(frame_of(150), tracks)
            params = ctrl.get_params(state)
        assert params == PARAM_TABLE[TrafficCondition.DENSE_PEAK]
