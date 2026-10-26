"""
tests/test_pipeline.py
───────────────────────
Integration coverage for the assembled pipeline. Until now every component
was tested in isolation and `Pipeline` itself had none, which hid two code
paths that have *never executed on real data*:

  NIGHT   the only labelled footage is an 08:00 clip, and tracker replay
          feeds a constant-brightness dummy frame, so the NIGHT row of
          PARAM_TABLE has never once been selected end-to-end.
  C3 accident branch
          the clip contains no accidents, so `tracks_accident` filtering
          and the alert hand-off have never been exercised on footage.

Both are blocked on data that has to come from elsewhere. These tests make
sure that when that data arrives, the data is the only unknown — the first
night clip should not also be the first execution of the night path.
"""

from __future__ import annotations

import numpy as np
import pytest

from code.models.adaptive_controller import PARAM_TABLE, TrafficCondition
from code.pipeline import ACCIDENT_CLASS_ID, Pipeline

CAR, MOTORCYCLE = 0, 1


def frame_of(value: int) -> np.ndarray:
    return np.full((450, 800, 3), value, dtype=np.uint8)


def dets(*rows) -> np.ndarray:
    """rows of (x1, y1, x2, y2, conf, class_id)."""
    return np.array(rows, dtype=float).reshape(-1, 6)


def moving_moto(n: int, x0: float = 100.0, dx: float = 4.0,
                conf: float = 0.9) -> list[np.ndarray]:
    return [dets((x0 + dx * i, 200, x0 + dx * i + 40, 260, conf, MOTORCYCLE))
            for i in range(n)]


@pytest.fixture
def pipe():
    p = Pipeline.from_config("configs/exp3_C1C2C3.yaml",
                             dry_run_alerts=True, build_detector=False)
    p.reset()
    return p


class TestNightPath:
    """PARAM_TABLE[NIGHT] -> tracker, driven only by frame content."""

    def test_dark_frames_select_night_params(self, pipe):
        pipe.context_estimator.hour_override = 22
        stream = moving_moto(60)
        for d in stream:
            out = pipe.process(frame_of(25), detections=d)
        night = PARAM_TABLE[TrafficCondition.NIGHT]
        assert out["state"].brightness < 80.0
        assert out["params"] == night
        # and it actually reached the tracker, not just the controller
        assert pipe.tracker.args.track_thresh == night.yolo_conf
        assert pipe.tracker.max_time_lost == night.buffer_len

    def test_bright_frames_do_not_select_night(self, pipe):
        pipe.context_estimator.hour_override = 8
        for d in moving_moto(60):
            out = pipe.process(frame_of(180), detections=d)
        assert out["state"].brightness > 80.0
        assert out["params"] != PARAM_TABLE[TrafficCondition.NIGHT]

    def test_brightness_is_ema_smoothed_not_instant(self, pipe):
        """A single dark frame must not flip the whole pipeline — alpha is
        0.1 and hysteresis is 15 frames, so a camera glitch cannot switch
        the system into night mode."""
        pipe.context_estimator.hour_override = 8
        for d in moving_moto(30):
            pipe.process(frame_of(200), detections=d)
        out = pipe.process(frame_of(0), detections=moving_moto(1)[0])
        assert out["state"].brightness > 80.0
        assert out["params"] != PARAM_TABLE[TrafficCondition.NIGHT]


class TestAccidentBranch:
    """C3's precision branch: the half of the dual-task claim that the
    current footage cannot test."""

    def test_accident_track_reaches_the_accident_branch(self, pipe):
        pipe.context_estimator.hour_override = 8
        acc = [dets((300 + 2 * i, 200, 360 + 2 * i, 260, 0.95, ACCIDENT_CLASS_ID))
               for i in range(40)]
        for d in acc:
            out = pipe.process(frame_of(128), detections=d)
        classes = {t.class_id for t in out["tracks_accident"]}
        assert ACCIDENT_CLASS_ID in classes

    def test_accident_branch_is_a_subset_of_mappo_branch(self, pipe):
        """tau_acc > tau_mappo by construction, so the precision branch can
        never contain a track the recall branch dropped. If this inverts,
        the dual-task split is broken."""
        pipe.context_estimator.hour_override = 8
        for d in moving_moto(40):
            out = pipe.process(frame_of(128), detections=d)
        acc_ids = {t.track_id for t in out["tracks_accident"]}
        mappo_ids = {t.track_id for t in out["tracks_mappo"]}
        assert acc_ids <= mappo_ids

    def test_no_alert_without_a_classifier(self, pipe):
        """Severity weights are absent in this repo, so the branch must
        degrade to silence rather than raise."""
        pipe.context_estimator.hour_override = 8
        for d in moving_moto(20):
            out = pipe.process(frame_of(128), detections=d)
        assert out["alerts"] == []


class TestAblationFlags:
    def test_c3_off_passes_everything_through(self):
        p = Pipeline.from_config("configs/exp2_C1C2.yaml",
                                 dry_run_alerts=True, build_detector=False)
        p.reset()
        p.context_estimator.hour_override = 8
        for d in moving_moto(20):
            out = p.process(frame_of(128), detections=d)
        assert out["scores"] == {}
        assert out["tracks_accident"] == out["tracks_mappo"]

    def test_c1_off_leaves_tracker_at_constructor_defaults(self):
        p = Pipeline.from_config("configs/exp1_C2.yaml",
                                 dry_run_alerts=True, build_detector=False)
        p.reset()
        for d in moving_moto(30):
            out = p.process(frame_of(25), detections=d)   # dark, but C1 off
        assert out["params"] is None
        assert p.tracker.args.track_thresh == 0.5

    def test_latency_is_reported_per_frame(self, pipe):
        out = pipe.process(frame_of(128), detections=moving_moto(1)[0])
        assert out["latency_ms"] > 0.0

    def test_stage_breakdown_accounts_for_the_whole_frame(self, pipe):
        out = pipe.process(frame_of(128), detections=moving_moto(1)[0])
        stages = out["stages_ms"]
        assert set(stages) == {"context", "detect", "track", "c3", "accident"}
        # The stages tile the frame: their sum is the total, up to the cost of
        # the perf_counter calls between them.
        assert sum(stages.values()) == pytest.approx(out["latency_ms"], abs=0.5)

    def test_supplied_detections_report_zero_detect_time(self, pipe):
        """The gap between a replay latency number and an end-to-end one.

        Replay runs feed cached detections, so `detect` is genuinely free —
        which is exactly why replay latency must not be quoted as real-time
        throughput.
        """
        out = pipe.process(frame_of(128), detections=moving_moto(1)[0])
        assert out["stages_ms"]["detect"] == pytest.approx(0.0, abs=0.05)
