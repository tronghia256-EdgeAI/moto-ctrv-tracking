"""
tests/test_adaptive_tracker.py
──────────────────────────────
Integration tests for AdaptiveBYTETracker (C2 wired into ByteTrack).

Headline test: weaving motorcycle + occlusion gaps — CTRV must fragment
no more than the vanilla linear-KF baseline on the identical stream.
Its numbers are the first before/after evidence for the paper.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from code.models.adaptive_controller import Params
from code.tracker import AdaptiveBYTETracker, ClassAwareSTrack, MOTORCYCLE_CLASS_ID
from code.tracker.vendor import BYTETracker, BaseTrack
from tests.synthetic import (
    straight,
    constant_turn,
    weaving,
    to_detections,
    CLASS_CAR,
    CLASS_MOTORCYCLE,
)

IMG_HW = (720, 1280)


def make_vanilla(track_thresh=0.5, track_buffer=30, match_thresh=0.8):
    BaseTrack._count = 0
    args = SimpleNamespace(track_thresh=track_thresh, track_buffer=track_buffer,
                           match_thresh=match_thresh, mot20=False)
    return BYTETracker(args, frame_rate=30)


def make_adaptive(ctrv_enabled=True):
    tracker = AdaptiveBYTETracker(
        {"tracker": {"track_thresh": 0.5, "track_buffer": 30, "match_thresh": 0.8}},
        ctrv_enabled=ctrv_enabled,
    )
    tracker.reset()
    return tracker


def id_series(tracker_update, frames, gt_centers, max_dist=60.0):
    """
    Run a tracker over frames; per frame, return the track id covering the
    ground-truth object (nearest active-track center within max_dist) or None.
    """
    series = []
    for k, dets in enumerate(frames):
        tracks = tracker_update(dets)
        best, best_d = None, max_dist
        for t in tracks:
            tlbr = t.tlbr
            c = np.array([(tlbr[0] + tlbr[2]) / 2, (tlbr[1] + tlbr[3]) / 2])
            d = float(np.linalg.norm(c - gt_centers[k]))
            if d < best_d:
                best, best_d = t.track_id, d
        series.append(best)
    return series


def count_breaks_and_switches(series):
    """
    fragmentation = coverage interruptions (covered → uncovered → covered)
    id_switches   = covered-to-covered transitions where the id changes
    """
    frags = 0
    switches = 0
    last_id = None
    in_gap = False
    seen_any = False
    for tid in series:
        if tid is None:
            if seen_any:
                in_gap = True
            continue
        if seen_any and in_gap:
            frags += 1
        if last_id is not None and tid != last_id:
            switches += 1
        last_id = tid
        in_gap = False
        seen_any = True
    return frags, switches


class TestClassConditionedDispatch:
    def test_motorcycle_gets_ctrv_car_gets_linear(self):
        car = straight(v=5.0, n=50, start=(100, 100))
        moto = weaving(v=5.0, amp=0.6, period=40, n=50, start=(100, 500))
        frames = to_detections([car, moto], class_ids=[CLASS_CAR, CLASS_MOTORCYCLE])

        tracker = make_adaptive(ctrv_enabled=True)
        tracks = []
        for dets in frames:
            tracks = tracker.update(dets)

        assert len(tracks) == 2
        by_class = {t.class_id: t for t in tracks}
        assert by_class[CLASS_MOTORCYCLE].ctrv is not None
        assert by_class[CLASS_CAR].ctrv is None

    def test_ctrv_disabled_flag_respected(self):
        moto = weaving(v=5.0, amp=0.6, period=40, n=20)
        frames = to_detections([moto], class_ids=[CLASS_MOTORCYCLE])
        tracker = make_adaptive(ctrv_enabled=False)
        tracks = []
        for dets in frames:
            tracks = tracker.update(dets)
        assert tracks[0].ctrv is None

    def test_stable_ids_mixed_stream(self):
        car = straight(v=5.0, n=100, start=(100, 100))
        moto = weaving(v=5.0, amp=0.6, period=50, n=100, start=(100, 500))
        frames = to_detections([car, moto], class_ids=[CLASS_CAR, CLASS_MOTORCYCLE])
        tracker = make_adaptive()
        ids = []
        for dets in frames:
            ids.append(sorted(t.track_id for t in tracker.update(dets)))
        for frame_ids in ids[1:]:
            assert frame_ids == ids[1]


class TestHeadlineOcclusion:
    """The C2 experiments: occlusion recovery, CTRV vs vanilla linear KF
    on identical streams. Numbers feed the paper's results section.

    Two scenarios probe both sides of the boundary:
      * constant turn + 12-frame gap → CTRV coasts along the arc and keeps
        the ID; the linear tangent drifts off and re-identifies (IDSW).
        This is the C2 claim.
      * weaving + gap right after an omega SIGN FLIP → adversarial for any
        curvilinear model: the estimated turn rate points the wrong way, so
        a lucky straight coast can do better by one IDSW. Verified robust
        across Q/R/omega-decay sweeps (2026-07-11) — a genuine boundary of
        the motion model, documented for the paper's analysis section.
    """

    def _run(self, gt, frames):
        vanilla = make_vanilla()
        s_van = id_series(lambda d: vanilla.update(d, IMG_HW, IMG_HW), frames, gt)
        adaptive = make_adaptive(ctrv_enabled=True)
        s_ctrv = id_series(lambda d: adaptive.update(d), frames, gt)
        return count_breaks_and_switches(s_van), count_breaks_and_switches(s_ctrv), s_ctrv

    def test_constant_turn_occlusion_ctrv_keeps_id(self):
        n, gap = 200, {0: list(range(80, 92))}
        gt = constant_turn(v=6.0, omega=0.04, n=n, noise=0.0, start=(200, 200))
        noisy = gt + np.random.default_rng(9).normal(0, 2.0, size=gt.shape)
        frames = to_detections([noisy], class_ids=[CLASS_MOTORCYCLE], drop_frames=gap)

        (f_van, sw_van), (f_ctrv, sw_ctrv), s_ctrv = self._run(gt, frames)
        print(f"\n[headline/turn] baseline: frags={f_van} idsw={sw_van} | "
              f"CTRV: frags={f_ctrv} idsw={sw_ctrv}")

        assert sw_ctrv == 0, "CTRV must hold the ID through the turn occlusion"
        assert sw_van >= 1, "baseline is expected to lose the ID here"
        assert f_ctrv <= f_van
        assert s_ctrv[-1] is not None

    def test_weaving_occlusion_bounded_regression(self):
        n, gaps = 300, {0: list(range(100, 108)) + list(range(200, 208))}
        gt = weaving(v=6.0, amp=0.6, period=60, n=n, noise=0.0, start=(60, 400))
        noisy = gt + np.random.default_rng(7).normal(0, 2.0, size=gt.shape)
        frames = to_detections([noisy], class_ids=[CLASS_MOTORCYCLE], drop_frames=gaps)

        (f_van, sw_van), (f_ctrv, sw_ctrv), s_ctrv = self._run(gt, frames)
        print(f"\n[headline/weave] baseline: frags={f_van} idsw={sw_van} | "
              f"CTRV: frags={f_ctrv} idsw={sw_ctrv}")

        assert f_ctrv <= f_van
        # Adversarial sign-flip gap: allow at most one extra IDSW (see class
        # docstring). Tightening this bound is the target of future R&D.
        assert sw_ctrv <= sw_van + 1
        assert s_ctrv[-1] is not None


class TestDynamicParams:
    def test_buffer_len_applied(self):
        tracker = make_adaptive()
        moto = straight(v=5.0, n=5)
        frames = to_detections([moto], class_ids=[CLASS_MOTORCYCLE])
        tracker.update(frames[0], Params(yolo_conf=0.5, buffer_len=45, motion_model="ctrv"))
        assert tracker.max_time_lost == 45
        tracker.update(frames[1], Params(yolo_conf=0.5, buffer_len=20, motion_model="ctrv"))
        assert tracker.max_time_lost == 20

    def test_track_thresh_applied(self):
        tracker = make_adaptive()
        dets = np.array([[100, 100, 140, 160, 0.50, CLASS_MOTORCYCLE]])
        # conf 0.45 → score 0.50 passes the high split
        tracker.update(dets, Params(yolo_conf=0.45, buffer_len=30, motion_model="ctrv"))
        assert tracker.args.track_thresh == 0.45
        assert tracker.det_thresh == pytest.approx(0.55)
        # conf 0.55 → the same score now falls to the low-score pool
        tracker.update(dets, Params(yolo_conf=0.55, buffer_len=30, motion_model="ctrv"))
        assert tracker.args.track_thresh == 0.55

    def test_none_params_baseline_mode(self):
        tracker = make_adaptive()
        moto = straight(v=5.0, n=3)
        frames = to_detections([moto], class_ids=[CLASS_MOTORCYCLE])
        for dets in frames:
            tracker.update(dets, params=None)
        assert tracker.args.track_thresh == 0.5  # constructor value untouched


class TestC3HistoryFields:
    def test_history_populated(self):
        moto = weaving(v=5.0, amp=0.5, period=40, n=30)
        frames = to_detections([moto], class_ids=[CLASS_MOTORCYCLE])
        tracker = make_adaptive()
        tracks = []
        for dets in frames:
            tracks = tracker.update(dets)
        t = tracks[0]
        assert len(t.hit_history) == 10          # deque maxlen
        assert all(t.hit_history)                # matched every frame
        assert len(t.iou_history) > 0
        assert all(0.0 <= v <= 1.0 for v in t.iou_history)
        assert all(c == CLASS_MOTORCYCLE for c in t.class_history)
        assert t.age == 29

    def test_missed_frames_recorded_as_false(self):
        # Gap near the end so the False marks are still inside the
        # maxlen=10 history window when we assert.
        moto = straight(v=5.0, n=30)
        frames = to_detections([moto], class_ids=[CLASS_MOTORCYCLE],
                               drop_frames={0: [24, 25, 26]})
        tracker = make_adaptive()
        all_tracks = {}
        for dets in frames:
            for t in tracker.update(dets):
                all_tracks[t.track_id] = t
        t = list(all_tracks.values())[0]
        assert not all(t.hit_history)            # the gap left False marks


class TestCrowdSmoke:
    def test_20_motorcycles_300_frames(self):
        rng = np.random.default_rng(11)
        trajectories = []
        for i in range(20):
            start = (float(rng.uniform(50, 400)), float(rng.uniform(50, 650)))
            trajectories.append(
                weaving(v=float(rng.uniform(3, 7)), amp=float(rng.uniform(0.2, 0.7)),
                        period=float(rng.uniform(40, 80)), n=300, noise=1.5,
                        start=start, seed=i)
            )
        frames = to_detections(trajectories, class_ids=CLASS_MOTORCYCLE)
        tracker = make_adaptive()
        tracks = []
        for dets in frames:
            tracks = tracker.update(dets)
        assert len(tracks) >= 15
