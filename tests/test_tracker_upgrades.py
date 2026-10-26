"""
tests/test_tracker_upgrades.py
───────────────────────────────
Tests for the two C2 upgrades (2026-07-14):
  1. CTRV uncertainty gating in the first association (P11 hook)
  2. Scale-adaptive process/measurement noise
"""

from __future__ import annotations

import numpy as np
import pytest

from code.models.motorcycle_motion_model import (
    CTRVMotionModel, REF_BOX_SIZE, SCALE_CLAMP,
)
from code.tracker.adaptive_bytetrack import (
    AdaptiveBYTETracker, ClassAwareSTrack,
)
from code.tracker.vendor import TrackState


def make_tracker(gating: bool) -> AdaptiveBYTETracker:
    cfg = {"tracker": {"uncertainty_gating": gating, "track_thresh": 0.5}}
    t = AdaptiveBYTETracker(cfg, ctrv_enabled=True)
    t.reset()
    return t


def make_lost_ctrv_track(tracker, tlbr, n_lost_predicts=15):
    ClassAwareSTrack.motion_model_this_frame = "ctrv"
    tr = ClassAwareSTrack(
        ClassAwareSTrack.tlbr_to_tlwh(np.asarray(tlbr, float)), 0.9, 1)
    tr.activate(tracker.kalman_filter, 1)
    assert tr.ctrv is not None, "motorcycle track must get a CTRV filter"
    tr.state = TrackState.Lost
    for _ in range(n_lost_predicts):   # occlusion: covariance inflates
        tr.ctrv.predict(lost=True)
    return tr


def make_detection(tlbr, class_id=1, score=0.8):
    return ClassAwareSTrack(
        ClassAwareSTrack.tlbr_to_tlwh(np.asarray(tlbr, float)),
        score, class_id)


class TestUncertaintyGating:
    def test_gate_rescues_nearby_same_class_detection(self):
        """A detection with ZERO box IoU but inside the inflated
        innovation ellipse must get a finite (matchable) cost."""
        tracker = make_tracker(gating=True)
        tr = make_lost_ctrv_track(tracker, [100, 100, 140, 160])
        det = make_detection([140, 100, 180, 160])  # 40 px off, IoU = 0
        dists = tracker._first_association_cost([tr], [det])
        assert dists[0, 0] < 1.0, "gate should rescue this detection"
        assert dists[0, 0] >= tracker.GATE_BASE_COST, \
            "rescued cost must never beat a genuine IoU match"

    # NB: the vendored IoU keeps upstream's +1 pixel-area convention, so
    # boxes must be separated by a gap (not merely touching) for IoU = 0.
    def test_gate_off_leaves_plain_iou(self):
        tracker = make_tracker(gating=False)
        tr = make_lost_ctrv_track(tracker, [100, 100, 140, 160])
        det = make_detection([150, 100, 190, 160])
        dists = tracker._first_association_cost([tr], [det])
        assert dists[0, 0] == pytest.approx(1.0)   # IoU 0 → cost 1

    def test_gate_ignores_other_classes(self):
        tracker = make_tracker(gating=True)
        tr = make_lost_ctrv_track(tracker, [100, 100, 140, 160])
        det_car = make_detection([150, 100, 190, 160], class_id=0)
        dists = tracker._first_association_cost([tr], [det_car])
        assert dists[0, 0] == pytest.approx(1.0)

    def test_gate_ignores_tracked_and_linear_tracks(self):
        tracker = make_tracker(gating=True)
        # Tracked (not lost) CTRV track: gate must not touch it.
        tr = make_lost_ctrv_track(tracker, [100, 100, 140, 160])
        tr.state = TrackState.Tracked
        det = make_detection([150, 100, 190, 160])
        assert tracker._first_association_cost(
            [tr], [det])[0, 0] == pytest.approx(1.0)
        # Linear-KF track (car): no ctrv attribute → untouched.
        ClassAwareSTrack.motion_model_this_frame = "ctrv"
        car = ClassAwareSTrack(
            ClassAwareSTrack.tlbr_to_tlwh(np.array([100., 100, 140, 160])),
            0.9, 0)
        car.activate(tracker.kalman_filter, 1)
        car.state = TrackState.Lost
        assert car.ctrv is None
        assert tracker._first_association_cost(
            [car], [make_detection([150, 100, 190, 160], class_id=0)]
        )[0, 0] == pytest.approx(1.0)

    def test_gate_rejects_far_detection(self):
        """A detection way outside the ellipse stays unmatched even for
        a lost CTRV track — the gate must not glue distant objects."""
        tracker = make_tracker(gating=True)
        tr = make_lost_ctrv_track(tracker, [100, 100, 140, 160],
                                  n_lost_predicts=3)
        det = make_detection([600, 350, 640, 410])
        dists = tracker._first_association_cost([tr], [det])
        assert dists[0, 0] == pytest.approx(1.0)

    def test_end_to_end_no_crash_and_no_regression(self):
        """Full update() loop with gating on: runs clean, keeps the ID
        through an occlusion on a constant turn."""
        from tests.synthetic import constant_turn, to_detections
        traj = constant_turn(v=5.0, omega=0.03, n=80, noise=0.5,
                             start=(200.0, 200.0))
        frames = to_detections([traj], class_ids=[1],
                               drop_frames={0: list(range(30, 44))})
        tracker = make_tracker(gating=True)
        ids = set()
        for dets in frames:
            for t in tracker.update(dets):
                ids.add(t.track_id)
        assert len(ids) == 1, f"expected a single ID, got {ids}"


class TestScaleAdaptiveNoise:
    def test_scale_factor_small_and_large(self):
        m_small = CTRVMotionModel(scale_adaptive=True)
        m_small.init(np.array([0, 0, 10, 10]))       # size 10 ≪ REF 40
        assert m_small._noise_scale2 == pytest.approx(SCALE_CLAMP[0] ** 2)

        m_big = CTRVMotionModel(scale_adaptive=True)
        m_big.init(np.array([0, 0, 160, 160]))       # size 160 = 4×REF
        assert m_big._noise_scale2 == pytest.approx(SCALE_CLAMP[1] ** 2)

        m_ref = CTRVMotionModel(scale_adaptive=True)
        m_ref.init(np.array([0, 0, REF_BOX_SIZE, REF_BOX_SIZE]))
        assert m_ref._noise_scale2 == pytest.approx(1.0)

    def test_angular_noise_unscaled(self):
        m = CTRVMotionModel(scale_adaptive=True)
        m.init(np.array([0, 0, 160, 160]))
        Q_eff = m._Q_eff()
        assert Q_eff[0, 0] == pytest.approx(m.Q[0, 0] * 16)
        assert Q_eff[3, 3] == pytest.approx(m.Q[3, 3])   # θ untouched
        assert Q_eff[4, 4] == pytest.approx(m.Q[4, 4])   # ω untouched

    def test_default_off_is_bit_identical(self):
        """scale_adaptive=False must reproduce the original filter."""
        za = CTRVMotionModel(scale_adaptive=False)
        zb = CTRVMotionModel()
        for m in (za, zb):
            m.init(np.array([100, 100, 120, 130]))
        for k in range(20):
            ba = za.predict()
            bb = zb.predict()
            np.testing.assert_allclose(ba, bb)
            z = np.array([100 + 2*k, 100 + k, 120 + 2*k, 130 + k], float)
            za.update(z)
            zb.update(z)
        np.testing.assert_allclose(za.x, zb.x)

    def test_nsa_default_off_is_bit_identical(self):
        """nsa=False (and nsa=True with score=None) must reproduce the
        original filter exactly."""
        plain = CTRVMotionModel()
        off = CTRVMotionModel(nsa=False)
        on_no_score = CTRVMotionModel(nsa=True)
        for m in (plain, off, on_no_score):
            m.init(np.array([100, 100, 120, 130]))
        for k in range(15):
            for m in (plain, off, on_no_score):
                m.predict()
            z = np.array([100 + 2*k, 100 + k, 120 + 2*k, 130 + k], float)
            plain.update(z)
            off.update(z, score=0.9)          # nsa off → score ignored
            on_no_score.update(z)             # nsa on, no score → plain R
        np.testing.assert_allclose(off.x, plain.x)
        np.testing.assert_allclose(on_no_score.x, plain.x)

    def test_nsa_confident_detection_trusted_more(self):
        """With NSA on, a high-score detection pulls the state harder and
        leaves a smaller posterior variance than a low-score one."""
        hi = CTRVMotionModel(nsa=True)
        lo = CTRVMotionModel(nsa=True)
        for m in (hi, lo):
            m.init(np.array([100, 100, 120, 130]))
            m.update(np.array([102, 100, 122, 130]), score=0.8)  # bootstrap
            m.predict()
        z = np.array([120, 110, 140, 140], float)   # offset detection
        hi.update(z, score=0.95)
        lo.update(z, score=0.15)
        z_c = np.array([130.0, 125.0])              # detection center
        d_hi = np.linalg.norm(hi.x[:2] - z_c)
        d_lo = np.linalg.norm(lo.x[:2] - z_c)
        assert d_hi < d_lo, "high-score update should land closer to z"
        assert hi.P[0, 0] < lo.P[0, 0]

    def test_small_far_object_gets_tighter_filter(self):
        """Same trajectory, small vs large box: the small object's
        position uncertainty must stay smaller after coasting."""
        small = CTRVMotionModel(scale_adaptive=True)
        small.init(np.array([100, 100, 110, 110]))
        big = CTRVMotionModel(scale_adaptive=True)
        big.init(np.array([100, 100, 260, 260]))
        for m in (small, big):
            m.update(np.array([102, 100, m.wh[0] + 102, m.wh[1] + 100]))
            for _ in range(10):
                m.predict(lost=True)
        assert small.P[0, 0] < big.P[0, 0]
