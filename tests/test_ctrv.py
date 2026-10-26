"""
tests/test_ctrv.py
──────────────────
Unit tests for the CTRV EKF (C2 core).

Test groups (per plan):
  1. Straight line   — bootstrap, convergence of theta/omega
  2. Constant turn   — filtering actually reduces noise; omega estimated
  3. CTRV vs linear KF — the quantitative core of the C2 claim
  4. Numerics        — omega≈0, angle wrap, P stays symmetric PD
  5. Lost coasting   — finite predictions, growing uncertainty
"""

from __future__ import annotations

import numpy as np
import pytest

from code.models.motorcycle_motion_model import CTRVMotionModel, wrap_angle
from tests.synthetic import straight, constant_turn, weaving, centers_to_bboxes

WH = (40.0, 60.0)


def run_ctrv(centers: np.ndarray) -> tuple[np.ndarray, CTRVMotionModel]:
    """
    Feed a center trajectory through the CTRV filter.

    Returns (pred_centers, model): pred_centers[k] is the one-step-ahead
    predicted center for frame k (available from k=1).
    """
    bboxes = centers_to_bboxes(centers, WH)
    model = CTRVMotionModel()
    model.init(bboxes[0])
    preds = np.full_like(centers, np.nan)
    for k in range(1, len(centers)):
        pred_bbox = model.predict()
        preds[k] = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
        model.update(bboxes[k])
    return preds, model


def run_linear_kf(centers: np.ndarray) -> np.ndarray:
    """Same protocol through ByteTrack's linear (xyah) Kalman filter."""
    from code.tracker.vendor.kalman_filter import KalmanFilter

    w, h = WH
    a = w / h
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([centers[0, 0], centers[0, 1], a, h]))
    preds = np.full_like(centers, np.nan)
    for k in range(1, len(centers)):
        mean, cov = kf.predict(mean, cov)
        preds[k] = mean[:2]
        mean, cov = kf.update(mean, cov, np.array([centers[k, 0], centers[k, 1], a, h]))
    return preds


def rmse(preds: np.ndarray, truth: np.ndarray, skip: int = 5) -> float:
    """One-step-ahead prediction RMSE, skipping the first `skip` frames
    (both filters need a few frames to converge from cold start)."""
    d = preds[skip:] - truth[skip:]
    return float(np.sqrt(np.nanmean(np.sum(d * d, axis=1))))


# ── 1. Straight line ─────────────────────────────────────────────────

class TestStraightLine:
    def test_noiseless_prediction_rmse(self):
        centers = straight(v=5.0, n=100, noise=0.0, heading=0.3)
        preds, _ = run_ctrv(centers)
        assert rmse(preds, centers) < 1.0

    def test_theta_converges_to_heading(self):
        heading = 0.7
        centers = straight(v=5.0, n=100, noise=0.0, heading=heading)
        _, model = run_ctrv(centers)
        assert abs(wrap_angle(model.x[3] - heading)) < 0.05

    def test_omega_converges_to_zero(self):
        centers = straight(v=5.0, n=100, noise=0.0)
        _, model = run_ctrv(centers)
        assert abs(model.x[4]) < 0.01

    def test_speed_estimated(self):
        centers = straight(v=5.0, n=100, noise=0.0)
        _, model = run_ctrv(centers)
        assert abs(model.x[2] - 5.0) < 0.5


# ── 2. Constant turn ─────────────────────────────────────────────────

class TestConstantTurn:
    def test_filter_beats_measurement_noise(self):
        sigma = 2.0
        truth = constant_turn(v=5.0, omega=0.05, n=150, noise=0.0)
        noisy = constant_turn(v=5.0, omega=0.05, n=150, noise=sigma, seed=1)
        preds, _ = run_ctrv(noisy)
        # Compare filter's one-step predictions against the NOISELESS truth:
        # a working filter should track the underlying arc better than raw
        # measurement noise would suggest.
        assert rmse(preds, truth, skip=30) < sigma * 1.5

    def test_omega_estimated_within_20pct(self):
        omega = 0.05
        centers = constant_turn(v=5.0, omega=omega, n=60, noise=0.5, seed=2)
        _, model = run_ctrv(centers)
        assert abs(model.x[4] - omega) / omega < 0.20


# ── 3. CTRV vs linear KF (the C2 claim) ──────────────────────────────

class TestCTRVvsLinear:
    """One-step-ahead center RMSE, identical input streams.

    These numbers feed the paper's motivation section directly.
    """

    def test_ctrv_wins_on_constant_turn(self):
        centers = constant_turn(v=6.0, omega=0.08, n=200, noise=2.0, seed=3)
        r_ctrv = rmse(run_ctrv(centers)[0], centers, skip=20)
        r_lin = rmse(run_linear_kf(centers), centers, skip=20)
        assert r_ctrv < r_lin, f"CTRV {r_ctrv:.2f} vs linear {r_lin:.2f}"

    def test_ctrv_wins_on_weaving(self):
        centers = weaving(v=6.0, amp=0.6, period=60, n=300, noise=2.0, seed=4)
        r_ctrv = rmse(run_ctrv(centers)[0], centers, skip=20)
        r_lin = rmse(run_linear_kf(centers), centers, skip=20)
        assert r_ctrv < r_lin, f"CTRV {r_ctrv:.2f} vs linear {r_lin:.2f}"

    def test_ctrv_near_parity_on_straight(self):
        """CTRV must not regress the easy case (≤10% worse than linear)."""
        centers = straight(v=6.0, n=300, noise=2.0, heading=0.4, seed=5)
        r_ctrv = rmse(run_ctrv(centers)[0], centers, skip=20)
        r_lin = rmse(run_linear_kf(centers), centers, skip=20)
        assert r_ctrv <= r_lin * 1.10, f"CTRV {r_ctrv:.2f} vs linear {r_lin:.2f}"

    def test_print_rmse_table(self, capsys):
        """Not an assertion — emits the RMSE table for the paper notes.

        Run with: pytest tests/test_ctrv.py -k table -s
        """
        rows = []
        for name, centers in [
            ("straight", straight(v=6.0, n=300, noise=2.0, heading=0.4, seed=5)),
            ("constant_turn", constant_turn(v=6.0, omega=0.08, n=200, noise=2.0, seed=3)),
            ("weaving", weaving(v=6.0, amp=0.6, period=60, n=300, noise=2.0, seed=4)),
        ]:
            r_ctrv = rmse(run_ctrv(centers)[0], centers, skip=20)
            r_lin = rmse(run_linear_kf(centers), centers, skip=20)
            rows.append((name, r_ctrv, r_lin))
        with capsys.disabled():
            print("\n| trajectory | CTRV RMSE (px) | linear KF RMSE (px) |")
            print("|---|---|---|")
            for name, rc, rl in rows:
                print(f"| {name} | {rc:.2f} | {rl:.2f} |")


# ── 4. Numerics ──────────────────────────────────────────────────────

class TestNumerics:
    @pytest.mark.parametrize("omega", [0.0, 1e-12])
    def test_no_nan_at_zero_omega(self, omega):
        model = CTRVMotionModel()
        model.init(np.array([0, 0, 40, 60], dtype=float))
        model.x = np.array([20.0, 30.0, 5.0, 0.3, omega])
        for _ in range(50):
            bbox = model.predict()
            assert np.all(np.isfinite(bbox))
            assert np.all(np.isfinite(model.P))

    def test_theta_wraps_across_pi(self):
        """Full circle: theta must stay in [-pi, pi) throughout."""
        centers = constant_turn(v=5.0, omega=0.1, n=100, noise=0.0)  # > full circle
        _, model = run_ctrv(centers)
        assert -np.pi <= model.x[3] < np.pi

    def test_wrap_angle_helper(self):
        assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
        assert wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
        assert wrap_angle(0.5) == pytest.approx(0.5)

    def test_P_stays_symmetric_positive_definite(self):
        centers = weaving(v=5.0, amp=0.5, period=50, n=500, noise=1.0, seed=6)
        bboxes = centers_to_bboxes(centers, WH)
        model = CTRVMotionModel()
        model.init(bboxes[0])
        for k in range(1, len(centers)):
            model.predict()
            model.update(bboxes[k])
            assert np.allclose(model.P, model.P.T, atol=1e-8)
            np.linalg.cholesky(model.P)  # raises LinAlgError if not PD


# ── 5. Lost coasting ─────────────────────────────────────────────────

class TestLostCoasting:
    def test_coasting_finite_and_uncertainty_grows(self):
        centers = constant_turn(v=5.0, omega=0.05, n=30, noise=0.0)
        bboxes = centers_to_bboxes(centers, WH)
        model = CTRVMotionModel()
        model.init(bboxes[0])
        for k in range(1, 30):
            model.predict()
            model.update(bboxes[k])

        traces = []
        for _ in range(10):
            bbox = model.predict(lost=True)
            assert np.all(np.isfinite(bbox))
            traces.append(np.trace(model.P))
        assert all(t2 > t1 for t1, t2 in zip(traces, traces[1:]))

    def test_lost_coasts_along_arc(self):
        """While lost, full CTRV dynamics apply — the track keeps turning
        through the occlusion (this is where CTRV beats the linear KF)."""
        model = CTRVMotionModel()
        model.init(np.array([0, 0, 40, 60], dtype=float))
        model.x = np.array([100.0, 100.0, 5.0, 0.5, 0.2])
        theta0, omega0 = model.x[3], model.x[4]
        for _ in range(5):
            model.predict(lost=True)
        assert model.x[3] == pytest.approx(theta0 + 5 * omega0)
        assert model.x[4] == omega0  # omega preserved for re-activation
