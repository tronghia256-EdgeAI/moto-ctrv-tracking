"""
motorcycle_motion_model.py
───────────────────────────
C2 Component: CTRV Motion Model for Motorcycles

Constant Turn Rate and Velocity (CTRV) model.
Replaces the linear Kalman filter in ByteTrack for tracks
whose class is 'motorcycle'.

State vector: x = [px, py, v, theta, omega]^T
  px    : x position (pixels)
  py    : y position (pixels)
  v     : speed (pixels/frame)
  theta : heading angle (radians)
  omega : turn rate (radians/frame)

Prediction (dt = 1 frame by default):
  px_{k+1}    = px_k + v_k * cos(theta_k) * dt
  py_{k+1}    = py_k + v_k * sin(theta_k) * dt
  v_{k+1}     = v_k
  theta_{k+1} = theta_k + omega_k * dt
  omega_{k+1} = omega_k

Euler discretization (as above) is used instead of the exact-integral CTRV
form, which sidesteps the omega -> 0 singularity; at 30 fps with dt = 1 frame
the discretization error is negligible.

Reference:
  Schubert, R., Richter, E., & Wanielik, G. (2008).
  "Comparison and Evaluation of Advanced Motion Models for Vehicle Tracking."
  FUSION 2008.

Author  : Trọng Nghĩa
Paper   : Context-Aware Adaptive Perception for Mixed Traffic
Target  : ICRA 2027 (primary) / IEEE IV 2027 (backup)
"""

from __future__ import annotations

import numpy as np


# ── Constants ────────────────────────────────────────────────────────

STATE_DIM  = 5   # [px, py, v, theta, omega]
MEASURE_DIM = 2  # [px, py]  (observation from detection bbox center)

# Process noise (diagonal). Tuned on synthetic trajectories (tests/test_ctrv.py);
# re-tune on VN footage later. Units per frame at 30 fps:
#   theta noise 1e-3 rad² ≈ 1.8°/frame std — the original placeholder (0.1 rad²
#   ≈ 18°/frame) drowned the heading estimate in process noise and made
#   predictions jitter worse than the linear KF.
Q_DIAG = [0.5, 0.5, 0.1, 1e-3, 1e-4]  # TODO: re-tune on real footage

# Measurement noise (diagonal, TODO: tune from real data)
R_DIAG = [10.0, 10.0]  # TODO: tune

# Initial covariance diagonal: position is observed directly (small),
# v / theta / omega are unobservable from a single detection (huge).
P0_DIAG = [10.0, 10.0, 100.0, np.pi ** 2, 1.0]

# Linear observation matrix: we observe only the bbox center [px, py].
H = np.zeros((MEASURE_DIM, STATE_DIM))
H[0, 0] = 1.0
H[1, 1] = 1.0

# Minimum displacement (px) between init and first update for the heading
# bootstrap to fire — below this, direction is noise, keep the EKF prior.
BOOTSTRAP_MIN_DISP = 1.0

# NSA floor: R_eff = R * clip(1 - score, NSA_FLOOR, 1), so a score of 1.0
# cannot make R singular (swept in code/scripts/sweep_c2_sensitivity.py).
NSA_FLOOR = 0.05

# Scale-adaptive noise: Q/R are calibrated in pixels for an object of
# REF_BOX_SIZE = sqrt(w*h) px. A motorcycle 3× closer to the camera is
# ~3× larger AND moves ~3× more px/frame, so pixel-space noise should
# scale with apparent size — variances scale with the SQUARE of the
# ratio. The scale factor is clamped so extreme boxes can't blow up or
# collapse the filter.
REF_BOX_SIZE = 40.0
SCALE_CLAMP = (0.25, 4.0)   # scale s ∈ [.25, 4] → variance factor s² ∈ [1/16, 16]


def wrap_angle(theta: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


# ── Main class ───────────────────────────────────────────────────────

class CTRVMotionModel:
    """
    Extended Kalman Filter with CTRV process model.

    One instance per motorcycle track.

    Usage:
        model = CTRVMotionModel()
        model.init(bbox)           # initialise from first detection
        pred_bbox = model.predict()
        model.update(bbox)         # update with new detection
    """

    def __init__(
        self,
        dt: float = 1.0,
        q_diag: list[float] | None = None,
        r_diag: list[float] | None = None,
        scale_adaptive: bool = False,
        nsa: bool = False,
    ) -> None:
        """
        Args:
            dt    : time step in frames (default 1.0).
            q_diag: process noise diagonal (default Q_DIAG; see configs/default.yaml).
            r_diag: measurement noise diagonal (default R_DIAG).
            scale_adaptive: scale the PIXEL-space noise entries (px, py, v
                    in Q; both in R) with (box_size/REF_BOX_SIZE)² so far
                    small objects get a tight filter and near large ones
                    a looser one. Angular entries (θ, ω) are unaffected —
                    heading dynamics do not depend on apparent size.
                    NOTE: measured harmful on NKKN replay (see
                    experiments/c2_variants/evidence.md); kept for ablation.
            nsa   : Noise-Scale-Adaptive measurement noise (StrongSORT):
                    R̃ = (1 − score)·R per update — a 0.9-confidence
                    detection is trusted ~10× more than a 0.1 one.
                    Detection score is a direct proxy for localization
                    quality, unlike box size (see scale_adaptive note).
        """
        self.dt = dt
        self.scale_adaptive = scale_adaptive
        self.nsa = nsa
        self._noise_scale2 = 1.0   # (box_size/REF)² — refreshed per box

        # State vector and covariance (initialised in .init())
        self.x: np.ndarray | None = None   # (5,)
        self.P: np.ndarray | None = None   # (5, 5)

        # Process and measurement noise covariance matrices
        self.Q = np.diag(q_diag if q_diag is not None else Q_DIAG)  # (5, 5)
        self.R = np.diag(r_diag if r_diag is not None else R_DIAG)  # (2, 2)

        # Width/height of the last matched detection — held constant while
        # the track is lost; association only needs a well-placed box.
        self.wh: np.ndarray | None = None  # (2,)

        # True until the first update() — triggers the heading bootstrap.
        self._bootstrap = False

    # ── Public API ───────────────────────────────────────────────────

    def init(self, bbox: np.ndarray) -> None:
        """
        Initialise state from first detection bounding box.

        Args:
            bbox: [x1, y1, x2, y2] in pixels.
        """
        cx, cy = self._bbox_to_state(bbox)
        self.x = np.array([cx, cy, 0.0, 0.0, 0.0])
        self.P = np.diag(P0_DIAG).astype(float)
        self.wh = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=float)
        self._refresh_noise_scale()
        self._bootstrap = True

    def predict(self, lost: bool = False) -> np.ndarray:
        """
        Apply CTRV prediction step.

        Args:
            lost: True while the track has no detections (ByteTrack Lost
                  state). The full CTRV dynamics still apply — coasting
                  ALONG THE ARC through occlusions is exactly where CTRV
                  beats the linear model (a weaving motorcycle keeps
                  turning while hidden). Note ByteTrack's linear KF also
                  keeps its positional velocities when lost; it only zeroes
                  the aspect-ratio velocity mean[7], which CTRV does not
                  model (wh is held fixed instead). `lost` is kept in the
                  signature for future damping experiments.

        Returns:
            Predicted bbox [x1, y1, x2, y2] for association.
        """
        F = self._F_jacobian(self.x)
        self.x = self._f(self.x)
        self.x[3] = wrap_angle(self.x[3])
        self.P = F @ self.P @ F.T + self._Q_eff()
        return self._state_to_bbox(self.x, self.wh)

    def update(self, bbox: np.ndarray, score: float | None = None) -> None:
        """
        EKF update step with a new detection.

        Args:
            bbox : [x1, y1, x2, y2] in pixels.
            score: matched-detection confidence (0–1). Only used when
                   nsa=True; None keeps the plain R.
        """
        z = self._bbox_to_state(bbox)

        if self._bootstrap:
            # Heading bootstrap: theta/v are unobservable from a single
            # detection. Set them directly from the first displacement
            # instead of letting the EKF grope for ~10 frames through a
            # diagonal (zero cross-term) P.
            dz = z - self.x[:2]
            disp = float(np.linalg.norm(dz))
            if disp > BOOTSTRAP_MIN_DISP:
                self.x[2] = disp / self.dt                 # v
                self.x[3] = np.arctan2(dz[1], dz[0])       # theta
                # v/theta were just set directly — declare them independent
                # by clearing their rows/columns too. Overwriting only the
                # diagonal would leave stale cross-covariances (|corr| > 1)
                # and P would lose positive definiteness on the next update.
                for i, var in ((2, 10.0), (3, 0.5)):
                    self.P[i, :] = 0.0
                    self.P[:, i] = 0.0
                    self.P[i, i] = var
            self._bootstrap = False

        # Standard linear-observation EKF update.
        R_eff = self._R_eff()
        if self.nsa and score is not None:
            # NSA (StrongSORT): confident detections shrink R (floored).
            R_eff = R_eff * float(np.clip(1.0 - score, NSA_FLOOR, 1.0))
        y = z - H @ self.x                       # innovation
        S = H @ self.P @ H.T + R_eff             # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)      # Kalman gain (5, 2)

        self.x = self.x + K @ y
        self.x[3] = wrap_angle(self.x[3])

        # Joseph form: preserves symmetry / positive-definiteness of P
        # over thousands of steps (plain (I-KH)P accumulates asymmetry).
        I_KH = np.eye(STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T

        # Refresh box size from the matched detection + the noise scale.
        self.wh = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=float)
        self._refresh_noise_scale()

    # ── Private helpers ──────────────────────────────────────────────

    def _refresh_noise_scale(self) -> None:
        """Recompute (box_size/REF)², clamped — see SCALE_CLAMP."""
        if not self.scale_adaptive or self.wh is None:
            self._noise_scale2 = 1.0
            return
        size = float(np.sqrt(max(self.wh[0] * self.wh[1], 1.0)))
        s = np.clip(size / REF_BOX_SIZE, *SCALE_CLAMP)
        self._noise_scale2 = float(s * s)

    def _Q_eff(self) -> np.ndarray:
        """Process noise with pixel-space entries (px, py, v) scaled;
        angular entries (θ, ω) are scale-free."""
        if self._noise_scale2 == 1.0:
            return self.Q
        Q = self.Q.copy()
        Q[0, 0] *= self._noise_scale2
        Q[1, 1] *= self._noise_scale2
        Q[2, 2] *= self._noise_scale2
        return Q

    def _R_eff(self) -> np.ndarray:
        """Measurement noise (both entries are pixel-space)."""
        if self._noise_scale2 == 1.0:
            return self.R
        return self.R * self._noise_scale2

    def _f(self, x: np.ndarray) -> np.ndarray:
        """CTRV nonlinear process function (Euler discretization)."""
        px, py, v, theta, omega = x
        return np.array([
            px + v * np.cos(theta) * self.dt,
            py + v * np.sin(theta) * self.dt,
            v,
            theta + omega * self.dt,
            omega,
        ])

    def _F_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of _f with respect to state x (5×5)."""
        _, _, v, theta, _ = x
        dt = self.dt
        F = np.eye(STATE_DIM)
        F[0, 2] = np.cos(theta) * dt      # ∂px'/∂v
        F[0, 3] = -v * np.sin(theta) * dt  # ∂px'/∂theta
        F[1, 2] = np.sin(theta) * dt      # ∂py'/∂v
        F[1, 3] = v * np.cos(theta) * dt   # ∂py'/∂theta
        F[3, 4] = dt                       # ∂theta'/∂omega
        return F

    def _bbox_to_state(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1,y1,x2,y2] to observation [cx, cy]."""
        bbox = np.asarray(bbox, dtype=float)
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])

    def _state_to_bbox(self, x: np.ndarray, wh: np.ndarray) -> np.ndarray:
        """Convert state [cx, cy, ...] back to [x1,y1,x2,y2]."""
        cx, cy = x[0], x[1]
        w, h = wh[0], wh[1]
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])
