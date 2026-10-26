"""
code/tracker/adaptive_bytetrack.py
────────────────────────────────────
Modified ByteTrack that integrates C1 + C2.

Wraps the vendored ByteTrack (code/tracker/vendor/, see PATCHES.md) and adds:
  1. Receives Params from AdaptiveController each frame → applies conf + buffer
     (Params.yolo_conf drives the tracker's track_thresh, NOT the YOLO
      inference threshold — BYTE's second association stage needs the
      low-score detections, so the detector always runs at conf ≈ 0.1).
  2. Routes each track to CTRVMotionModel (motorcycle) or default Kalman
     (others). The choice is made once, when the track is activated.
  3. Maintains per-track history fields (hit/iou/class) consumed by the
     C3 TrackReliabilityScorer.

Limitation (by design of the vendor `strack_cls` hook): CTRV settings are
class-level on ClassAwareSTrack, so only one AdaptiveBYTETracker should be
live at a time. Ablation runs are sequential — call reset() between runs.

Author  : Trọng Nghĩa
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np

from code.models.motorcycle_motion_model import CTRVMotionModel, H as CTRV_H
from code.tracker.vendor import BYTETracker, STrack, BaseTrack, TrackState
from code.tracker.vendor import matching

# Project class id for motorcycles (see code/detection contract).
MOTORCYCLE_CLASS_ID = 1

# Window for the C3 history deques (reliability scorer reads these).
HISTORY_WINDOW = 10


class ClassAwareSTrack(STrack):
    """
    STrack with a class-conditioned motion model (C2) and C3 history fields.

    Motorcycle tracks get a per-track CTRV EKF; every other class keeps the
    vendored linear Kalman filter (vectorized shared_kalman fast path).
    """

    # ── Class-level config, set by AdaptiveBYTETracker each update() ──
    ctrv_enabled: bool = False           # ablation flag (C2 on/off)
    motion_model_this_frame: str = "ctrv"  # from Params.motion_model
    ctrv_dt: float = 1.0
    ctrv_q_diag: list | None = None
    ctrv_r_diag: list | None = None
    ctrv_scale_adaptive: bool = False
    ctrv_nsa: bool = False

    def __init__(self, tlwh, score, class_id=None):
        super().__init__(tlwh, score, class_id)
        self.ctrv: CTRVMotionModel | None = None
        # C3 history fields (consumed by TrackReliabilityScorer)
        self.hit_history: deque = deque(maxlen=HISTORY_WINDOW)
        self.iou_history: deque = deque(maxlen=HISTORY_WINDOW)
        self.class_history: deque = deque(maxlen=HISTORY_WINDOW)
        # Matched-detection confidence per hit. Catches "ghost" tracks
        # sustained purely by low-score detections through BYTE's second
        # association — those match every frame (hit_history all True),
        # so only the detection score separates them from real objects.
        self.score_history: deque = deque(maxlen=HISTORY_WINDOW)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def age(self) -> int:
        """Track age in frames (0 for a brand-new track)."""
        return self.frame_id - self.start_frame

    @property
    def tlwh(self):
        """CTRV tracks derive their box from the EKF state + stored wh."""
        if self.ctrv is not None and self.ctrv.x is not None:
            tlbr = self.ctrv._state_to_bbox(self.ctrv.x, self.ctrv.wh)
            return self.tlbr_to_tlwh(tlbr)
        return STrack.tlwh.fget(self)

    # ── Track lifecycle overrides ─────────────────────────────────────

    def activate(self, kalman_filter, frame_id):
        # Parent initializes the linear KF state (harmless for CTRV tracks —
        # their box comes from the EKF via the tlwh override) plus ids/state.
        super().activate(kalman_filter, frame_id)
        if self._should_use_ctrv():
            self.ctrv = CTRVMotionModel(
                dt=self.ctrv_dt,
                q_diag=self.ctrv_q_diag,
                r_diag=self.ctrv_r_diag,
                scale_adaptive=self.ctrv_scale_adaptive,
                nsa=self.ctrv_nsa,
            )
            self.ctrv.init(self.tlwh_to_tlbr(self._tlwh))
        self.hit_history.append(True)
        self.class_history.append(self.class_id)
        self.score_history.append(float(self.score))

    def re_activate(self, new_track, frame_id, new_id=False):
        self._record_history(new_track)
        if self.ctrv is not None:
            self.ctrv.update(self.tlwh_to_tlbr(new_track.tlwh),
                             score=float(new_track.score))
            # Mirror the parent's bookkeeping (state machine unchanged).
            self.tracklet_len = 0
            self.state = TrackState.Tracked
            self.is_activated = True
            self.frame_id = frame_id
            if new_id:
                self.track_id = self.next_id()
            self.score = new_track.score
        else:
            super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        self._record_history(new_track)
        if self.ctrv is not None:
            self.ctrv.update(self.tlwh_to_tlbr(new_track.tlwh),
                             score=float(new_track.score))
            self.frame_id = frame_id
            self.tracklet_len += 1
            self.state = TrackState.Tracked
            self.is_activated = True
            self.score = new_track.score
        else:
            super().update(new_track, frame_id)

    # ── Prediction dispatch ───────────────────────────────────────────

    def predict(self):
        if self.ctrv is not None:
            self.ctrv.predict(lost=self.state != TrackState.Tracked)
        else:
            super().predict()

    @staticmethod
    def multi_predict(stracks):
        """Split: CTRV tracks get per-track EKF prediction, the rest keep
        the vectorized shared-Kalman fast path."""
        linear = [st for st in stracks if getattr(st, "ctrv", None) is None]
        STrack.multi_predict(linear)
        for st in stracks:
            if getattr(st, "ctrv", None) is not None:
                st.ctrv.predict(lost=st.state != TrackState.Tracked)

    # ── Private ───────────────────────────────────────────────────────

    def _should_use_ctrv(self) -> bool:
        """Decided once, at activation time. A track activated under a
        non-CTRV regime keeps the linear model for its lifetime."""
        return (
            self.ctrv_enabled
            and self.class_id == MOTORCYCLE_CLASS_ID
            and self.motion_model_this_frame == "ctrv"
        )

    def _record_history(self, new_track) -> None:
        """Record predicted-vs-detected IoU BEFORE the state update — this
        measures prediction quality, which C3 uses as its smoothness signal."""
        iou = float(matching.ious([self.tlbr], [new_track.tlbr])[0, 0])
        self.iou_history.append(iou)
        self.hit_history.append(True)
        self.class_history.append(new_track.class_id)
        self.score_history.append(float(new_track.score))


class AdaptiveBYTETracker(BYTETracker):
    """
    ByteTrack wrapper with:
      - Dynamic confidence threshold (from Params → track_thresh, D3)
      - Dynamic buffer length        (from Params → max_time_lost)
      - Class-conditioned motion model (CTRV for motorcycle, Kalman for others)
      - Optional CTRV uncertainty gating (`tracker.uncertainty_gating`):
        lost motorcycle tracks may re-associate to same-class detections
        that fall inside the EKF's innovation ellipse even when box IoU
        is ~0 — the covariance grows along the predicted arc during an
        occlusion, which is exactly where the reappearing detection lands.

    Usage:
        tracker = AdaptiveBYTETracker(config, ctrv_enabled=True)
        tracks = tracker.update(detections_n6, params)   # params may be None
    """

    strack_cls = ClassAwareSTrack

    # χ²(0.95, 2 dof): standard Mahalanobis gate for a 2-D measurement.
    GATE_CHI2 = 5.991
    # Rescued matches enter the cost matrix at >= GATE_BASE_COST, so any
    # genuine IoU overlap (cost < 0.5 in the first association) always
    # outranks a gate rescue — the gate only fills in where IoU failed.
    GATE_BASE_COST = 0.5

    def __init__(self, config: dict | None = None, ctrv_enabled: bool = True,
                 img_size: tuple[int, int] = (720, 1280)) -> None:
        config = config or {}
        tr = config.get("tracker", {})
        args = SimpleNamespace(
            track_thresh=tr.get("track_thresh", 0.5),
            track_buffer=tr.get("track_buffer", 30),
            match_thresh=tr.get("match_thresh", 0.8),
            mot20=False,
        )
        self.frame_rate = tr.get("frame_rate", 30)
        super().__init__(args, frame_rate=self.frame_rate)

        self.ctrv_enabled = ctrv_enabled
        self.img_size = img_size
        self.uncertainty_gating = bool(tr.get("uncertainty_gating", False))

        # Push CTRV config to the track class (single-tracker limitation —
        # see module docstring).
        ctrv_cfg = config.get("ctrv", {})
        ClassAwareSTrack.ctrv_enabled = ctrv_enabled
        ClassAwareSTrack.ctrv_dt = ctrv_cfg.get("dt", 1.0)
        ClassAwareSTrack.ctrv_q_diag = ctrv_cfg.get("Q_diag")
        ClassAwareSTrack.ctrv_r_diag = ctrv_cfg.get("R_diag")
        ClassAwareSTrack.ctrv_scale_adaptive = ctrv_cfg.get(
            "scale_adaptive", False)
        ClassAwareSTrack.ctrv_nsa = ctrv_cfg.get("nsa_noise", False)

    def _first_association_cost(self, strack_pool, detections):
        """P11 hook override: IoU cost + CTRV uncertainty gate.

        For each LOST CTRV track, detections of the same class whose
        center lies inside the EKF innovation ellipse (Mahalanobis
        d² < χ²₀.₉₅) get their cost capped at
            GATE_BASE_COST + (1 − GATE_BASE_COST) · d²/χ²
        i.e. rescued costs live in [0.5, 1): they lose to any real IoU
        overlap and merely make a match *possible* where IoU alone is 0.
        Tracked (non-lost) tracks and linear-KF tracks are untouched.
        """
        dists = matching.iou_distance(strack_pool, detections)
        if not self.uncertainty_gating or not len(detections):
            return dists

        det_centers = np.array(
            [[d.tlwh[0] + d.tlwh[2] / 2.0, d.tlwh[1] + d.tlwh[3] / 2.0]
             for d in detections])
        det_classes = np.array([d.class_id for d in detections])

        for i, tr in enumerate(strack_pool):
            if tr.ctrv is None or tr.state != TrackState.Lost:
                continue
            P, R = tr.ctrv.P, tr.ctrv._R_eff()
            S = CTRV_H @ P @ CTRV_H.T + R          # innovation covariance
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            mu = tr.ctrv.x[:2]
            diff = det_centers - mu                 # (N, 2)
            d2 = np.einsum("nj,jk,nk->n", diff, S_inv, diff)
            inside = (d2 < self.GATE_CHI2) & (det_classes == tr.class_id)
            if not inside.any():
                continue
            rescue = (self.GATE_BASE_COST
                      + (1.0 - self.GATE_BASE_COST) * d2 / self.GATE_CHI2)
            dists[i, inside] = np.minimum(dists[i, inside], rescue[inside])
        return dists

    def update(self, detections, params=None):  # params: Params | None
        """
        Args:
            detections : N×6 numpy [x1, y1, x2, y2, score, class_id] in
                         frame-native pixels (from code/detection/detector.py).
            params     : Params from AdaptiveController, or None to run with
                         the fixed constructor thresholds (baseline / Exp1).

        Returns:
            List of active ClassAwareSTrack objects.
        """
        if params is not None:
            # D3: yolo_conf drives the tracker's high/low split, not the
            # detector. det_thresh mirrors upstream's "+ 0.1" rule.
            self.args.track_thresh = params.yolo_conf
            self.det_thresh = params.yolo_conf + 0.1
            self.max_time_lost = int(self.frame_rate / 30.0 * params.buffer_len)
            ClassAwareSTrack.motion_model_this_frame = params.motion_model
        else:
            ClassAwareSTrack.motion_model_this_frame = "ctrv"
        ClassAwareSTrack.ctrv_enabled = self.ctrv_enabled

        detections = np.asarray(detections, dtype=float)
        if detections.size == 0:
            detections = np.empty((0, 6))

        # img_info == img_size → scale == 1: coords stay frame-native.
        output = super().update(detections, self.img_size, self.img_size)

        # C3 bookkeeping: tracks not matched this frame were predicted-only.
        for t in self.tracked_stracks + self.lost_stracks:
            if t.frame_id != self.frame_id:
                t.hit_history.append(False)

        return output

    def reset(self) -> None:
        """Full reset so sequential ablation runs are independent."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        BaseTrack._count = 0
