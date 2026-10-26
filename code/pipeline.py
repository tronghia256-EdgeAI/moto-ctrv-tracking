"""
code/pipeline.py
─────────────────
Main inference pipeline — ties all components together.

Data flow:
    Camera frame
        → ContextEstimator   → TrafficState        (C1, ablation flag)
        → AdaptiveController → Params              (C1)
        → Detector           → raw detections (always low conf — D3)
        → AdaptiveBYTETracker→ active tracks  (CTRV or Kalman by class, C2)
        → TrackReliabilityScorer → ReliabilityScore per track (C3)
        → AccidentClassifier (tracks reliable_for_accident, class accident)
        → TelegramAlert      (if severity MINOR or SEVERE)
        → MAPPO Observations (tracks reliable_for_mappo)

Ablation flags (config `ablation:`):
    use_C1_adaptive=False   → params=None: tracker keeps its fixed
                              constructor thresholds every frame
    use_C2_ctrv=False       → tracker built with ctrv_enabled=False
    use_C3_reliability=False→ every track passes both downstream branches

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from code.models import (
    ContextEstimator,
    AdaptiveController,
    TrackReliabilityScorer,
)
from code.tracker.adaptive_bytetrack import AdaptiveBYTETracker
from code.detection.detector import Detector
from code.detection.accident_classifier import AccidentClassifier, Severity
from code.alert.telegram_alert import TelegramAlert

logger = logging.getLogger(__name__)

ACCIDENT_CLASS_ID = 4


def load_config(config_path: str) -> dict:
    """Load an experiment config, two-level-merged over configs/default.yaml."""
    config_path = Path(config_path)
    default_path = config_path.parent / "default.yaml"

    config: dict = {}
    if default_path.exists() and default_path != config_path:
        with open(default_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    with open(config_path, encoding="utf-8") as f:
        override = yaml.safe_load(f) or {}

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


class Pipeline:
    """
    End-to-end adaptive perception pipeline.

    Usage:
        pipeline = Pipeline.from_config('configs/default.yaml')
        for frame in video_source:
            result = pipeline.process(frame)
    """

    def __init__(
        self,
        detector: Detector | None,
        tracker: AdaptiveBYTETracker,
        context_estimator: ContextEstimator,
        adaptive_controller: AdaptiveController,
        reliability_scorer: TrackReliabilityScorer,
        accident_classifier: AccidentClassifier | None,
        alert: TelegramAlert | None,
        use_c1: bool = True,
        use_c3: bool = True,
    ) -> None:
        self.detector            = detector
        self.tracker             = tracker
        self.context_estimator   = context_estimator
        self.adaptive_controller = adaptive_controller
        self.reliability_scorer  = reliability_scorer
        self.accident_classifier = accident_classifier
        self.alert               = alert
        self.use_c1              = use_c1
        self.use_c3              = use_c3

        self._last_tracks: list = []
        self.roi = None

    @classmethod
    def from_config(cls, config_path: str, dry_run_alerts: bool = True,
                    build_detector: bool = True) -> "Pipeline":
        """Build pipeline from a YAML config (merged over default.yaml).

        Args:
            dry_run_alerts : never hit the real Telegram API (experiments).
            build_detector : False for --synthetic runs (no torch needed).
        """
        cfg = load_config(config_path)
        roi = None
        roi_cfg=cfg.get('roi',{})
        if roi_cfg.get('file') or roi_cfg.get('review_bundle'):
            from code.scripts.gt_roi import load_roi
            from pathlib import Path
            if roi_cfg.get('review_bundle')=='active_mixed':
                from code.scripts.gt_roi_scope import active_bundle
                roi_path=active_bundle()/'roi.json'
            elif roi_cfg.get('file'):roi_path=Path(roi_cfg['file'])
            else:raise ValueError('Unknown ROI review bundle selector')
            if not roi_path.is_file():raise ValueError(f'ROI file is missing: {roi_path}')
            roi=load_roi(roi_path.parent)
            if not roi or not roi['enabled']:raise ValueError('ROI config requires an enabled saved ROI')
        ablation = cfg.get("ablation", {})
        use_c1 = ablation.get("use_C1_adaptive", True)
        use_c2 = ablation.get("use_C2_ctrv", True)
        use_c3 = ablation.get("use_C3_reliability", True)

        detector = None
        if build_detector:
            det_cfg = cfg.get("detector", {})
            detector = Detector(
                weights_path=cfg.get("weights", {}).get("yolo11", ""),
                img_size=det_cfg.get("img_size", 640),
            )

        tracker = AdaptiveBYTETracker(cfg, ctrv_enabled=use_c2)

        ac_cfg = cfg.get("adaptive_controller", {})
        controller = AdaptiveController(
            density_threshold=ac_cfg.get("density_threshold", 30),
            brightness_night=ac_cfg.get("brightness_night", 80.0),
            peak_hours=range(ac_cfg.get("peak_hours_start", 7),
                             ac_cfg.get("peak_hours_end", 20) + 1),
        )

        rel_cfg = cfg.get("reliability", {})
        scorer = TrackReliabilityScorer(weights=rel_cfg.get("weights"))

        classifier = None
        if build_detector:
            classifier = AccidentClassifier(
                weights_path=cfg.get("weights", {}).get("efficientnet", ""),
                input_size=cfg.get("accident_classifier", {}).get("input_size", 224),
            )

        alert = TelegramAlert.from_config(cfg, dry_run=dry_run_alerts)

        pipeline = cls(
            detector=detector,
            tracker=tracker,
            context_estimator=ContextEstimator(),
            adaptive_controller=controller,
            reliability_scorer=scorer,
            accident_classifier=classifier,
            alert=alert,
            use_c1=use_c1,
            use_c3=use_c3,
        )
        pipeline.roi=roi
        if detector is not None:detector.roi=roi
        return pipeline

    def process(self, frame: np.ndarray,
                detections: np.ndarray | None = None) -> dict:
        """
        Process one frame end-to-end.

        Args:
            frame      : BGR frame (H, W, 3).
            detections : optional pre-computed N×6 detections (synthetic
                         runs); if None the Detector is invoked.

        Returns:
            {
              'tracks_accident' : list of track objects (reliable_for_accident),
              'tracks_mappo'    : list of track objects (reliable_for_mappo),
              'scores'          : dict {track_id: ReliabilityScore},
              'params'          : Params used this frame (None = fixed baseline),
              'state'           : TrafficState this frame,
              'alerts'          : list of (Severity, confidence) fired this frame,
              'latency_ms'      : wall-clock processing time,
              'stages_ms'       : per-stage breakdown {context, detect, track,
                                  c3, accident}. 'detect' is 0.0 when
                                  `detections` was supplied (replay runs), which
                                  is exactly the gap between a replay latency
                                  number and an end-to-end one.
            }

        The stage timers are four perf_counter() calls on a path that already
        costs milliseconds; they are always on so that a latency number can
        never be reported without knowing what is inside it.
        """
        t0 = time.perf_counter()
        stages: dict[str, float] = {}

        # C1: context → params (previous frame's tracks feed the estimator).
        from code.scripts.gt_roi import pixel_mask,filter_detections,contains_box
        mask=pixel_mask(frame,self.roi)
        state = self.context_estimator.estimate(frame, self._last_tracks,mask) if mask is not None else self.context_estimator.estimate(frame,self._last_tracks)
        params = self.adaptive_controller.get_params(state) if self.use_c1 else None
        t_ctx = time.perf_counter()
        stages["context"] = (t_ctx - t0) * 1000.0

        # Detection (always low conf — D3) + tracking (C2).
        if detections is None:
            detections = self.detector.detect(frame)
        detections=filter_detections(detections,self.roi)
        t_det = time.perf_counter()
        stages["detect"] = (t_det - t_ctx) * 1000.0

        tracks = self.tracker.update(detections, params)
        if self.roi and self.roi['enabled']:
            # Remove state outside the scope, rather than retaining hidden tracks
            # that can re-enter or contribute to C1/C3 history.
            for attr in ('tracked_stracks','lost_stracks'):
                retained=[]
                for track in getattr(self.tracker,attr):
                    if contains_box(track.tlbr,self.roi):retained.append(track)
                    else:
                        track.mark_removed();self.tracker.removed_stracks.append(track)
                setattr(self.tracker,attr,retained)
            tracks=[t for t in tracks if contains_box(t.tlbr,self.roi)]
        self._last_tracks = tracks
        t_trk = time.perf_counter()
        stages["track"] = (t_trk - t_det) * 1000.0

        # C3: dual-task reliability split.
        if self.use_c3:
            scores = self.reliability_scorer.score_all(tracks)
            tracks_accident = [t for t in tracks
                               if scores[t.track_id].reliable_for_accident]
            tracks_mappo = [t for t in tracks
                            if scores[t.track_id].reliable_for_mappo]
        else:
            scores = {}
            tracks_accident = list(tracks)
            tracks_mappo = list(tracks)
        t_c3 = time.perf_counter()
        stages["c3"] = (t_c3 - t_trk) * 1000.0

        # Accident branch: severity classification + alert.
        alerts = []
        if self.accident_classifier is not None and self.accident_classifier.enabled:
            for t in tracks_accident:
                if t.class_id != ACCIDENT_CLASS_ID:
                    continue
                x1, y1, x2, y2 = np.clip(t.tlbr, 0, None).astype(int)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                severity, conf = self.accident_classifier.classify(roi)
                if severity in (Severity.MINOR, Severity.SEVERE):
                    alerts.append((severity, conf))
                    if self.alert is not None:
                        asyncio.run(self.alert.send(severity, frame))

        return {
            "tracks_accident": tracks_accident,
            "tracks_mappo": tracks_mappo,
            "scores": scores,
            "params": params,
            "state": state,
            "alerts": alerts,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "stages_ms": {**stages,
                          "accident": (time.perf_counter() - t_c3) * 1000.0},
        }

    def reset(self) -> None:
        """Reset all stateful components between experiment runs."""
        self.tracker.reset()
        self.context_estimator.reset()
        self.adaptive_controller.reset()
        self._last_tracks = []
