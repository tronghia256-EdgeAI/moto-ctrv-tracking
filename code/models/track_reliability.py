"""
track_reliability.py
─────────────────────
C3 Component: Dual-Task Track Reliability Scorer

Assigns a scalar reliability score S ∈ [0, 1] to each active track.
The score combines five components:

  S(τ) = w1·age(τ) + w2·consistency(τ) + w3·smoothness(τ)
       + w4·class_stability(τ) + w5·det_conf(τ)

  age(τ)             : normalised track age (saturates at MAX_AGE_FRAMES)
  consistency(τ)     : detection hit rate over last WINDOW_SIZE frames
  smoothness(τ)      : MEAN predicted-vs-detected IoU over the window
  class_stability(τ) : fraction of frames with the dominant class label
  det_conf(τ)        : mean matched-detection confidence over the window

Why det_conf exists (found on real footage, 2026-07-14): BYTE's second
association can sustain a track indefinitely on sub-threshold detections
(shadows, road texture). Such "ghost" tracks are matched every frame, so
consistency is high and hit_history is blind to them — on the NKKN
segment, 40 tracks (~1% of track-frames) lived with <30% high-confidence
support. Only the matched detection score separates them from real
objects, so it is a first-class component.

Design note — smoothness deviates from the original "inverse of IoU
variance" idea on purpose: pure variance scores a *consistently badly
predicted* track as "smooth". Mean IoU is monotone in prediction quality
and already punishes erratic motion (erratic → prediction misses → low
IoU). Mention this in the paper's method section.

Cold start: with fewer than MIN_HISTORY samples, the history-based
components return a neutral 0.5 — combined with the near-zero age
component this keeps brand-new tracks below τ_acc by construction,
which is exactly the false-alert suppression C3 exists for.

Two task-specific thresholds split the output:
  τ_acc   : high threshold → precision branch (accident detection)
  τ_mappo : lower threshold → recall branch   (MAPPO observations)

Author  : Trọng Nghĩa
Paper   : Context-Aware Adaptive Perception for Mixed Traffic
Target  : ICRA 2027 (primary) / IEEE IV 2027 (backup)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


# ── Weights (must sum to 1.0) ────────────────────────────────────────
# TODO: verify empirically; initial guess from domain intuition.
# det_conf carries the highest weight: it is the only component a ghost
# track cannot fake (see module docstring) — sized so that a maximally
# plausible ghost (perfect age/consistency/class, smooth 0.7, conf 0.2)
# lands at 0.685, under τ_acc = 0.70.

W_AGE             = 0.10
W_CONSISTENCY     = 0.20
W_SMOOTHNESS      = 0.25
W_CLASS_STABILITY = 0.15
W_DET_CONF        = 0.30

assert abs(W_AGE + W_CONSISTENCY + W_SMOOTHNESS + W_CLASS_STABILITY
           + W_DET_CONF - 1.0) < 1e-6


# ── Downstream thresholds ────────────────────────────────────────────
# TODO: tune via ablation study (Section V)

THRESHOLD_ACCIDENT = 0.70   # τ_acc   : precision branch
THRESHOLD_MAPPO    = 0.40   # τ_mappo : recall branch


# ── Window / normalisation constants ─────────────────────────────────

WINDOW_SIZE = 10       # last N frames (matches ClassAwareSTrack deques)
MAX_AGE_FRAMES = 30    # 1 s @ 30 fps saturates the age component
MIN_HISTORY = 3        # below this → neutral 0.5 (cold start)
NEUTRAL = 0.5


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class ReliabilityScore:
    """Full breakdown of a track's reliability."""
    total: float                     # final weighted score (0.0–1.0)
    age_component: float
    consistency_component: float
    smoothness_component: float
    class_stability_component: float
    det_conf_component: float

    # Downstream flags
    reliable_for_accident: bool = field(init=False)
    reliable_for_mappo: bool    = field(init=False)

    def __post_init__(self) -> None:
        self.reliable_for_accident = self.total >= THRESHOLD_ACCIDENT
        self.reliable_for_mappo    = self.total >= THRESHOLD_MAPPO


# ── Main class ───────────────────────────────────────────────────────

class TrackReliabilityScorer:
    """
    Scores all active tracks and returns a dict of ReliabilityScore objects.

    Usage:
        scorer = TrackReliabilityScorer()
        scores = scorer.score_all(tracks)
        accident_ids = [tid for tid, s in scores.items() if s.reliable_for_accident]
        mappo_ids    = [tid for tid, s in scores.items() if s.reliable_for_mappo]
    """

    def __init__(self,
                 weights: dict | None = None,
                 max_age_frames: int = MAX_AGE_FRAMES) -> None:
        w = weights or {}
        self.w_age = w.get("age", W_AGE)
        self.w_consistency = w.get("consistency", W_CONSISTENCY)
        self.w_smoothness = w.get("smoothness", W_SMOOTHNESS)
        self.w_class_stability = w.get("class_stability", W_CLASS_STABILITY)
        self.w_det_conf = w.get("det_conf", W_DET_CONF)
        total = (self.w_age + self.w_consistency + self.w_smoothness
                 + self.w_class_stability + self.w_det_conf)
        assert abs(total - 1.0) < 1e-6, f"weights must sum to 1, got {total}"
        self.max_age_frames = max_age_frames

    def score_all(self, tracks: List) -> Dict[int, ReliabilityScore]:
        """
        Args:
            tracks: list of active track objects.
                    Each track must expose: .track_id, .age,
                    .hit_history (List[bool]), .iou_history (List[float]),
                    .class_history (List of class ids),
                    .score_history (List[float], matched det confidences).

        Returns:
            Dict mapping track_id → ReliabilityScore.
        """
        return {t.track_id: self.score(t) for t in tracks}

    def score(self, track) -> ReliabilityScore:
        """Score a single track."""
        age_c   = self._age_component(track)
        cons_c  = self._consistency_component(track)
        smooth_c = self._smoothness_component(track)
        cls_c   = self._class_stability_component(track)
        conf_c  = self._det_conf_component(track)

        total = (self.w_age * age_c
                 + self.w_consistency * cons_c
                 + self.w_smoothness * smooth_c
                 + self.w_class_stability * cls_c
                 + self.w_det_conf * conf_c)

        return ReliabilityScore(
            total=float(np.clip(total, 0.0, 1.0)),
            age_component=age_c,
            consistency_component=cons_c,
            smoothness_component=smooth_c,
            class_stability_component=cls_c,
            det_conf_component=conf_c,
        )

    # ── Component calculators ────────────────────────────────────────

    def _age_component(self, track) -> float:
        """Normalised track age, saturating at max_age_frames."""
        return float(min(track.age / self.max_age_frames, 1.0))

    def _consistency_component(self, track) -> float:
        """Detection hit rate over the last WINDOW_SIZE frames."""
        hist = list(track.hit_history)
        if len(hist) < MIN_HISTORY:
            return NEUTRAL
        return float(np.mean(hist))

    def _smoothness_component(self, track) -> float:
        """Mean predicted-vs-detected IoU (see module docstring for why
        this replaces inverse IoU variance)."""
        hist = list(track.iou_history)
        if len(hist) < MIN_HISTORY:
            return NEUTRAL
        return float(np.mean(hist))

    def _class_stability_component(self, track) -> float:
        """Fraction of frames where the detected class equals the mode class."""
        hist = list(track.class_history)
        if len(hist) < MIN_HISTORY:
            return NEUTRAL
        mode_count = Counter(hist).most_common(1)[0][1]
        return mode_count / len(hist)

    def _det_conf_component(self, track) -> float:
        """Mean matched-detection confidence — the anti-ghost signal (see
        module docstring). Tracks without a score_history score neutral."""
        hist = list(getattr(track, "score_history", []))
        if len(hist) < MIN_HISTORY:
            return NEUTRAL
        return float(np.clip(np.mean(hist), 0.0, 1.0))
