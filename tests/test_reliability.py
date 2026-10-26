"""
tests/test_reliability.py
─────────────────────────
Unit + integration tests for C3: TrackReliabilityScorer.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from code.models.track_reliability import (
    TrackReliabilityScorer,
    THRESHOLD_ACCIDENT,
    THRESHOLD_MAPPO,
    W_AGE, W_CONSISTENCY, W_SMOOTHNESS, W_CLASS_STABILITY, W_DET_CONF,
)
from code.tracker import AdaptiveBYTETracker
from tests.synthetic import weaving, to_detections, CLASS_MOTORCYCLE


def mock_track(track_id=1, age=30, hits=None, ious=None, classes=None,
               scores=None):
    return SimpleNamespace(
        track_id=track_id,
        age=age,
        hit_history=hits if hits is not None else [True] * 10,
        iou_history=ious if ious is not None else [0.9] * 10,
        class_history=classes if classes is not None else [1] * 10,
        score_history=scores if scores is not None else [0.8] * 10,
    )


class TestComponents:
    def test_age_saturates(self):
        s = TrackReliabilityScorer()
        assert s._age_component(mock_track(age=15)) == pytest.approx(0.5)
        assert s._age_component(mock_track(age=30)) == 1.0
        assert s._age_component(mock_track(age=300)) == 1.0
        assert s._age_component(mock_track(age=0)) == 0.0

    def test_consistency_exact(self):
        s = TrackReliabilityScorer()
        t = mock_track(hits=[True] * 7 + [False] * 3)
        assert s._consistency_component(t) == pytest.approx(0.7)

    def test_smoothness_is_mean_iou(self):
        s = TrackReliabilityScorer()
        t = mock_track(ious=[0.8, 0.6, 1.0])
        assert s._smoothness_component(t) == pytest.approx(0.8)

    def test_smoothness_punishes_bad_predictions(self):
        """The reason mean-IoU replaced inverse-variance: a consistently
        BADLY predicted track must not score as 'smooth'."""
        s = TrackReliabilityScorer()
        consistently_bad = mock_track(ious=[0.1] * 10)   # zero variance!
        good = mock_track(ious=[0.85, 0.9, 0.8] * 3)
        assert (s._smoothness_component(consistently_bad)
                < s._smoothness_component(good))

    def test_class_stability(self):
        s = TrackReliabilityScorer()
        t = mock_track(classes=[1, 1, 1, 1, 1, 1, 1, 0, 0, 1])  # 8/10 mode
        assert s._class_stability_component(t) == pytest.approx(0.8)

    def test_det_conf_is_mean_score(self):
        s = TrackReliabilityScorer()
        t = mock_track(scores=[0.9, 0.8, 0.7])
        assert s._det_conf_component(t) == pytest.approx(0.8)

    def test_det_conf_neutral_without_history_field(self):
        """Graceful degradation for track objects lacking score_history."""
        s = TrackReliabilityScorer()
        t = mock_track()
        del t.score_history
        assert s._det_conf_component(t) == 0.5

    def test_cold_start_neutral(self):
        s = TrackReliabilityScorer()
        t = mock_track(age=1, hits=[True], ious=[0.9], classes=[1],
                       scores=[0.9])
        assert s._consistency_component(t) == 0.5
        assert s._smoothness_component(t) == 0.5
        assert s._class_stability_component(t) == 0.5
        assert s._det_conf_component(t) == 0.5

    def test_weights_sum_to_one(self):
        assert abs(W_AGE + W_CONSISTENCY + W_SMOOTHNESS + W_CLASS_STABILITY
                   + W_DET_CONF - 1.0) < 1e-6

    def test_custom_weights_validated(self):
        with pytest.raises(AssertionError):
            TrackReliabilityScorer(weights={"age": 0.5, "consistency": 0.5,
                                            "smoothness": 0.5, "class_stability": 0.5})


class TestDualTaskSplit:
    def test_old_stable_track_passes_both(self):
        s = TrackReliabilityScorer()
        score = s.score(mock_track(age=60))
        assert score.total > THRESHOLD_ACCIDENT
        assert score.reliable_for_accident and score.reliable_for_mappo

    def test_flickering_baby_track_fails_both(self):
        s = TrackReliabilityScorer()
        t = mock_track(age=2, hits=[True, False, True],
                       ious=[0.2, 0.1, 0.3], classes=[1, 0, 2],
                       scores=[0.2, 0.15, 0.2])
        score = s.score(t)
        assert score.total < THRESHOLD_MAPPO
        assert not score.reliable_for_accident
        assert not score.reliable_for_mappo

    def test_mid_quality_track_mappo_only(self):
        """The dual-task split: a mediocre track feeds MAPPO (recall)
        but is filtered from accident alerting (precision)."""
        s = TrackReliabilityScorer()
        t = mock_track(age=12,                       # age_c = 0.4
                       hits=[True] * 6 + [False] * 4,  # cons = 0.6
                       ious=[0.5] * 10,                # smooth = 0.5
                       classes=[1] * 8 + [0] * 2,      # cls = 0.8
                       scores=[0.5] * 10)              # conf = 0.5
        score = s.score(t)
        # total = .10*.4 + .20*.6 + .25*.5 + .15*.8 + .30*.5 = 0.555
        assert score.total == pytest.approx(0.555)
        assert score.reliable_for_mappo
        assert not score.reliable_for_accident

    def test_brand_new_track_below_accident_by_construction(self):
        """Cold start: however clean, a 1-frame track cannot pass τ_acc."""
        s = TrackReliabilityScorer()
        t = mock_track(age=0, hits=[True], ious=[], classes=[1], scores=[0.9])
        score = s.score(t)
        # 0.10*0 + (0.20+0.25+0.15+0.30)*0.5 = 0.45 < 0.70
        assert score.total == pytest.approx(0.45)
        assert not score.reliable_for_accident

    def test_ghost_track_blind_spot_closed(self):
        """The real-footage failure mode this component exists for: a track
        sustained purely by low-score detections matches EVERY frame, so
        age/consistency/class are all perfect — only det_conf can (and
        must) keep it out of the accident branch."""
        s = TrackReliabilityScorer()
        ghost = mock_track(age=60,                    # old → age_c = 1.0
                           hits=[True] * 10,          # matched every frame
                           ious=[0.7] * 10,           # plausibly smooth
                           classes=[1] * 10,          # class-stable
                           scores=[0.2] * 10)         # ...on garbage dets
        score = s.score(ghost)
        # .10 + .20 + .25*.7 + .15 + .30*.2 = 0.685 < τ_acc = 0.70
        assert not score.reliable_for_accident
        # ...while the same track on real detections passes:
        real = mock_track(age=60, hits=[True] * 10, ious=[0.7] * 10,
                          classes=[1] * 10, scores=[0.75] * 10)
        assert s.score(real).reliable_for_accident

    def test_score_all_returns_dict(self):
        s = TrackReliabilityScorer()
        tracks = [mock_track(track_id=i, age=40) for i in (1, 2, 3)]
        scores = s.score_all(tracks)
        assert set(scores.keys()) == {1, 2, 3}


class TestIntegrationWithTracker:
    """Ghost-vs-persistent split on a real AdaptiveBYTETracker run."""

    def test_ghost_track_never_passes_accident_threshold(self):
        # Persistent weaving motorcycle + a 2-frame ghost detection burst.
        moto = weaving(v=5.0, amp=0.5, period=50, n=60, noise=1.0, seed=3)
        frames = to_detections([moto], class_ids=[CLASS_MOTORCYCLE])
        ghost_box = np.array([[600, 100, 640, 160, 0.9, CLASS_MOTORCYCLE]])
        frames[30] = np.vstack([frames[30], ghost_box])
        frames[31] = np.vstack([frames[31], ghost_box + [3, 0, 3, 0, 0, 0]])

        tracker = AdaptiveBYTETracker({"tracker": {}}, ctrv_enabled=True)
        tracker.reset()
        scorer = TrackReliabilityScorer()

        ghost_passed_acc = False
        moto_final_score = None
        for k, dets in enumerate(frames):
            tracks = tracker.update(dets)
            scores = scorer.score_all(tracks)
            for t in tracks:
                sc = scores[t.track_id]
                is_ghost = t.tlbr[0] > 500
                if is_ghost and sc.reliable_for_accident:
                    ghost_passed_acc = True
                if not is_ghost:
                    moto_final_score = sc

        assert not ghost_passed_acc, "2-frame ghost must never pass τ_acc"
        assert moto_final_score is not None
        assert moto_final_score.reliable_for_accident, (
            f"persistent track should pass τ_acc, got {moto_final_score.total:.2f}"
        )
