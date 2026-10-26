"""
tests/test_metrics.py
─────────────────────
Cross-validate self-implemented MOTA/IDF1 against py-motmetrics (D5),
plus detection/severity metric unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from code.evaluation.metrics import (
    compute_iou,
    compute_tracking_metrics,
    compute_detection_metrics,
    compute_severity_metrics,
)


def box(cx, cy, w=40, h=60):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ── Scenarios (shared with the motmetrics oracle) ────────────────────

def scenario_perfect(n=20):
    """One object, tracked perfectly."""
    gt, pred = [], []
    for k in range(n):
        b = box(100 + 5 * k, 100)
        gt.append([(1, b)])
        pred.append([(7, b)])
    return pred, gt


def scenario_id_switch(n=20):
    """One object; the tracker changes id halfway."""
    gt, pred = [], []
    for k in range(n):
        b = box(100 + 5 * k, 100)
        gt.append([(1, b)])
        pred.append([(7 if k < n // 2 else 8, b)])
    return pred, gt


def scenario_fp_fn(n=20):
    """One object; one missed frame (FN) and one spurious box (FP)."""
    gt, pred = [], []
    for k in range(n):
        b = box(100 + 5 * k, 100)
        gt.append([(1, b)])
        if k == 5:
            pred.append([])                             # miss
        elif k == 10:
            pred.append([(7, b), (99, box(500, 500))])  # extra FP
        else:
            pred.append([(7, b)])
    return pred, gt


def motmetrics_oracle(pred_frames, gt_frames):
    # motmetrics 1.4 still calls np.asfarray, removed in numpy 2.0 —
    # restore it before import (dev-only oracle, not production code).
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for gts, preds in zip(gt_frames, pred_frames):
        gt_ids = [g[0] for g in gts]
        pr_ids = [p[0] for p in preds]
        gt_boxes = np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, b in gts]) \
            if gts else np.empty((0, 4))
        pr_boxes = np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, b in preds]) \
            if preds else np.empty((0, 4))
        dist = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
        acc.update(gt_ids, pr_ids, dist)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches"], name="x")
    return (float(summary["mota"].iloc[0]),
            float(summary["idf1"].iloc[0]),
            int(summary["num_switches"].iloc[0]))


class TestIoU:
    def test_identical(self):
        b = np.array([0, 0, 10, 10])
        assert compute_iou(b, b) == 1.0

    def test_half_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 0, 15, 10])
        assert compute_iou(a, b) == pytest.approx(50 / 150)

    def test_disjoint(self):
        assert compute_iou(np.array([0, 0, 1, 1]), np.array([5, 5, 6, 6])) == 0.0


class TestCrossValidationAgainstMotmetrics:
    @pytest.mark.parametrize("scenario", [scenario_perfect, scenario_id_switch, scenario_fp_fn])
    def test_mota_idf1_match(self, scenario):
        pred, gt = scenario()
        ours = compute_tracking_metrics(pred, gt)
        mota_ref, idf1_ref, sw_ref = motmetrics_oracle(pred, gt)
        assert ours.mota == pytest.approx(mota_ref, abs=1e-6), scenario.__name__
        assert ours.idf1 == pytest.approx(idf1_ref, abs=1e-6), scenario.__name__
        assert ours.num_switches == sw_ref, scenario.__name__


class TestTrackingMetricsDirect:
    def test_perfect_is_one(self):
        pred, gt = scenario_perfect()
        m = compute_tracking_metrics(pred, gt)
        assert m.mota == 1.0 and m.idf1 == 1.0
        assert m.num_switches == 0 and m.num_fragmentations == 0

    def test_id_switch_counted(self):
        pred, gt = scenario_id_switch(20)
        m = compute_tracking_metrics(pred, gt)
        assert m.num_switches == 1
        assert m.mota == pytest.approx(1.0 - 1 / 20)
        assert m.idf1 == pytest.approx(0.5)  # best ID covers half the frames

    def test_fragmentation_counted(self):
        pred, gt = scenario_fp_fn(20)
        m = compute_tracking_metrics(pred, gt)
        # miss at k=5 breaks coverage once → 1 fragmentation, 1 GT track
        assert m.num_fragmentations == 1
        assert m.track_fragmentation_rate == 1.0
        assert m.num_misses == 1 and m.num_false_positives == 1


class TestDetectionMetrics:
    def test_perfect_detections(self):
        gt = [np.array([[0, 0, 10, 10, 1]])] * 5
        pred = [np.array([[0, 0, 10, 10, 0.9, 1]])] * 5
        m = compute_detection_metrics(pred, gt)
        assert m.precision == 1.0 and m.recall == 1.0
        assert m.f1 == pytest.approx(1.0) and m.map50 == pytest.approx(1.0)

    def test_one_fp_one_fn(self):
        gt = [np.array([[0, 0, 10, 10, 1], [50, 50, 60, 60, 1]])]
        pred = [np.array([[0, 0, 10, 10, 0.9, 1], [200, 200, 210, 210, 0.8, 1]])]
        m = compute_detection_metrics(pred, gt)
        assert m.precision == 0.5   # 1 TP, 1 FP
        assert m.recall == 0.5      # 1 TP, 1 FN

    def test_class_confusion_is_fp(self):
        gt = [np.array([[0, 0, 10, 10, 1]])]
        pred = [np.array([[0, 0, 10, 10, 0.9, 2]])]  # right box, wrong class
        m = compute_detection_metrics(pred, gt)
        assert m.precision == 0.0 and m.recall == 0.0


class TestSeverityMetrics:
    def test_accuracy_and_confusion(self):
        gt = [0, 0, 1, 1, 2, 2]
        pred = [0, 1, 1, 1, 2, 0]
        m = compute_severity_metrics(pred, gt)
        assert m.accuracy == pytest.approx(4 / 6)
        assert m.confusion_matrix[0, 0] == 1
        assert m.confusion_matrix[0, 1] == 1
        assert m.confusion_matrix[2, 0] == 1
        assert m.confusion_matrix.sum() == 6
