"""
code/evaluation/metrics.py
───────────────────────────
All evaluation metrics for Paper 2 — self-implemented (a project
learning goal), cross-validated against py-motmetrics in
tests/test_metrics.py.

Tracking : MOTA, IDF1, Track Fragmentation Rate
Detection : Precision, Recall, F1, mAP@0.5
Severity  : Accuracy, Confusion Matrix (3-class)
System    : FPS, Latency

References:
  Bernardin & Stiefelhagen, EURASIP 2008 (CLEAR MOT metrics)
  Ristani et al., ECCV 2016 workshops (IDF1)

Frame data format used throughout:
  a "frame list" is a list over frames; each frame is a list of
  (track_id, box) tuples with box = [x1, y1, x2, y2].

Author  : Trọng Nghĩa
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.optimize


IOU_MATCH_THRESHOLD = 0.5

FrameList = List[List[Tuple[int, Sequence[float]]]]


# ── Result containers ────────────────────────────────────────────────

@dataclass
class TrackingMetrics:
    mota: float                  # Multiple Object Tracking Accuracy
    idf1: float                  # ID F1 Score
    track_fragmentation_rate: float  # fragmentations / GT tracks
    num_switches: int
    num_fragmentations: int
    num_misses: int
    num_false_positives: int

@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    map50: float                 # mAP @ IoU 0.5

@dataclass
class SeverityMetrics:
    accuracy: float
    confusion_matrix: np.ndarray  # (3, 3)

@dataclass
class SystemMetrics:
    fps: float
    latency_ms: float


# ── Geometry ─────────────────────────────────────────────────────────

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two boxes [x1, y1, x2, y2] (no +1 pixel convention —
    evaluation uses plain geometric area, unlike the vendored tracker's
    cython_bbox-compatible IoU)."""
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _iou_matrix(gts, preds) -> np.ndarray:
    m = np.zeros((len(gts), len(preds)))
    for i, (_, gb) in enumerate(gts):
        for j, (_, pb) in enumerate(preds):
            m[i, j] = compute_iou(np.asarray(gb), np.asarray(pb))
    return m


# ── Tracking: CLEAR MOT + IDF1 ───────────────────────────────────────

def compute_tracking_metrics(pred_frames: FrameList,
                             gt_frames: FrameList,
                             iou_threshold: float = IOU_MATCH_THRESHOLD) -> TrackingMetrics:
    """
    Args:
        pred_frames: per-frame lists of (pred_track_id, box)
        gt_frames  : per-frame lists of (gt_track_id, box)

    Returns:
        TrackingMetrics (MOTA per CLEAR MOT, IDF1 per Ristani et al.,
        fragmentation rate = coverage interruptions / #GT tracks)
    """
    assert len(pred_frames) == len(gt_frames), "frame counts differ"

    misses = 0
    false_positives = 0
    switches = 0
    total_gt = 0

    # CLEAR MOT state: gt_id → last matched pred_id (persists across gaps)
    last_match: Dict[int, int] = {}
    # For fragmentation: gt_id → was it matched in the previous *visible* frame,
    # and has it ever been matched
    frag_state: Dict[int, dict] = {}
    fragmentations = 0

    for gts, preds in zip(gt_frames, pred_frames):
        total_gt += len(gts)
        iou = _iou_matrix(gts, preds)

        matched_gt: Dict[int, int] = {}   # gt index → pred index
        used_pred = set()

        # Step 1 (CLEAR): keep still-valid previous correspondences.
        pred_id_to_idx = {pid: j for j, (pid, _) in enumerate(preds)}
        for i, (gid, _) in enumerate(gts):
            prev_pid = last_match.get(gid)
            if prev_pid is not None and prev_pid in pred_id_to_idx:
                j = pred_id_to_idx[prev_pid]
                if j not in used_pred and iou[i, j] >= iou_threshold:
                    matched_gt[i] = j
                    used_pred.add(j)

        # Step 2: Hungarian on the rest (maximize IoU ⇔ minimize 1 − IoU).
        free_gt = [i for i in range(len(gts)) if i not in matched_gt]
        free_pred = [j for j in range(len(preds)) if j not in used_pred]
        if free_gt and free_pred:
            sub = 1.0 - iou[np.ix_(free_gt, free_pred)]
            rows, cols = scipy.optimize.linear_sum_assignment(sub)
            for r, c in zip(rows, cols):
                i, j = free_gt[r], free_pred[c]
                if iou[i, j] >= iou_threshold:
                    matched_gt[i] = j
                    used_pred.add(j)

        # Step 3: bookkeeping.
        for i, (gid, _) in enumerate(gts):
            st = frag_state.setdefault(gid, {"ever": False, "prev": False})
            if i in matched_gt:
                pid = preds[matched_gt[i]][0]
                if gid in last_match and last_match[gid] != pid:
                    switches += 1
                if st["ever"] and not st["prev"]:
                    fragmentations += 1
                last_match[gid] = pid
                st["ever"] = True
                st["prev"] = True
            else:
                misses += 1
                st["prev"] = False

        false_positives += len(preds) - len(used_pred)

    mota = 1.0 - (misses + false_positives + switches) / max(total_gt, 1)

    idf1 = _compute_idf1(pred_frames, gt_frames, iou_threshold)

    num_gt_tracks = len(frag_state) if frag_state else 1
    return TrackingMetrics(
        mota=mota,
        idf1=idf1,
        track_fragmentation_rate=fragmentations / num_gt_tracks,
        num_switches=switches,
        num_fragmentations=fragmentations,
        num_misses=misses,
        num_false_positives=false_positives,
    )


def _compute_idf1(pred_frames: FrameList, gt_frames: FrameList,
                  iou_threshold: float) -> float:
    """IDF1: global one-to-one ID matching maximizing per-frame overlaps.

    counts[g][p] = #frames where gt g and pred p overlap with IoU ≥ thr;
    Hungarian-maximize the total → IDTP.
    IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)
         = 2·IDTP / (total_pred_dets + total_gt_dets)
    """
    gt_ids: Dict[int, int] = {}
    pred_ids: Dict[int, int] = {}
    pair_counts: Dict[Tuple[int, int], int] = {}
    total_gt_dets = 0
    total_pred_dets = 0

    for gts, preds in zip(gt_frames, pred_frames):
        total_gt_dets += len(gts)
        total_pred_dets += len(preds)
        for gid, gb in gts:
            gt_ids.setdefault(gid, len(gt_ids))
        for pid, pb in preds:
            pred_ids.setdefault(pid, len(pred_ids))
        for gid, gb in gts:
            for pid, pb in preds:
                if compute_iou(np.asarray(gb), np.asarray(pb)) >= iou_threshold:
                    pair_counts[(gid, pid)] = pair_counts.get((gid, pid), 0) + 1

    if not pair_counts:
        return 0.0

    counts = np.zeros((len(gt_ids), len(pred_ids)))
    for (gid, pid), c in pair_counts.items():
        counts[gt_ids[gid], pred_ids[pid]] = c

    rows, cols = scipy.optimize.linear_sum_assignment(-counts)
    idtp = counts[rows, cols].sum()
    return float(2.0 * idtp / (total_pred_dets + total_gt_dets))


# ── Detection metrics ────────────────────────────────────────────────

def compute_detection_metrics(pred_boxes: List[np.ndarray],
                              gt_boxes: List[np.ndarray],
                              iou_threshold: float = IOU_MATCH_THRESHOLD) -> DetectionMetrics:
    """
    Args:
        pred_boxes: per-frame arrays of [x1,y1,x2,y2,conf,class_id]
        gt_boxes  : per-frame arrays of [x1,y1,x2,y2,class_id]

    Returns:
        DetectionMetrics. P/R/F1 counted over all frames at the given IoU
        threshold (confidence-descending greedy matching within class);
        mAP@0.5 is all-point AP averaged over classes present in GT.
    """
    # Collect per-class scored predictions and match flags for AP;
    # simultaneously count TP/FP/FN for P/R/F1.
    tp = fp = fn = 0
    ap_records: Dict[int, List[Tuple[float, bool]]] = {}
    gt_count_per_class: Dict[int, int] = {}

    for preds, gts in zip(pred_boxes, gt_boxes):
        preds = np.asarray(preds, dtype=float).reshape(-1, 6)
        gts = np.asarray(gts, dtype=float).reshape(-1, 5)
        classes = set(gts[:, 4].astype(int)) | set(preds[:, 5].astype(int))
        for cls in classes:
            p = preds[preds[:, 5] == cls]
            g = gts[gts[:, 4] == cls]
            gt_count_per_class[cls] = gt_count_per_class.get(cls, 0) + len(g)
            order = np.argsort(-p[:, 4])
            matched_gt = set()
            for idx in order:
                best_iou, best_j = 0.0, -1
                for j in range(len(g)):
                    if j in matched_gt:
                        continue
                    iou = compute_iou(p[idx, :4], g[j, :4])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                is_tp = best_iou >= iou_threshold
                if is_tp:
                    matched_gt.add(best_j)
                    tp += 1
                else:
                    fp += 1
                ap_records.setdefault(cls, []).append((float(p[idx, 4]), is_tp))
            fn += len(g) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    # All-point AP per class with GT instances.
    aps = []
    for cls, n_gt in gt_count_per_class.items():
        if n_gt == 0:
            continue
        recs = sorted(ap_records.get(cls, []), key=lambda r: -r[0])
        tps = np.array([r[1] for r in recs], dtype=float)
        if len(tps) == 0:
            aps.append(0.0)
            continue
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(1.0 - tps)
        rec = cum_tp / n_gt
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        # all-point interpolation
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))

    map50 = float(np.mean(aps)) if aps else 0.0
    return DetectionMetrics(precision=precision, recall=recall, f1=f1, map50=map50)


# ── Severity metrics ─────────────────────────────────────────────────

def compute_severity_metrics(pred_labels: List[int],
                             gt_labels: List[int]) -> SeverityMetrics:
    """Accuracy + 3×3 confusion matrix (rows = GT, cols = predicted)."""
    pred = np.asarray(pred_labels, dtype=int)
    gt = np.asarray(gt_labels, dtype=int)
    assert pred.shape == gt.shape
    cm = np.zeros((3, 3), dtype=int)
    for g, p in zip(gt, pred):
        cm[g, p] += 1
    acc = float((pred == gt).mean()) if len(gt) else 0.0
    return SeverityMetrics(accuracy=acc, confusion_matrix=cm)
