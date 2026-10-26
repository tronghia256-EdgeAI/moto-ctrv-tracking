# Vendored from ByteTrack (ifzhang/ByteTrack) yolox/tracker/matching.py
# Patched for Windows / modern numpy — see PATCHES.md for the full list:
#   - lap.lapjv        -> scipy.optimize.linear_sum_assignment (equivalent
#                         extended-matrix construction, P3)
#   - cython_bbox      -> pure-numpy pairwise IoU (P4, keeps the +1 pixel
#                         area convention of cython_bbox.bbox_overlaps)
#   - np.float         -> float (P2)
#   - removed functions unused by byte_tracker.py (P8)
import numpy as np
import scipy.optimize


def linear_assignment(cost_matrix, thresh):
    """Drop-in replacement for lap.lapjv(cost, extend_cost=True, cost_limit=thresh).

    Uses the standard extended-square-matrix construction: each real
    assignment (r, c) competes against leaving both r and c unassigned at
    thresh/2 + thresh/2 = thresh, so a real pair is selected iff its cost
    beats the cost limit — the same semantics lapjv implements internally.
    Returns (matches, unmatched_a, unmatched_b) in the upstream format.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    n, m = cost_matrix.shape
    extended = np.full((n + m, n + m), thresh / 2.0)
    extended[n:, m:] = 0.0
    extended[:n, :m] = cost_matrix
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(extended)

    matches = [[r, c] for r, c in zip(row_ind, col_ind) if r < n and c < m]
    matched_a = {r for r, _ in matches}
    matched_b = {c for _, c in matches}
    unmatched_a = np.array([i for i in range(n) if i not in matched_a], dtype=int)
    unmatched_b = np.array([j for j in range(m) if j not in matched_b], dtype=int)
    matches = np.asarray(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray

    Pure-numpy replacement for cython_bbox.bbox_overlaps. Keeps the +1
    pixel-area convention of the original (area = (x2-x1+1)*(y2-y1+1)) so
    values are bit-comparable with upstream.
    """
    result = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if result.size == 0:
        return result

    a = np.ascontiguousarray(atlbrs, dtype=float)
    b = np.ascontiguousarray(btlbrs, dtype=float)

    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(br - tl + 1.0, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = (a[:, 2] - a[:, 0] + 1.0) * (a[:, 3] - a[:, 1] + 1.0)
    area_b = (b[:, 2] - b[:, 0] + 1.0) * (b[:, 3] - b[:, 1] + 1.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
