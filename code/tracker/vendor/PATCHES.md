# Vendored ByteTrack — patch log

Source: https://github.com/ifzhang/ByteTrack, files `yolox/tracker/{byte_tracker,
kalman_filter, basetrack, matching}.py` (local clone at `lib/ByteTrack/`,
gitignored).

**Guarantee:** the BYTE association logic — two-stage high/low-score matching,
all thresholds, the track state machine, duplicate removal — is byte-for-byte
identical to upstream. Only imports, input format, and two hook lines changed.
Exp0 (baseline) therefore remains a legitimate vanilla-ByteTrack comparison.

## Why vendor at all

Upstream cannot run on this machine (Windows 11, Python 3.10, numpy ≥ 1.24):
`np.float` was removed in numpy 1.24; `cython_bbox` needs MSVC compilation;
`lap` has stale Windows wheels; `byte_tracker.py` imports `torch` and assumes
a YOLOX-specific detection format. `lib/` is also gitignored, so a fresh clone
of this repo must work without the ByteTrack clone present.

## Patch list

| # | File | Change | Behavioral impact |
|---|------|--------|-------------------|
| P1 | byte_tracker.py | `from yolox.tracker import matching` → `from . import matching`; dropped `torch`, `os`, `copy`, `deque` imports | none |
| P2 | byte_tracker.py, matching.py | `np.float` → `float` | none (alias removal) |
| P3 | matching.py | `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)` → `scipy.optimize.linear_sum_assignment` on the standard extended square matrix (`[[C, τ/2·1], [τ/2·1ᵀ, 0]]`) | equivalent optimum; ties at exactly cost == thresh may resolve differently (upstream behavior at the boundary is likewise unspecified) |
| P4 | matching.py | `cython_bbox.bbox_overlaps` → pure-numpy pairwise IoU, keeping the original `+1` pixel-area convention | none (numerically identical) |
| P5 | byte_tracker.py | `update()` accepts numpy N×5 `[x1,y1,x2,y2,score]` or N×6 `[...,class_id]`; removed the upstream torch branch (`obj_conf * cls_conf` YOLOX format) | input contract only; scoring/thresholds unchanged. Pipeline always passes frame-native coords with `img_info == img_size` → `scale == 1` |
| P6 | byte_tracker.py | `STrack.__init__(tlwh, score)` → `(tlwh, score, class_id=None)`; stores `self.class_id` | metadata only, never used in association |
| P7 | byte_tracker.py | class attribute `BYTETracker.strack_cls = STrack`, used at the two detection-construction sites and for the `multi_predict` call | none with default; enables `AdaptiveBYTETracker` to inject `ClassAwareSTrack` without copying `update()` |
| P8 | matching.py | removed functions unused by `byte_tracker.py` (`merge_matches`, `_indices_to_matches`, `v_iou_distance`, `embedding_distance`, `gate_cost_matrix`, `fuse_motion`, `fuse_iou`) and their `cv2`/`cdist`/`time` imports | none (dead code for this pipeline) |
| P9 | kalman_filter.py | copied verbatim | none |
| P10 | basetrack.py | copied verbatim | none |
| P11 | byte_tracker.py | first-association cost extracted into overridable `BYTETracker._first_association_cost(strack_pool, detections)` (base = plain `matching.iou_distance`, identical to upstream) | none with default; enables `AdaptiveBYTETracker` to add CTRV uncertainty gating for lost motorcycle tracks without copying `update()` |
