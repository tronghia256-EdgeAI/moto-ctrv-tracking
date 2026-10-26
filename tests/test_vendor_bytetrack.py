"""
tests/test_vendor_bytetrack.py
──────────────────────────────
Sanity tests for the vendored ByteTrack (code/tracker/vendor/).

Verifies the Windows/numpy compat patches (PATCHES.md) did not change
tracking behavior: stable IDs on clean streams, no crash on empty frames,
and the scipy linear_assignment shim matches hand-computed optima.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from code.tracker.vendor import BYTETracker, BaseTrack
from code.tracker.vendor.matching import linear_assignment, ious
from tests.synthetic import straight, to_detections, CLASS_CAR, CLASS_MOTORCYCLE

IMG_HW = (720, 1280)  # img_info == img_size -> scale == 1 (frame-native coords)


def make_tracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8):
    BaseTrack._count = 0  # independent runs
    args = SimpleNamespace(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        mot20=False,
    )
    return BYTETracker(args, frame_rate=30)


class TestVanillaTracking:
    def test_two_objects_stable_ids_100_frames(self):
        t1 = straight(v=5.0, n=100, start=(100, 100), heading=0.0)
        t2 = straight(v=5.0, n=100, start=(100, 500), heading=0.0)
        frames = to_detections([t1, t2], class_ids=[CLASS_CAR, CLASS_MOTORCYCLE])

        tracker = make_tracker()
        ids_per_frame = []
        for dets in frames:
            tracks = tracker.update(dets, IMG_HW, IMG_HW)
            ids_per_frame.append(sorted(t.track_id for t in tracks))

        # From frame 2 on, exactly the same 2 IDs every frame.
        stable = ids_per_frame[1]
        assert len(stable) == 2
        for ids in ids_per_frame[1:]:
            assert ids == stable

    def test_class_id_carried_through(self):
        t1 = straight(v=5.0, n=10, start=(100, 100))
        frames = to_detections([t1], class_ids=[CLASS_MOTORCYCLE])
        tracker = make_tracker()
        tracks = []
        for dets in frames:
            tracks = tracker.update(dets, IMG_HW, IMG_HW)
        assert tracks[0].class_id == CLASS_MOTORCYCLE

    def test_empty_frames_no_crash(self):
        tracker = make_tracker()
        t1 = straight(v=5.0, n=10, start=(100, 100))
        frames = to_detections([t1])
        for k, dets in enumerate(frames):
            # every other frame empty
            tracker.update(dets if k % 2 == 0 else np.empty((0, 6)), IMG_HW, IMG_HW)

    def test_n5_input_without_class(self):
        tracker = make_tracker()
        dets5 = np.array([[100, 100, 140, 160, 0.9]])
        tracks = tracker.update(dets5, IMG_HW, IMG_HW)
        assert len(tracks) == 1
        assert tracks[0].class_id == -1.0


class TestLinearAssignmentShim:
    """Hand-checked 3×3 matrices, including the cost_limit edge case
    lap.lapjv(extend_cost=True, cost_limit=thresh) implements."""

    def test_simple_diagonal(self):
        cost = np.array([[0.1, 0.9, 0.9],
                         [0.9, 0.2, 0.9],
                         [0.9, 0.9, 0.3]])
        matches, ua, ub = linear_assignment(cost, thresh=0.8)
        assert sorted(map(tuple, matches)) == [(0, 0), (1, 1), (2, 2)]
        assert len(ua) == 0 and len(ub) == 0

    def test_cost_limit_excludes_expensive_pair(self):
        # Optimal full assignment needs pair (2,2) at 0.95 > limit 0.8:
        # the solver must leave row 2 / col 2 unmatched instead.
        cost = np.array([[0.1, 0.9, 0.9],
                         [0.9, 0.2, 0.9],
                         [0.9, 0.9, 0.95]])
        matches, ua, ub = linear_assignment(cost, thresh=0.8)
        assert sorted(map(tuple, matches)) == [(0, 0), (1, 1)]
        assert list(ua) == [2] and list(ub) == [2]

    def test_rectangular_matrix(self):
        cost = np.array([[0.1, 0.9],
                         [0.9, 0.2],
                         [0.5, 0.6]])
        matches, ua, ub = linear_assignment(cost, thresh=0.8)
        assert sorted(map(tuple, matches)) == [(0, 0), (1, 1)]
        assert list(ua) == [2]
        assert len(ub) == 0

    def test_prefers_cheaper_global_solution(self):
        # Greedy would take (0,0)=0.1 then forces (1,1)=0.7; optimal total
        # is (0,1)=0.2 + (1,0)=0.3 = 0.5 < 0.1+0.7 = 0.8.
        cost = np.array([[0.1, 0.2],
                         [0.3, 0.7]])
        matches, _, _ = linear_assignment(cost, thresh=0.8)
        assert sorted(map(tuple, matches)) == [(0, 1), (1, 0)]

    def test_empty_matrix(self):
        matches, ua, ub = linear_assignment(np.empty((0, 3)), thresh=0.8)
        assert len(matches) == 0
        assert tuple(ub) == (0, 1, 2)


class TestIoUReplacement:
    def test_identical_boxes(self):
        a = [np.array([0, 0, 10, 10], dtype=float)]
        assert ious(a, a)[0, 0] == 1.0

    def test_known_overlap_plus_one_convention(self):
        # cython_bbox convention: area = (x2-x1+1)*(y2-y1+1).
        # Boxes [0,0,9,9] and [5,0,14,9]: inter = 5*10=50 (widths 9-5+1),
        # areas 100 each -> IoU = 50/150.
        a = [np.array([0, 0, 9, 9], dtype=float)]
        b = [np.array([5, 0, 14, 9], dtype=float)]
        assert ious(a, b)[0, 0] == np.float64(50) / 150

    def test_disjoint_boxes(self):
        a = [np.array([0, 0, 10, 10], dtype=float)]
        b = [np.array([100, 100, 110, 110], dtype=float)]
        assert ious(a, b)[0, 0] == 0.0

    def test_empty(self):
        assert ious([], []).shape == (0, 0)
