"""
tests/test_gt_review_queue.py
──────────────────────────────
Phase 2 GT infrastructure: the review queue routes a human's attention,
so the failure that matters is a *wrong frame number* or a check that
fires on ordinary traffic. Both waste the annotation session and, worse,
teach the annotator to distrust the list and skim it.

Each check is exercised on a hand-built scene where the right answer is
known by construction.
"""

from __future__ import annotations

import pytest

from code.scripts.gt_review_queue import (
    build_tracks, find_dups, find_ghosts, find_jumps, find_merges,
    find_orphans, robust_threshold, to_cvat,
)

W, H = 800, 450


def scene(tracks: dict[int, list]) -> dict[int, list]:
    """{track_id: [(frame, box, cls)]} -> the {frame: [...]} shape the
    detectors consume."""
    frames: dict[int, list] = {}
    for tid, entries in tracks.items():
        for f, box, cls in entries:
            frames.setdefault(f, []).append((tid, list(box), cls))
    return frames


def moving(tid: int, f0: int, n: int, x: float, y: float,
           dx: float = 4.0, w: float = 40.0, h: float = 60.0, cls: int = 1):
    """A track drifting at a constant velocity, well inside the frame."""
    return [(f0 + i, (x + dx * i, y, x + dx * i + w, y + h), cls)
            for i in range(n)]


class TestFrameNumbering:
    """CVAT counts frames from 0, gt.txt from 1. Every number in the
    report is typed into CVAT, so the conversion must be total."""

    def test_at_refs_shift_by_one(self):
        assert to_cvat("chet o frame @465, sinh o frame @467") == \
            "chet o frame 464, sinh o frame 466"

    def test_untagged_numbers_untouched(self):
        # gaps, distances and IDs are counts, not frame indices
        assert to_cvat("ID 7036 gap 2f, lech 0.2 box") == \
            "ID 7036 gap 2f, lech 0.2 box"

    def test_frame_one_maps_to_zero(self):
        assert to_cvat("frame @1") == "frame 0"


class TestMerge:
    def test_pairs_a_fragmented_track(self):
        """The C2 failure case: one vehicle, ID dropped and re-created a
        few frames later further along its own heading."""
        tracks = build_tracks(scene({
            10: moving(10, 1, 20, 300, 200),
            11: moving(11, 24, 20, 300 + 4 * 23, 200),
        }))
        found = find_merges(tracks, 60, W, H)
        assert len(found) == 1
        assert found[0].ids == (10, 11)
        assert found[0].frame == 20        # the death frame, MOT numbering

    def test_ignores_border_exit(self):
        """A vehicle leaving the frame edge and an unrelated one entering
        elsewhere is ordinary traffic, not a fragmented track."""
        tracks = build_tracks(scene({
            10: [(f, (740 + 4 * i, 200, 795 + 4 * i, 260), 1)
                 for i, f in enumerate(range(1, 15))],
            11: moving(11, 16, 10, 20, 400),
        }))
        assert find_merges(tracks, 60, W, H) == []

    def test_one_successor_per_track(self):
        """Two newborns near one death must not both be merged into it —
        a chain of contradictory merges is worse than none."""
        tracks = build_tracks(scene({
            10: moving(10, 1, 20, 300, 200),
            11: moving(11, 23, 10, 388, 200),
            12: moving(12, 23, 10, 392, 205),
        }))
        found = find_merges(tracks, 60, W, H)
        assert len(found) == 1
        used = {i for f in found for i in f.ids}
        assert 10 in used and len(used) == 2


class TestJump:
    def test_flags_an_id_teleport(self):
        """ID kept, box moved onto a different vehicle: no birth, no
        death, so this is the check that catches silent switches."""
        t = moving(20, 1, 30, 200, 200, dx=2.0)
        t[15] = (16, (600.0, 380.0, 640.0, 440.0), 1)   # teleport
        found, calib = find_jumps(build_tracks(scene({20: t})), W, H)
        assert [f.frame for f in found] == [16]
        assert calib["jump"] > 0

    def test_steady_traffic_is_silent(self):
        tracks = build_tracks(scene({
            20: moving(20, 1, 40, 100, 200),
            21: moving(21, 1, 40, 400, 300, dx=-3.0),
        }))
        found, _ = find_jumps(tracks, W, H)
        assert found == []

    def test_ignores_clipping_at_frame_edge(self):
        """A car driving out of frame gets clipped and its area collapses
        — the two largest area changes in the real `dense` segment were
        exactly this, and they are not identity errors."""
        t = [(1, (700.0, 380.0, 780.0, 448.0), 0),
             (2, (720.0, 400.0, 795.0, 449.0), 0),
             (3, (740.0, 430.0, 799.0, 449.0), 0)]
        found, _ = find_jumps(build_tracks(scene({30: t})), W, H)
        assert found == []


class TestDupGhostOrphan:
    def test_dup_reported_once_per_pair(self):
        a = moving(40, 1, 12, 200, 200)
        b = [(f, (x + 2, y + 2, x2 + 2, y2 + 2), c)
             for f, (x, y, x2, y2), c in a]
        frames = scene({40: a, 41: b})
        found = find_dups(frames, build_tracks(frames))
        assert len(found) == 1
        assert found[0].ids == (40, 41)
        assert found[0].frame == 1          # first frame they overlap

    def test_ghost_is_a_short_interior_track(self):
        tracks = build_tracks(scene({50: moving(50, 5, 3, 300, 200)}))
        found = find_ghosts(tracks, 60, W, H)
        assert len(found) == 1 and found[0].ids == (50,)

    def test_ghost_ignores_frame_edge_clip(self):
        tracks = build_tracks(scene({
            51: [(f, (0.0, 200.0, 30.0, 260.0), 1) for f in (1, 2, 3)],
        }))
        assert find_ghosts(tracks, 60, W, H) == []

    def test_orphan_suppressed_below_min_size(self):
        """Interior breaks on tiny distant vehicles are the common case in
        dense traffic (261 of them in `dense`); the guide says not to
        label those at all, so they must not fill the queue."""
        big = moving(60, 1, 20, 300, 200, w=60, h=80)
        small = moving(61, 1, 20, 100, 100, w=8, h=10)
        tracks = build_tracks(scene({60: big, 61: small}))
        found, suppressed = find_orphans(tracks, [], 60, W, H, min_diag=40.0)
        assert {f.ids[0] for f in found} == {60}
        assert suppressed >= 1

    def test_orphan_skips_already_merged(self):
        tracks = build_tracks(scene({60: moving(60, 1, 20, 300, 200,
                                                w=60, h=80)}))
        merged = find_merges(tracks, 60, W, H)      # none, but shape it
        found, _ = find_orphans(tracks, merged, 60, W, H)
        assert found                                 # unpaired -> reported
        fake = [type(found[0])("MERGE", 1, (60,), "car", 1.0, "", "")]
        assert find_orphans(tracks, fake, 60, W, H)[0] == []


class TestRobustThreshold:
    def test_outlier_cannot_raise_its_own_bar(self):
        """The reason this is not a percentile: p99.9 of these 30 values
        *is* the 4.0, so a percentile threshold would never flag it."""
        vals = [0.05] * 29 + [4.0]
        assert robust_threshold(vals, 10.0, 0.0) < 4.0

    def test_floor_wins_on_a_clean_segment(self):
        """Near-identical values give MAD ~ 0; without the floor every
        tiny deviation would be flagged."""
        assert robust_threshold([1.0] * 50, 10.0, 2.3) == 2.3

    def test_empty_falls_back_to_floor(self):
        assert robust_threshold([], 10.0, 0.35) == 0.35
