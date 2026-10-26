"""
tests/test_gt_roundtrip.py
───────────────────────────
Phase 2 GT pipeline: export_prelabels.write_mot ↔ import_gt.load_mot
must round-trip exactly, and a file scored against itself must give
perfect MOTA/IDF1 — the smoke test that validates the whole
export → CVAT → import → metrics path before humans spend hours in it.
"""

from __future__ import annotations

import numpy as np
import pytest

from code.evaluation.metrics import compute_tracking_metrics
from code.scripts.export_prelabels import write_mot
from code.scripts.import_gt import load_mot, to_framelist, tracks_json_slice


# Two tracks (moto id 7, car id 3) over 3 frames; moto skips frame 2.
FRAMES = [
    [(7, 1, [100.0, 100.0, 140.0, 160.0]), (3, 0, [300.0, 200.0, 380.0, 260.0])],
    [(3, 0, [305.0, 202.0, 385.0, 262.0])],
    [(7, 1, [110.0, 105.0, 150.0, 165.0]), (3, 0, [310.0, 204.0, 390.0, 264.0])],
]


@pytest.fixture
def mot_dir(tmp_path):
    stats = write_mot(FRAMES, tmp_path)
    assert stats == {"frames": 3, "boxes": 5, "track_ids": 2}
    return tmp_path


class TestRoundTrip:
    def test_boxes_ids_classes_survive(self, mot_dir):
        loaded = load_mot(mot_dir / "gt" / "gt.txt")
        assert set(loaded) == {1, 2, 3}          # 1-based frame ids
        for k, entries in enumerate(FRAMES, start=1):
            got = sorted(loaded[k])
            want = sorted((tid, [b[0], b[1], b[2], b[3]], cls)
                          for tid, cls, b in entries)
            for (gt_id, gt_box, gt_cls), (w_id, w_box, w_cls) in zip(got, want):
                assert gt_id == w_id and gt_cls == w_cls
                np.testing.assert_allclose(gt_box, w_box, atol=0.11)

    def test_ignored_lines_skipped(self, tmp_path):
        p = tmp_path / "gt.txt"
        p.write_text("1,1,10,10,20,20,1,2,1.0\n"
                     "1,2,50,50,20,20,0,2,1.0\n", encoding="utf-8")
        loaded = load_mot(p)
        assert len(loaded[1]) == 1 and loaded[1][0][0] == 1

    def test_short_line_raises(self, tmp_path):
        p = tmp_path / "gt.txt"
        p.write_text("1,1,10,10,20,20\n", encoding="utf-8")
        with pytest.raises(ValueError, match="columns"):
            load_mot(p)


class TestSmokeSelfScore:
    def test_self_score_is_perfect(self, mot_dir):
        """The Phase 2 acceptance gate: pre-labels scored against
        themselves through the full import path must give MOTA = IDF1 = 1
        with zero errors of any kind."""
        frames = load_mot(mot_dir / "gt" / "gt.txt")
        fl = to_framelist(frames)
        m = compute_tracking_metrics(fl, fl)
        assert m.mota == pytest.approx(1.0)
        assert m.idf1 == pytest.approx(1.0)
        assert (m.num_switches, m.num_misses, m.num_false_positives) == (0, 0, 0)

    def test_class_filter(self, mot_dir):
        frames = load_mot(mot_dir / "gt" / "gt.txt")
        moto = to_framelist(frames, class_id=1)
        assert [len(f) for f in moto] == [1, 0, 1]
        car = to_framelist(frames, class_id=0)
        assert [len(f) for f in car] == [1, 1, 1]


class TestTracksJsonSlice:
    def test_slice_matches_written_mot(self, tmp_path, mot_dir):
        import json
        dump = {"video": "v.asf", "start_frame": 0,
                "num_frames": 5,
                "frames": [[]] + FRAMES + [[]]}   # segment is [1:4]
        p = tmp_path / "tracks_x.json"
        p.write_text(json.dumps(dump), encoding="utf-8")
        sliced = tracks_json_slice(p, 1, 4)
        mot = load_mot(mot_dir / "gt" / "gt.txt")
        fl_a = to_framelist(sliced, 3)
        fl_b = to_framelist(mot, 3)
        m = compute_tracking_metrics(fl_a, fl_b)
        assert m.mota == pytest.approx(1.0)
        assert m.idf1 == pytest.approx(1.0)
