"""New-camera GT (code/scripts/gt_newcam.py): the pre-registered window rule and the held-out label."""

import json

import pytest

from code.scripts import gt_mixed_review, gt_newcam
from code.scripts.gt_newcam import densest_window, union_seed
from tests.test_gt_mixed_review import complete_audit, factory  # noqa: F401  (fixture)


def test_densest_window_skips_the_warmup_and_finds_the_dense_block():
    counts = {f: 1 for f in range(0, 3000, 5)}
    counts.update({f: 50 for f in range(0, 600, 5)})       # denser, but inside the warm-up
    counts.update({f: 20 for f in range(1500, 2100, 5)})
    assert densest_window(counts, 3000) == (1500, 20.0)


def test_union_seed_fills_gaps_under_the_matched_identity_and_adds_the_rest_fresh():
    a, b = [0, 0, 10, 10], [50, 50, 60, 60]
    base = [[[1, 1, a]], [[1, 1, a]], [], [[1, 1, a]]]
    other = [[[7, 1, a]], [[7, 1, a], [8, 1, b]], [[7, 1, a]], [[7, 1, a], [9, 0, b]]]
    seed, stats = union_seed([base, other], 0, 4)
    assert seed[0] == [(1, 1, a)] and seed[3][0] == (1, 1, a)     # shared boxes are not duplicated
    assert seed[2] == [(1, 1, a)]                                  # gap filled under the matched identity
    assert seed[1] == [(1, 1, a), (2, 1, b)]                       # never-matched track: fresh identity
    assert seed[3][1] == (3, 0, b)                                 # another track, another fresh one
    assert stats == {"kept": 3, "linked": 1, "fresh_boxes": 2, "fresh_ids": 2}


def write_run(root, store):
    """window.json and a baseline dump reproducing the store's GT, under root/cam."""
    (root / "cam/dumps/yolo11_vn_1280").mkdir(parents=True)
    (root / "cam/window.json").write_text(
        json.dumps({"start": 0, "end": 6, "rule": "r", "profile": {"0": 1}}), encoding="utf-8")
    frames = [[] for _ in range(6)]
    for (f, _), p in store.preview()[0].items():
        x, y, w, h = map(float, p[2:6])
        frames[f - 1].append([int(p[1]), int(p[7]) - 1, [x, y, x + w, y + h]])
    (root / "cam/dumps/yolo11_vn_1280/tracks_baseline.json").write_text(
        json.dumps(dict(start_frame=0, frames=frames)), encoding="utf-8")


def test_finalize_relabels_the_mixed_role_as_held_out(factory, tmp_path, monkeypatch):
    store = factory()
    complete_audit(store)
    monkeypatch.setattr(gt_mixed_review, "MixedStore", lambda bundle: store)
    monkeypatch.setattr(gt_newcam, "ROOT", tmp_path)
    write_run(tmp_path, store)

    gt_newcam.finalize("cam", "Nghĩa")

    report = json.loads((tmp_path / "cam/reviewed/review.json").read_text(encoding="utf-8"))
    assert report["evaluation_role"] == gt_newcam.HELD_OUT_ROLE
    assert report["window"] == {"start": 0, "end": 6, "rule": "r"}
    assert report["prelabel_seed"]["trackers"] == ["yolo11_vn_1280/baseline"]   # no seed.json: legacy seed
    assert report["scored_detector"]["name"] == "yolo11_vn_1280"
    summary = (tmp_path / "cam/reviewed/summary.md").read_text(encoding="utf-8")
    assert gt_newcam.HELD_OUT_ROLE in summary and "development" not in summary


def test_scope_then_finalize_use_the_roi_bundle_and_leave_the_mixed_pointer(factory, tmp_path, monkeypatch):
    from code.scripts import gt_roi_scope
    from code.scripts.gt_roi import save_roi
    parent = factory()
    monkeypatch.setattr(gt_newcam, "parent_bundle", lambda camera: parent.bundle)
    monkeypatch.setattr(gt_newcam, "ROOT", tmp_path / "newcam")
    monkeypatch.setattr(gt_roi_scope, "POINTER", tmp_path / "mixed_pointer.json")
    monkeypatch.setattr(gt_roi_scope, "REPO_ROOT", tmp_path)        # ROI seed snapshots
    assert gt_newcam.roi_child("cam") is None
    save_roi(parent.bundle, [[0, 0], [200, 0], [200, 450], [0, 450]], True, "Nghia", "test", 1, None)
    (tmp_path / "newcam/cam").mkdir(parents=True)
    seed = tmp_path / "newcam/cam/seed.json"
    seed.write_text(json.dumps({"trackers": ["baseline"]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="seeded from one tracker"):
        gt_newcam.scope("cam")                                     # a one-tracker seed never reaches review
    seed.write_text(json.dumps({"trackers": ["baseline", "ctrv_gate_nsa"]}), encoding="utf-8")

    gt_newcam.scope("cam")

    assert not (tmp_path / "mixed_pointer.json").exists()
    child = gt_mixed_review.MixedStore(gt_newcam.roi_child("cam"))
    child.record_seen(range(1, 7))
    for i in range(len(child.rows)):
        child.save(i, "KEEP", "checked")
    write_run(tmp_path / "newcam", child)
    gt_newcam.finalize("cam", "Nghĩa")
    report = json.loads((tmp_path / "newcam/cam/reviewed/review.json").read_text(encoding="utf-8"))
    assert report["review_scope_provenance"]["mode"] == "ROI_ONLY_REVIEW"
    assert report["evaluation_role"] == gt_newcam.HELD_OUT_ROLE
    assert report["prelabel_seed"]["trackers"] == ["baseline", "ctrv_gate_nsa"]
