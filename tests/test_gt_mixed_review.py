"""Meaningful integrity checks for mixed identity/geometry edits and finalization."""
import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from code.scripts.gt_mixed_review import (
    DECISION_FIELDS, MixedStore, apply_review, finalize_mixed, json_bytes, prepare,
)
from code.scripts.gt_review import sha256


def case(kind, before, after, proposal=None, rank=1):
    return dict(rank=rank, kind=kind, before=list(before), after=list(after), proposal=proposal,
                ids=sorted({before[0], after[0]} - {0}), question="Question, not a verdict",
                window=[1, 6], sheet=f"{rank:03d}_{kind}.png")


def merge_case(a=(1, 2), b=(2, 4), rank=1, overlap=False):
    return case("DUP" if overlap else "MERGE", a, b,
                dict(op="merge", anchors=[list(a), list(b)], allow_overlap=overlap), rank)


def split_case(tid=1, before=1, after=2, rank=1):
    return case("SPLIT", (tid, before), (tid, after), dict(op="split", tid=tid, frame=after), rank)


@pytest.fixture
def factory(tmp_path):
    def build(cases=None, raw=None):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        raw = raw or ("1,1,100,100,20,40,1,2,1\n2,1,105,100,20,40,1,2,0.7\n"
                      "4,2,115,100,20,40,1,2,1\n5,2,120,100,20,40,1,2,1\n"
                      "3,3,250,100,20,40,1,1,1\n")
        (bundle / "source_gt.txt").write_text(raw, encoding="utf-8")
        images = bundle / "img1"
        images.mkdir()
        picture = Image.new("RGB", (800, 450), "white")
        hashes = {}
        for frame in range(1, 7):
            path = images / f"{frame:06d}.jpg"
            picture.save(path)
            hashes[path.name] = sha256(path)
        cases = cases or []
        cases = cases + [case("AUDIT", (0, 1), (0, 6), rank=len(cases) + 1)]
        for c in cases:
            picture.save(bundle / c["sheet"])
        manifest = dict(schema_version=1, source=dict(path="original.txt", sha256=sha256(bundle / "source_gt.txt")),
                        segment=[0, 6], n_frames=6, image_size=[800, 450], frames_dir=str(images),
                        images_sha256=hashes, cases=cases)
        (bundle / "manifest.json").write_bytes(json_bytes(manifest))
        with (bundle / "decisions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DECISION_FIELDS)
            writer.writeheader()
            for c in cases:
                writer.writerow(dict(rank=c["rank"], sheet=c["sheet"], kind=c["kind"],
                                     ids=" ".join(map(str, c["ids"])), frame_before=c["before"][1],
                                     frame_after=c["after"][1], decision="", note=""))
        (bundle / "edits.json").write_bytes(json_bytes(dict(schema_version=1, events=[], seen_frames=[])))
        return MixedStore(bundle)
    return build


def complete_audit(store):
    store.record_seen(range(1, 7))
    store.save(len(store.rows) - 1, "KEEP", "Checked full clip")


def test_pending_never_applies_proposals_and_original_unchanged(factory):
    store = factory([merge_case(), split_case(rank=2)])
    before = store.source_path.read_bytes()
    rows, stats = store.preview()
    assert rows == store.source_rows
    assert stats["identity_changed_boxes"] == 0
    assert store.source_path.read_bytes() == before
    assert store.tally["PENDING"] == 3


def test_merge_changes_only_id_fields_and_preserves_visibility(factory):
    store = factory([merge_case()])
    store.save(0, "APPLY", "Same motorcycle")
    rows, stats = store.preview()
    assert rows[(4, 2)][1] == "1"
    assert rows[(3, 3)][1] == "3"
    assert stats["identities_after"] == 2
    assert all(p[:1] + p[2:] == store.source_rows[k][:1] + store.source_rows[k][2:] for k, p in rows.items())
    assert rows[(2, 1)][8] == "0.7"


def test_split_preserves_prefix_and_allocates_noncolliding_id(factory):
    store = factory([split_case()])
    store.save(0, "APPLY", "Different vehicles")
    rows, stats = store.preview()
    assert rows[(1, 1)][1] == "1"
    assert rows[(2, 1)][1] == "4"
    assert stats["identities_after"] == 4


def test_merge_after_split_uses_only_anchored_tail_segment(factory):
    raw = ("1,1,100,100,20,40,1,2,1\n2,1,105,100,20,40,1,2,1\n"
           "3,1,110,100,20,40,1,2,1\n4,1,115,100,20,40,1,2,1\n6,2,125,100,20,40,1,2,1\n")
    store = factory([split_case(before=2, after=3), merge_case((1, 4), (2, 6), rank=2)], raw)
    store.save(0, "APPLY", "")
    store.save(1, "APPLY", "")
    rows, _ = store.preview()
    assert rows[(1, 1)][1] == rows[(2, 1)][1] == "1"
    assert rows[(3, 1)][1] == rows[(6, 2)][1] == "2"


def test_transitive_merge_cannot_undo_accepted_split(factory):
    raw = ("1,1,100,100,20,40,1,2,1\n2,1,105,100,20,40,1,2,1\n"
           "3,1,110,100,20,40,1,2,1\n4,1,115,100,20,40,1,2,1\n6,2,125,100,20,40,1,2,1\n")
    store = factory([split_case(before=2, after=3), merge_case((1, 4), (2, 6), rank=2),
                     merge_case((1, 1), (2, 6), rank=3)], raw)
    store.save(0, "APPLY", "")
    store.save(1, "APPLY", "")
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="undo this accepted split"):
        store.save(2, "APPLY", "")
    assert store.path.read_bytes() == before


def test_manual_merge_cannot_contradict_keep_separate(factory):
    store = factory([merge_case()])
    store.save(0, "KEEP", "Two vehicles")
    with pytest.raises(ValueError, match="contradicts"):
        store.add_event(dict(op="merge", anchors=[[1, 2], [2, 4]], allow_overlap=False))
    assert not store.journal["events"]


@pytest.mark.parametrize("overlap", [False, True])
def test_conflicting_simultaneous_boxes_cannot_be_merged(factory, overlap):
    raw = "2,1,100,100,20,40,1,2,1\n2,2,300,100,20,40,1,2,1\n"
    store = factory([merge_case((1, 2), (2, 2), overlap=overlap)], raw)
    with pytest.raises(ValueError, match="conflicting boxes"):
        store.save(0, "APPLY", "")
    assert store.rows[0]["decision"] == ""


def test_confirmed_duplicate_keeps_one_box_and_nonoverlapping_observations(factory):
    raw = ("1,1,100,100,20,40,1,2,0.8\n2,1,102,100,20,40,1,2,1\n"
           "2,2,103,100,20,40,1,2,0.7\n3,2,105,100,20,40,1,2,1\n")
    store = factory([merge_case((1, 2), (2, 2), overlap=True)], raw)
    store.save(0, "APPLY", "Same vehicle throughout")
    rows, stats = store.preview()
    assert set(rows) == {(1, 1), (2, 1), (3, 2)}
    assert rows[(3, 2)][1] == "1"
    assert stats["duplicate_boxes_removed"] == 1
    assert rows[(2, 1)][2:] == store.source_rows[(2, 1)][2:]


def test_ghost_removes_only_explicitly_accepted_track(factory):
    ghost = case("GHOST", (3, 3), (3, 3), dict(op="delete_track", tid=3))
    store = factory([ghost])
    store.save(0, "APPLY", "No vehicle")
    rows, stats = store.preview()
    assert (3, 3) not in rows
    assert len(rows) == 4
    assert stats["deleted_source_boxes"] == 1


def test_delete_track_and_merge_conflict_is_not_silently_applied(factory):
    ghost = case("GHOST", (2, 4), (2, 5), dict(op="delete_track", tid=2), rank=2)
    store = factory([merge_case(), ghost])
    store.save(0, "APPLY", "")
    with pytest.raises(ValueError, match="deleted track"):
        store.save(1, "APPLY", "")


def test_edit_requires_explicit_relevant_event(factory):
    store = factory([case("ORPHAN", (1, 2), (1, 2))])
    with pytest.raises(ValueError, match="no saved manual edit"):
        store.save(0, "EDIT", "")
    fields = list(store.source_rows[(2, 1)])
    fields[2] = "109"
    store.add_event(dict(op="upsert", fields=fields, case_rank=1))
    store.save(0, "EDIT", "Adjusted box")
    assert store.preview()[0][(2, 1)][2] == "109"
    assert MixedStore(store.bundle).rows[0]["decision"] == "EDIT"


def test_manual_add_box_retains_new_identity_and_can_be_undone(factory):
    store = factory()
    tid = store.new_track_id()
    assert tid == 4
    store.add_event(dict(op="upsert", fields=[6, tid, 200, 100, 25, 45, 1, 2, 1], case_rank=1))
    assert store.preview()[1]["added_boxes"] == 1
    assert store.new_track_id() == 5
    store.undo()
    assert store.preview()[1]["added_boxes"] == 0


def test_undo_resets_edit_verdict_if_its_last_edit_is_removed(factory):
    store = factory([case("ORPHAN", (1, 2), (1, 2))])
    fields = list(store.source_rows[(2, 1)])
    fields[2] = "109"
    store.add_event(dict(op="upsert", fields=fields, case_rank=1))
    store.save(0, "EDIT", "fixed")
    store.undo()
    assert store.rows[0]["decision"] == ""
    assert store.rows[0]["note"] == "fixed"
    assert not store.journal["events"]


def test_audit_requires_every_frame_displayed(factory):
    store = factory()
    store.record_seen([1, 6])
    with pytest.raises(ValueError, match="unseen"):
        store.save(0, "KEEP", "")
    complete_audit(store)
    assert store.tally["PENDING"] == 0


def test_pending_finalization_does_not_publish_any_output(factory, tmp_path):
    store = factory([merge_case()])
    complete_audit(store)
    with pytest.raises(ValueError, match="Complete every verdict"):
        finalize_mixed(store, "Nghĩa", out_dir=tmp_path / "result", evaluate=False)
    assert not (tmp_path / "result").exists()


def test_external_csv_change_is_never_overwritten(factory):
    store = factory([merge_case()])
    store.path.write_bytes(store.path.read_bytes() + b"\n")
    original = store.path.read_bytes()
    with pytest.raises(ValueError, match="outside"):
        store.save(0, "APPLY", "")
    assert store.path.read_bytes() == original


def test_external_edit_journal_change_blocks_csv_save(factory):
    store = factory([merge_case()])
    store.edit_path.write_bytes(store.edit_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="outside"):
        store.save(0, "KEEP", "")


def test_mixed_merge_needs_class_correction_before_acceptance(factory):
    raw = "2,1,100,100,20,40,1,2,1\n4,2,115,100,20,40,1,1,1\n"
    store = factory([merge_case()], raw)
    with pytest.raises(ValueError, match="multiple classes") as error:
        store.save(0, "APPLY", "")
    assert "source IDs 1, 2" in str(error.value)
    assert "car: MOT frame 4" in str(error.value)
    assert "motorcycle: MOT frame 2" in str(error.value)
    store.add_event(dict(op="reclassify", keys=[[4, 2]], class_id=2, case_rank=1))
    store.save(0, "APPLY", "")
    assert {p[7] for p in store.preview()[0].values()} == {"2"}


def test_same_class_at_merge_anchors_reports_hidden_conflicting_frame(factory):
    raw = ("1,1,100,100,20,40,1,1,1\n"
           "2,1,105,100,20,40,1,2,1\n"
           "4,2,115,100,20,40,1,2,1\n")
    store = factory([merge_case()], raw)
    before = store.path.read_bytes(), store.edit_path.read_bytes()
    assert store.source_rows[(2, 1)][7] == store.source_rows[(4, 2)][7] == "2"
    with pytest.raises(ValueError, match="multiple classes") as error:
        store.save(0, "APPLY", "")
    assert "car: MOT frame 1" in str(error.value)
    assert "motorcycle: MOT frame 2, 4" in str(error.value)
    assert (store.path.read_bytes(), store.edit_path.read_bytes()) == before
    store.add_event(dict(op="reclassify", keys=[[1, 1]], class_id=2, case_rank=1))
    store.save(0, "APPLY", "Human corrected hidden frame")
    assert {p[7] for p in store.preview()[0].values()} == {"2"}


def test_initial_classflip_can_be_inspected_and_corrected(factory):
    raw = "1,1,100,100,20,40,1,2,1\n2,1,105,100,20,40,1,1,1\n"
    store = factory([case("CLASSCHECK", (1, 1), (1, 2))], raw)
    assert len(store.preview()[0]) == 2
    store.add_event(dict(op="reclassify", keys=[[1, 1], [2, 1]], class_id=2, case_rank=1))
    store.save(0, "EDIT", "Motorcycle")
    complete_audit(store)
    assert {p[7] for p in store.preview(strict=True)[0].values()} == {"2"}


@pytest.mark.parametrize("fields", [[6, 4, 200, 100, -25, 45, 1, 2, 1],
                                   [7, 4, 200, 100, 25, 45, 1, 2, 1],
                                   [6, 4, float('nan'), 100, 25, 45, 1, 2, 1]])
def test_invalid_drawn_box_cannot_be_saved(factory, fields):
    store = factory()
    with pytest.raises(ValueError):
        store.add_event(dict(op="upsert", fields=fields))
    assert not store.journal["events"]


def test_finalization_snapshot_hashes_and_metrics_match_actual_labels(factory, tmp_path):
    store = factory([merge_case()])
    source_hash = sha256(store.source_path)
    store.save(0, "APPLY", "Cùng xe\nghi chú, hai dòng")
    complete_audit(store)
    frames = [[] for _ in range(6)]
    for (f, _), p in store.preview()[0].items():
        x, y, w, h = map(float, p[2:6])
        frames[f - 1].append([int(p[1]), int(p[7]) - 1, [x, y, x + w, y + h]])
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "tracks_baseline.json").write_text(json.dumps(dict(start_frame=0, frames=frames)), encoding="utf-8")
    out = finalize_mixed(store, "Nghĩa", tracks, tmp_path / "result")
    report = json.loads((out / "review.json").read_text(encoding="utf-8"))
    assert report["reviewer"] == "Nghĩa"
    assert report["variants"]["baseline"]["metrics"]["motorcycle"]["idf1"] == 1
    assert report["displayed_frames"] == 6
    assert report["reviewed_gt_sha256"] == sha256(out / "gt_reviewed.txt")
    assert report["snapshot_sha256"]["decisions.csv"] == sha256(store.path) == sha256(out / "decisions.csv")
    assert sha256(store.source_path) == source_hash
    assert "validation" in report["evaluation_role"]
    assert (out / "edits.json").exists()
    with pytest.raises(ValueError, match="previous review runs"):
        finalize_mixed(store, "Nghĩa", tracks, out)


def test_changed_source_frame_blocks_finalization(factory, tmp_path):
    store = factory()
    complete_audit(store)
    (Path(store.manifest["frames_dir"]) / "000003.jpg").write_bytes(b"changed image")
    with pytest.raises(ValueError, match="changed after preparation"):
        finalize_mixed(store, "Nghĩa", out_dir=tmp_path / "result", evaluate=False)


def test_incomplete_dump_never_publishes_a_partial_run(factory, tmp_path):
    store = factory()
    complete_audit(store)
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "tracks_baseline.json").write_text('{"frames": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover"):
        finalize_mixed(store, "Nghĩa", tracks, tmp_path / "result")
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".mixed-*"))


def test_prepare_is_resumable_without_resetting_human_decisions(factory):
    store = factory([merge_case()])
    store.save(0, "KEEP", "Review complete")
    source = store.bundle / "original"
    (source / "gt").mkdir(parents=True)
    (source / "gt/gt.txt").write_bytes(store.source_path.read_bytes())
    assert prepare(source, store.bundle, segment=(0, 6)) == store.bundle
    assert MixedStore(store.bundle).rows[0]["note"] == "Review complete"


def test_fresh_preparation_validates_images_and_does_not_guess_verdicts(tmp_path):
    source = tmp_path / "source"
    (source / "gt").mkdir(parents=True)
    (source / "img1").mkdir()
    gt = source / "gt/gt.txt"
    gt.write_text("1,1,100,100,20,40,1,2,1\n2,1,105,100,20,40,1,2,1\n", encoding="utf-8")
    original = gt.read_bytes()
    for frame in range(1, 7):
        Image.new("RGB", (800, 450), "white").save(source / f"img1/{frame:06d}.jpg")
    bundle = prepare(source, tmp_path / "bundle", segment=(0, 6), progress=None)
    store = MixedStore(bundle)
    store.check_inputs(images=True)
    assert store.tally["PENDING"] == len(store.rows)
    assert store.journal == {"schema_version": 1, "events": [], "seen_frames": []}
    assert gt.read_bytes() == original
    assert store.preview()[0] == store.source_rows
