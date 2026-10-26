"""Guard human decisions and GT integrity through the complete review workflow."""
import csv
import json
from types import SimpleNamespace

import pytest

from code.scripts.gt_review import DecisionStore, check_inputs, finalize


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sample(tmp_path):
    decisions = tmp_path / "decisions.csv"
    row = dict(rank="1", sheet="001_id1_f0-1.png", gt_id="1", cut_at_frame="2",
               decision="", note="", extra="preserve this metadata")
    write_csv(decisions, list(row), [row])
    candidates = tmp_path / "candidates.csv"
    candidate = dict(rank="1", gt_id_INTERNAL_not_cvat="1", cvat_before="0", cvat_after="1",
                     mot_before="1", mot_after="2", abs_before="0", abs_after="1")
    write_csv(candidates, list(candidate), [candidate])
    (tmp_path / row["sheet"]).write_bytes(b"fixture-sheet")
    gt = tmp_path / "gt.txt"
    gt.write_text("1,1,10,20,30,40,1,2,1\n2,1,12,22,30,40,1,2,0.8\n", encoding="utf-8")
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    baseline = {"start_frame": 0, "frames": [[[1, 1, [10, 20, 40, 60]]], [[1, 1, [12, 22, 42, 62]]]]}
    (tracks / "tracks_baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    split = {"start_frame": 0, "frames": [[[1, 1, [10, 20, 40, 60]]], [[2, 1, [12, 22, 42, 62]]]]}
    (tracks / "tracks_split.json").write_text(json.dumps(split), encoding="utf-8")
    return SimpleNamespace(root=tmp_path, decisions=decisions, candidates=candidates,
                           gt=gt, tracks=tracks, row=row, candidate=candidate)


def finish(sample, **kwargs):
    return finalize(DecisionStore(sample.decisions), sample.candidates, sample.gt,
                    "Human reviewer", sample.tracks, (0, 2), sample.root / "result", **kwargs)


def test_resume_preserve_metadata_and_multiline_vietnamese_notes(sample):
    store = DecisionStore(sample.decisions)
    assert store.next_pending() == 0
    store.save(0, "cut", "Khác xe\nkiểm tra lại, góc trái")
    reopened = DecisionStore(sample.decisions)
    assert reopened.next_pending() is None
    assert reopened.rows[0]["extra"] == "preserve this metadata"
    assert reopened.rows[0]["note"] == "Khác xe\nkiểm tra lại, góc trái"
    assert reopened.tally == {"CUT": 1, "KEEP": 0, "PENDING": 0}
    reopened.save(0, "", "chưa rõ")
    assert reopened.tally["PENDING"] == 1


def test_external_edit_is_never_overwritten(sample):
    store = DecisionStore(sample.decisions)
    sample.decisions.write_bytes(sample.decisions.read_bytes() + b"\n")
    external = sample.decisions.read_bytes()
    with pytest.raises(ValueError, match="outside"):
        store.save(0, "KEEP", "")
    assert sample.decisions.read_bytes() == external
    assert store.rows[0]["decision"] == ""


def test_failed_atomic_save_retains_disk_and_memory(sample, monkeypatch):
    store = DecisionStore(sample.decisions)
    original = sample.decisions.read_bytes()

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr("code.scripts.gt_review.os.replace", fail)
    with pytest.raises(OSError):
        store.save(0, "CUT", "test")
    assert sample.decisions.read_bytes() == original
    assert store.rows[0]["decision"] == ""
    assert not list(sample.root.glob(".decisions-*.tmp"))


@pytest.mark.parametrize("change", [{"decision": "AUTO"}, {"sheet": "../outside.png"},
                                   {"sheet": "..\\outside.png"}, {"rank": "0"}])
def test_reject_invalid_decision_or_sheet(sample, change):
    row = dict(sample.row, **change)
    write_csv(sample.decisions, list(row), [row])
    with pytest.raises(ValueError):
        DecisionStore(sample.decisions)


def test_complete_input_check_is_read_only(sample):
    before = sample.decisions.read_bytes(), sample.gt.read_bytes()
    checked = check_inputs(DecisionStore(sample.decisions), sample.candidates, sample.gt, (0, 2))
    assert checked == {"candidates": 1, "boxes": 2, "identities": 1, "CUT": 0, "KEEP": 0, "PENDING": 1}
    assert before == (sample.decisions.read_bytes(), sample.gt.read_bytes())


def test_wrong_candidate_mapping_is_rejected(sample):
    row = dict(sample.row, gt_id="17")
    write_csv(sample.decisions, list(row), [row])
    with pytest.raises(ValueError, match="metadata"):
        check_inputs(DecisionStore(sample.decisions), sample.candidates, sample.gt, (0, 2))


def test_wrong_absolute_segment_is_rejected(sample):
    with pytest.raises(ValueError, match="metadata"):
        check_inputs(DecisionStore(sample.decisions), sample.candidates, sample.gt, (10, 12))


def test_missing_sheet_is_rejected(sample):
    (sample.root / sample.row["sheet"]).unlink()
    with pytest.raises(ValueError, match="missing"):
        check_inputs(DecisionStore(sample.decisions), sample.candidates, sample.gt, (0, 2))


def test_missing_join_in_gt_is_rejected(sample):
    sample.gt.write_text("1,1,10,20,30,40,1,2,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="existing join"):
        check_inputs(DecisionStore(sample.decisions), sample.candidates, sample.gt, (0, 2))


def test_pending_review_does_not_publish_gt(sample):
    with pytest.raises(ValueError, match="pending"):
        finish(sample)
    assert not (sample.root / "result").exists()
    assert not list(sample.root.glob(".review-*"))


def test_anonymous_review_is_not_finalized(sample):
    store = DecisionStore(sample.decisions)
    store.save(0, "KEEP", "")
    with pytest.raises(ValueError, match="name"):
        finalize(store, sample.candidates, sample.gt, "", segment=(0, 2))


def test_finalize_preserves_original_and_scores_exact_reviewed_gt(sample):
    original = sample.gt.read_bytes()
    DecisionStore(sample.decisions).save(0, "CUT", "Different vehicles")
    out = finish(sample)
    assert sample.gt.read_bytes() == original
    reviewed = (out / "gt_reviewed.txt").read_text(encoding="utf-8").splitlines()
    assert reviewed == ["1,1,10,20,30,40,1,2,1", "2,2,12,22,30,40,1,2,0.8"]
    report = json.loads((out / "review.json").read_text(encoding="utf-8"))
    assert report["reviewer"] == "Human reviewer"
    assert report["variants"]["split"]["metrics"]["motorcycle"]["idf1"] == 1
    assert report["variants"]["baseline"]["metrics"]["motorcycle"]["idf1"] == 0.5
    assert (out / "decisions.csv").read_bytes() == sample.decisions.read_bytes()
    assert len(report["reviewed_gt_sha256"]) == 64
    assert "split | motorcycle" in (out / "summary.md").read_text(encoding="utf-8")


def test_refuse_to_overwrite_previous_run(sample):
    DecisionStore(sample.decisions).save(0, "KEEP", "")
    out = finish(sample)
    original = (out / "review.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        finish(sample)
    assert (out / "review.json").read_bytes() == original


def test_incomplete_dump_does_not_publish_partial_results(sample):
    DecisionStore(sample.decisions).save(0, "CUT", "")
    (sample.tracks / "tracks_baseline.json").write_text('{"frames": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="cover"):
        finish(sample)
    assert not (sample.root / "result").exists()
    assert not list(sample.root.glob(".review-*"))


def test_explicit_gt_only_finalization_records_skipped_evaluation(sample):
    DecisionStore(sample.decisions).save(0, "KEEP", "")
    out = finish(sample, evaluate=False)
    report = json.loads((out / "review.json").read_text(encoding="utf-8"))
    assert report["evaluation_performed"] is False
    assert report["variants"] == {}
    assert "explicitly skipped" in (out / "summary.md").read_text(encoding="utf-8")


def test_corrupted_box_never_publishes_reviewed_gt(sample, monkeypatch):
    from code.scripts.gt_review import apply_cuts

    DecisionStore(sample.decisions).save(0, "CUT", "")

    def corrupt(gt, cuts, out):
        stats = apply_cuts(gt, cuts, out)
        out.write_text(out.read_text(encoding="utf-8").replace(",10,20,", ",999,20,"), encoding="utf-8")
        return stats

    monkeypatch.setattr("code.scripts.gt_review.apply_cuts", corrupt)
    with pytest.raises(ValueError, match="integrity"):
        finish(sample)
    assert not (sample.root / "result").exists()
