"""Trial isolation, heuristic split constraints and source preservation."""
import csv
import json
from unittest.mock import patch

import pytest

from tests.test_clean_detection_dataset import dataset, snapshot
from code.scripts.train_detection_trial import MODE, plan, prepare_trial, select_smoke, verify_trial


def test_trial_preserves_review_sources_and_groups(dataset, tmp_path):
    source, _, bundle, store = dataset
    before = snapshot(source), snapshot(bundle)
    output = prepare_trial(store, tmp_path / "trial", smoke_train=3, smoke_val=2)
    assert (snapshot(source), snapshot(bundle)) == before
    report = json.loads((output / "report.json").read_text())
    assert report["mode"] == MODE
    assert report["review_status"]["pending"] > 0
    assert report["review_status"]["metadata_confirmed"] == 0
    assert verify_trial(output)["mode"] == MODE
    assert not (output / "test").exists()
    records = list(csv.DictReader((output / "manifest.csv").open()))
    groups, pixels = {}, {}
    for row in records:
        groups.setdefault(row["group_id"], set()).add(row["split"])
        pixels.setdefault(store.rows[row["source_path"]]["pixel_sha256"], set()).add(row["split"])
    assert all(len(values) == 1 for values in [*groups.values(), *pixels.values()])
    assert set(row["split"] for row in records) == {"train", "val"}
    assert plan(store) == plan(store)


def test_trial_skips_collisions_and_respects_user_exclusions(dataset, tmp_path):
    _, _, bundle, store = dataset
    paths = sorted(store.rows)
    box = [0, .5, .5, .2, .2]
    store.update_images([paths[0]], boxes=[box, box])
    store.update_images([paths[1]], status="QUARANTINE", reason="Synthetic questionable labels")
    before = snapshot(bundle)
    labels, assignment, skipped = plan(store)
    assert paths[0] not in assignment and paths[1] not in labels
    assert len(skipped) == 2
    prepare_trial(store, tmp_path / "trial")
    assert snapshot(bundle) == before


def test_trial_output_safety_and_atomic_failure(dataset, tmp_path):
    source, _, bundle, store = dataset
    for output in (source / "trial", bundle / "trial", source.parent):
        with pytest.raises(ValueError, match="separate"):
            prepare_trial(store, output)
    output = tmp_path / "trial"
    with patch("code.scripts.train_detection_trial.shutil.copy2", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            prepare_trial(store, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".trial-*"))


def test_trial_snapshot_change_is_rejected(dataset, tmp_path):
    _, _, _, store = dataset
    original = store.verify_files
    def changed(**kwargs):
        original(**kwargs)
        store.path.write_bytes(store.path.read_bytes() + b"\n")
    with patch.object(store, "verify_files", side_effect=changed):
        with pytest.raises(ValueError, match="Review changed"):
            prepare_trial(store, tmp_path / "trial")
    assert not (tmp_path / "trial").exists()


def test_smoke_selection_covers_classes_and_is_deterministic(dataset):
    _, _, _, store = dataset
    paths = sorted(store.rows)
    selected = select_smoke(store, paths, 3, 42)
    assert selected == select_smoke(store, paths, 3, 42)
    assert len(selected) == 3
    assert {int(b[0]) for path in selected for b in store.boxes(path)} == {0, 1}


def test_trial_tampering_or_extra_images_blocks_training(dataset, tmp_path):
    _, _, _, store = dataset
    output = prepare_trial(store, tmp_path / "trial")
    original = next((output / "train/images").glob("*"))
    extra = original.with_name("extra.png")
    extra.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="inventory"):
        verify_trial(output)
    extra.unlink()
    original.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file changed"):
        verify_trial(output)
