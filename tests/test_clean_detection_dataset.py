"""Source-preserving conversion, human gates, grouped splits and export safety."""
import copy
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from PIL import Image

from code.scripts.audit_detection_dataset import audit
from code.scripts.clean_detection_dataset import (
    CleanStore, assign_groups, child, export, final_groups, label_text, prepare,
)
from code.scripts.gt_review import sha256


def snapshot(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    (source / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 2\nnames: [Accident, car]\n")
    rng = np.random.default_rng(123)
    shared = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for split in ("train", "valid", "test"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir()
        for i in range(4):
            pixels = shared if i == 0 else rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(source / split / "images" / f"{split}_{i}.png")
            (source / split / "labels" / f"{split}_{i}.txt").write_text(
                "0 .1 .1 .4 .1 .4 .4 .1 .4\n1 .7 .7 .2 .2\n")
    report = tmp_path / "audit"
    audit(source, report, workers=1)
    bundle = tmp_path / "clean/v1"
    prepare(source, report, bundle, progress=None)
    return source, report, bundle, CleanStore(bundle)


def approve_all(store):
    for case in store.cases:
        for path in store.review_requirement(case):
            store.seen(case["rank"], path)
        store.review_case(case["rank"], note="Synthetic test only")


def confirm_sources(store):
    for path in store.rows:
        meta = dict(source_id="source:"+path, clip_id="", event_id="", camera_id="",
                    domain="cctv", provenance_confirmed=True, unaugmented_confirmed=True)
        store.update_images([path], reviewer="Synthetic reviewer", metadata=meta)


def test_conversion_retains_every_image_class_geometry_and_source(dataset):
    source, _, bundle, store = dataset
    assert len(store.rows) == 12
    assert store.tally["KEEP"] == 12  # even three identical pixel images retained
    assert store.tally["pending"] == len(store.cases)
    assert not (bundle / "data.yaml").exists()
    for path, row in store.rows.items():
        assert sha256(source / path) == sha256(bundle / "pool/images" / row["name"])
        values = (bundle / "pool/labels" / (Path(row["name"]).stem+".txt")).read_text().splitlines()
        assert all(len(line.split()) == 5 for line in values)
        assert store.boxes(path)[0] == pytest.approx([0, .25, .25, .3, .3])
        assert store.boxes(path)[1] == pytest.approx([1, .7, .7, .2, .2])
    store.verify_files(original=True)


def test_resume_never_resets_notes_exclusions_metadata_or_box_edits(dataset):
    source, report, bundle, store = dataset
    path = next(iter(store.rows))
    store.update_images([path], status="EXCLUDE", reason="Confirmed duplicate in synthetic fixture", boxes=[[1,.5,.5,.1,.1]])
    case = store.cases[0]
    store.review_case(case["rank"], False, "ghi chú có dấu")
    saved = (bundle / "review.json").read_bytes()
    prepare(source, report, bundle, progress=None)
    assert (bundle / "review.json").read_bytes() == saved
    assert CleanStore(bundle).boxes(path) == [[1,.5,.5,.1,.1]]


def test_case_view_gate_and_edits_invalidate_earlier_review(dataset):
    _, _, _, store = dataset
    case = store.cases[0]
    with pytest.raises(ValueError, match="View"):
        store.review_case(case["rank"])
    for path in store.review_requirement(case):
        store.seen(case["rank"], path)
    store.review_case(case["rank"])
    assert store.state["cases"][str(case["rank"])]["reviewed"]
    store.update_images([case["members"][0]], boxes=[[0,.5,.5,.2,.2]])
    assert not store.state["cases"][str(case["rank"])]["reviewed"]


def test_export_requires_review_and_provenance_but_training_only_is_explicit(dataset):
    source, _, _, store = dataset
    before = snapshot(source)
    with pytest.raises(ValueError, match="pending"):
        export(store, progress=None)
    approve_all(store)
    with pytest.raises(ValueError, match="confirmed provenance"):
        export(store, progress=None)
    result = export(store, training_only=True, progress=None)
    config = yaml.safe_load((result / "data.yaml").read_text())
    assert "val" not in config and "test" not in config
    assert json.loads((result / "report.json").read_text())["mode"] == "TRAINING_ONLY_NOT_FOR_EVALUATION"
    assert len(list((result / "train/images").glob("*.png"))) == 12
    assert snapshot(source) == before


def test_grouped_split_is_reproducible_keeps_related_sources_together(dataset):
    source, _, _, store = dataset
    confirm_sources(store); approve_all(store)
    a = assign_groups(store); b = assign_groups(store)
    assert a == b and {split for split, _ in a.values()} == {"train","val","test"}
    identical = [path for path in store.rows if Path(path).stem.endswith("_0")]
    assert len({a[p][0] for p in identical}) == 1
    before = snapshot(source)
    result = export(store, reviewer="Nghĩa", progress=None)
    rows = list(csv.DictReader((result / "manifest.csv").open(encoding="utf-8")))
    assert len(rows) == 12
    for row in rows:
        assert sha256(result / row["image"]) == row["image_sha256"]
        assert sha256(result / row["label"]) == row["final_label_sha256"]
    assert yaml.safe_load((result/"data.yaml").read_text())["path"] == str(result)
    assert snapshot(source) == before


def test_camera_and_event_constraints_join_different_image_groups(dataset):
    _, _, _, store = dataset
    paths = [p for p in store.rows if not Path(p).stem.endswith("_0")][:2]
    assert not any(set(paths).issubset(g) for g in final_groups(store))
    meta = dict(source_id="", clip_id="", event_id="same_event", camera_id="",
                domain="web", provenance_confirmed=True, unaugmented_confirmed=True)
    store.update_images(paths, metadata=meta)
    assert any(set(paths).issubset(g) for g in final_groups(store))
    meta.update(event_id="", camera_id="same_camera")
    store.update_images(paths, metadata=meta)
    assert any(set(paths).issubset(g) for g in final_groups(store))


def test_excluding_one_member_does_not_remove_a_whole_component(dataset):
    _, _, _, store = dataset
    identical = [path for path in store.rows if Path(path).stem.endswith("_0")]
    store.update_images([identical[0]], status="EXCLUDE", reason="Reviewed redundant image")
    approve_all(store)
    result = export(store, training_only=True, progress=None)
    report = json.loads((result / "report.json").read_text())
    assert report["after_images"] == 11
    assert report["excluded_or_quarantined"] == 1
    assert all(store.image(p)["status"] == "KEEP" for p in identical[1:])


def test_bad_boxes_metadata_and_empty_exclusion_reason_do_not_save(dataset):
    _, _, bundle, store = dataset
    path = next(iter(store.rows)); before = (bundle / "review.json").read_bytes()
    for updates in ({"boxes": [[0,.99,.5,.2,.2]]}, {"boxes": [[9,.5,.5,.2,.2]]},
                    {"status": "EXCLUDE", "reason": ""}, {"metadata": {"domain": "unknown"}}):
        with pytest.raises(ValueError):
            store.update_images([path], **updates)
        assert (bundle / "review.json").read_bytes() == before


def test_external_review_change_is_not_overwritten(dataset):
    _, _, bundle, store = dataset
    path = next(iter(store.rows)); review = bundle / "review.json"
    review.write_bytes(review.read_bytes() + b"\n")
    external = review.read_bytes()
    with pytest.raises(ValueError, match="changed outside"):
        store.update_images([path], status="QUARANTINE", reason="Pending annotation review")
    assert review.read_bytes() == external


def test_original_or_pool_tampering_blocks_export(dataset):
    source, _, bundle, store = dataset
    approve_all(store)
    path = next(iter(store.rows)); row = store.rows[path]
    original = source / path; original_bytes = original.read_bytes()
    original.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Original source changed"):
        export(store, training_only=True, progress=None)
    original.write_bytes(original_bytes)
    prepared_label = bundle / "pool/labels" / (Path(row["name"]).stem+".txt")
    prepared_label.write_text("0 .5 .5 .2 .2\n")
    with pytest.raises(ValueError, match="Prepared pool changed"):
        export(store, training_only=True, progress=None)


def test_failed_export_has_no_half_published_release(dataset):
    source, _, bundle, store = dataset
    approve_all(store); before = snapshot(source)
    with patch("code.scripts.clean_detection_dataset.shutil.copyfile", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            export(store, training_only=True, progress=None)
    assert list((bundle / "releases").iterdir()) == []
    assert snapshot(source) == before


def test_rejects_source_output_overlap_and_path_escape(dataset):
    source, report, _, _ = dataset
    with pytest.raises(ValueError, match="outside"):
        prepare(source, report, source / "clean", progress=None)
    with pytest.raises(ValueError, match="escapes"):
        child(source, "../other.txt")


def test_immutable_manifest_change_is_detected_on_resume(dataset):
    _, _, bundle, _ = dataset
    path = bundle / "manifest.json"
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="Immutable manifest"):
        CleanStore(bundle)


def test_distinct_polygons_with_same_box_are_preserved_and_require_review(dataset, tmp_path):
    source, _, _, _ = dataset
    original = source / "train/labels/train_1.txt"
    original.write_text(original.read_text()+"0 .1 .1 .4 .1 .4 .4\n")
    report, bundle = tmp_path / "audit2", tmp_path / "clean/v2"
    audit(source, report, workers=1)
    prepare(source, report, bundle, progress=None)
    store = CleanStore(bundle)
    path = "train/images/train_1.png"
    assert len(store.boxes(path)) == 3
    assert any("polygon khác" in flag for flag in store.rows[path]["flags"])
    case = next(c for c in store.cases if c["kind"] == "QUALITY" and path in c["members"])
    for member in case["members"]:
        store.seen(case["rank"], member)
    with pytest.raises(ValueError, match="duplicate_annotation_row"):
        store.review_case(case["rank"])
    # Human correction removes exactly the selected redundant detection annotation.
    store.update_images([path], boxes=store.boxes(path)[:2])
    store.review_case(case["rank"])
    assert len(original.read_text().splitlines()) == 3


def test_provenance_csv_import_is_atomic_and_resets_affected_reviews(dataset):
    _, _, bundle, store = dataset
    approve_all(store)
    template = bundle / "provenance_template.csv"
    rows = list(csv.DictReader(template.open(encoding="utf-8")))
    for row in rows:
        row.update(source_id="confirmed:"+row["source_path"], domain="cctv",
                   provenance_confirmed="true", unaugmented_confirmed="1")
    fields = list(rows[0])
    csv_path = bundle / "filled.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)
    assert store.import_provenance(csv_path) == 12
    assert store.tally["metadata_confirmed"] == 12
    assert store.tally["pending"] == len(store.cases)
    assert store.import_provenance(csv_path) == 0
    before = store.path.read_bytes()
    rows[0]["source_id"] = "this change must not be saved"
    rows[-1]["unaugmented_confirmed"] = "guess"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="Confirmation"):
        store.import_provenance(csv_path)
    assert store.path.read_bytes() == before
