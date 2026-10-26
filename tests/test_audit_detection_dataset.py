"""Check leakage detection and the audit's source-preservation contract."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from code.scripts.audit_detection_dataset import audit, near_pairs, parse_label, source_key


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def make_dataset(root):
    root.mkdir()
    (root / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: [car, Accident]\n")
    pixels = np.random.default_rng(9).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    for split, cid in (("train", 0), ("valid", 1), ("test", 0)):
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        # Same decoded image; PNG ancillary metadata/compression can differ.
        Image.fromarray(pixels).save(images / f"same.rf.{split}.png", compress_level=len(split))
        (labels / f"same.rf.{split}.txt").write_text(f"{cid} 0.5 0.5 0.5 0.5\n")


def test_audit_cross_split_duplicates_label_conflicts_and_no_input_writes(tmp_path):
    root = tmp_path / "dataset"
    make_dataset(root)
    before = snapshot(root)
    result = audit(root, tmp_path / "report", workers=1)
    assert snapshot(root) == before
    assert result["images"] == result["readable"] == 3
    assert result["exact_pixels"]["cross_split_groups"] == 1
    assert result["exact_pixels"]["redundant_copies"] == 2
    assert result["exact_pixels"]["different_labels_groups"] == 1
    assert result["source_name_hints"]["cross_split_groups"] == 1
    assert result["annotation_formats"] == {"box": 3}
    assert result["near_candidates"]["pairs"] == 0  # exact groups reported separately
    assert all(v["exists"] for v in result["yaml_paths"].values())


def test_missing_orphan_invalid_labels_and_corrupt_images_are_reported(tmp_path):
    root = tmp_path / "dataset"
    make_dataset(root)
    (root / "train/labels/same.rf.train.txt").unlink()
    (root / "train/labels/orphan.txt").write_text("0 0.5 0.5 0.1 0.1")
    (root / "test/images/same.rf.test.png").write_bytes(b"not an image")
    (root / "valid/labels/same.rf.valid.txt").write_text("9 0.5 0.5 0.1 0.1\n")
    before = snapshot(root)
    result = audit(root, tmp_path / "report", workers=1)
    assert result["readable"] == 2
    assert result["issues"]["missing_label"] == 1
    assert result["issues"]["invalid class"] == 1
    assert any(key.startswith("image_decode:") for key in result["issues"])
    assert result["orphan_labels"] == ["train/labels/orphan.txt"]
    assert snapshot(root) == before


@pytest.mark.parametrize("text,expected", [
    ("0 .5 .5 0 .2", "nonpositive box"),
    ("0 .5 .5 nan .2", "nonfinite value"),
    ("0 .1 .1 .1 .1 .1 .1", "degenerate polygon"),
    ("0 .1 .1 .2", "invalid coordinate count"),
    (".5 .5 .5 .2 .2", "invalid class"),
    ("0 .99 .5 .2 .2", "bounds_outside_image"),
])
def test_label_error_classification(text, expected):
    _, issues, _ = parse_label(text, 2, 640, 640)
    assert expected in [issue for _, issue in issues]


def test_polygon_geometry_and_order_independent_signature():
    poly = "1 .1 .2 .4 .2 .4 .6 .1 .6"
    box = "0 .5 .5 .01 .001"
    objects, issues, signature = parse_label(poly + "\n" + box, 2, 640, 640)
    assert not issues
    assert objects[0]["kind"] == "polygon"
    assert objects[0]["bounds"] == pytest.approx([.1, .2, .4, .6])
    assert objects[0]["area_px"] == pytest.approx(.12 * 640**2)
    assert objects[1]["tiny"]
    assert parse_label(box + "\n" + poly, 2, 640, 640)[2] == signature
    assert parse_label(poly + "\n" + poly, 2, 640, 640)[1] == [(2, "duplicate_annotation_row")]


def test_near_search_checks_dhash_and_omits_exact_pixel_copies():
    def row(path, pixel, phash, dhash):
        return dict(path=path, split=path, pixel_sha256=pixel, phash=f"{phash:016x}",
                    dhash=f"{dhash:016x}", source_hint="same")
    rows = [row("a", "one", 0, 0), row("b", "two", 3, 1),
            row("c", "three", 1, 2**32-1), row("copy", "one", 0, 0)]
    assert list(near_pairs(rows)) == [dict(a="a", b="b", split_a="a", split_b="b",
                                         phash_distance=2, dhash_distance=1, same_source_hint=True)]


def test_rejects_output_within_source_and_existing_output(tmp_path):
    root = tmp_path / "dataset"
    make_dataset(root)
    before = snapshot(root)
    with pytest.raises(ValueError, match="outside"):
        audit(root, root / "report")
    output = tmp_path / "report"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        audit(root, output)
    assert snapshot(root) == before
    assert source_key("frame_jpg.rf.abc.jpg") == "frame_jpg"
