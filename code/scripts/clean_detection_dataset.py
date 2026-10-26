"""Prepare, review and publish a local detection dataset without changing sources.

No perceptual-hash component is automatically deduplicated. Unknown provenance
or augmentation prevents publishing an evaluation split; training-only output
is an explicit alternative. All detailed records stay in gitignored data/.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from code.scripts.audit_detection_dataset import parse_label
from code.scripts.gt_review import atomic_write, sha256

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "data/annotations/accident_vehicle"
AUDIT = REPO / "data/processed/dataset_audits/accident_vehicle_20261010_v2"
BUNDLE = REPO / "data/annotations/accident_vehicle_clean/v1"
META = ("source_id", "clip_id", "event_id", "camera_id", "domain",
        "provenance_confirmed", "unaugmented_confirmed")
DOMAINS = ("unknown", "cctv", "dashcam", "web", "other")


def log(message):
    print(message, flush=True)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def payload(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def child(root, relative):
    path = (root / relative).resolve()
    if path == root.resolve() or root.resolve() not in path.parents:
        raise ValueError(f"Path escapes dataset directory: {relative}")
    return path


class Components:
    def __init__(self, values):
        self.parents = {value: value for value in values}

    def find(self, value):
        if value not in self.parents:
            raise ValueError(f"Unknown image in grouping: {value}")
        trail = []
        while self.parents[value] != value:
            trail.append(value)
            value = self.parents[value]
        for old in trail:
            self.parents[old] = value
        return value

    def join(self, a, b):
        a, b = sorted((self.find(a), self.find(b)))
        self.parents[b] = a

    def groups(self):
        result = defaultdict(list)
        for value in sorted(self.parents):
            result[self.find(value)].append(value)
        return list(result.values())


def boxes_from_objects(objects):
    boxes = []
    for obj in objects:
        x1, y1, x2, y2 = (float(x) for x in obj["bounds"])
        # Only normalize floating-point boundary noise already accepted by audit.
        x1, y1, x2, y2 = (min(1., max(0., x)) for x in (x1, y1, x2, y2))
        boxes.append([int(obj["class"]), (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
    return boxes


def label_text(boxes, nc, allow_duplicates=False):
    lines = []
    for box in boxes:
        if len(box) != 5:
            raise ValueError("Detection labels must have five values")
        lines.append(f"{box[0]} " + " ".join(f"{float(v):.12g}" for v in box[1:]))
    text = "\n".join(lines) + ("\n" if lines else "")
    objects, issues, _ = parse_label(text, nc, 640, 640)
    invalid = [issue for issue in issues if not (allow_duplicates and issue[1] == "duplicate_annotation_row")]
    if invalid or len(objects) != len(boxes):
        raise ValueError(f"Invalid detection boxes: {issues}")
    return text


def metadata_hints(path):
    name = Path(path).name.lower()
    match = re.match(r"(cam_\d+)_", name)
    camera = match.group(1) if match else ""
    if name.startswith("7_20_2017"):
        camera = "7_20_2017: source clip needs confirmation"
    return camera


def prepare(source=SOURCE, audit=AUDIT, bundle=BUNDLE, progress=log):
    source, audit, bundle = Path(source).resolve(), Path(audit).resolve(), Path(bundle).resolve()
    if bundle == source or source in bundle.parents:
        raise ValueError("Clean version must be outside the original source directory")
    if bundle.exists():
        store = CleanStore(bundle)
        store.verify_files(original=True)
        return bundle  # Resume without changing review decisions or corrected boxes.
    rows = json.loads((audit / "inventory.json").read_text(encoding="utf-8"))
    summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    if Path(summary["source"]).resolve() != source:
        raise ValueError("Audit belongs to a different source dataset")
    if sha256(source / "data.yaml") != summary["source_data_yaml_sha256"]:
        raise ValueError("Source YAML changed since audit")
    names = summary["names"]
    if not rows or len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate inventory")
    actual_images = {p.relative_to(source).as_posix() for p in source.rglob("*")
                     if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}}
    if actual_images != {r["path"] for r in rows}:
        raise ValueError("Source image inventory changed since audit")
    expected_labels = {(Path(r["path"]).parent.parent / "labels" / (Path(r["path"]).stem + ".txt")).as_posix() for r in rows}
    actual_labels = {p.relative_to(source).as_posix() for p in source.rglob("labels/*.txt")}
    if actual_labels != expected_labels:
        raise ValueError("Source label inventory changed or contains orphan labels")
    groups_data = json.loads((audit / "groups.json").read_text(encoding="utf-8"))
    pairs = list(csv.DictReader((audit / "near_pairs.csv").open(encoding="utf-8", newline="")))
    components = Components(r["path"] for r in rows)
    for pair in pairs:
        components.join(pair["a"], pair["b"])
    for kind in ("pixels", "source_hints"):
        for group in groups_data[kind]:
            for member in group["paths"][1:]:
                components.join(group["paths"][0], member)
    related = components.groups()
    index = {r["path"]: r for r in rows}
    cases = []
    clusters = [g for g in related if len(g) > 1]
    clusters.sort(key=lambda g: (len({index[p]["split"] for p in g}) == 1, -len(g), g[0]))
    for group in clusters:
        cases.append({"kind": "RELATED", "members": group,
                      "cross_split": len({index[p]["split"] for p in group}) > 1,
                      "question": "So sánh nguồn/biến thể. Giữ ảnh hữu ích; chỉ loại từng ảnh đã xác nhận dư. Cụm được giữ cùng split."})
    flags = defaultdict(list)
    for row in rows:
        converted = label_text(boxes_from_objects(row["objects"]), len(names), allow_duplicates=True)
        _, conversion_issues, _ = parse_label(converted, len(names), row["width"], row["height"])
        if conversion_issues:
            flags[row["path"]].append("polygon khác nhau chuyển thành box cùng class/tọa độ: cần kiểm tra và sửa nhãn dư")
        if not row["objects"]:
            flags[row["path"]].append("nhãn rỗng: background hay thiếu nhãn?")
        if any(o["tiny"] for o in row["objects"]):
            flags[row["path"]].append("box có cạnh dưới 4px")
        if row["blur_laplacian"] is not None and row["blur_laplacian"] <= summary["quality_percentiles"]["blur_laplacian"]["p05"]:
            flags[row["path"]].append("nhóm sắc nét thấp: cần xem ảnh")
        if row["gray_mean"] is not None and row["gray_mean"] <= summary["quality_percentiles"]["gray_mean"]["p01"]:
            flags[row["path"]].append("nhóm tối nhất: cần xem ảnh")
    # Batches amortize navigation while requiring every flagged image to be viewed.
    flagged = sorted(flags)
    for start in range(0, len(flagged), 12):
        cases.append({"kind": "QUALITY", "members": flagged[start:start+12],
                      "cross_split": False, "question": "Xem từng ảnh/box. Giữ ca khó hợp lệ; sửa nhãn sai hoặc cách ly ảnh chưa rõ."})
    rng = np.random.default_rng(42)
    sampled = set()
    for cid in range(len(names)):
        for split in sorted(summary["splits"]):
            candidates = [r["path"] for r in rows if r["split"] == split and any(o["class"] == cid for o in r["objects"])]
            if candidates:
                sampled.update(candidates[i] for i in rng.choice(len(candidates), min(6, len(candidates)), replace=False))
    sample = sorted(sampled - set(flagged))
    for start in range(0, len(sample), 12):
        cases.append({"kind": "LABEL_AUDIT", "members": sample[start:start+12],
                      "cross_split": False, "question": "Mẫu theo class/split: kiểm tra thiếu/sai nhãn và định nghĩa Accident. Không chứng nhận toàn bộ dataset."})
    for rank, case in enumerate(cases, 1):
        case["rank"] = rank
    bundle.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prepare-", dir=bundle.parent))
    try:
        (stage / "pool/images").mkdir(parents=True)
        (stage / "pool/labels").mkdir()
        records = []
        automatic = []
        for number, row in enumerate(rows, 1):
            original = child(source, row["path"])
            label_rel = (Path(row["path"]).parent.parent / "labels" / (original.stem + ".txt")).as_posix()
            label = child(source, label_rel)
            if row["issues"] or not row["pixel_sha256"]:
                raise ValueError(f"Unresolved audit error in {row['path']}: {row['issues']}")
            if sha256(original) != row["sha256"] or sha256(label) != row["label_sha256"]:
                raise ValueError(f"Source changed since audit: {row['path']}")
            objects, issues, _ = parse_label(label.read_text(encoding="utf-8-sig"), len(names), row["width"], row["height"])
            if issues or len(objects) != len(row["objects"]):
                raise ValueError(f"Source labels failed validation: {row['path']}")
            name = row["split"] + "__" + original.name
            image_dest, label_dest = stage / "pool/images" / name, stage / "pool/labels" / (Path(name).stem + ".txt")
            shutil.copyfile(original, image_dest)
            boxes = boxes_from_objects(objects)
            label_dest.write_text(label_text(boxes, len(names), allow_duplicates=True), encoding="utf-8")
            if sha256(image_dest) != row["sha256"]:
                raise ValueError("Image copy integrity check failed")
            records.append({"path": row["path"], "label_path": label_rel, "name": name,
                            "original_split": row["split"], "image_sha256": row["sha256"],
                            "source_label_sha256": row["label_sha256"], "pixel_sha256": row["pixel_sha256"],
                            "normalized_label_sha256": sha256(label_dest), "boxes": boxes,
                            "source_hint": row["source_hint"], "camera_hint": metadata_hints(row["path"]),
                            "flags": flags[row["path"]], "width": row["width"], "height": row["height"]})
            automatic.append({"path": row["path"], "action": "normalize_detection",
                              "polygon_count": sum(o["kind"] == "polygon" for o in objects),
                              "box_count": sum(o["kind"] == "box" for o in objects)})
            if progress and number % 500 == 0:
                progress(f"Copied and normalized {number}/{len(rows)} images")
        manifest = {"schema": 1, "source": str(source), "audit": str(audit), "created": stamp(),
                    "names": names, "source_yaml_sha256": sha256(source / "data.yaml"),
                    "rows": records, "conservative_groups": related,
                    "grouping_basis": "All near-hash links and repeated source-name hints; conservative constraints, not verified event identities.",
                    "annotation_scope": "Format conversion plus human review of flagged images and stratified samples; no global semantic certification."}
        (stage / "manifest.json").write_bytes(payload(manifest))
        (stage / "cases.json").write_bytes(payload(cases))
        (stage / "automatic_actions.json").write_bytes(payload(automatic))
        (stage / "provenance_template.csv").write_bytes(csv_bytes([
            {"source_path": r["path"], "original_split": r["original_split"],
             "source_hint": r["source_hint"], "camera_hint": r["camera_hint"],
             **{k: False if k.endswith("_confirmed") else "unknown" if k == "domain" else "" for k in META}}
            for r in records], ["source_path", "original_split", "source_hint", "camera_hint", *META]))
        state = {"schema": 1, "reviewer": "Nghĩa", "cases": {str(c["rank"]): {"reviewed": False, "seen": [], "note": ""} for c in cases},
                 "images": {}, "events": []}
        (stage / "review.json").write_bytes(payload(state))
        (stage / "receipt.json").write_bytes(payload({"manifest_sha256": sha256(stage / "manifest.json"),
                                                       "cases_sha256": sha256(stage / "cases.json")}))
        (stage / "preparation_report.json").write_bytes(payload({
            "status": "PREPARED_PENDING_HUMAN_REVIEW", "source_images": len(records),
            "normalized_images": len(records), "annotations": sum(len(r["boxes"]) for r in records),
            "images_removed": 0, "cases": dict(Counter(c["kind"] for c in cases)),
            "cross_split_related_cases": sum(c["kind"] == "RELATED" and c["cross_split"] for c in cases),
            "converted_box_collision_images": sum(any(f.startswith("polygon ") for f in r["flags"]) for r in records),
            "source_format_counts": summary["annotation_formats"], "class_mapping": dict(enumerate(names)),
            "evaluation_ready": False, "metadata_confirmed": 0,
            "conversion": "Polygon envelopes become boxes; existing boxes and class IDs preserved; only numerical boundary noise clamped.",
            "source_hashes": "Recorded per image/label and verified before exposing this prepared version."}))
        (stage / "preview_data.yaml").write_text(yaml.safe_dump({"path": str(bundle / "pool"),
            "train": "images", "nc": len(names), "names": names,
            "status": "DRAFT: missing reviewed provenance and frozen evaluation splits"}, sort_keys=False), encoding="utf-8")
        (stage / "README.md").write_text("# Prepared detection dataset — NOT a finalized evaluation dataset\n\n"
            "Original sources preserved. All images retained; polygons converted to boxes.\n"
            "Open review_accident_dataset.cmd to review candidates, edit boxes and record provenance.\n"
            "preview_data.yaml is an inventory preview, not a train/val/test configuration.\n"
            "Final releases are published under releases/ only after explicit human review.\n", encoding="utf-8")
        # Detect changes during copying before exposing a completed version.
        staged = CleanStore(stage)
        staged.verify_files(original=True)
        if bundle.exists():
            raise ValueError("Another process created this version; reload it")
        os.replace(stage, bundle)
    except Exception:
        # Only our verified sibling staging directory is removed.
        if stage.exists() and stage.resolve().parent == bundle.parent.resolve() and stage.resolve().name.startswith(".prepare-"):
            shutil.rmtree(stage)
        raise
    if progress:
        progress(f"Prepared {len(records)} images; {len(cases)} review cases; no images removed")
    return bundle


class CleanStore:
    def __init__(self, bundle=BUNDLE):
        self.bundle = Path(bundle).resolve()
        self.reload()

    def reload(self):
        receipt = json.loads((self.bundle / "receipt.json").read_text(encoding="utf-8"))
        for name in ("manifest", "cases"):
            if sha256(self.bundle / f"{name}.json") != receipt[f"{name}_sha256"]:
                raise ValueError(f"Immutable {name} changed; do not continue this review")
        self.manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        self.cases = json.loads((self.bundle / "cases.json").read_text(encoding="utf-8"))
        self.rows = {r["path"]: r for r in self.manifest["rows"]}
        self.names = self.manifest["names"]
        self.path = self.bundle / "review.json"
        raw = self.path.read_bytes()
        self.state, self.fingerprint = json.loads(raw), hashlib.sha256(raw).hexdigest()
        if set(self.state["cases"]) != {str(c["rank"]) for c in self.cases} or set(self.state["images"]) - set(self.rows):
            raise ValueError("Review keys do not match immutable manifest")
        for path in self.state["images"]:
            entry = self.image(path)
            if entry["status"] not in ("KEEP", "EXCLUDE", "QUARANTINE"):
                raise ValueError("Unknown image status")
            if entry["status"] != "KEEP" and not entry["reason"].strip():
                raise ValueError("Excluded/quarantined image requires a reason")
            self._validate_metadata(entry["metadata"])
            label_text(self.boxes(path), len(self.names), allow_duplicates=True)
        for case in self.cases:
            record = self.state["cases"][str(case["rank"])]
            if type(record["reviewed"]) is not bool or not isinstance(record["seen"], list) or not isinstance(record["note"], str):
                raise ValueError("Malformed case review fields")
            if set(record["seen"]) - set(case["members"]):
                raise ValueError("Viewed images do not match case")
            if record["reviewed"] and not self.review_requirement(case).issubset(set(record["seen"])):
                raise ValueError("Reviewed case is missing required viewed images")

    def image(self, path):
        if path not in self.rows:
            raise ValueError("Unknown image")
        default = {"status": "KEEP", "reason": "", "metadata": {k: False if k.endswith("_confirmed") else "unknown" if k == "domain" else "" for k in META},
                   "boxes": None}
        return self.state["images"].get(path, default)

    def boxes(self, path):
        override = self.image(path)["boxes"]
        return copy.deepcopy(self.rows[path]["boxes"] if override is None else override)

    @staticmethod
    def _validate_metadata(meta):
        if set(meta) != set(META) or meta["domain"] not in DOMAINS:
            raise ValueError("Invalid provenance fields/domain")
        if any(type(meta[k]) is not bool for k in META if k.endswith("_confirmed")):
            raise ValueError("Provenance/original-image confirmation must be boolean")
        if any(not isinstance(meta[k], str) for k in META if not k.endswith("_confirmed")):
            raise ValueError("Source identities must be strings")

    def _save(self, new_state, action):
        for name in ("manifest", "cases"):
            receipt = json.loads((self.bundle / "receipt.json").read_text(encoding="utf-8"))
            if sha256(self.bundle / f"{name}.json") != receipt[f"{name}_sha256"]:
                raise ValueError("Immutable review input changed")
        new_state["events"].append({"id": uuid.uuid4().hex, "at": stamp(), "reviewer": new_state["reviewer"], **action})
        data = payload(new_state)
        atomic_write(self.path, data, self.fingerprint)
        self.state, self.fingerprint = new_state, hashlib.sha256(data).hexdigest()

    def seen(self, rank, path, reviewer=None):
        case = self.cases[rank - 1]
        if path not in case["members"]:
            raise ValueError("Image is not part of this case")
        if path in self.state["cases"][str(rank)]["seen"]:
            return
        new = copy.deepcopy(self.state)
        if reviewer is not None:
            if not reviewer.strip():
                raise ValueError("Reviewer name required")
            new["reviewer"] = reviewer.strip()
        new["cases"][str(rank)]["seen"].append(path)
        self._save(new, {"action": "view", "rank": rank, "path": path})

    def review_requirement(self, case):
        # Approving a related group only accepts the conservative split constraint;
        # it does not attest to all member labels or call every member a duplicate.
        if case["kind"] != "RELATED":
            return set(case["members"])
        first = case["members"][0]
        different = next((p for p in case["members"] if self.rows[p]["original_split"] != self.rows[first]["original_split"]), None)
        return {first, different or case["members"][1]}

    def review_case(self, rank, reviewed=True, note="", reviewer=None):
        case = self.cases[rank - 1]
        record = self.state["cases"][str(rank)]
        if reviewed and not self.review_requirement(case).issubset(record["seen"]):
            raise ValueError("View both initial comparison images (related group), or every image (quality/label audit), before accepting")
        if reviewed and case["kind"] != "RELATED":
            for path in case["members"]:
                if self.image(path)["status"] == "KEEP":
                    label_text(self.boxes(path), len(self.names))
        new = copy.deepcopy(self.state)
        if reviewer is not None:
            if not reviewer.strip():
                raise ValueError("Reviewer name required")
            new["reviewer"] = reviewer.strip()
        new["cases"][str(rank)].update(reviewed=bool(reviewed), note=note)
        self._save(new, {"action": "review_case", "rank": rank, "reviewed": bool(reviewed), "note": note})

    def update_images(self, paths, reviewer="Nghĩa", **updates):
        paths = list(paths)
        if not paths or set(updates) - {"status", "reason", "metadata", "boxes"}:
            raise ValueError("Invalid image update")
        if not reviewer.strip():
            raise ValueError("Reviewer name required")
        new = copy.deepcopy(self.state)
        changed_paths = []
        for path in paths:
            entry = copy.deepcopy(self.image(path))
            for key, value in updates.items():
                entry[key] = copy.deepcopy(value)
            if entry["status"] not in ("KEEP", "EXCLUDE", "QUARANTINE") or (entry["status"] != "KEEP" and not entry["reason"].strip()):
                raise ValueError("Choose KEEP, EXCLUDE or QUARANTINE; removal needs a reason")
            self._validate_metadata(entry["metadata"])
            if entry["boxes"] is not None:
                label_text(entry["boxes"], len(self.names), allow_duplicates=True)
            if entry != self.image(path):
                new["images"][path] = entry
                changed_paths.append(path)
        if not changed_paths:
            return
        # A subsequent edit cannot silently retain an earlier case approval.
        for case in self.cases:
            if set(changed_paths).intersection(case["members"]):
                new["cases"][str(case["rank"])]["reviewed"] = False
        new["reviewer"] = reviewer.strip()
        self._save(new, {"action": "update_images", "paths": changed_paths, "updates": updates})

    def import_provenance(self, path, reviewer="Nghĩa"):
        """Apply an explicit human-supplied CSV mapping in a single transaction."""
        if not reviewer.strip():
            raise ValueError("Reviewer name required")
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"source_path", *META}.issubset(reader.fieldnames or []):
                raise ValueError("Provenance CSV is missing columns")
            rows = list(reader)
        new, changed, seen_paths = copy.deepcopy(self.state), {}, set()
        for row in rows:
            key = row["source_path"]
            if key not in self.rows or key in seen_paths or None in row or any(v is None for v in row.values()):
                raise ValueError("Unknown/duplicate image or malformed row in provenance CSV")
            seen_paths.add(key)
            meta = {k: row[k].strip() for k in META}
            for k in META:
                if k.endswith("_confirmed"):
                    if meta[k].lower() not in ("", "false", "true", "0", "1"):
                        raise ValueError("Confirmation values must be true/false/1/0")
                    meta[k] = meta[k].lower() in ("true", "1")
            meta["domain"] = meta["domain"].lower() or "unknown"
            self._validate_metadata(meta)
            if meta != self.image(key)["metadata"]:
                entry = copy.deepcopy(self.image(key)); entry["metadata"] = meta
                new["images"][key], changed[key] = entry, meta
        if not changed:
            return 0
        for case in self.cases:
            if set(changed).intersection(case["members"]):
                new["cases"][str(case["rank"])]["reviewed"] = False
        new["reviewer"] = reviewer.strip()
        self._save(new, {"action": "import_provenance", "changes": changed, "csv_sha256": sha256(Path(path))})
        return len(changed)

    @property
    def tally(self):
        return {"cases": len(self.cases), "pending": sum(not c["reviewed"] for c in self.state["cases"].values()),
                **dict(Counter(self.image(p)["status"] for p in self.rows)),
                "metadata_confirmed": sum(self.eligible(p) for p in self.rows)}

    def eligible(self, path):
        meta = self.image(path)["metadata"]
        return (meta["provenance_confirmed"] and meta["unaugmented_confirmed"] and
                meta["domain"] != "unknown" and bool(meta["source_id"] or meta["clip_id"] or meta["event_id"]))

    def verify_files(self, original=False):
        receipt = json.loads((self.bundle / "receipt.json").read_text(encoding="utf-8"))
        for key in ("manifest", "cases"):
            if sha256(self.bundle / f"{key}.json") != receipt[f"{key}_sha256"]:
                raise ValueError("Immutable review inputs changed")
        source = Path(self.manifest["source"])
        if original and sha256(source / "data.yaml") != self.manifest["source_yaml_sha256"]:
            raise ValueError("Original source YAML changed")
        if original:
            source_images = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()
                             and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}}
            source_labels = {p.relative_to(source).as_posix() for p in source.rglob("labels/*.txt")}
            if source_images != set(self.rows) or source_labels != {r["label_path"] for r in self.rows.values()}:
                raise ValueError("Original source inventory changed")
        for path, row in self.rows.items():
            for target, expected in ((self.bundle / "pool/images" / row["name"], row["image_sha256"]),
                                     (self.bundle / "pool/labels" / (Path(row["name"]).stem + ".txt"), row["normalized_label_sha256"])):
                if not target.is_file() or sha256(target) != expected:
                    raise ValueError(f"Prepared pool changed: {target}")
            if original and (sha256(child(source, path)) != row["image_sha256"] or
                             sha256(child(source, row["label_path"])) != row["source_label_sha256"]):
                raise ValueError(f"Original source changed: {path}")


def final_groups(store):
    comps = Components(store.rows)
    for group in store.manifest["conservative_groups"]:
        for member in group[1:]:
            comps.join(group[0], member)
    identities = {}
    for path, row in store.rows.items():
        # Filename camera hints are conservative constraints, NOT verified metadata.
        if row["camera_hint"]:
            key = ("camera_hint", row["camera_hint"])
            if key in identities:
                comps.join(identities[key], path)
            identities[key] = path
        meta = store.image(path)["metadata"]
        if meta["provenance_confirmed"]:
            for kind in ("source_id", "clip_id", "event_id", "camera_id"):
                if meta[kind]:
                    key = (kind, meta[kind])
                    if key in identities:
                        comps.join(identities[key], path)
                    identities[key] = path
    return comps.groups()


def assign_groups(store, training_only=False, seed=42):
    kept = {p for p in store.rows if store.image(p)["status"] == "KEEP"}
    if not kept:
        raise ValueError("No retained images")
    if not training_only:
        missing = [p for p in sorted(kept) if not store.eligible(p)]
        if missing:
            raise ValueError(f"{len(missing)} retained images need confirmed provenance/domain and original-before-augmentation status. First: {missing[0]}. Use training-only export explicitly if originals/provenance are unavailable.")
    groups = [sorted(set(g) & kept) for g in final_groups(store)]
    groups = [g for g in groups if g]
    groups.sort(key=lambda g: hashlib.sha256((str(seed) + "|" + "|".join(g)).encode()).hexdigest())
    groups.sort(key=len, reverse=True)  # Stable seeded tie ordering.
    if not training_only and len(groups) < 3:
        raise ValueError("At least three independent conservative groups are required for train/val/test")
    names = ("train",) if training_only else ("train", "val", "test")
    ratios = np.array([1.] if training_only else [.70, .15, .15])
    class_total = np.zeros(len(store.names))
    for p in kept:
        for box in store.boxes(p):
            class_total[int(box[0])] += 1
    target = np.concatenate(([len(kept)], class_total))
    used = np.zeros((len(names), len(target)))
    assignments = {}
    for number, group in enumerate(groups):
        features = np.zeros(len(target)); features[0] = len(group)
        for p in group:
            for box in store.boxes(p):
                features[int(box[0])+1] += 1
        # Guarantee nonempty splits, then minimize count/class imbalance at group level.
        remaining = len(groups) - number
        empty = [i for i in range(len(names)) if not used[i, 0]]
        options = empty if remaining == len(empty) else list(range(len(names)))
        scores = []
        for index in options:
            candidate = used.copy(); candidate[index] += features
            scores.append(np.square((candidate - ratios[:, None]*target) / np.maximum(target, 1)).sum())
        chosen = options[int(np.argmin(scores))]
        used[chosen] += features
        gid = hashlib.sha256("|".join(group).encode()).hexdigest()[:16]
        for p in group:
            assignments[p] = (names[chosen], gid)
    return assignments


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def export(store, reviewer="Nghĩa", training_only=False, seed=42, progress=log):
    if store.tally["pending"]:
        raise ValueError(f"{store.tally['pending']} review cases are still pending")
    if not reviewer.strip():
        raise ValueError("Reviewer name required")
    if sha256(store.path) != store.fingerprint:
        raise ValueError("Review changed outside the app; reload before exporting")
    store.verify_files(original=True)
    frozen = store.fingerprint
    assignments = assign_groups(store, training_only, seed)
    releases = store.bundle / "releases"
    releases.mkdir(exist_ok=True)
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    result = releases / run
    stage = Path(tempfile.mkdtemp(prefix=".export-", dir=releases))
    try:
        manifest, quarantine = [], []
        split_names = ("train",) if training_only else ("train", "val", "test")
        for split in split_names:
            (stage / split / "images").mkdir(parents=True)
            (stage / split / "labels").mkdir()
        counts = {split: {"images": 0, "objects": 0, "classes": {name: 0 for name in store.names}} for split in split_names}
        for path, row in store.rows.items():
            entry = store.image(path)
            record = {"source_path": path, "original_split": row["original_split"], "status": entry["status"],
                      "reason": entry["reason"], "image_sha256": row["image_sha256"],
                      "original_label_sha256": row["source_label_sha256"], **entry["metadata"]}
            if path not in assignments:
                quarantine.append(record)
                continue
            split, gid = assignments[path]
            image = stage / split / "images" / row["name"]
            label = stage / split / "labels" / (Path(row["name"]).stem + ".txt")
            shutil.copyfile(store.bundle / "pool/images" / row["name"], image)
            boxes = store.boxes(path)
            label.write_text(label_text(boxes, len(store.names)), encoding="utf-8")
            if sha256(image) != row["image_sha256"]:
                raise ValueError("Export image integrity failed")
            counts[split]["images"] += 1; counts[split]["objects"] += len(boxes)
            for box in boxes:
                counts[split]["classes"][store.names[int(box[0])]] += 1
            record.update(split=split, group_id=gid, image=image.relative_to(stage).as_posix(),
                          label=label.relative_to(stage).as_posix(), final_label_sha256=sha256(label),
                          boxes_edited=entry["boxes"] is not None)
            manifest.append(record)
        config = {"path": str(result), "train": "train/images", "nc": len(store.names), "names": store.names}
        if not training_only:
            config.update(val="val/images", test="test/images")
        (stage / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        fields = ["source_path", "original_split", "split", "group_id", "status", "reason", "image", "label",
                  "image_sha256", "original_label_sha256", "final_label_sha256", "boxes_edited", *META]
        (stage / "manifest.csv").write_bytes(csv_bytes(manifest, fields))
        (stage / "groups.csv").write_bytes(csv_bytes(manifest, ["source_path", "group_id", "split", *META]))
        (stage / "splits.csv").write_bytes(csv_bytes(manifest, ["source_path", "split", "group_id"]))
        (stage / "quarantine.csv").write_bytes(csv_bytes(quarantine, ["source_path", "status", "reason", *META]))
        (stage / "review_decisions.csv").write_bytes(csv_bytes([
            {"rank": c["rank"], "kind": c["kind"], **store.state["cases"][str(c["rank"])]} for c in store.cases],
            ["rank", "kind", "reviewed", "seen", "note"]))
        (stage / "review_snapshot.json").write_bytes(payload(store.state))
        (stage / "source_manifest.json").write_bytes(payload(store.manifest))
        evaluation_class_gaps = {split: [name for name, number in counter["classes"].items() if not number]
                                 for split, counter in counts.items()}
        report = {"created": stamp(), "reviewer": reviewer.strip(), "seed": seed,
                  "mode": "TRAINING_ONLY_NOT_FOR_EVALUATION" if training_only else "GROUPED_DETECTION",
                  "before_images": len(store.rows), "before_objects": sum(len(r["boxes"]) for r in store.rows.values()),
                  "after_images": len(manifest), "after_objects": sum(c["objects"] for c in counts.values()),
                  "excluded_or_quarantined": len(quarantine), "splits": counts, "class_gaps": evaluation_class_gaps,
                  "domain_counts": {split: dict(Counter(r["domain"] for r in manifest if r["split"] == split)) for split in split_names},
                  "cctv_images_without_confirmed_camera_id": sum(r["domain"] == "cctv" and not r["camera_id"] for r in manifest),
                  "review_sha256": frozen, "source_preserved": True,
                  "limitations": [store.manifest["annotation_scope"], store.manifest["grouping_basis"],
                                   "Original status and provenance are explicit human attestations, not inferred from image appearance.",
                                   "No severity labels or video event boundaries are provided.",
                                   "This tool does not check historical detector training overlap with other benchmark datasets."]}
        (stage / "report.json").write_bytes(payload(report))
        (stage / "README.md").write_text(f"# {report['mode']}\n\n"
            f"Retained images: {len(manifest)}/{len(store.rows)}. Reviewer: {reviewer}. Seed: {seed}.\n"
            "Original images/annotations are preserved outside this release.\n"
            "See report.json for class gaps, scope, provenance limitations and counts.\n"
            + ("This release has only a training pool; it supplies no validation/test protocol.\n"
               "Use it only with an independently supplied clean evaluation set.\n" if training_only else
               "Splits use conservative image/source/clip/event/camera constraints; inspect domain distributions and class support before reporting metrics.\n"), encoding="utf-8")
        if sha256(store.path) != frozen:
            raise ValueError("Review changed during export; retry after reloading")
        store.verify_files(original=True)
        os.replace(stage, result)
    except Exception:
        if stage.exists() and stage.resolve().parent == releases.resolve() and stage.resolve().name.startswith(".export-"):
            shutil.rmtree(stage)
        raise
    if progress:
        progress(f"Published {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--reviewer", default="Nghĩa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--import-provenance", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.source, args.audit, args.bundle)
    store = CleanStore(args.bundle)
    if args.import_provenance:
        print(f"Updated provenance for {store.import_provenance(args.import_provenance, args.reviewer)} images")
    if args.check:
        store.verify_files(original=True)
        print(json.dumps(store.tally, ensure_ascii=False))
    if args.export:
        export(store, args.reviewer, args.training_only, args.seed)


if __name__ == "__main__":
    main()
