"""Prepare and persist a local human review of mixed prelabels.

Proposals are never verdicts. Only explicit human decisions/edits change labels.
Splits precede merges; anchors name source IDs and frames, preserving lineage.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from code.scripts.gt_review import REPO_ROOT, TRACKS_DIR, atomic_write, score_variants, sha256
from code.scripts.gt_review_queue import (
    CLASS_NAMES, _iou, build_tracks, find_classflips, find_dups, find_ghosts,
    find_jumps, find_merges, find_orphans, load_mot,
)
from code.scripts.render_split_candidates import find_candidates

SOURCE = REPO_ROOT / "data/annotations/prelabels/mixed"
BUNDLE = REPO_ROOT / "experiments/gt_qa/mixed_review"
RESULTS = REPO_ROOT / "experiments/gt_qa/mixed_reviewed"
DECISION_FIELDS = ["rank", "sheet", "kind", "ids", "frame_before", "frame_after", "decision", "note"]
VALID = {"", "APPLY", "KEEP", "EDIT"}


def now():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def safe_cleanup(staging: Path, parent: Path):
    if staging.resolve().parent != parent.resolve() or not staging.name.startswith(".mixed-"):
        raise ValueError("Refusing to clean a temporary directory outside the output parent.")
    shutil.rmtree(staging)


def read_rows(path: Path, n_frames: int) -> dict:
    rows = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.strip().split(",")
        validate_fields(fields, n_frames)
        key = (int(fields[0]), int(fields[1]))
        if key in rows:
            raise ValueError(f"Duplicate source ID/frame on GT line {number}.")
        rows[key] = fields
    if not rows:
        raise ValueError("Source GT is empty.")
    return rows


def validate_fields(fields, n_frames):
    if len(fields) < 8:
        raise ValueError("A MOT box needs at least eight fields.")
    frame, tid = int(fields[0]), int(fields[1])
    values = [float(x) for x in fields[2:6]]
    if not 1 <= frame <= n_frames or tid < 1 or not all(math.isfinite(x) for x in values):
        raise ValueError("Box frame, ID or coordinates are invalid.")
    if min(values[2:]) <= 0 or int(fields[7]) not in range(1, 6):
        raise ValueError("Box dimensions must be positive and the class must be known.")
    if not math.isfinite(float(fields[6])):
        raise ValueError("Invalid MOT ignore/confidence field.")
    if len(fields) > 8 and not math.isfinite(float(fields[8])):
        raise ValueError("Invalid visibility field.")


def make_cases(frames, n_frames, width, height, audit_block=20):
    tracks = build_tracks(frames)
    merges = find_merges(tracks, n_frames, width, height)
    jumps, _ = find_jumps(tracks, width, height)
    orphans, _ = find_orphans(tracks, merges, n_frames, width, height)
    cases = []

    def add(kind, a, b, proposal, question, low=None, high=None):
        cases.append(dict(kind=kind, before=list(a), after=list(b), proposal=proposal,
                          ids=sorted({a[0], b[0]} - {0}), question=question,
                          window=[max(1, min(a[1], b[1]) - 15) if low is None else low,
                                  min(n_frames, max(a[1], b[1]) + 15) if high is None else high]))

    # Review all suspicious gaps, not just the single worst gap per source ID.
    splits = find_candidates(tracks)
    split_keys = set()
    for candidate in splits:
        tid, before, after = candidate["tid"], candidate["f_before"], candidate["f_after"]
        add("SPLIT", (tid, before), (tid, after), {"op": "split", "tid": tid, "frame": after},
            "Có phải hai xe khác nhau? A: tách ID từ ảnh sau. K: cùng xe, giữ ID.")
        split_keys.add((tid, after))
    for finding in sorted(merges, key=lambda f: -f.conf):
        a, b = finding.ids
        before, after = tracks[a].last, tracks[b].first
        add("MERGE", (a, before), (b, after),
            {"op": "merge", "anchors": [[a, before], [b, after]], "allow_overlap": False},
            "Hai ID này có phải cùng một xe? A: nối ID. K: hai xe, giữ riêng.")
    for finding in jumps:
        tid = finding.ids[0]
        if (tid, finding.frame) in split_keys:
            continue
        previous = max(f for f in tracks[tid].frames if f < finding.frame)
        add("JUMP", (tid, previous), (tid, finding.frame),
            {"op": "split", "tid": tid, "frame": finding.frame},
            "Box có nhảy sang xe khác? A: tách ID. Nếu chỉ lệch box, V để sửa rồi E.")
    for finding in sorted(find_dups(frames, tracks), key=lambda f: -f.conf):
        a, b = finding.ids
        add("DUP", (a, finding.frame), (b, finding.frame),
            {"op": "merge", "anchors": [[a, finding.frame], [b, finding.frame]], "allow_overlap": True},
            f"Hai box có cùng một xe suốt track? A: nối ID, frame trùng giữ box ID {a}. K: hai xe.",
            max(1, min(tracks[a].first, tracks[b].first) - 5),
            min(n_frames, max(tracks[a].last, tracks[b].last) + 5))
    for finding in find_ghosts(tracks, n_frames, width, height):
        tid = finding.ids[0]
        add("GHOST", (tid, tracks[tid].first), (tid, tracks[tid].last),
            {"op": "delete_track", "tid": tid},
            "Có xe thật trong box không? A: không có xe, xóa cả track. K: có xe thật.")
    check_ids = {t.tid for t in tracks.values() if t.cls >= 2}
    check_ids.update(f.ids[0] for f in find_classflips(tracks))
    for tid in sorted(check_ids):
        t = tracks[tid]
        add("CLASSCHECK", (tid, t.first), (tid, t.last), None,
            f"Kiểm tra nhãn {CLASS_NAMES[t.cls]}. K: đúng nhãn. V: đổi class hoặc sửa box; sau đó E.")
    for finding in orphans:
        tid = finding.ids[0]
        add("ORPHAN", (tid, finding.frame), (tid, finding.frame), None,
            "Xe có bị thiếu box trước/sau điểm này? K: không cần sửa. V: thêm/chỉnh box rồi E.")
    for low in range(1, n_frames + 1, audit_block):
        high = min(n_frames, low + audit_block - 1)
        add("AUDIT", (0, low), (0, high), None,
            "V: xem toàn đoạn, kiểm tra xe thiếu, box và class. A/K: đã xem, không cần sửa. E: đã sửa.",
            low, high)
    for rank, case in enumerate(cases, 1):
        case.update(rank=rank, sheet=f"{rank:03d}_{case['kind']}.png")
    return cases


def prepare(source: Path = SOURCE, bundle: Path = BUNDLE, segment=(2500, 3100), progress=print, frames_dir=None):
    source, bundle = Path(source), Path(bundle)
    if bundle.exists():
        store = MixedStore(bundle)
        store.check_inputs(images=True)
        if store.manifest["source"]["sha256"] != sha256(source / "gt/gt.txt"):
            raise ValueError("Existing review uses different prelabels. Choose a new bundle directory.")
        return bundle  # Resuming never clears any human decision, edit or viewed frame.
    n_frames = segment[1] - segment[0]
    if segment[0] < 0 or n_frames < 1:
        raise ValueError("Invalid absolute segment.")
    gt = source / "gt/gt.txt"
    source_hash = sha256(gt)
    read_rows(gt, n_frames)
    images = Path(frames_dir) if frames_dir is not None else source / "img1"
    image_hashes = {}
    size = None
    for frame in range(1, n_frames + 1):
        path = images / f"{frame:06d}.jpg"
        image_hashes[path.name] = sha256(path)
        with Image.open(path) as picture:
            picture.verify()
            if size is None:
                size = picture.size
            if picture.size != size:
                raise ValueError("All review frames must have the same size.")
    frames = load_mot(gt)
    cases = make_cases(frames, n_frames, *size)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mixed-", dir=bundle.parent))
    try:
        shutil.copyfile(gt, staging / "source_gt.txt")
        manifest = dict(schema_version=1, created_utc=now(), segment=list(segment),
                        n_frames=n_frames, image_size=list(size),
                        source={"path": str(gt.resolve()), "sha256": source_hash,
                                "seed": "baseline tracker prelabels; human decisions still required"},
                        frames_dir=str(images.resolve()), images_sha256=image_hashes, cases=cases)
        (staging / "manifest.json").write_bytes(json_bytes(manifest))
        from code.scripts.gt_mixed_sheets import render_sheet

        for index, case in enumerate(cases, 1):
            render_sheet(case, frames, images, staging / case["sheet"])
            if progress and (index % 20 == 0 or index == len(cases)):
                progress(f"Rendered {index}/{len(cases)} review sheets")
        with (staging / "decisions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DECISION_FIELDS)
            writer.writeheader()
            for case in cases:
                writer.writerow(dict(rank=case["rank"], sheet=case["sheet"], kind=case["kind"],
                                     ids=" ".join(map(str, case["ids"])), frame_before=case["before"][1],
                                     frame_after=case["after"][1], decision="", note=""))
        (staging / "edits.json").write_bytes(json_bytes({"schema_version": 1, "events": [], "seen_frames": []}))
        counts = dict(Counter(c["kind"] for c in cases))
        lines = ["# Mixed review", "", "Every case is a question, not an automatic annotation.", "",
                 f"Frames: {n_frames}; absolute segment: {list(segment)}; cases: {counts}", "",
                 "Open review_gt_mixed.cmd. A: accept proposal, K: keep, U: uncertain, V: video/editor, E: edited.", "",
                 "| Rank | Type | MOT before → after | Source IDs | Sheet |", "|---|---|---|---|---|"]
        for c in cases:
            lines.append(f"| {c['rank']} | {c['kind']} | {c['before'][1]} → {c['after'][1]} | {c['ids']} | [{c['sheet']}]({c['sheet']}) |")
        (staging / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if sha256(gt) != source_hash or sha256(staging / "source_gt.txt") != source_hash:
            raise ValueError("Prelabels changed while review sheets were prepared.")
        for name, fingerprint in image_hashes.items():
            if sha256(images / name) != fingerprint:
                raise ValueError("A source image changed during preparation.")
        staging.rename(bundle)
    finally:
        if staging.exists():
            safe_cleanup(staging, bundle.parent)
    return bundle


class MixedStore:
    def __init__(self, bundle: Path):
        self.bundle = Path(bundle)
        self.path = self.bundle / "decisions.csv"
        self.manifest_path = self.bundle / "manifest.json"
        self.edit_path = self.bundle / "edits.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest_hash = sha256(self.manifest_path)
        self.source_path = self.bundle / "source_gt.txt"
        self.source_rows = read_rows(self.source_path, self.manifest["n_frames"])
        self.reload()

    def reload(self):
        raw = self.path.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        if reader.fieldnames != DECISION_FIELDS:
            raise ValueError("Mixed decision CSV columns differ from the prepared review.")
        rows = list(reader)
        cases = self.manifest["cases"]
        if len(rows) != len(cases):
            raise ValueError("Mixed review rows are missing or duplicated.")
        for row, case in zip(rows, cases):
            expected = dict(rank=str(case["rank"]), sheet=case["sheet"], kind=case["kind"],
                            ids=" ".join(map(str, case["ids"])), frame_before=str(case["before"][1]),
                            frame_after=str(case["after"][1]))
            if any(row.get(k) != value for k, value in expected.items()) or None in row:
                raise ValueError("Mixed decision metadata was modified; use the editor for label changes.")
            row["decision"] = (row.get("decision") or "").strip().upper()
            if row["decision"] not in VALID or row.get("note") is None:
                raise ValueError("Expected APPLY, KEEP, EDIT or a blank mixed verdict.")
        edits_raw = self.edit_path.read_bytes()
        journal = json.loads(edits_raw.decode("utf-8"))
        if journal.get("schema_version") != 1 or not isinstance(journal.get("events"), list):
            raise ValueError("Invalid edit journal.")
        seen = journal.get("seen_frames", [])
        if any(not isinstance(f, int) or not 1 <= f <= self.manifest["n_frames"] for f in seen):
            raise ValueError("Invalid displayed-frame log.")
        self.rows, self.journal = rows, journal
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.edit_hash = hashlib.sha256(edits_raw).hexdigest()
        self.check_inputs()

    @property
    def tally(self):
        counts = Counter(row["decision"] or "PENDING" for row in self.rows)
        return {key: counts[key] for key in ("APPLY", "KEEP", "EDIT", "PENDING")}

    def next_pending(self, start=0):
        for offset in range(len(self.rows)):
            index = (start + offset) % len(self.rows)
            if not self.rows[index]["decision"]:
                return index
        return None

    def assert_unchanged(self):
        for path, expected in ((self.path, self.fingerprint), (self.edit_path, self.edit_hash),
                               (self.manifest_path, self.manifest_hash),
                               (self.source_path, self.manifest["source"]["sha256"])):
            if sha256(path) != expected:
                raise ValueError("Mixed review changed outside the app. Reload before saving.")

    def check_inputs(self, images=False):
        self.assert_unchanged()
        n = self.manifest["n_frames"]
        if self.manifest["segment"][1] - self.manifest["segment"][0] != n:
            raise ValueError("Manifest segment/frame length mismatch.")
        for rank, case in enumerate(self.manifest["cases"], 1):
            sheet = self.bundle / case["sheet"]
            if case["rank"] != rank or sheet.resolve().parent != self.bundle.resolve() or not sheet.is_file():
                raise ValueError("Missing or invalid mixed review sheet.")
            lo, hi = case["window"]
            if not 1 <= lo <= hi <= n:
                raise ValueError("Invalid review window.")
            for tid, frame in (case["before"], case["after"]):
                if tid and (frame, tid) not in self.source_rows:
                    raise ValueError("Review anchor does not refer to its source annotation.")
        audits = [c for c in self.manifest["cases"] if c["kind"] == "AUDIT"]
        if {f for c in audits for f in range(c["window"][0], c["window"][1] + 1)} != set(range(1, n + 1)):
            raise ValueError("Full-frame audit does not cover the whole segment.")
        if images:
            directory = Path(self.manifest["frames_dir"])
            expected_names = {f"{f:06d}.jpg" for f in range(1, n + 1)}
            if set(self.manifest["images_sha256"]) != expected_names:
                raise ValueError("Manifest image inventory does not cover the segment.")
            for name, fingerprint in self.manifest["images_sha256"].items():
                if sha256(directory / name) != fingerprint:
                    raise ValueError(f"Review image {name} changed after preparation.")

    def save(self, index, decision, note):
        self.assert_unchanged()
        decision = decision.strip().upper()
        if decision not in VALID:
            raise ValueError("Expected APPLY, KEEP, EDIT or a blank decision.")
        replacement = dict(self.rows[index], decision=decision, note=note)
        if replacement == self.rows[index]:
            return
        rows = list(self.rows)
        rows[index] = replacement
        self.preview(rows=rows)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        payload = buffer.getvalue().encode("utf-8")
        atomic_write(self.path, payload, self.fingerprint)
        self.rows, self.fingerprint = rows, hashlib.sha256(payload).hexdigest()

    def update_journal(self, journal):
        self.assert_unchanged()
        self.preview(journal=journal)
        payload = json_bytes(journal)
        atomic_write(self.edit_path, payload, self.edit_hash)
        self.journal, self.edit_hash = journal, hashlib.sha256(payload).hexdigest()

    def add_event(self, event):
        if self.manifest.get('review_scope')=='ROI_ONLY' and event.get('op')=='upsert':
            from code.scripts.gt_roi import load_roi,contains_box
            p=event['fields'];x,y,w,h=map(float,p[2:6])
            if not contains_box([x,y,x+w,y+h],load_roi(self.bundle)):
                raise ValueError('Box lies outside the frozen ROI; no annotation was saved')
        event = dict(event, event_id=uuid.uuid4().hex, created_utc=now())
        journal = dict(self.journal, events=self.journal["events"] + [event])
        self.update_journal(journal)

    def undo(self):
        if self.journal["events"]:
            remaining = self.journal["events"][:-1]
            for index, (case, row) in enumerate(zip(self.manifest["cases"], self.rows)):
                if row["decision"] == "EDIT" and not any(event_relevant(e, case) for e in remaining):
                    self.save(index, "", row["note"])
            self.update_journal(dict(self.journal, events=remaining))

    def record_seen(self, frames):
        seen = sorted(set(self.journal["seen_frames"]) | set(frames))
        if seen != self.journal["seen_frames"]:
            self.update_journal(dict(self.journal, seen_frames=seen))

    def new_track_id(self):
        tids = [tid for _, tid in self.source_rows]
        tids += [int(e["fields"][1]) for e in self.journal["events"] if e["op"] == "upsert"]
        return max(tids, default=0) + 1

    def preview(self, rows=None, journal=None, strict=False):
        return apply_review(self.source_rows, self.manifest, self.rows if rows is None else rows,
                            self.journal if journal is None else journal, strict)


class Groups:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        if node not in self.parent:
            raise ValueError(f"Identity segment {node} no longer exists; revise the conflicting decision.")
        while node != self.parent[node]:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def event_relevant(event, case):
    if event.get("case_rank") == case["rank"]:
        return True
    if event["op"] in {"upsert", "delete_box"}:
        frame = int(event["fields"][0]) if event["op"] == "upsert" else event["frame"]
        tid = int(event["fields"][1]) if event["op"] == "upsert" else event["tid"]
        return case["window"][0] <= frame <= case["window"][1] and (case["kind"] == "AUDIT" or tid in case["ids"])
    if event["op"] == "reclassify" and "keys" in event:
        return any(case["window"][0] <= f <= case["window"][1]
                   and (case["kind"] == "AUDIT" or tid in case["ids"]) for f, tid in event["keys"])
    return bool(set(event.get("tids", [event.get("tid")])) & set(case["ids"]))


def apply_review(source_rows, manifest, decisions, journal, strict=False):
    """Return {(frame, source ID): output fields}, recording every removed box."""
    n = manifest["n_frames"]
    records = {key: list(fields) for key, fields in source_rows.items()}
    cuts, merges, deletions = defaultdict(set), [], set()
    cases = manifest["cases"]
    for case, row in zip(cases, decisions):
        verdict = row["decision"]
        if verdict == "APPLY":
            proposal = case["proposal"]
            if proposal is None:
                raise ValueError(f"Case {case['rank']}: inspect/edit this case instead of accepting an automatic fix.")
            if proposal["op"] == "split":
                cuts[proposal["tid"]].add(proposal["frame"])
            elif proposal["op"] == "merge":
                merges.append(proposal)
            elif proposal["op"] == "delete_track":
                deletions.add(proposal["tid"])
        if verdict in {"KEEP", "EDIT"} and case["kind"] == "AUDIT":
            lo, hi = case["window"]
            missing = set(range(lo, hi + 1)) - set(journal["seen_frames"])
            if missing:
                raise ValueError(f"Case {case['rank']}: open V and inspect every frame in this block first ({len(missing)} unseen).")
        if verdict == "EDIT" and not any(event_relevant(e, case) for e in journal["events"]):
            raise ValueError(f"Case {case['rank']}: no saved manual edit belongs to this case.")
    for event in journal["events"]:
        op = event["op"]
        if op == "upsert":
            fields = [str(x) for x in event["fields"]]
            validate_fields(fields, n)
            records[(int(fields[0]), int(fields[1]))] = fields
        elif op == "delete_box":
            key = (int(event["frame"]), int(event["tid"]))
            if key not in records:
                raise ValueError("The box to remove no longer exists; undo/revise the edit.")
            del records[key]
        elif op == "reclassify":
            if event["class_id"] not in range(1, 6):
                raise ValueError("Unknown edited class.")
            keys = {tuple(k) for k in event.get("keys", [])}
            tids = set(event.get("tids", []))
            targets = {key for key in records if key in keys or key[1] in tids}
            if not targets:
                raise ValueError("No existing track was selected for reclassification.")
            for key in targets:
                records[key][7] = str(event["class_id"])
        elif op == "split":
            cuts[int(event["tid"])].add(int(event["frame"]))
        elif op == "merge":
            merges.append(event)
        else:
            raise ValueError(f"Unknown manual edit {op}.")
    original_and_added = {**source_rows, **records}
    for tid, frames in cuts.items():
        available = sorted(f for f, t in original_and_added if t == tid)
        for frame in frames:
            if frame not in available or not any(f < frame for f in available):
                raise ValueError(f"ID {tid}: split must start at an existing box after its first frame.")
    sorted_cuts = {tid: sorted(values) for tid, values in cuts.items()}

    def node(tid, frame):
        return (tid, bisect.bisect_right(sorted_cuts.get(tid, []), frame))

    nodes = {node(tid, frame) for frame, tid in original_and_added}
    groups, duplicates = Groups(nodes), Groups(nodes)
    preferred = []
    for merge in merges:
        anchors = merge["anchors"]
        if len(anchors) != 2 or any((frame, tid) not in original_and_added for tid, frame in anchors):
            raise ValueError("Merge anchors must refer to existing source boxes.")
        a, b = [node(tid, frame) for tid, frame in anchors]
        if a == b:
            raise ValueError("Cannot merge an identity segment with itself.")
        if any(tid in deletions for tid, _ in anchors):
            raise ValueError("A merge conflicts with a deleted track. Revise that case first.")
        groups.join(a, b)
        if merge.get("allow_overlap"):
            duplicates.join(a, b)
            preferred.append(a)
    # Reject a transitive merge that accidentally reconnects an accepted split,
    # or contradicts a human 'these are different vehicles' judgement.
    for case, row in zip(cases, decisions):
        if not case["ids"]:
            continue
        a, b = [node(tid, frame) for tid, frame in (case["before"], case["after"]) if tid]
        same = groups.find(a) == groups.find(b)
        if row["decision"] == "APPLY" and case["kind"] in {"SPLIT", "JUMP"} and same:
            raise ValueError(f"Case {case['rank']}: a merge would undo this accepted split.")
        if row["decision"] == "KEEP" and case["kind"] in {"MERGE", "DUP"} and same:
            raise ValueError(f"Case {case['rank']}: a merge contradicts the decision to keep vehicles separate.")
        if row["decision"] == "KEEP" and case["kind"] in {"SPLIT", "JUMP"} and not same:
            raise ValueError(f"Case {case['rank']}: a split contradicts the decision to keep this identity.")
    max_id = max(tid for _, tid in original_and_added)
    node_ids = {}
    for nd in sorted(nodes):
        if nd[1] == 0:
            node_ids[nd] = nd[0]
        else:
            max_id += 1
            node_ids[nd] = max_id
    component_ids = {}
    for nd in nodes:
        root = groups.find(nd)
        component_ids[root] = min(component_ids.get(root, node_ids[nd]), node_ids[nd])
    buckets = defaultdict(list)
    for key, fields in records.items():
        frame, tid = key
        if tid not in deletions:
            buckets[(frame, groups.find(node(tid, frame)))].append((key, fields))
    output, removed_duplicates = {}, []
    for (frame, root), bucket in buckets.items():
        if len(bucket) > 1:
            duplicate_roots = {duplicates.find(node(key[1], frame)) for key, _ in bucket}
            boxes = [[float(p[2]), float(p[3]), float(p[2]) + float(p[4]), float(p[3]) + float(p[5])]
                     for _, p in bucket]
            if len(duplicate_roots) != 1 or any(_iou(a, b) < 0.65
                                              for i, a in enumerate(boxes) for b in boxes[i + 1:]):
                raise ValueError(f"Frame {frame}: merged tracks contain conflicting boxes. Inspect V, fix the overlap, then E.")
            def preference(item):
                nd = node(item[0][1], frame)
                return (preferred.index(nd) if nd in preferred else len(preferred), item[0][1])
            bucket.sort(key=preference)
            removed_duplicates.extend(key for key, _ in bucket[1:])
        key, fields = bucket[0]
        result = list(fields)
        result[1] = str(component_ids[root])
        output[key] = result
    component_classes = defaultdict(set)
    for fields in output.values():
        component_classes[int(fields[1])].add(int(fields[7]))
    # Merging across different labels always needs explicit class correction.
    merged_ids = {component_ids[root] for root, count in Counter(groups.find(nd) for nd in nodes).items() if count > 1}
    conflicts = sorted(tid for tid, values in component_classes.items()
                       if len(values) > 1 and (strict or tid in merged_ids))
    if conflicts:
        details = []
        for tid in conflicts[:5]:
            affected = [(key, fields) for key, fields in output.items() if int(fields[1]) == tid]
            sources = sorted({key[1] for key, _ in affected})
            classes = defaultdict(list)
            for _, fields in affected:
                classes[int(fields[7])].append(int(fields[0]))
            descriptions = []
            for cid, frames in sorted(classes.items()):
                samples = ', '.join(map(str, sorted(frames)[:3]))
                descriptions.append(f"{CLASS_NAMES.get(cid - 1, str(cid))}: MOT frame {samples} ({len(frames)} boxes)")
            details.append(f"ID {tid}, source IDs {', '.join(map(str, sources))}: " + '; '.join(descriptions))
        if len(conflicts) > 5:
            details.append(f"... and {len(conflicts) - 5} more conflicting identities")
        raise ValueError("An identity contains multiple classes.\n" + '\n'.join(details)
                         + "\nThe operation was not saved. Inspect these IDs/frames in V; if they are different vehicles, keep them separate. "
                           "If one vehicle has a wrong label, use the whole-ID class correction button, not only the class dropdown. "
                           "After a saved manual merge, use E in the main review; A applies the original automatic proposal.")
    if strict and (any(not r["decision"] for r in decisions) or len(set(journal["seen_frames"])) != n):
        raise ValueError("Complete every verdict and full-frame audit before finalizing mixed GT.")
    if strict and not output:
        raise ValueError("Reviewed GT has no boxes.")
    stats = dict(boxes_before=len(source_rows), boxes_after=len(output),
                 identities_before=len({tid for _, tid in source_rows}),
                 identities_after=len({int(p[1]) for p in output.values()}),
                 duplicate_boxes_removed=len(removed_duplicates),
                 deleted_source_boxes=len(set(source_rows) - set(output)),
                 added_boxes=len(set(output) - set(source_rows)),
                 manually_changed_boxes=sum(p[2:] != source_rows[k][2:]
                                            for k, p in output.items() if k in source_rows),
                 identity_changed_boxes=sum(p[1] != source_rows[k][1]
                                           for k, p in output.items() if k in source_rows))
    return output, stats


def finalize_mixed(store: MixedStore, reviewer: str, tracks_dir=TRACKS_DIR, out_dir=None,
                   evaluate=True, progress=None):
    if not reviewer.strip():
        raise ValueError("Reviewer name is required.")
    store.check_inputs(images=True)
    from code.scripts.gt_roi import load_roi, contains_box
    roi = load_roi(store.bundle)
    roi_path = store.bundle / 'roi.json'
    roi_raw = roi_path.read_bytes() if roi else None
    records, stats = store.preview(strict=True)
    snapshots = {"decisions.csv": store.path.read_bytes(), "edits.json": store.edit_path.read_bytes(),
                 "manifest.json": store.manifest_path.read_bytes(), "source_gt.txt": store.source_path.read_bytes()}
    if (store.bundle/'scope_provenance.json').exists():
        snapshots['scope_provenance.json']=(store.bundle/'scope_provenance.json').read_bytes()
    out_dir = Path(out_dir) if out_dir else RESULTS / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    if out_dir.exists():
        raise ValueError("Choose a new output directory; previous review runs are preserved.")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mixed-", dir=out_dir.parent))
    try:
        reviewed = staging / "gt_reviewed.txt"
        payload = "".join(",".join(fields) + "\n" for _, fields in sorted(records.items()))
        reviewed.write_text(payload, encoding="utf-8", newline="")
        parsed = read_rows(reviewed, store.manifest["n_frames"])
        if len(parsed) != len(records):
            raise ValueError("Reviewed GT has duplicate identity/frame pairs.")
        roi_kept = {}
        if roi and roi['enabled']:
            for key,p in records.items():
                x,y,w,h = map(float,p[2:6])
                if contains_box([x,y,x+w,y+h],roi):roi_kept[key]=p
            if not roi_kept:
                raise ValueError('ROI contains no reviewed GT boxes; revise the ROI before exporting')
            (staging/'gt_roi.txt').write_text(''.join(','.join(p)+'\n' for _,p in sorted(roi_kept.items())),encoding='utf-8')
        scores = score_variants(reviewed, Path(tracks_dir), tuple(store.manifest["segment"]), progress, roi=roi) if evaluate else {}
        store.check_inputs(images=True)
        if (roi_path.read_bytes() if roi_path.exists() else None) != roi_raw:
            raise ValueError('ROI changed during export/scoring; retry with the current ROI')
        if roi_raw is not None:
            snapshots['roi.json'] = roi_raw
        for name, snapshot in snapshots.items():
            if (store.bundle / name).read_bytes() != snapshot:
                raise ValueError("Review changed during scoring; reload and retry.")
            (staging / name).write_bytes(snapshot)
        report = dict(schema_version=1, reviewer=reviewer.strip(), created_utc=now(),
                      scope="Human decisions on generated cases, manual edits, and full-frame playback audit. "
                            "Displayed frames are logged; this does not certify annotator agreement or test independence.",
                      evaluation_role="development/validation: mixed segment lies in the sequence used to develop the method",
                      segment_zero_based_end_exclusive=store.manifest["segment"],
                      source_gt_sha256=store.manifest["source"]["sha256"],
                      reviewed_gt_sha256=sha256(reviewed), verdicts=store.tally, gt_changes=stats,
                      displayed_frames=len(set(store.journal["seen_frames"])),
                      manual_events=len(store.journal["events"]),
                      snapshot_sha256={k: hashlib.sha256(v).hexdigest() for k, v in snapshots.items()},
                      metric_engine="code.evaluation.metrics.compute_tracking_metrics; IoU=0.5",
                      metric_engine_sha256=sha256(REPO_ROOT / "code/evaluation/metrics.py"),
                      evaluation_performed=evaluate, variants=scores)
        report['evaluation_scope'] = 'ROI' if roi and roi['enabled'] else 'FULL_FRAME'
        report['roi'] = roi
        if 'scope_provenance.json' in snapshots:
            report['review_scope_provenance']=json.loads(snapshots['scope_provenance.json'])
        if roi and roi['enabled']:
            report['roi_gt'] = dict(path='gt_roi.txt',sha256=sha256(staging/'gt_roi.txt'),
                                    boxes_included=len(roi_kept),boxes_excluded=len(records)-len(roi_kept))
        (staging / "review.json").write_bytes(json_bytes(report))
        lines = ["# Mixed GT review", "", f"Reviewer: {reviewer.strip()}", "", report["scope"], "",
                 report["evaluation_role"], "", f"GT changes: {stats}", "",
                 f"Evaluation scope: {report['evaluation_scope']}; GT and predictions use the same ROI rule.", "",
                 "| Variant | Class | MOTA | IDF1 | ID switches |", "|---|---|---:|---:|---:|"]
        for variant, data in scores.items():
            for cls, m in data["metrics"].items():
                lines.append(f"| {variant} | {cls} | {m['mota']:.4f} | {m['idf1']:.4f} | {m['num_switches']} |")
        if not evaluate:
            lines += ["", "Evaluation explicitly skipped."]
        (staging / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        staging.rename(out_dir)
    finally:
        if staging.exists():
            safe_cleanup(staging, out_dir.parent)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--reviewer", default="Nghĩa")
    parser.add_argument("--tracks-dir", type=Path, default=TRACKS_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    try:
        if args.prepare:
            print(prepare(args.source, args.bundle))
        else:
            store = MixedStore(args.bundle)
            if args.check:
                store.check_inputs(images=True)
                print(json.dumps({"cases": dict(Counter(c['kind'] for c in store.manifest['cases'])),
                                  "verdicts": store.tally, "displayed_frames": len(store.journal['seen_frames'])}, indent=2))
            else:
                print(finalize_mixed(store, args.reviewer, args.tracks_dir, args.out_dir,
                                     not args.skip_evaluation, progress=lambda s: print(s, flush=True)))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Mixed review: {error}\n")


if __name__ == "__main__":
    main()
