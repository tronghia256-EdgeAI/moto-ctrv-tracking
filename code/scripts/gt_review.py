"""Safe persistence and finalization for human review of GT split candidates.

No verdict is inferred from a tracker or from the candidate geometry. A completed
candidate review is an identity-split audit, not a certification of all GT boxes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from code.scripts.gt_apply_splits import (
    CANDIDATES, DECISIONS, GT_IN, REPO_ROOT, apply_cuts, load_candidates, sheet_name,
)

TRACKS_DIR = REPO_ROOT / "experiments/gt_ablation"
RESULTS_DIR = REPO_ROOT / "experiments/gt_qa/reviewed"
FIELDS = {"rank", "sheet", "gt_id", "cut_at_frame", "decision", "note"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(
            hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def atomic_write(path: Path, payload: bytes, expected_hash: str) -> None:
    """Replace a local review file only if the version read by the caller remains."""
    if sha256(path) != expected_hash:
        raise ValueError("Review file changed outside the reviewer. Reload before saving.")
    fd, name = tempfile.mkstemp(prefix=".decisions-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if sha256(path) != expected_hash:
            raise ValueError("Review file changed outside the reviewer. Reload before saving.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class DecisionStore:
    """Preserve CSV metadata and refuse to overwrite an external edit."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.reload()

    def reload(self) -> None:
        raw = self.path.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        fields = reader.fieldnames or []
        if not FIELDS.issubset(fields) or len(set(fields)) != len(fields):
            raise ValueError("Decision CSV has missing or duplicate columns.")
        rows = list(reader)
        if not rows:
            raise ValueError("Decision CSV has no candidates.")
        ranks, joins = set(), set()
        for row in rows:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Decision CSV has a row with the wrong column count.")
            rank, tid, frame = (int(row[k]) for k in ("rank", "gt_id", "cut_at_frame"))
            if min(rank, tid, frame) < 1 or rank in ranks or (tid, frame) in joins:
                raise ValueError("Candidate ranks and joins must be positive and unique.")
            ranks.add(rank)
            joins.add((tid, frame))
            name = row["sheet"]
            if not name or "/" in name or "\\" in name or Path(name).suffix.lower() != ".png":
                raise ValueError("Candidate sheet must be a PNG filename without directories.")
            row["decision"] = row["decision"].strip().upper()
            if row["decision"] not in {"", "CUT", "KEEP"}:
                raise ValueError(f"Rank {rank}: expected CUT, KEEP or a blank decision.")
        self.fields, self.rows = fields, rows
        self.fingerprint = hashlib.sha256(raw).hexdigest()

    @property
    def tally(self) -> dict:
        counts = Counter(row["decision"] or "PENDING" for row in self.rows)
        return {key: counts[key] for key in ("CUT", "KEEP", "PENDING")}

    def next_pending(self, start: int = 0) -> int | None:
        for offset in range(len(self.rows)):
            index = (start + offset) % len(self.rows)
            if not self.rows[index]["decision"]:
                return index
        return None

    def assert_unchanged(self) -> None:
        if sha256(self.path) != self.fingerprint:
            raise ValueError("CSV changed outside the reviewer. Reload before saving.")

    def save(self, index: int, decision: str, note: str) -> None:
        decision = decision.strip().upper()
        if decision not in {"", "CUT", "KEEP"}:
            raise ValueError("Expected CUT, KEEP or a blank decision.")
        self.assert_unchanged()
        replacement = dict(self.rows[index], decision=decision, note=note)
        if replacement == self.rows[index]:
            return
        rows = list(self.rows)
        rows[index] = replacement
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=self.fields)
        writer.writeheader()
        writer.writerows(rows)
        payload = buffer.getvalue().encode("utf-8")
        atomic_write(self.path, payload, self.fingerprint)
        self.rows = rows
        self.fingerprint = hashlib.sha256(payload).hexdigest()


def check_inputs(store: DecisionStore, candidates: Path, gt: Path,
                 segment: tuple[int, int]) -> dict:
    """Check candidate mapping, all sheets and every proposed cut against GT."""
    store.assert_unchanged()
    if segment[0] < 0 or segment[1] <= segment[0]:
        raise ValueError("Segment must be a nonempty, zero-based [start, end) interval.")
    source = load_candidates(candidates)
    by_rank = {int(row["rank"]): row for row in source}
    if len(by_rank) != len(source) or set(by_rank) != {int(r["rank"]) for r in store.rows}:
        raise ValueError("Decisions and candidates must contain exactly the same ranks.")
    identities = defaultdict(set)
    boxes = 0
    with gt.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 8:
                raise ValueError(f"GT line {number}: expected at least 8 columns.")
            frame, tid = int(parts[0]), int(parts[1])
            if not 1 <= frame <= segment[1] - segment[0] or tid < 1:
                raise ValueError(f"GT line {number}: frame or identity out of range.")
            if frame in identities[tid]:
                raise ValueError(f"Duplicate GT identity {tid} in frame {frame}.")
            identities[tid].add(frame)
            boxes += 1
    if not boxes:
        raise ValueError("GT contains no boxes.")
    for row in store.rows:
        rank, tid, frame = (int(row[k]) for k in ("rank", "gt_id", "cut_at_frame"))
        candidate = by_rank[rank]
        if (tid != int(candidate["gt_id_INTERNAL_not_cvat"])
                or frame != int(candidate["mot_after"])
                or int(candidate["abs_after"]) != segment[0] + frame - 1
                or int(candidate["abs_before"]) != segment[0] + int(candidate["mot_before"]) - 1
                or row["sheet"] != sheet_name(candidate, rank)):
            raise ValueError(f"Rank {rank}: decision metadata disagrees with candidates.")
        image_path = store.path.parent / row["sheet"]
        if not image_path.is_file() or image_path.resolve().parent != store.path.parent.resolve():
            raise ValueError(f"Rank {rank}: missing or invalid sheet {row['sheet']}.")
        before = int(candidate["mot_before"])
        if frame not in identities[tid] or before not in identities[tid] or before >= frame:
            raise ValueError(f"Rank {rank}: cut must refer to an existing join in GT.")
    return {"candidates": len(store.rows), "boxes": boxes,
            "identities": len(identities), **store.tally}


def score_variants(gt: Path, tracks_dir: Path, segment: tuple[int, int], progress=None, roi=None) -> dict:
    """Score frozen track dumps with the repository's existing metrics."""
    from code.evaluation.metrics import compute_tracking_metrics
    from code.scripts.import_gt import CLASS_NAMES, load_mot, to_framelist

    paths = sorted(tracks_dir.glob("tracks_*.json"))
    if not paths or not (tracks_dir / "tracks_baseline.json").is_file():
        raise ValueError("Expected saved tracks_*.json including tracks_baseline.json.")
    start, end = segment
    truth = load_mot(gt)
    from code.scripts.gt_roi import filter_frames
    truth_before = sum(map(len, truth.values()))
    truth = filter_frames(truth, roi)
    class_ids = sorted({cls for entries in truth.values() for _, _, cls in entries})
    classes = [("ALL", None)] + [(CLASS_NAMES.get(cid, str(cid)), cid) for cid in class_ids]
    gt_lists = {name: to_framelist(truth, end - start, cid) for name, cid in classes}
    result = {}
    for number, path in enumerate(paths, 1):
        if progress:
            progress(f"Đang chấm {number}/{len(paths)}: {path.stem.removeprefix('tracks_')}")
        fingerprint = sha256(path)
        dump = json.loads(path.read_text(encoding="utf-8"))
        if dump.get("start_frame", 0) != 0 or len(dump["frames"]) < end:
            raise ValueError(f"{path.name}: dump does not cover the requested absolute segment.")
        predictions = {k: [(tid, box, cls) for tid, cls, box in entries]
                       for k, entries in enumerate(dump["frames"][start:end], 1)}
        predictions_before = sum(map(len, predictions.values()))
        predictions = filter_frames(predictions, roi)
        metrics = {name: asdict(compute_tracking_metrics(
            to_framelist(predictions, end - start, cid), gt_lists[name]))
            for name, cid in classes}
        if sha256(path) != fingerprint:
            raise ValueError(f"{path.name}: track dump changed during evaluation.")
        result[path.stem.removeprefix("tracks_")] = {
            "path": str(path.resolve()), "sha256": fingerprint, "metrics": metrics,
        }
        if roi and roi['enabled']:
            result[path.stem.removeprefix('tracks_')]['roi_counts'] = dict(
                gt_included=sum(map(len,truth.values())), gt_excluded=truth_before-sum(map(len,truth.values())),
                predictions_included=sum(map(len,predictions.values())),
                predictions_excluded=predictions_before-sum(map(len,predictions.values())))
    return result


def finalize(store: DecisionStore, candidates: Path, gt: Path, reviewer: str,
             tracks_dir: Path = TRACKS_DIR, segment: tuple[int, int] = (10700, 11300),
             out_dir: Path | None = None, evaluate: bool = True, progress=None) -> Path:
    """Publish a complete new run only after validation and evaluation succeed."""
    if store.tally["PENDING"]:
        raise ValueError(f"{store.tally['PENDING']} candidates remain pending; blanks are not KEEP.")
    if not reviewer.strip():
        raise ValueError("Enter the name of the person who reviewed the candidates.")
    decision_snapshot = store.path.read_bytes()
    if hashlib.sha256(decision_snapshot).hexdigest() != store.fingerprint:
        raise ValueError("CSV changed outside the reviewer. Reload before finalizing.")
    hashes = {"gt": sha256(gt), "candidates": sha256(candidates), "decisions": sha256(store.path)}
    check_inputs(store, candidates, gt, segment)
    created = datetime.now(timezone.utc).isoformat()
    out_dir = out_dir or RESULTS_DIR / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError("Output directory already exists. Choose a new run directory.")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".review-", dir=out_dir.parent))
    try:
        reviewed = staging / "gt_reviewed.txt"
        cuts = [(int(r["gt_id"]), int(r["cut_at_frame"]))
                for r in store.rows if r["decision"] == "CUT"]
        stats = apply_cuts(gt, cuts, reviewed)
        # Verify independently that no row or field other than the identity changed.
        def fields(path):
            return [line.strip().split(",") for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        before, after = fields(gt), fields(reviewed)
        if len(before) != len(after) or any(a[:1] + a[2:] != b[:1] + b[2:]
                                          for a, b in zip(before, after)):
            raise ValueError("GT integrity check failed: a box or metadata field changed.")
        scores = score_variants(reviewed, tracks_dir, segment, progress) if evaluate else {}
        store.assert_unchanged()
        for name, path in (("gt", gt), ("candidates", candidates), ("decisions", store.path)):
            if sha256(path) != hashes[name]:
                raise ValueError(f"{name} changed during finalization. Retry after reloading.")
        (staging / "decisions.csv").write_bytes(decision_snapshot)
        report = {
            "schema_version": 1, "created_utc": created, "reviewer": reviewer.strip(),
            "scope": "Human adjudication of generated identity-split candidates only; "
                     "not independent validation of all boxes, missed objects or class labels.",
            "segment_zero_based_end_exclusive": list(segment), "verdicts": store.tally,
            "sources": {name: {"path": str(path.resolve()), "sha256": hashes[name]}
                        for name, path in (("gt", gt), ("candidates", candidates), ("decisions", store.path))},
            "reviewed_gt_sha256": sha256(reviewed), "gt_changes": stats,
            "metric_engine": "code.evaluation.metrics.compute_tracking_metrics; IoU=0.5",
            "metric_engine_sha256": sha256(REPO_ROOT / "code/evaluation/metrics.py"),
            "evaluation_performed": evaluate, "variants": scores,
        }
        (staging / "review.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = ["# GT split review", "", f"Reviewer: {reviewer.strip()}", "",
                 report["scope"], "", f"CUT: {store.tally['CUT']}; KEEP: {store.tally['KEEP']}",
                 f"Boxes preserved: {stats['boxes']}; IDs: {stats['ids_before']} -> {stats['ids_after']}", "",
                 "Metrics use the repository's current implementation on frozen predictions.", "",
                 "| Variant | Class | MOTA | IDF1 | ID switches |", "|---|---|---:|---:|---:|"]
        for variant, data in scores.items():
            for cls, m in data["metrics"].items():
                lines.append(f"| {variant} | {cls} | {m['mota']:.4f} | {m['idf1']:.4f} | {m['num_switches']} |")
        if not evaluate:
            lines += ["", "Evaluation was explicitly skipped."]
        (staging / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        staging.rename(out_dir)
    finally:
        if staging.exists():
            if staging.resolve().parent != out_dir.parent.resolve() or not staging.name.startswith(".review-"):
                raise ValueError("Refusing to clean a staging directory outside the output parent.")
            shutil.rmtree(staging)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate inputs without changing them.")
    mode.add_argument("--finalize", action="store_true", help="Apply a completed human review and score all dumps.")
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--gt", type=Path, default=GT_IN)
    parser.add_argument("--tracks-dir", type=Path, default=TRACKS_DIR)
    parser.add_argument("--segment", nargs=2, type=int, default=(10700, 11300))
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    try:
        store = DecisionStore(args.decisions)
        if args.check:
            print(json.dumps(check_inputs(store, args.candidates, args.gt, tuple(args.segment)), indent=2))
        else:
            print(finalize(store, args.candidates, args.gt, args.reviewer, args.tracks_dir,
                           tuple(args.segment), args.out_dir, not args.skip_evaluation))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"GT review: {error}\n")


if __name__ == "__main__":
    main()
