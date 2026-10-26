"""
code/scripts/gt_apply_splits.py
───────────────────────────────
Turn a human verdict on the 105 SPLIT candidate sheets into a corrected
`gt.txt`, without going through CVAT.

Why not CVAT
------------
Three reasons, and the first one is disqualifying:

1. CVAT renumbers every track `1..N` in export order on each MOT 1.1 write.
   Editing there and re-exporting invalidates the id mapping that
   `split_candidates/candidates.csv` and the whole sensitivity analysis are
   keyed on — you would no longer be able to say which candidate you acted on.
2. Locating 105 joins by frame-and-pixel (the only way, given point 1) is slow
   and misclick-prone.
3. The edit is not an annotation edit. "Cut this join" means: from frame X
   onward, this identity carries a new id. That is a transformation of
   `gt.txt`, and nothing needs to be redrawn.

So the human does only the part a machine cannot — look at the sheet, decide
whether it is one vehicle or several — and this script does the rest.

Workflow
--------
    # 1. write a decision sheet next to the rendered candidates
    python -m code.scripts.gt_apply_splits --init

    # 2. open experiments/gt_qa/split_candidates/decisions.csv, look at each
    #    PNG, and put CUT or KEEP in the `decision` column.
    #      CUT  = these are different vehicles, break the identity here
    #      KEEP = same vehicle, the heuristic was wrong
    #      (blank = not reviewed yet; --apply refuses to run until none are)

    # 3. apply exactly those cuts
    python -m code.scripts.gt_apply_splits --apply

Writes `gt_reviewed.txt` beside the original `gt.txt` and never overwrites it,
so the as-annotated version stays available as a row in the sensitivity table.

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANDIDATES = REPO_ROOT / "experiments/gt_qa/split_candidates/candidates.csv"
DECISIONS = REPO_ROOT / "experiments/gt_qa/split_candidates/decisions.csv"
GT_IN = REPO_ROOT / "data/annotations/NKKN_gt/dense/gt/gt.txt"

CUT, KEEP = "CUT", "KEEP"
VALID = {CUT, KEEP}


def show(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (--out may point
    anywhere, and a display helper must never be the thing that crashes a
    run that already wrote its output)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ── Decision sheet ───────────────────────────────────────────────────

def load_candidates(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sheet_name(row: dict, rank: int) -> str:
    """Match the filenames render_split_candidates.py writes."""
    return (f"{rank:03d}_id{row['gt_id_INTERNAL_not_cvat']}"
            f"_f{row['cvat_before']}-{row['cvat_after']}.png")


def init_decisions(cand_path: Path, out_path: Path, force: bool) -> None:
    rows = load_candidates(cand_path)
    if out_path.exists() and not force:
        raise SystemExit(
            f"{show(out_path)} already exists — refusing to "
            f"overwrite your verdicts. Pass --force if you really mean to.")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "sheet", "gt_id", "cut_at_frame", "gap",
                    "move_diag", "area_ratio", "decision", "note"])
        for i, r in enumerate(rows, start=1):
            w.writerow([r["rank"], sheet_name(r, i),
                        r["gt_id_INTERNAL_not_cvat"], r["mot_after"],
                        r["gap"], r["move_diag"], r["area_ratio"], "", ""])

    print(f"Wrote {show(out_path)} — {len(rows)} candidates, "
          f"all blank.\n"
          f"Open it next to {out_path.parent.name}/ and fill `decision` with "
          f"CUT or KEEP.\n"
          f"Sheets are sorted worst-first, so the early rows are the ones most "
          f"likely to be genuine.")


# ── Applying verdicts ────────────────────────────────────────────────

def load_decisions(path: Path, allow_partial: bool = False
                   ) -> tuple[list[tuple[int, int]], dict]:
    """-> ([(gt_id, cut_at_frame)] for CUT rows, tally).

    allow_partial=True is for --bound, which is *about* the unreviewed rows and
    so must read a half-filled sheet. --apply keeps refusing one.
    """
    if not path.exists():
        raise SystemExit(f"{show(path)} not found — run "
                         f"--init first.")

    cuts, tally, blank = [], defaultdict(int), []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r.get("decision") or "").strip().upper()
            if not d:
                blank.append(r["rank"])
                continue
            if d not in VALID:
                raise SystemExit(
                    f"row {r['rank']}: decision {d!r} is not {CUT} or {KEEP}.")
            tally[d] += 1
            if d == CUT:
                cuts.append((int(r["gt_id"]), int(r["cut_at_frame"])))

    if blank and not allow_partial:
        raise SystemExit(
            f"{len(blank)} candidates still blank (ranks "
            f"{', '.join(blank[:8])}{'...' if len(blank) > 8 else ''}).\n"
            f"Every row needs a verdict — a blank is not the same as KEEP, and "
            f"guessing on your behalf is the one thing this script must not do.")
    tally = dict(tally)
    tally["BLANK"] = len(blank)
    return cuts, tally


def unreviewed_cuts(path: Path) -> list[tuple[int, int]]:
    """The cuts that would apply if every unreviewed row were judged CUT."""
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not (r.get("decision") or "").strip():
                out.append((int(r["gt_id"]), int(r["cut_at_frame"])))
    return out


def apply_cuts(gt_path: Path, cuts: list[tuple[int, int]],
               out_path: Path) -> dict:
    """Rewrite gt.txt so each cut gives the identity a fresh id from that
    frame on. Column order and every other field are preserved byte-for-byte;
    only the id column changes."""
    cut_frames: dict[int, set[int]] = defaultdict(set)
    for tid, frame in cuts:
        cut_frames[tid].add(frame)

    rows = []
    max_id = 0
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",")
            fid, tid = int(parts[0]), int(parts[1])
            max_id = max(max_id, tid)
            rows.append((fid, tid, parts))

    next_id = max_id + 1
    # Segment index per identity: how many cut points lie at or before this
    # frame. Segment 0 keeps the original id so untouched identities are
    # bit-identical to the input.
    remap: dict[tuple[int, int], int] = {}
    seg_counts: dict[int, int] = {}

    for fid, tid, parts in rows:
        seg = sum(1 for cf in cut_frames.get(tid, ()) if fid >= cf)
        if seg:
            key = (tid, seg)
            if key not in remap:
                remap[key] = next_id
                next_id += 1
            parts[1] = str(remap[key])
        seg_counts[tid] = max(seg_counts.get(tid, 0), seg)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for _, _, parts in rows:
            f.write(",".join(parts) + "\n")

    ids_before = len({tid for _, tid, _ in rows})
    return {
        "boxes": len(rows),
        "ids_before": ids_before,
        "ids_after": ids_before + len(remap),
        "identities_touched": sum(1 for v in seg_counts.values() if v),
        "new_ids": len(remap),
    }


# ── Bounding the answer with a partly-reviewed sheet ─────────────────

def bound(gt_path: Path, decisions: Path, tracks_dir: Path,
          segment: tuple[int, int]) -> dict:
    """Margin under both extreme readings of the unreviewed rows.

    This is what makes the review stoppable. Reviewing all 105 candidates to
    three decimal places is only worth it if the answer is still in doubt --
    so after each batch, score the ground truth twice: once treating every
    unreviewed row as KEEP, once as CUT. The truth is between them by
    construction, no matter what the remaining sheets turn out to show. When
    both ends support the same statement, more reviewing cannot change it.
    """
    from code.evaluation.metrics import compute_tracking_metrics
    from code.scripts.import_gt import load_mot, to_framelist, tracks_json_slice

    MOTORCYCLE = 1
    decided, tally = load_decisions(decisions, allow_partial=True)
    pending = unreviewed_cuts(decisions)

    gt0 = load_mot(gt_path)
    n = max(gt0)
    preds = {v: tracks_json_slice(str(tracks_dir / f"tracks_{v}.json"), *segment)
             for v in ("baseline", "ctrv_gate_nsa")}

    def moto_idf1(pred, gt):
        return compute_tracking_metrics(to_framelist(pred, n, MOTORCYCLE),
                                        to_framelist(gt, n, MOTORCYCLE)).idf1

    out = {"reviewed": tally.get(CUT, 0) + tally.get(KEEP, 0),
           "cut": tally.get(CUT, 0), "keep": tally.get(KEEP, 0),
           "pending": len(pending), "ends": {}}

    tmp = gt_path.with_name("_bound_tmp.txt")
    try:
        for end, cuts in (("keep", decided), ("cut", decided + pending)):
            apply_cuts(gt_path, cuts, tmp)
            gt = load_mot(tmp)
            b = moto_idf1(preds["baseline"], gt)
            c = moto_idf1(preds["ctrv_gate_nsa"], gt)
            out["ends"][end] = {"baseline": b, "proposed": c, "margin": c - b,
                                "gt_ids": len({t for f in gt.values()
                                               for t, _, _ in f})}
    finally:
        tmp.unlink(missing_ok=True)
    return out


def print_bound(r: dict) -> None:
    lo = min(r["ends"]["keep"]["margin"], r["ends"]["cut"]["margin"])
    hi = max(r["ends"]["keep"]["margin"], r["ends"]["cut"]["margin"])
    print(f"\nReviewed {r['reviewed']}/{r['reviewed'] + r['pending']}  "
          f"({r['cut']} CUT, {r['keep']} KEEP, {r['pending']} pending)\n")
    print(f"{'unreviewed read as':<20} {'GT ids':>7} {'baseline':>9} "
          f"{'proposed':>9} {'margin':>8}")
    print("-" * 58)
    for end in ("keep", "cut"):
        e = r["ends"][end]
        print(f"{'all ' + end.upper():<20} {e['gt_ids']:>7} "
              f"{e['baseline']:>9.3f} {e['proposed']:>9.3f} "
              f"{e['margin']:>+8.3f}")
    print("-" * 58)
    print(f"Motorcycle IDF1 margin is between {lo:+.3f} and {hi:+.3f}, "
          f"whatever the {r['pending']} unreviewed sheets show.")
    if lo > 0:
        print("The proposed method leads at BOTH ends: that conclusion is "
              "already settled and no further review can overturn it.")
    else:
        print("The bracket still straddles zero — the lead is not yet "
              "established. Keep reviewing.")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply a human verdict on SPLIT candidates to gt.txt.")
    ap.add_argument("--init", action="store_true",
                    help="Write the blank decision sheet.")
    ap.add_argument("--apply", action="store_true",
                    help="Apply CUT verdicts and write gt_reviewed.txt.")
    ap.add_argument("--bound", action="store_true",
                    help="Score both extreme readings of the unreviewed rows. "
                         "Tells you whether more reviewing can still change "
                         "the answer. Works on a half-filled sheet.")
    ap.add_argument("--tracks-dir", type=Path,
                    default=REPO_ROOT / "experiments/gt_ablation")
    ap.add_argument("--segment", nargs=2, type=int, default=[10700, 11300])
    ap.add_argument("--candidates", type=Path, default=CANDIDATES)
    ap.add_argument("--decisions", type=Path, default=DECISIONS)
    ap.add_argument("--gt", type=Path, default=GT_IN)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: gt_reviewed.txt beside --gt.")
    ap.add_argument("--force", action="store_true",
                    help="--init: overwrite an existing decision sheet.")
    args = ap.parse_args()

    if sum([args.init, args.apply, args.bound]) != 1:
        raise SystemExit("Pass exactly one of --init, --apply or --bound.")

    if args.init:
        init_decisions(args.candidates, args.decisions, args.force)
        return

    if args.bound:
        print_bound(bound(args.gt, args.decisions, args.tracks_dir,
                          tuple(args.segment)))
        return

    cuts, tally = load_decisions(args.decisions)
    out = args.out or args.gt.with_name("gt_reviewed.txt")
    if not out.exists():
        shutil.copy(args.gt, args.gt.with_name("gt_as_annotated.txt"))

    stats = apply_cuts(args.gt, cuts, out)

    print(f"Verdicts: {tally.get(CUT, 0)} CUT, {tally.get(KEEP, 0)} KEEP")
    print(f"Identities touched : {stats['identities_touched']}")
    print(f"Identities          : {stats['ids_before']} -> {stats['ids_after']} "
          f"(+{stats['new_ids']})")
    print(f"Boxes (unchanged)   : {stats['boxes']}")
    print(f"\nWrote {show(out)}")
    print(f"Original preserved as {args.gt.with_name('gt_as_annotated.txt').name} "
          f"— it stays the 'as annotated' row of the sensitivity table.")
    print("\nNext:\n"
          "  python -m code.scripts.import_gt --gt "
          f"{Path(show(out)).as_posix()}\n"
          "  python -m code.scripts.gt_review_queue   # SPLIT hits should drop\n"
          "  python -m code.scripts.gt_bias_sensitivity")


if __name__ == "__main__":
    main()
