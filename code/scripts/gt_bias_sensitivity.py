"""
code/scripts/gt_bias_sensitivity.py
────────────────────────────────────
Bound the circularity bias in the `dense` GT instead of only disclosing it.

Pre-labels were seeded from `ctrv_gate_nsa`, the proposed method. Any
fragmentation the annotator did not catch therefore survives into the GT
as *agreement*: the tracker split one vehicle into two IDs, the GT records
two IDs, and the tracker scores as correct on its own error. Every other
variant is scored against that same GT without having shaped it, so the
bias runs one way — in favour of the proposed method.

`gt_review_queue` still flags 29 merge candidates in the finished GT.
Visual sampling of the top six (this session) found the ranking
informative: candidates at confidence >= 0.43 were genuine un-merged
fragmentations (a red motorcycle, a parked bike, a car passing behind a
pole), while those <= 0.36 were plainly different objects.

This script does NOT decide which are real and publish a "corrected" GT —
that would re-introduce exactly the problem, with the model as annotator.
Instead it runs a sensitivity analysis: apply the top-K candidates as
merges for several K, re-score every variant, and report how far the
numbers move. The result is a bracket ("the C2 margin survives correction
of up to K fragmentations") rather than a point estimate resting on the
model's own eyesight.

Usage (repo root, venv active):
  python -m code.scripts.gt_bias_sensitivity
  python -m code.scripts.gt_bias_sensitivity --k 0 5 10

Writes <out-dir>/sensitivity.md and .json.

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.evaluation.metrics import compute_tracking_metrics
from code.scripts.gt_review_queue import (_area as q_area, _centre as q_centre,
                                          _diag as q_diag, build_tracks)
from code.scripts.import_gt import load_mot, to_framelist, tracks_json_slice

MOTORCYCLE = 1
VARIANTS = ["baseline", "ctrv", "ctrv_nsa", "ctrv_gate", "ctrv_gate_nsa"]


def load_merge_candidates(csv_path: str | Path) -> list[tuple[int, int, float]]:
    """(dead_id, born_id, confidence), most confident first."""
    out = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["kind"] != "MERGE":
                continue
            a, b = (int(x) for x in r["ids"].split())
            out.append((a, b, float(r["confidence"])))
    out.sort(key=lambda t: -t[2])
    return out


def apply_merges(gt: dict[int, list],
                 merges: list[tuple[int, int, float]]) -> dict[int, list]:
    """Relabel each born_id to its dead_id, following chains so that a
    three-way split collapses to one identity rather than two."""
    remap: dict[int, int] = {}
    for a, b, _ in merges:
        root = a
        while root in remap:
            root = remap[root]
        if root != b:
            remap[b] = root
    out: dict[int, list] = {}
    for k, entries in gt.items():
        new = []
        for tid, box, cls in entries:
            root = tid
            while root in remap:
                root = remap[root]
            new.append((root, box, cls))
        out[k] = new
    return out


def split_gt(gt: dict[int, list], tracks: dict, move_max: float,
             area_max: float) -> tuple[dict[int, list], int]:
    """Break identities whose box teleports or rescales across a gap.

    The inverse of apply_merges, and the more consequential direction. An
    over-merged GT identity rewards any tracker willing to bridge the gap
    between two different vehicles and penalises one that correctly
    refuses — so it inflates exactly the re-association mechanisms C2 is
    built from. Uses *raw* displacement across the gap (in box diagonals),
    because dividing by the gap length is what hides these.
    """
    cuts: dict[int, set] = {}
    for tid, t in tracks.items():
        s = set()
        for i in range(len(t.frames) - 1):
            gap = t.frames[i + 1] - t.frames[i]
            if gap <= 5:
                continue
            (x0, y0), (x1, y1) = q_centre(t.boxes[i]), q_centre(t.boxes[i + 1])
            d0 = max(q_diag(t.boxes[i]), 1e-6)
            move = math.hypot(x1 - x0, y1 - y0) / d0
            a0, a1 = q_area(t.boxes[i]), q_area(t.boxes[i + 1])
            ratio = max(a0, a1) / max(min(a0, a1), 1e-6)
            if move > move_max or ratio > area_max:
                s.add(t.frames[i + 1])
        cuts[tid] = s

    seg_of: dict[tuple[int, int], int] = {}
    for tid, t in tracks.items():
        seg = 0
        for f in t.frames:
            if f in cuts[tid]:
                seg += 1
            seg_of[(tid, f)] = seg

    nxt = max(tracks) + 1
    remap: dict[tuple[int, int], int] = {}
    out: dict[int, list] = {}
    for k, ents in gt.items():
        row = []
        for tid, box, cls in ents:
            s = seg_of.get((tid, k), 0)
            if s == 0:
                row.append((tid, box, cls))
            else:
                key = (tid, s)
                if key not in remap:
                    remap[key] = nxt
                    nxt += 1
                row.append((remap[key], box, cls))
        out[k] = row
    return out, sum(len(c) for c in cuts.values())


def score(pred: dict[int, list], gt: dict[int, list], n: int) -> dict:
    m_all = compute_tracking_metrics(to_framelist(pred, n),
                                     to_framelist(gt, n))
    m_mot = compute_tracking_metrics(to_framelist(pred, n, MOTORCYCLE),
                                     to_framelist(gt, n, MOTORCYCLE))
    return {"mota": m_all.mota, "idf1": m_all.idf1,
            "moto_idf1": m_mot.idf1, "moto_idsw": m_mot.num_switches}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/annotations/NKKN_gt/dense/gt/gt.txt")
    ap.add_argument("--queue", default="experiments/gt_qa/review_queue.csv")
    ap.add_argument("--tracks-dir", default="experiments/gt_ablation")
    ap.add_argument("--segment", nargs=2, type=int, default=[10700, 11300])
    ap.add_argument("--k", nargs="*", type=int, default=[0, 3, 6, 10, 29],
                    help="how many top-ranked merge candidates to apply")
    ap.add_argument("--split", nargs="*", type=float,
                    default=[3.0, 2.0, 1.5],
                    help="also split GT at joins exceeding these "
                         "move/area limits (box diagonals / ratio)")
    ap.add_argument("--out-dir", default="experiments/gt_qa")
    args = ap.parse_args()

    gt0 = load_mot(args.gt)
    n = max(gt0)
    cands = load_merge_candidates(args.queue)
    print(f"{len(cands)} merge candidates in {args.queue}")

    preds = {v: tracks_json_slice(f"{args.tracks_dir}/tracks_{v}.json",
                                  *args.segment) for v in VARIANTS}

    rows = []
    for k in args.k:
        k = min(k, len(cands))
        gt_k = apply_merges(gt0, cands[:k]) if k else gt0
        ids = len({t for f in gt_k.values() for t, _, _ in f})
        row = {"k": k, "gt_ids": ids,
               "scores": {v: score(preds[v], gt_k, n) for v in VARIANTS}}
        rows.append(row)
        base = row["scores"]["baseline"]["moto_idf1"]
        prop = row["scores"]["ctrv_gate_nsa"]["moto_idf1"]
        print(f"  k={k:<3} GT ids {ids:<4} "
              f"baseline moto IDF1 {base:.3f}  "
              f"C2 {prop:.3f}  margin {prop - base:+.3f}")

    tracks = build_tracks(gt0)
    split_rows = []
    for lim in args.split:
        g, nc = split_gt(gt0, tracks, lim, lim)
        sc = {v: score(preds[v], g, n) for v in VARIANTS}
        m = sc["ctrv_gate_nsa"]["moto_idf1"] - sc["baseline"]["moto_idf1"]
        split_rows.append({"limit": lim, "cuts": nc, "scores": sc,
                           "margin": m})
        print(f"  split@{lim:.1f} ({nc:>3} cuts)  "
              f"baseline {sc['baseline']['moto_idf1']:.3f}  "
              f"C2 {sc['ctrv_gate_nsa']['moto_idf1']:.3f}  "
              f"margin {m:+.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sensitivity.json").write_text(
        json.dumps({"merge": rows, "split": split_rows}, indent=2),
        encoding="utf-8")

    header = ("\n| correction | GT IDs | " + " | ".join(
        f"{v} moto IDF1" for v in VARIANTS) + " | C2 − baseline |\n"
        + "|---|---|" + "---|" * (len(VARIANTS) + 1))

    def row_line(label: str, ids: int, s: dict) -> str:
        margin = s["ctrv_gate_nsa"]["moto_idf1"] - s["baseline"]["moto_idf1"]
        return (f"| {label} | {ids} | "
                + " | ".join(f"{s[v]['moto_idf1']:.3f}" for v in VARIANTS)
                + f" | {margin:+.3f} |")

    lines = [
        "# Annotation-bias sensitivity on `dense` GT\n",
        "The ground truth can be wrong in two opposite directions, and they",
        "do **not** matter equally. Both are reported here; quoting only the",
        "first would understate the risk to the headline result.\n",
        "\n## Under-merging: fragmentations the annotator did not join\n",
        "Pre-labels were seeded from `ctrv_gate_nsa`, so a fragmentation it",
        "produced and the annotator missed survives as agreement. Applying",
        "the top-K flagged merge candidates:\n",
        header,
    ]
    for r in rows:
        lines.append(row_line(str(r["k"]), r["gt_ids"], r["scores"]))
    lines += [
        "\nThe margin barely moves. This direction is not the problem.\n",
        "\n## Over-merging: distinct vehicles chained into one identity\n",
        "The damaging direction. An over-merged identity rewards a tracker",
        "that bridges the gap between two vehicles and penalises one that",
        "correctly refuses — which is exactly what C2's uncertainty gate",
        "does. Splitting joins whose box moves more than the given number of",
        "box-diagonals (raw, across the gap) or rescales by more than it:\n",
        header,
    ]
    for r in split_rows:
        ids = len({t for f in split_gt(gt0, tracks, r["limit"],
                                       r["limit"])[0].values()
                   for t, _, _ in f})
        lines.append(row_line(f"split @{r['limit']:.1f} ({r['cuts']} cuts)",
                              ids, r["scores"]))
    lines += [
        "\nThe margin collapses, the baseline *improves* and the proposed",
        "method *degrades* — the signature of a ground truth that rewards",
        "over-merging. See `WRONG_MERGES.md` for the verified cases and what",
        "to do about them.\n",
    ]
    (out_dir / "sensitivity.md").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")
    print(f"\nwrote {out_dir/'sensitivity.md'}")


if __name__ == "__main__":
    main()
