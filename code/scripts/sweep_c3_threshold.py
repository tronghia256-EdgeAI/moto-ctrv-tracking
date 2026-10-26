"""
code/scripts/sweep_c3_threshold.py
──────────────────────────────────
Tune C3's τ_mappo against human GT.

track_reliability.THRESHOLD_MAPPO ships at 0.40 with a "TODO: tune via
ablation study" — but the lowest reliability score any track reaches on
the `dense` segment is 0.435, so the recall branch never rejects
anything. `tracks_c1c2c3` comes out byte-identical to `tracks_c1_ctrv_gate`
on all 13726 frames: C3's MAPPO half is inert, not merely untuned.

This sweep walks τ_mappo up through the observed score distribution
(p1 0.453, p5 0.621, median 0.854) and measures what filtering actually
buys on the branch it governs. A reliability filter can only trade misses
for false positives, so the question is whether any threshold trades them
favourably, or whether every level of filtering costs more recall than the
precision it returns.

SCOPE — read before quoting any number from here. This measures C3 on the
**MAPPO branch only**. C3's headline claim is dual-task: high τ_acc for
accident-alert precision, lower τ_mappo for MAPPO recall. The accident
branch cannot be evaluated on this clip at all, because the clip contains
no accidents. Nothing here supports or refutes the precision half of the
contribution.

Usage (repo root, venv active):
  python -m code.scripts.sweep_c3_threshold
  python -m code.scripts.sweep_c3_threshold --tau 0.45 0.55 0.65

Writes <out-dir>/sweep.md and <out-dir>/sweep.json.

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.evaluation.metrics import compute_tracking_metrics
from code.models import track_reliability as tr
from code.pipeline import Pipeline
from code.scripts.import_gt import load_mot, to_framelist

DEFAULT_DETS = "experiments/c2_real_vnweights_full/detections_full13726.npz"
DEFAULT_GT = "data/annotations/NKKN_gt/dense/gt/gt.txt"
CONFIG_C1C2C3 = "configs/exp3_C1C2C3.yaml"
MOTORCYCLE = 1


def replay(dets: list, end: int, seg: tuple[int, int],
           tau_mappo: float) -> tuple[dict[int, list], int, int]:
    """Replay with τ_mappo patched; return (segment frames, kept, total).

    ReliabilityScore sets its flags in __post_init__ by reading the module
    constants, so patching the module attribute is enough — no per-track
    plumbing needed.
    """
    tr.THRESHOLD_MAPPO = tau_mappo

    pipe = Pipeline.from_config(CONFIG_C1C2C3, dry_run_alerts=True,
                                build_detector=False)
    pipe.reset()
    pipe.context_estimator.hour_override = 8
    dummy = np.full((450, 800, 3), 128, dtype=np.uint8)

    a, b = seg
    out: dict[int, list] = {}
    kept = total = 0
    for i in range(end):
        r = pipe.process(dummy, detections=dets[i])
        if a <= i < b:
            out[i - a + 1] = [(int(t.track_id),
                               [float(v) for v in t.tlbr],
                               int(t.class_id))
                              for t in r["tracks_mappo"]]
            total += len(r["scores"])
            kept += len(r["tracks_mappo"])
    return out, kept, total


def diagnose(dets: list, end: int, seg: tuple[int, int],
             gt: dict[int, list]) -> dict:
    """Can the reliability score tell a true track from a false one at all?

    Every threshold question is downstream of this one. Label each scored
    track TP/FP by IoU>=0.5 against GT, then measure how well the score
    ranks TPs above FPs (ROC AUC via the Mann-Whitney rank statistic —
    no sklearn dependency). AUC 0.5 means the score is noise and no
    threshold can help; the sweep table then has a mechanical explanation
    rather than just a shape.
    """
    from code.scripts.gt_review_queue import _iou

    tr.THRESHOLD_MAPPO = 0.0          # score everything, filter nothing
    pipe = Pipeline.from_config(CONFIG_C1C2C3, dry_run_alerts=True,
                                build_detector=False)
    pipe.reset()
    pipe.context_estimator.hour_override = 8
    dummy = np.full((450, 800, 3), 128, dtype=np.uint8)

    a, b = seg
    s_list, y_list = [], []
    for i in range(end):
        r = pipe.process(dummy, detections=dets[i])
        if i < a:
            continue
        gts = gt.get(i - a + 1, [])
        used: set[int] = set()
        for t in r["tracks_mappo"]:
            box = tuple(float(v) for v in t.tlbr)
            best, bj = 0.0, -1
            for j, (_, gb, _) in enumerate(gts):
                if j in used:
                    continue
                v = _iou(box, tuple(gb))
                if v > best:
                    best, bj = v, j
            if best >= 0.5:
                used.add(bj)
            sc = r["scores"].get(t.track_id)
            if sc is not None:
                s_list.append(sc.total)
                y_list.append(1 if best >= 0.5 else 0)

    s, y = np.asarray(s_list), np.asarray(y_list)
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

    enrich = {}
    for tau in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80):
        drop = s < tau
        if drop.sum():
            enrich[tau] = {"dropped": int(drop.sum()),
                           "fp_share": float((1 - y)[drop].mean())}
    return {"n": len(s), "fp_base_rate": float((1 - y).mean()), "auc": auc,
            "tp_mean": float(s[y == 1].mean()),
            "fp_mean": float(s[y == 0].mean()), "enrichment": enrich}


def score(pred: dict[int, list], gt: dict[int, list], n: int) -> dict:
    row = {}
    for label, cid in (("all", None), ("moto", MOTORCYCLE)):
        m = compute_tracking_metrics(to_framelist(pred, n, cid),
                                     to_framelist(gt, n, cid))
        row[label] = {"mota": m.mota, "idf1": m.idf1,
                      "idsw": m.num_switches, "frag": m.num_fragmentations,
                      "miss": m.num_misses, "fp": m.num_false_positives}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default=DEFAULT_DETS)
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--segment", nargs=2, type=int, default=[10700, 11300])
    ap.add_argument("--tau", nargs="*", type=float,
                    default=[0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80])
    ap.add_argument("--diagnose", action="store_true",
                    help="also measure whether the score separates TP "
                         "from FP at all (ROC AUC)")
    ap.add_argument("--out-dir", default="experiments/c3_sweep")
    args = ap.parse_args()

    a, b = args.segment
    gt = load_mot(args.gt)
    n = max(gt)
    npz = np.load(args.detections)
    dets = [npz[f"f{i}"] for i in range(b)]
    baseline_tau = tr.THRESHOLD_MAPPO
    print(f"shipped tau_mappo = {baseline_tau}; sweeping {args.tau}",
          flush=True)

    rows = []
    diag = None
    try:
        if args.diagnose:
            diag = diagnose(dets, b, (a, b), gt)
            print(f"  discriminability: ROC AUC {diag['auc']:.3f} "
                  f"(0.5 = no signal) | FP base rate "
                  f"{100*diag['fp_base_rate']:.1f}% | "
                  f"score mean TP {diag['tp_mean']:.3f} vs FP "
                  f"{diag['fp_mean']:.3f}", flush=True)
            for tau, e in diag["enrichment"].items():
                print(f"    tau {tau:.2f}: drops {e['dropped']:5d} tracks, "
                      f"{100*e['fp_share']:4.1f}% of them FP", flush=True)
        for tau in args.tau:
            t0 = time.perf_counter()
            pred, kept, total = replay(dets, b, (a, b), tau)
            r = score(pred, gt, n)
            pct = 100.0 * kept / max(total, 1)
            rows.append({"tau": tau, "kept": kept, "total": total,
                         "kept_pct": pct, **r})
            print(f"  tau {tau:.2f}  kept {pct:5.1f}%  "
                  f"MOTA {r['all']['mota']:.3f}  "
                  f"moto IDF1 {r['moto']['idf1']:.3f}  "
                  f"miss {r['all']['miss']:5d}  FP {r['all']['fp']:5d}   "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    finally:
        tr.THRESHOLD_MAPPO = baseline_tau   # never leak into another import

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(
        json.dumps({"segment": args.segment, "rows": rows,
                    "diagnostics": diag}, indent=2),
        encoding="utf-8")

    best = max(rows, key=lambda r: r["all"]["mota"])
    lines = [
        "# C3 tau_mappo sweep on human GT\n",
        f"Segment `dense` (frames {a}-{b}) vs `{args.gt}`.\n",
        "**MAPPO branch only.** The accident branch — where C3's precision "
        "claim lives — cannot be tested on this clip, which contains no "
        "accidents.\n",
        "\n| tau_mappo | tracks kept | MOTA | IDF1 | IDsw | miss | FP | "
        "moto IDF1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tag = " (shipped)" if r["tau"] == baseline_tau else ""
        lines.append(
            f"| {r['tau']:.2f}{tag} | {r['kept_pct']:.1f}% | "
            f"{r['all']['mota']:.3f} | {r['all']['idf1']:.3f} | "
            f"{r['all']['idsw']} | {r['all']['miss']} | {r['all']['fp']} | "
            f"{r['moto']['idf1']:.3f} |")
    lines.append(
        f"\nBest MOTA: tau {best['tau']:.2f} "
        f"({best['all']['mota']:.3f}, keeping {best['kept_pct']:.1f}%).\n")
    (out_dir / "sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_dir/'sweep.md'}\n      {out_dir/'sweep.json'}")


if __name__ == "__main__":
    main()
