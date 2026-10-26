"""
code/scripts/sweep_c1_params.py
────────────────────────────────
Tune C1's DENSE_PEAK row against human GT.

adaptive_controller.PARAM_TABLE ships with "TODO: all numeric values are
initial guesses — tune via ablation". Measured against the `dense` GT,
the shipped guess (conf 0.35, buffer 45) is not merely untuned but net
harmful: MOTA 0.917 -> 0.731, false positives 1388 -> 6255, because the
segment sits in DENSE_PEAK for all 600 frames so track_thresh is pinned
at 0.35 throughout. Recall does improve (miss 873 -> 808); it just does
not come close to paying for the extra detections.

No no-GT proxy could have found this — false positives are only countable
against ground truth — which is why this sweep waited for GT rather than
running on re-id events.

This grid answers the question the ablation cannot: is C1 recoverable at
some other operating point, or does adaptive thresholding lose to a fixed
threshold on this footage regardless? The comparison row is C2 alone
(C1 off), which uses the tracker's fixed track_thresh = 0.5.

Usage (repo root, venv active):
  python -m code.scripts.sweep_c1_params
  python -m code.scripts.sweep_c1_params --conf 0.45 0.50 --buffer 30

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
from code.models import adaptive_controller as ac
from code.pipeline import Pipeline
from code.scripts.import_gt import load_mot, to_framelist

DEFAULT_DETS = "experiments/c2_real_vnweights_full/detections_full13726.npz"
DEFAULT_GT = "data/annotations/NKKN_gt/dense/gt/gt.txt"
CONFIG_C1C2 = "configs/exp2_C1C2.yaml"
CONFIG_C2 = "configs/exp1_C2.yaml"      # C1 off — the row to beat
MOTORCYCLE = 1


def replay(config: str, dets: list, end: int, seg: tuple[int, int],
           dense_conf: float | None = None,
           dense_buffer: int | None = None) -> dict[int, list]:
    """Replay the tracker over [0, end) and return the segment's frames in
    load_mot() shape. Warm-up from frame 0 matters: track identities at
    frame 10700 depend on the whole history before it."""
    if dense_conf is not None or dense_buffer is not None:
        row = ac.PARAM_TABLE[ac.TrafficCondition.DENSE_PEAK]
        ac.PARAM_TABLE[ac.TrafficCondition.DENSE_PEAK] = ac.Params(
            yolo_conf=dense_conf if dense_conf is not None else row.yolo_conf,
            buffer_len=(dense_buffer if dense_buffer is not None
                        else row.buffer_len),
            motion_model=row.motion_model,
        )

    pipe = Pipeline.from_config(config, dry_run_alerts=True,
                                build_detector=False)
    pipe.reset()
    pipe.context_estimator.hour_override = 8      # 08:00 footage, no wall clock
    dummy = np.full((450, 800, 3), 128, dtype=np.uint8)

    a, b = seg
    out: dict[int, list] = {}
    for i in range(end):
        r = pipe.process(dummy, detections=dets[i])
        if a <= i < b:
            out[i - a + 1] = [(int(t.track_id),
                               [float(v) for v in t.tlbr],
                               int(t.class_id))
                              for t in r["tracks_mappo"]]
    return out


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
    ap.add_argument("--conf", nargs="*", type=float,
                    default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    ap.add_argument("--buffer", nargs="*", type=int, default=[45, 30])
    ap.add_argument("--out-dir", default="experiments/c1_sweep")
    args = ap.parse_args()

    a, b = args.segment
    gt = load_mot(args.gt)
    n = max(gt)
    npz = np.load(args.detections)
    dets = [npz[f"f{i}"] for i in range(b)]
    print(f"loaded {len(dets)} detection frames; GT {n} frames, "
          f"{sum(len(v) for v in gt.values())} boxes", flush=True)

    rows = []

    # Reference: C1 off (fixed track_thresh = 0.5). This is what C1 must beat.
    t0 = time.perf_counter()
    ref = score(replay(CONFIG_C2, dets, b, (a, b)), gt, n)
    rows.append({"c1": False, "conf": None, "buffer": None, **ref})
    print(f"  C1 off (fixed 0.50)            "
          f"MOTA {ref['all']['mota']:.3f}  moto IDF1 {ref['moto']['idf1']:.3f}  "
          f"FP {ref['all']['fp']:5d}   [{time.perf_counter()-t0:.0f}s]",
          flush=True)

    shipped = (0.35, 45)
    for buf in args.buffer:
        for conf in args.conf:
            t0 = time.perf_counter()
            r = score(replay(CONFIG_C1C2, dets, b, (a, b),
                             dense_conf=conf, dense_buffer=buf), gt, n)
            rows.append({"c1": True, "conf": conf, "buffer": buf, **r})
            tag = "  <- shipped" if (conf, buf) == shipped else ""
            print(f"  C1 dense conf {conf:.2f} buffer {buf:2d}      "
                  f"MOTA {r['all']['mota']:.3f}  "
                  f"moto IDF1 {r['moto']['idf1']:.3f}  "
                  f"FP {r['all']['fp']:5d}   "
                  f"[{time.perf_counter()-t0:.0f}s]{tag}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(
        json.dumps({"segment": args.segment, "rows": rows}, indent=2),
        encoding="utf-8")

    best = max((r for r in rows if r["c1"]), key=lambda r: r["all"]["mota"])
    lines = [
        "# C1 DENSE_PEAK sweep on human GT\n",
        f"Segment `dense` (frames {a}-{b}), scored against "
        f"`{args.gt}`.\n",
        "The shipped row (conf 0.35, buffer 45) was an untuned guess "
        "(`PARAM_TABLE` TODO). This grid asks whether C1 is recoverable at "
        "another operating point.\n",
        "\n| C1 | dense conf | buffer | MOTA | IDF1 | IDsw | miss | FP | "
        "moto MOTA | moto IDF1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        name = "off" if not r["c1"] else "on"
        conf = "fixed 0.50" if r["conf"] is None else f"{r['conf']:.2f}"
        buf = "30" if r["buffer"] is None else str(r["buffer"])
        lines.append(
            f"| {name} | {conf} | {buf} | {r['all']['mota']:.3f} | "
            f"{r['all']['idf1']:.3f} | {r['all']['idsw']} | "
            f"{r['all']['miss']} | {r['all']['fp']} | "
            f"{r['moto']['mota']:.3f} | {r['moto']['idf1']:.3f} |")
    lines.append(
        f"\nBest C1 point: conf {best['conf']:.2f}, buffer {best['buffer']} "
        f"-> MOTA {best['all']['mota']:.3f} vs {rows[0]['all']['mota']:.3f} "
        f"with C1 off.\n")
    (out_dir / "sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_dir/'sweep.md'}\n      {out_dir/'sweep.json'}")


if __name__ == "__main__":
    main()
