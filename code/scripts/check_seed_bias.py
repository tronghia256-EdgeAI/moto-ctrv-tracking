"""
code/scripts/check_seed_bias.py
───────────────────────────────
Does a reviewed GT favour the tracker it was pre-labelled from?

Both reviewed NKKN GTs inherit their box geometry from one tracker's
filtered output (dense: ctrv_gate_nsa, mixed: baseline); the reviewers fixed
identities, not geometry. Every frozen dump in experiments/gt_ablation is
scored on both, motorcycles only, in each review's own scope (ROI or full
frame):
  published  boxes as dumped, the repository metric (IoU 0.5) plus HOTA
  anchored   every GT and predicted box replaced by the cached detection it
             overlaps one-to-one with IoU >= SNAP_IOU, so both sides carry the
             detector's geometry and only association differs

    python -m code.scripts.check_seed_bias
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from code.evaluation.metrics import compute_tracking_metrics
from code.scripts.eval_hota_ci import ALPHAS, exact_box_share, frame_data, hota, hota_data, iou_matrix
from code.scripts.gt_roi import filter_frames, load_roi
from code.scripts.import_gt import load_mot, to_framelist
from code.scripts.sweep_c2_sensitivity import DEFAULT_NPZ, moto_framelist

MOTO, SNAP_IOU = 1, 0.5
ALPHA_05 = int(np.argmin(np.abs(ALPHAS - 0.5)))
DUMPS = Path("experiments/gt_ablation")
OUT = Path("experiments/seed_bias")
# name: (reviewed run, absolute segment, tracker the pre-labels came from). The final
# ROI exports listed in docs/GT_COMPLETION_20261012.md; freeflow's pre-labels came
# from export_prelabels.py, i.e. the ctrv_gate_nsa dump.
REVIEWS = {
    "dense": ("experiments/gt_qa/mixed_reviewed/20261012T112827Z_7b620b49", (10700, 11300), "ctrv_gate_nsa"),
    "freeflow": ("experiments/gt_qa/mixed_reviewed/20261012T112802Z_e7771a10", (6400, 7000), "ctrv_gate_nsa"),
    "mixed": ("experiments/gt_qa/mixed_reviewed/20261012T094827Z_7779f93a", (2500, 3100), "baseline"),
}


def snap(fl: list, dets, start: int) -> tuple[list, int]:
    """Each box -> the detection it overlaps one-to-one (IoU >= SNAP_IOU).
    Returns the snapped FrameList and the number of boxes left unsnapped."""
    out, unsnapped = [], 0
    for k, frame in enumerate(fl):
        boxes, new = dets[f"f{start + k}"][:, :4], list(frame)
        if frame and len(boxes):
            iou = iou_matrix([b for _, b in frame], boxes)
            for r, c in zip(*linear_sum_assignment(-iou)):
                if iou[r, c] >= SNAP_IOU:
                    new[r] = (frame[r][0], boxes[c].tolist())
        unsnapped += sum(a is b for a, b in zip(new, frame))
        out.append(new)
    return out, unsnapped


def score(gt_fl: list, pr_fl: list) -> dict:
    m = compute_tracking_metrics(pr_fl, gt_fl)
    d = frame_data(gt_fl, pr_fl)
    h = hota(hota_data(d))
    return dict(idf1=m.idf1, idsw=m.num_switches, mota=m.mota, exact_box_share=exact_box_share(d),
                hota_alpha05=float(h["HOTA"][ALPHA_05]), hota=float(np.mean(h["HOTA"])),
                assa=float(np.mean(h["AssA"])))


def main() -> None:
    dets = np.load(DEFAULT_NPZ, allow_pickle=True)
    result = {"snap_iou": SNAP_IOU, "detections": DEFAULT_NPZ, "dumps": str(DUMPS)}
    for name, (run, (start, end), seed) in REVIEWS.items():
        roi = load_roi(Path(run))
        truth = filter_frames(load_mot(Path(run) / "gt_reviewed.txt"), roi)
        gt_fl = to_framelist(truth, end - start, MOTO)
        gt_snap, gt_left = snap(gt_fl, dets, start)
        rows = {}
        for dump in sorted(DUMPS.glob("tracks_*.json")):
            pr_fl = moto_framelist(dump, start, end, roi)
            pr_snap, pr_left = snap(pr_fl, dets, start)
            rows[dump.stem.removeprefix("tracks_")] = dict(
                published=score(gt_fl, pr_fl), anchored=score(gt_snap, pr_snap),
                pred_boxes=sum(map(len, pr_fl)), pred_unsnapped=pr_left)
        result[name] = dict(run=run, segment=[start, end], seeded_from=seed,
                            roi=bool(roi and roi["enabled"]), gt_boxes=sum(map(len, gt_fl)),
                            gt_unsnapped=gt_left, variants=rows)
        print(f"\n{name} (seeded from {seed}; {result[name]['gt_boxes']} GT boxes, {gt_left} unsnapped)")
        print(f"{'variant':<15}{'exact':>7}{'IDF1':>8}{'IDSW':>6}{'HOTA.5':>8} | "
              f"{'exact':>7}{'IDF1':>8}{'IDSW':>6}{'HOTA.5':>8}{'HOTA':>8}{'AssA':>8}")
        for v, r in rows.items():
            p, a = r["published"], r["anchored"]
            print(f"{v:<15}{p['exact_box_share']:>7.3f}{p['idf1']:>8.4f}{p['idsw']:>6}{p['hota_alpha05']:>8.4f} | "
                  f"{a['exact_box_share']:>7.3f}{a['idf1']:>8.4f}{a['idsw']:>6}{a['hota_alpha05']:>8.4f}"
                  f"{a['hota']:>8.4f}{a['assa']:>8.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "seed_bias.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
