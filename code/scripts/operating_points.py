"""
code/scripts/operating_points.py
────────────────────────────────
Accuracy at detector operating points on the reconciled ROI GT.

  label=<run>/<size>  a paper run's size dir (ROI mask before the detector, state expiry)
  label=old           old cached yolo11_vn@1280 tracks (experiments/gt_ablation,
                      external_trackers), tracked on the full frame, ROI only when scoring
Per segment, class and tracker: MOTA, IDF1, HOTA@0.5, FN, FP, IDSW (repository
evaluator, IoU 0.5). For motorcycles also the miss rate by GT box height and
where the unmatched predictions lie (max IoU with any GT box of any class;
< 0.1 = nothing in the GT there, the detector-level seed caveat), both from
per-frame one-to-one matching at IoU >= 0.5.

    python -m code.scripts.operating_points best_640=<run>/640 best_448=<run>/448 yolo11_vn_1280=old
        -> experiments/operating_point/operating_points.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from code.evaluation.metrics import compute_tracking_metrics
from code.scripts.audit_disagreements import unmatched
from code.scripts.eval_hota_ci import ALPHAS, frame_data, hota, hota_data, iou_matrix
from code.scripts.gt_roi import filter_frames, load_roi
from code.scripts.import_gt import load_mot, to_framelist, tracks_json_slice

A05 = int(np.argmin(np.abs(ALPHAS - 0.5)))
CLASSES = {"motorcycle": 1, "car": 0, "ALL": None}
TRACKERS = ("baseline", "ctrv_gate_nsa", "c1_ctrv_gate", "ocsort_matched")
HEIGHTS = ((0, 20), (20, 40), (40, 10_000))


def old_dump(name: str) -> Path:
    folder = "external_trackers" if name.startswith(("ocsort", "botsort")) else "gt_ablation"
    return Path(f"experiments/{folder}/tracks_{name}.json")


def main() -> None:
    ops = {label: None if path == "old" else Path(path) for label, path in (a.split("=", 1) for a in sys.argv[1:])}
    protocol = json.loads((next(f for f in ops.values() if f).parent / "protocol.json").read_text(encoding="utf-8"))
    for folder in filter(None, ops.values()):
        if json.loads((folder.parent / "protocol.json").read_text(encoding="utf-8"))["segments"] != protocol["segments"]:
            raise ValueError(f"{folder}: different GT segments")
    result = {"note": __doc__.split("\n\n")[1].strip(), "operating_points": {}}
    for op, folder in ops.items():
        if folder is not None and not (folder / "tracks_baseline.json").exists():
            print(f"{op}: dumps not ready, skipped")
            continue
        result["operating_points"][op] = {}
        for seg, item in protocol["segments"].items():
            (start, end), n = item["segment"], item["segment"][1] - item["segment"][0]
            truth = load_mot(item["gt"])
            roi = load_roi(Path(item["gt"]).parent)
            rows = {}
            for t in TRACKERS:
                path = old_dump(t) if folder is None else folder / f"tracks_{t}.json"
                preds = tracks_json_slice(path, start, end)
                if folder is None:
                    preds = filter_frames(preds, roi)        # full-frame tracking, ROI only at scoring
                rows[t] = {}
                for cls, cid in CLASSES.items():
                    gt, pr = to_framelist(truth, n, cid), to_framelist(preds, n, cid)
                    m = compute_tracking_metrics(pr, gt)
                    rows[t][cls] = dict(mota=m.mota, idf1=m.idf1, idsw=m.num_switches, fn=m.num_misses,
                                        fp=m.num_false_positives,
                                        hota05=float(hota(hota_data(frame_data(gt, pr)))["HOTA"][A05]))
                gt, pr, gt_all = to_framelist(truth, n, 1), to_framelist(preds, n, 1), to_framelist(truth, n)
                miss = {f"{lo}-{hi}": [0, 0] for lo, hi in HEIGHTS}
                where = {"nothing (<0.1)": 0, "partial (0.1-0.5)": 0, "overlaps (>=0.5)": 0}
                for k in range(n):
                    g_left = set(unmatched(gt[k], pr[k]))
                    for i, (_, b) in enumerate(gt[k]):
                        key = next(f"{lo}-{hi}" for lo, hi in HEIGHTS if lo <= b[3] - b[1] < hi)
                        miss[key][0] += 1
                        miss[key][1] += i in g_left
                    for j in unmatched(pr[k], gt[k]):
                        o = iou_matrix([pr[k][j][1]], [b for _, b in gt_all[k]]).max() if gt_all[k] else 0.0
                        where["nothing (<0.1)" if o < 0.1 else "partial (0.1-0.5)" if o < 0.5 else "overlaps (>=0.5)"] += 1
                rows[t]["motorcycle"]["miss_by_height"] = {
                    k: dict(gt=v[0], missed=v[1], rate=round(v[1] / max(1, v[0]), 3)) for k, v in miss.items()}
                rows[t]["motorcycle"]["unmatched_pred_location"] = where
            result["operating_points"][op][seg] = rows
            for t in TRACKERS:
                r = rows[t]["motorcycle"]
                mh = " ".join(f"{k}:{v['rate']:.2f}" for k, v in r["miss_by_height"].items())
                w = r["unmatched_pred_location"]
                print(f"{op:<15}{seg:<9}{t:<15}IDF1 {r['idf1']:.3f} HOTA.5 {r['hota05']:.3f} FN {r['fn']:>5} "
                      f"FP {r['fp']:>5} IDSW {r['idsw']:>3} | miss by height {mh} | "
                      f"FP where GT empty {w['nothing (<0.1)'] / max(1, sum(w.values())):.2f}")
    out = Path("experiments/operating_point")
    out.mkdir(parents=True, exist_ok=True)
    (out / "operating_points.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"-> {out / 'operating_points.json'}")


if __name__ == "__main__":
    main()
