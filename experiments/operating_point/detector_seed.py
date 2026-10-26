"""Detector-level seed check for the operating-point table (motorcycle class, ByteTrack
baseline tracker on each detector, one-to-one matching at IoU >= 0.5 per frame).

Every NKKN GT was pre-labelled from trackers on yolo11_vn@1280 detections, so an object
only another detector finds was never shown to a reviewer, and yolo11_vn's misses are
understated.
  extras_by_height  unmatched predictions by predicted box height and by where they lie
                    (max IoU with any GT box of any class; < 0.1 = the GT has nothing there)
  miss_by_height    miss rate by GT box height on the GT as is ("gt") and on the GT plus
                    the reference point's extras that lie where the GT has nothing
                    ("gt_plus_ref", i.e. as if all of them were real)
No audit share exists for these extras (the blind audit sampled the yolo11_vn@1280 trackers),
so the two columns are bounds, not a corrected value. Kept outside code/ while paper runs
are pending; move to code/scripts afterwards.
    PYTHONPATH=. python experiments/operating_point/detector_seed.py ref=best_1280 \
        best_448=<run>/448 ... best_1280=<run>/1280 yolo11_vn_1280=<run>/1280|old
"""
import json
import sys
from pathlib import Path

from code.scripts.audit_disagreements import unmatched
from code.scripts.eval_hota_ci import iou_matrix
from code.scripts.gt_roi import filter_frames, load_roi
from code.scripts.import_gt import load_mot, to_framelist, tracks_json_slice
from code.scripts.operating_points import HEIGHTS, old_dump

args = dict(a.split("=", 1) for a in sys.argv[1:])
ref = args.pop("ref")
ops = {label: None if path == "old" else Path(path) for label, path in args.items()}
protocol = json.loads((next(f for f in ops.values() if f).parent / "protocol.json").read_text(encoding="utf-8"))


def height_bin(box):
    return next(f"{lo}-{hi}" for lo, hi in HEIGHTS if lo <= box[3] - box[1] < hi)


def where(box, gt_all):
    o = iou_matrix([box], [b for _, b in gt_all]).max() if gt_all else 0.0
    return "nothing" if o < 0.1 else "partial" if o < 0.5 else "overlaps"


result = {"note": __doc__.split("\n\n")[1].strip(), "reference": ref, "tracker": "baseline", "segments": {}}
for seg, item in protocol["segments"].items():
    (start, end), n = item["segment"], item["segment"][1] - item["segment"][0]
    truth, roi = load_mot(item["gt"]), load_roi(Path(item["gt"]).parent)
    gt, gt_all = to_framelist(truth, n, 1), to_framelist(truth, n)
    preds = {}
    for op, folder in ops.items():
        p = tracks_json_slice(old_dump("baseline") if folder is None else folder / "tracks_baseline.json", start, end)
        preds[op] = to_framelist(filter_frames(p, roi) if folder is None else p, n, 1)
    # the reference point's extras where the GT has nothing, added as motorcycle GT
    added = [[preds[ref][k][j] for j in unmatched(preds[ref][k], gt[k])
              if where(preds[ref][k][j][1], gt_all[k]) == "nothing"] for k in range(n)]
    rows = {}
    for op in ops:
        extras = {f"{lo}-{hi}": {"nothing": 0, "partial": 0, "overlaps": 0} for lo, hi in HEIGHTS}
        miss = {g: {f"{lo}-{hi}": [0, 0] for lo, hi in HEIGHTS} for g in ("gt", "gt_plus_ref")}
        for k in range(n):
            pr = preds[op][k]
            for j in unmatched(pr, gt[k]):
                extras[height_bin(pr[j][1])][where(pr[j][1], gt_all[k])] += 1
            for g, boxes in (("gt", gt[k]), ("gt_plus_ref", gt[k] + added[k])):
                left = set(unmatched(boxes, pr))
                for i, (_, b) in enumerate(boxes):
                    miss[g][height_bin(b)][0] += 1
                    miss[g][height_bin(b)][1] += i in left
        rows[op] = dict(
            extras_by_height={h: dict(v, share_nothing=round(v["nothing"] / max(1, sum(v.values())), 3))
                              for h, v in extras.items()},
            miss_by_height={g: {h: dict(gt=c[0], missed=c[1], rate=round(c[1] / max(1, c[0]), 3))
                                for h, c in m.items()} for g, m in miss.items()})
        e, m = rows[op]["extras_by_height"], rows[op]["miss_by_height"]
        print(f"{seg:<9}{op:<15} extras where GT empty "
              + " ".join(f"{h}:{e[h]['share_nothing']:.2f}({sum(e[h][w] for w in ('nothing', 'partial', 'overlaps'))})"
                         for h in e)
              + " | miss gt " + " ".join(f"{m['gt'][h]['rate']:.2f}" for h in e)
              + " | gt+ref " + " ".join(f"{m['gt_plus_ref'][h]['rate']:.2f}" for h in e))
    result["segments"][seg] = dict(added_boxes=sum(map(len, added)), operating_points=rows)
out = Path("experiments/operating_point/detector_seed.json")
out.write_text(json.dumps(result, indent=1), encoding="utf-8")
print(f"-> {out}")
