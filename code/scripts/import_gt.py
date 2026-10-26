"""
code/scripts/import_gt.py
──────────────────────────
Phase 2 GT infrastructure: read MOT 1.1 annotation files (the format
CVAT exports and export_prelabels.py writes) and evaluate tracker output
against them with the self-implemented metrics in code/evaluation/.

Line format accepted (extra trailing columns ignored):
  frame_id, track_id, x, y, w, h, not_ignored, class_id, visibility
frame_id 1-based, class_id 1-based into labels.txt, (x, y) top-left.
Lines with not_ignored == 0 are skipped.

CLI (repo root, venv active):
  # smoke test: a file against itself must give MOTA = IDF1 = 1.0
  python -m code.scripts.import_gt --gt A/gt/gt.txt --pred A/gt/gt.txt

  # evaluate a tracker dump slice against human-corrected GT
  python -m code.scripts.import_gt \
      --gt data/annotations/NKKN_gt/dense/gt/gt.txt \
      --pred-tracks experiments/c2_final/tracks_baseline.json \
      --segment 10700 11300

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from code.evaluation.metrics import (FrameList, compute_detection_metrics,
                                     compute_tracking_metrics)

CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck",
               4: "accident"}


def load_mot(path: str | Path) -> dict[int, list]:
    """Parse a MOT 1.1 file -> {frame_id: [(track_id, box, class_id0)]}
    with box = [x1, y1, x2, y2] and class_id0 the 0-based project class."""
    frames: dict[int, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                raise ValueError(f"{path}:{line_no}: expected >=8 columns, "
                                 f"got {len(parts)}")
            fid, tid = int(parts[0]), int(parts[1])
            x, y, w, h = (float(v) for v in parts[2:6])
            if int(float(parts[6])) == 0:      # ignored region
                continue
            cls0 = int(parts[7]) - 1
            frames[fid].append((tid, [x, y, x + w, y + h], cls0))
    return dict(frames)


def to_framelist(frames: dict[int, list], num_frames: int | None = None,
                 class_id: int | None = None) -> FrameList:
    """Convert load_mot() output to the FrameList the metrics expect
    (dense list over frames 1..num_frames of (track_id, box) tuples),
    optionally keeping a single class."""
    if num_frames is None:
        num_frames = max(frames) if frames else 0
    out: FrameList = []
    for k in range(1, num_frames + 1):
        out.append([(tid, box) for tid, box, cls in frames.get(k, [])
                    if class_id is None or cls == class_id])
    return out


def tracks_json_slice(path: str | Path, start: int, end: int) -> dict[int, list]:
    """Slice a tracks_*.json dump into load_mot()-shaped frames, so a
    variant can be scored against segment GT without re-running anything."""
    with open(path, encoding="utf-8") as f:
        dump = json.load(f)
    assert dump.get("start_frame", 0) == 0, "dump must start at frame 0"
    frames: dict[int, list] = {}
    for k, entries in enumerate(dump["frames"][start:end], start=1):
        frames[k] = [(tid, [x1, y1, x2, y2], cls)
                     for tid, cls, (x1, y1, x2, y2) in entries]
    return frames


def gt_to_det_arrays(gt: dict[int, list], num_frames: int) -> list:
    """GT frames -> per-frame [x1, y1, x2, y2, class_id] for detection metrics."""
    return [np.array([[*box, cls] for _, box, cls in gt.get(k, ())],
                     dtype=float).reshape(-1, 5)
            for k in range(1, num_frames + 1)]


def npz_to_det_arrays(path: str | Path, start: int, end: int,
                      conf_min: float = 0.0) -> list:
    """Cached detector dump -> per-frame [x1, y1, x2, y2, conf, class_id].

    This is the *detector's* raw output, before the tracker's track_thresh
    filter, so it measures the ceiling every tracker row is working under:
    a vehicle the detector never proposes cannot be tracked by anything
    downstream.
    """
    z = np.load(path)
    out = []
    for i in range(start, end):
        d = np.asarray(z[f"f{i}"], dtype=float).reshape(-1, 6)
        out.append(d[d[:, 4] >= conf_min] if conf_min > 0 else d)
    return out


def tracks_to_det_arrays(pred: dict[int, list], num_frames: int) -> list:
    """Tracker output -> detection arrays. Confidence is not carried in the
    dumps, so every box scores 1.0: P/R/F1 stay meaningful, mAP does not
    (a single-point PR curve), and the caller is told so."""
    return [np.array([[*box, 1.0, cls] for _, box, cls in pred.get(k, ())],
                     dtype=float).reshape(-1, 6)
            for k in range(1, num_frames + 1)]


def evaluate_detection(gt: dict[int, list], preds: list, n: int,
                       label: str, show_map: bool = True,
                       note: str = "") -> None:
    """One row of the detection table.

    mAP is only printed when it means something. It is threshold-free by
    definition, so a confidence-filtered prediction set yields a truncated
    PR curve, not a different mAP — printing one per cut-off would invite
    reading the drop as a quality change when it is just truncation.
    """
    g = gt_to_det_arrays(gt, n)
    m = compute_detection_metrics(preds, g)
    map_cell = f"{m.map50:>10.3f}" if show_map else f"{'-':>10}"
    print(f"{label:<28}{m.precision:>10.3f}{m.recall:>10.3f}"
          f"{m.f1:>10.3f}{map_cell}"
          + (f"   {note}" if note else ""))


def evaluate(gt: dict[int, list], pred: dict[int, list]) -> None:
    n = max(max(gt, default=0), max(pred, default=0))
    print(f"{'class':<12}{'MOTA':>8}{'IDF1':>8}{'IDsw':>6}{'frag':>6}"
          f"{'miss':>7}{'FP':>7}")
    rows = [("ALL", None)] + [(name, cid)
                              for cid, name in CLASS_NAMES.items()]
    for label, cid in rows:
        gt_fl = to_framelist(gt, n, cid)
        if not any(gt_fl):
            continue
        m = compute_tracking_metrics(to_framelist(pred, n, cid), gt_fl)
        print(f"{label:<12}{m.mota:>8.3f}{m.idf1:>8.3f}"
              f"{m.num_switches:>6}{m.num_fragmentations:>6}"
              f"{m.num_misses:>7}{m.num_false_positives:>7}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="MOT gt.txt (human GT)")
    ap.add_argument("--pred", help="MOT gt.txt to score against --gt")
    ap.add_argument("--pred-tracks", help="tracks_*.json dump to score")
    ap.add_argument("--segment", nargs=2, type=int, metavar=("START", "END"),
                    help="absolute frame range for --pred-tracks / --det-npz")
    ap.add_argument("--detection", action="store_true",
                    help="report Precision/Recall/F1/mAP@0.5 instead of "
                         "tracking metrics")
    ap.add_argument("--det-npz", help="cached detector dump to score in "
                                      "--detection mode (carries confidences)")
    ap.add_argument("--conf", nargs="*", type=float,
                    default=[0.10, 0.30, 0.50, 0.70],
                    help="confidence cut-offs for P/R/F1 in --detection mode")
    args = ap.parse_args()

    gt = load_mot(args.gt)

    if args.detection:
        n = max(gt) if gt else 0
        print(f"{'source':<28}{'P':>10}{'R':>10}{'F1':>10}{'mAP@.5':>10}")
        if args.det_npz:
            assert args.segment, "--det-npz needs --segment START END"
            # mAP once, over every detection: this is the detector ceiling
            # each tracker row works under. P/R/F1 then per operating point.
            evaluate_detection(
                gt, npz_to_det_arrays(args.det_npz, *args.segment), n,
                "detector (all dets)", note="<- mAP = detector ceiling")
            for c in args.conf:
                evaluate_detection(
                    gt, npz_to_det_arrays(args.det_npz, *args.segment, c), n,
                    f"detector @conf>={c:.2f}", show_map=False)
        if args.pred_tracks:
            assert args.segment, "--pred-tracks needs --segment START END"
            pred = tracks_json_slice(args.pred_tracks, *args.segment)
            evaluate_detection(gt, tracks_to_det_arrays(pred, n), n,
                               Path(args.pred_tracks).stem, show_map=False,
                               note="<- after tracking")
        if not (args.det_npz or args.pred_tracks):
            ap.error("--detection needs --det-npz and/or --pred-tracks")
        return

    if args.pred:
        pred = load_mot(args.pred)
    elif args.pred_tracks:
        assert args.segment, "--pred-tracks needs --segment START END"
        pred = tracks_json_slice(args.pred_tracks, *args.segment)
    else:
        ap.error("need --pred or --pred-tracks")
    evaluate(gt, pred)


if __name__ == "__main__":
    main()
