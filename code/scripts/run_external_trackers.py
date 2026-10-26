"""
code/scripts/run_external_trackers.py
─────────────────────────────────────
Replay the SAME cached detections every ByteTrack variant in
experiments/gt_ablation/ received through two external trackers, each with
its published MOT17 default hyperparameters (no tuning on this data):

  ocsort  - OC-SORT, noahcao/OC_SORT @2d34c67 (code/tracker/external/ocsort)
  botsort - BoT-SORT without ReID and without camera-motion compensation
            (fixed camera), NirAharon/BoT-SORT @2519854
            (code/tracker/external/botsort)

Input and frame range match compare_motion_models.run_replay(): every frame
of detections_full13726.npz from frame 0, so each tracker gets the same
10 700-frame warm-up before the dense GT segment. Dumps follow the
gt_ablation schema (tracks_<name>.json, per frame a list of
(track_id, class_id, [x1, y1, x2, y2]) rounded to 0.1 px), so any scorer of
the gt_ablation dumps reads them unchanged.

Usage (repo root):
  python -m code.scripts.run_external_trackers
  python -m code.scripts.run_external_trackers \
      --trackers ocsort_matched,ocsort_byte,botsort_matched  # ByteTrack's thresholds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DETS = "experiments/c2_real_vnweights_full/detections_full13726.npz"
OUT_DIR = "experiments/external_trackers"

# OC-SORT MOT17: tools/run_ocsort.py builds OCSort(det_thresh=track_thresh,
# iou_threshold, asso_func, delta_t, inertia) from utils/args.py defaults;
# max_age and min_hits keep the OCSort() defaults; use_byte is off.
OCSORT_MOT17 = dict(det_thresh=0.6, max_age=30, min_hits=3, iou_threshold=0.3,
                    delta_t=3, asso_func="iou", inertia=0.2, use_byte=False)

# BoT-SORT MOT17: tools/track.py --default-parameters (generic values; the
# per-sequence MOT17 overrides do not apply to other footage), fps = 30.
BOTSORT_MOT17 = dict(track_high_thresh=0.6, track_low_thresh=0.1,
                     new_track_thresh=0.7, track_buffer=30, match_thresh=0.8,
                     proximity_thresh=0.5, appearance_thresh=0.25,
                     with_reid=False, mot20=False, cmc_method="none")
BOTSORT_FRAME_RATE = 30

# Threshold-matched controls: the detection-score thresholds of the
# repository's ByteTrack (first association > 0.5, second stage 0.1-0.5,
# birth >= 0.6), everything else as above. OC-SORT has a single threshold for
# association and birth, so it gets its own BYTE second stage with that
# threshold at ByteTrack's association value (*_matched, 0.5) or at its
# birth value (*_byte, 0.6, which is also OC-SORT's published value).
OCSORT_MATCHED = {**OCSORT_MOT17, "det_thresh": 0.5, "use_byte": True}
OCSORT_BYTE = {**OCSORT_MOT17, "use_byte": True}
BOTSORT_MATCHED = {**BOTSORT_MOT17, "track_high_thresh": 0.5,
                   "new_track_thresh": 0.6}
PARAMS = {"ocsort": OCSORT_MOT17, "ocsort_matched": OCSORT_MATCHED,
          "ocsort_byte": OCSORT_BYTE,
          "botsort": BOTSORT_MOT17, "botsort_matched": BOTSORT_MATCHED}

SOURCES = {
    "ocsort": "https://github.com/noahcao/OC_SORT/tree/"
              "2d34c67d58f89b0762ce912c38f94746175e2903",
    "botsort": "https://github.com/NirAharon/BoT-SORT/tree/"
               "251985436d6712aaf682aaaf5f71edb4987224bd",
}


def _box(v) -> list:
    return [round(float(x), 1) for x in v]


def run(name: str, dets: list) -> list:
    """One pass over all frames; returns per-frame (id, class, box) lists."""
    frames = []
    if name not in PARAMS:
        raise ValueError(f"unknown tracker {name!r}")
    if name.startswith("ocsort"):
        from code.tracker.external.ocsort import OCSort
        tracker = OCSort(**PARAMS[name])
        for d in dets:
            out = tracker.update(np.asarray(d, dtype=float))
            frames.append([(int(r[4]), int(r[5]), _box(r[:4])) for r in out])
    else:
        from code.tracker.external.botsort import BoTSORT
        tracker = BoTSORT(SimpleNamespace(**PARAMS[name]),
                          frame_rate=BOTSORT_FRAME_RATE)
        for d in dets:
            out = tracker.update(np.asarray(d, dtype=float))
            frames.append([(int(t.track_id), int(t.class_id), _box(t.tlbr))
                           for t in out])
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay cached detections through OC-SORT and BoT-SORT.")
    ap.add_argument("--dets", default=DETS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--trackers", default="ocsort,botsort")
    args = ap.parse_args()

    npz = np.load(REPO_ROOT / args.dets, allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    dets = [npz[f"f{i}"] for i in range(meta["num_frames"])]
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in [t.strip() for t in args.trackers.split(",")]:
        frames = run(name, dets)
        assert len(frames) == meta["num_frames"]
        params = PARAMS[name] if name.startswith("ocsort") else {
            **PARAMS[name], "frame_rate": BOTSORT_FRAME_RATE}
        with open(out_dir / f"tracks_{name}.json", "w", encoding="utf-8") as f:
            json.dump({**meta, "variant": name, "detections": args.dets,
                       "params": params, "source": SOURCES[name.split("_")[0]],
                       "frames": frames}, f)
        print(f"{name}: {len(frames)} frames -> {out_dir / f'tracks_{name}.json'}",
              flush=True)


if __name__ == "__main__":
    main()
