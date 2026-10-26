"""
code/scripts/verify_external_upstream.py
────────────────────────────────────────
Checks that the vendored OC-SORT / BoT-SORT (code/tracker/external) produce
exactly what the UNMODIFIED upstream code produces on the same cached
detections: same track ids and same boxes (0.1 px, the dump precision) on
every frame. This covers the input adapter, the class metadata, the copied
filterpy helpers, lap -> scipy, np.float -> float and the removed
identity-warp CMC calls.

Upstream runs with its own dependencies (lap, filterpy) installed OUTSIDE the
project environment (`pip install --no-deps --target DIR lap filterpy`).
Two shims are unavoidable and in-memory only: cython_bbox (needs a C
compiler) is replaced by the +1-pixel numpy IoU of
code/tracker/vendor/matching.ious, and fast_reid by an unused stub (ReID off).
TrackEval-style np.float/np.int aliases are restored for BoT-SORT, which
predates NumPy 1.24.

Usage (repo root; SRC = upstream checkout root at the pinned commit):
  python -m code.scripts.verify_external_upstream --tracker ocsort  --src SRC --pydeps DIR
  python -m code.scripts.verify_external_upstream --tracker botsort --src SRC --pydeps DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.scripts.run_external_trackers import (BOTSORT_FRAME_RATE, BOTSORT_MOT17,
                                                DETS, OCSORT_MOT17, OUT_DIR, _box)


def upstream_frames(tracker: str, src: str, dets: list) -> list:
    sys.path.insert(0, src)
    frames = []
    if tracker == "ocsort":
        from trackers.ocsort_tracker.ocsort import OCSort
        t = OCSort(**OCSORT_MOT17)
        for d in dets:
            # upstream 5-column path; img_info == img_size -> scale 1
            out = t.update(np.asarray(d, dtype=float)[:, :5].copy(), (1, 1), (1, 1))
            frames.append(sorted((int(r[4]), tuple(_box(r[:4]))) for r in out))
    else:
        from code.tracker.vendor.matching import ious
        for alias, typ in {"float": float, "int": int}.items():
            if not hasattr(np, alias):
                setattr(np, alias, typ)
        cb = types.ModuleType("cython_bbox")
        cb.bbox_overlaps = ious
        sys.modules["cython_bbox"] = cb
        fr = types.ModuleType("fast_reid.fast_reid_interfece")
        fr.FastReIDInterface = object
        sys.modules["fast_reid"] = types.ModuleType("fast_reid")
        sys.modules["fast_reid.fast_reid_interfece"] = fr
        from tracker.bot_sort import BoTSORT
        t = BoTSORT(SimpleNamespace(**BOTSORT_MOT17, name="verify", ablation=False),
                    frame_rate=BOTSORT_FRAME_RATE)
        for d in dets:
            out = t.update(np.asarray(d, dtype=float)[:, :5].copy(), None)
            frames.append(sorted((int(s.track_id), tuple(_box(s.tlbr))) for s in out))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Vendored vs unmodified upstream tracker.")
    ap.add_argument("--tracker", choices=("ocsort", "botsort"), required=True)
    ap.add_argument("--src", required=True, help="upstream checkout root")
    ap.add_argument("--pydeps", required=True, help="dir with lap + filterpy (pip --target)")
    ap.add_argument("--dets", default=DETS)
    ap.add_argument("--dump", default=None, help="default: OUT_DIR/tracks_<tracker>.json")
    ap.add_argument("--segment", nargs=2, type=int, default=(10700, 11300))
    args = ap.parse_args()

    sys.path.insert(0, args.pydeps)
    npz = np.load(REPO_ROOT / args.dets, allow_pickle=True)
    n = json.loads(str(npz["meta"]))["num_frames"]
    dets = [npz[f"f{i}"] for i in range(n)]

    dump = Path(args.dump or REPO_ROOT / OUT_DIR / f"tracks_{args.tracker}.json")
    vend = [sorted((tid, tuple(box)) for tid, _, box in f)
            for f in json.loads(dump.read_text(encoding="utf-8"))["frames"]]
    up = upstream_frames(args.tracker, args.src, dets)
    assert len(up) == len(vend) == n
    bad = [k for k in range(n) if up[k] != vend[k]]
    seg = [k for k in bad if args.segment[0] <= k < args.segment[1]]
    print(json.dumps({"tracker": args.tracker, "src": args.src, "frames": n,
                      "frames_differing": len(bad), "frames_differing_in_segment": len(seg),
                      "first_differing_frame": bad[0] if bad else None,
                      "boxes_compared": sum(len(f) for f in vend)}, indent=2))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
