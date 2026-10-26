"""
code/scripts/render_qualitative.py
──────────────────────────────────
Qualitative figure for the paper: one ground-truth motorcycle identity,
followed across the annotated segment, tracked by the baseline (top row)
and by C2 (bottom row).

Example selection is grounded in ground truth rather than in a proxy. For
every GT motorcycle identity we count how many distinct tracker IDs cover
it (best IoU >= 0.5 per frame) under each configuration, and keep the
identities the baseline splits while C2 keeps whole. On the annotated
segment there are 111 such identities; the default figure uses the most
extreme, where the baseline spends 11 IDs on one motorcycle over 353
frames and C2 spends 1.

An earlier attempt picked the example from the re-id *event* list instead
and produced a case where both configurations fragmented — the event
proxy matches on frame proximity, not on vehicle identity, so it cannot
tell "C2 fixed this" from "C2 broke it somewhere else". GT is what makes
the claim checkable.

Usage (repo root, venv active):
  python -m code.scripts.render_qualitative
  python -m code.scripts.render_qualitative --gt-id 85 --cols 5

Writes paper/figures/fig_qualitative_reid.png.

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.scripts.gt_review_queue import _iou
from code.scripts.import_gt import load_mot, tracks_json_slice

SEGMENT = (10700, 11300)
MOTORCYCLE = 1
IOU_MATCH = 0.5
BASE_COLOR = (60, 60, 235)      # BGR: baseline, red
C2_COLOR = (70, 200, 70)        # BGR: proposed, green


def cover_map(gt: dict, pred: dict) -> dict[int, list]:
    """GT id -> [(frame, matched pred id, gt box)] for motorcycles."""
    out: dict[int, list] = defaultdict(list)
    for k in sorted(gt):
        for gid, gb, gc in gt[k]:
            if gc != MOTORCYCLE:
                continue
            best_t, best_v = None, 0.0
            for t, pb, _ in pred.get(k, ()):
                v = _iou(tuple(gb), tuple(pb))
                if v > best_v:
                    best_t, best_v = t, v
            if best_v >= IOU_MATCH:
                out[gid].append((k, best_t, gb))
    return out


def pick_frames(rows: list, n: int) -> list:
    """A tight window around one baseline identity switch.

    Spanning the whole track reads badly: these motorcycles are occluded
    for 20-30 frames at a time, so evenly spaced panels sit far apart and
    the vehicle changes scale and pose between them — the reader cannot
    confirm by eye that it is the same object, which is the entire point
    of the figure. Sampling consecutive matched frames around a single
    switch keeps continuity visible.

    Among candidate switches we take the one with the largest boxes: a
    foreground motorcycle is legible at print size, a 15 px one is not.
    """
    switches = [i for i in range(1, len(rows)) if rows[i][1] != rows[i - 1][1]]
    if not switches:
        return rows[:n]
    diag = lambda box: max(box[2] - box[0], box[3] - box[1])
    i = max(switches, key=lambda j: diag(rows[j][2]))
    half = n // 2
    lo = max(0, min(i - half, len(rows) - n))
    return rows[lo:lo + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/annotations/NKKN_gt/dense/gt/gt.txt")
    ap.add_argument("--images",
                    default="data/annotations/prelabels/dense/img1")
    ap.add_argument("--tracks-dir", default="experiments/gt_ablation")
    ap.add_argument("--gt-id", type=int, default=None,
                    help="GT identity to show (default: most-split one)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--pad", type=int, default=34)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out", default="paper/figures/fig_qualitative_reid.png")
    args = ap.parse_args()

    a, b = SEGMENT
    gt = load_mot(args.gt)
    base = tracks_json_slice(f"{args.tracks_dir}/tracks_baseline.json", a, b)
    prop = tracks_json_slice(f"{args.tracks_dir}/tracks_ctrv_gate_nsa.json",
                             a, b)
    mb, mc = cover_map(gt, base), cover_map(gt, prop)

    cands = [(g, len({t for _, t, _ in mb[g]}), len({t for _, t, _ in mc.get(g, [])}))
             for g in mb if len({t for _, t, _ in mc.get(g, [])}) == 1]
    cands = [c for c in cands if c[1] >= 2]
    if not cands:
        raise SystemExit("no GT identity is split by baseline but kept by C2")
    print(f"{len(cands)} GT motorcycle identities: baseline splits, C2 keeps")

    gid = args.gt_id or max(cands, key=lambda c: c[1])[0]
    n_base = len({t for _, t, _ in mb[gid]})
    print(f"showing GT id {gid}: baseline {n_base} ids, C2 1 id, "
          f"{len(mb[gid])} matched frames")

    rows = pick_frames(mb[gid], args.cols)
    tiles_top, tiles_bot = [], []
    for k, bid, gbox in rows:
        x1, y1 = max(0, int(gbox[0] - args.pad)), max(0, int(gbox[1] - args.pad))
        x2, y2 = int(gbox[2] + args.pad), int(gbox[3] + args.pad)
        img = cv2.imread(f"{args.images}/{k:06d}.jpg")
        if img is None:
            raise SystemExit(f"missing frame {k}")
        cid = next((t for kk, t, _ in mc[gid] if kk == k), None)
        for pred, tid, color, bucket in (
                (base, bid, BASE_COLOR, tiles_top),
                (prop, cid, C2_COLOR, tiles_bot)):
            im = img.copy()
            for t, pb, _ in pred.get(k, ()):
                if t != tid:
                    continue
                cv2.rectangle(im, (int(pb[0]), int(pb[1])),
                              (int(pb[2]), int(pb[3])), color, 1)
            crop = im[y1:min(y2, im.shape[0]), x1:min(x2, im.shape[1])]
            crop = cv2.resize(crop, None, fx=args.scale, fy=args.scale,
                              interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, 18), color, -1)
            cv2.putText(crop, f"f{k}  ID {tid}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            bucket.append(crop)

    h = min(t.shape[0] for t in tiles_top + tiles_bot)
    w = min(t.shape[1] for t in tiles_top + tiles_bot)
    fit = lambda ts: [cv2.resize(t, (w, h)) for t in ts]
    fig = cv2.vconcat([cv2.hconcat(fit(tiles_top)),
                       cv2.hconcat(fit(tiles_bot))])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), fig)
    print(f"wrote {out}  ({fig.shape[1]}x{fig.shape[0]})")


if __name__ == "__main__":
    main()
