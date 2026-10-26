"""
code/scripts/audit_disagreements.py
───────────────────────────────────
Blind human audit of where a tracker and a reviewed GT disagree.

A reviewed GT holds its seeding tracker's boxes and identities, each shown to
the reviewer. What only the OTHER tracker outputs was never shown, so its
absence from the GT is a default, not a judgement (experiments/seed_bias).

sample    membership: runs (consecutive frames of one ID) of
            extra    tracker box with no GT box at IoU >= 0.5 (an FP if the GT is right)
            missing  GT box with no tracker box at IoU >= 0.5 (an FN if the GT is right)
          as numbered crops of each run's middle frame: red = audited box,
          green = the other GT boxes. Verdicts in verdicts.csv:
            Y  a real motorcycle at the red box, boxed about right
            N  no motorcycle there (ghost, other object, background)
            D  the motorcycle is already boxed in green (duplicate)
            ?  cannot tell
switches  identity: ID switches exactly as compute_tracking_metrics counts
          them, as crop pairs of the GT identity (red) at its last match before
          the switch and at the switch. Verdicts in switch_verdicts.csv:
            Y  the same vehicle in both red boxes (a real switch)
            N  two different vehicles (the GT identity jumps: an artefact)
            ?  cannot tell
score     works on partly filled files. Until every stratum in both files has
          met the stopping rule (STOP_N below) it prints only, per file,
          "continue" or "you may stop", so the judge never sees interim shares;
          then the per-stratum shares with Wilson 95% intervals ->
          experiments/seed_bias/audit.json

The judge sees numbers only; key.csv and switch_key.csv map them to GT and tracker.

    python -m code.scripts.audit_disagreements sample|switches|score [--unblind]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from code.evaluation.metrics import _iou_matrix, compute_tracking_metrics
from code.scripts.check_seed_bias import DUMPS, MOTO
from code.scripts.compare_motion_models import open_video
from code.scripts.eval_hota_ci import iou_matrix
from code.scripts.gt_roi import filter_frames, load_roi
from code.scripts.import_gt import load_mot, to_framelist
from code.scripts.sweep_c2_sensitivity import moto_framelist

VIDEO = "data/raw/videos/NKKN-VoThiSau 2017-07-18_08_00_00_000.asf"
# Retired 2026-10-12 before any verdict: its dense strata came from the full-frame
# dense GT, since superseded by the ROI export, and paper_gt_audit
# (docs/PAPER_STEP1_PROTOCOL.md) samples all three final ROI GTs. Pinned to the
# GTs the sample was drawn from, so the files under AUDIT stay reproducible.
REVIEWS = {
    "dense": ("experiments/gt_qa/reviewed/20261007T161301Z_126e0b66", (10700, 11300), "ctrv_gate_nsa"),
    "mixed": ("experiments/gt_qa/mixed_reviewed/20261012T094827Z_7779f93a", (2500, 3100), "baseline"),
}
# (review, tracker that did NOT seed it): the unseen side of each GT
STRATA = [("mixed", "ctrv_gate_nsa"), ("dense", "baseline")]
# (review, tracker, sample size or None = all). Each seed's own switches are
# reviewer corrections, so they double as a control on the judge.
SWITCH_STRATA = [("dense", "baseline", 30), ("dense", "ctrv_gate_nsa", None),
                 ("mixed", "baseline", None), ("mixed", "ctrv_gate_nsa", None)]
PER_STRATUM, SEED, IOU = 40, 20261012, 0.5
# Stopping rule, fixed 2026-10-12 before any verdict was entered. The judge
# works in sheet order (1, 2, ... and S1, S2, ...); the order is a seeded
# shuffle, so every prefix is a random subsample of each stratum. A stratum
# stops early once STOP_N items are judged (? excluded) and its Wilson
# interval at STOP_Z excludes 0.5; otherwise it runs to its end. The early
# boundary is 99%, not 95%, because score is checked after every sheet and
# repeated looks at 95% inflate the error rate; the reported interval is 95%.
STOP_N, STOP_Z = 20, 2.576
AUDIT = Path("data/annotations/seed_bias_audit")      # gitignored: footage crops
OUT = Path("experiments/seed_bias/audit.json")
FIELDS = ["n", "review", "tracker", "kind", "track_id", "run_frames", "abs_frame",
          "x1", "y1", "x2", "y2"]
SWITCH_FIELDS = ["n", "review", "tracker", "gt_id", "frame_before", "abs_frame",
                 "bx1", "by1", "bx2", "by2", "x1", "y1", "x2", "y2"]


def load(review: str, tracker: str):
    """Segment, motorcycle GT, all-class GT and motorcycle predictions, in the review's scope."""
    run_dir, (s, e), _ = REVIEWS[review]
    roi = load_roi(Path(run_dir))
    truth = filter_frames(load_mot(Path(run_dir) / "gt_reviewed.txt"), roi)
    return (s, e, to_framelist(truth, e - s, MOTO), to_framelist(truth, e - s),
            moto_framelist(DUMPS / f"tracks_{tracker}.json", s, e, roi))


def unmatched(a: list, b: list) -> list[int]:
    """Indices of boxes in frame list `a` with no one-to-one partner in `b` at IoU >= IOU."""
    if not a:
        return []
    if not b:
        return list(range(len(a)))
    iou = iou_matrix([x for _, x in a], [x for _, x in b])
    hit = {r for r, c in zip(*linear_sum_assignment(-iou)) if iou[r, c] >= IOU}
    return [i for i in range(len(a)) if i not in hit]


def runs(fl: list, flags: list[list[int]]) -> list[list[tuple]]:
    """Group flagged boxes into runs of (frame, id, box): same ID, consecutive frames."""
    open_, done = {}, []
    for k, idx in enumerate(flags):
        now = {}
        for i in idx:
            tid, box = fl[k][i]
            now[tid] = open_.pop(tid, []) + [(k, tid, box)]
        done += open_.values()
        open_ = now
    return done + list(open_.values())


def switch_events(gt_fl: list, pr_fl: list) -> list[tuple]:
    """(frame before, frame, gt id, box before, box) for every ID switch, replicating
    compute_tracking_metrics' CLEAR matching step for step (keep still-valid
    correspondences, Hungarian on the rest; the last match persists across gaps)."""
    last, events = {}, []                  # gt id -> (pred id, frame, box) of its last match
    for k, (gts, preds) in enumerate(zip(gt_fl, pr_fl)):
        iou, matched, used = _iou_matrix(gts, preds), {}, set()
        index = {pid: j for j, (pid, _) in enumerate(preds)}
        for i, (gid, _) in enumerate(gts):
            j = index.get(last[gid][0]) if gid in last else None
            if j is not None and j not in used and iou[i, j] >= IOU:
                matched[i] = j
                used.add(j)
        free_g = [i for i in range(len(gts)) if i not in matched]
        free_p = [j for j in range(len(preds)) if j not in used]
        if free_g and free_p:
            for r, c in zip(*linear_sum_assignment(1.0 - iou[np.ix_(free_g, free_p)])):
                if iou[free_g[r], free_p[c]] >= IOU:
                    matched[free_g[r]] = free_p[c]
                    used.add(free_p[c])
        for i, j in matched.items():
            (gid, box), pid = gts[i], preds[j][0]
            if gid in last and last[gid][0] != pid:
                events.append((last[gid][1], k, gid, last[gid][2], box))
            last[gid] = (pid, k, box)
    return events


def guard(path: Path) -> None:
    """Never overwrite verdicts someone has started to fill in."""
    if path.exists() and any(r["verdict"].strip() for r in csv.DictReader(path.open(encoding="utf-8"))):
        raise SystemExit(f"{path} already holds verdicts; move it away to resample")


def sample() -> None:
    guard(AUDIT / "verdicts.csv")
    rng = np.random.default_rng(SEED)
    rows, gt_boxes = [], {}
    for review, tracker in STRATA:
        s, e, gt, gt_all, pr = load(review, tracker)
        for kind, a, b in (("extra", pr, gt), ("missing", gt, pr)):
            found = runs(a, [unmatched(a[k], b[k]) for k in range(e - s)])
            pick = rng.choice(len(found), size=min(PER_STRATUM, len(found)), replace=False)
            for j in sorted(pick):
                k, tid, box = found[j][len(found[j]) // 2]
                rows.append(dict(review=review, tracker=tracker, kind=kind, track_id=tid,
                                 run_frames=len(found[j]), abs_frame=s + k,
                                 x1=round(box[0], 1), y1=round(box[1], 1),
                                 x2=round(box[2], 1), y2=round(box[3], 1)))
                gt_boxes[s + k] = [x for _, x in gt_all[k]]
            print(f"{review}/{tracker}/{kind}: {len(found)} runs, sampled {len(pick)}")
    order = rng.permutation(len(rows))            # strata interleaved: the judge sees no labels
    rows = [dict(n=i + 1, **rows[j]) for i, j in enumerate(order)]
    frames = frames_at({r["abs_frame"] for r in rows})
    write_sheets([label(crop(frames[r["abs_frame"]], box(r), gt_boxes[r["abs_frame"]]), str(r["n"]))
                  for r in rows], 5, 4, "sheet")
    write_key(AUDIT / "key.csv", FIELDS, rows)
    (AUDIT / "verdicts.csv").write_text("n,verdict\n" + "".join(f"{r['n']},\n" for r in rows),
                                        encoding="utf-8")
    print(f"{len(rows)} crops -> {AUDIT}/sheet_*.png; verdicts go in {AUDIT / 'verdicts.csv'}")


def switches() -> None:
    guard(AUDIT / "switch_verdicts.csv")
    rng = np.random.default_rng(SEED + 1)
    rows, gt_boxes = [], {}
    for review, tracker, size in SWITCH_STRATA:
        s, e, gt, gt_all, pr = load(review, tracker)
        events = switch_events(gt, pr)
        if len(events) != compute_tracking_metrics(pr, gt).num_switches:
            raise SystemExit("switch_events no longer replicates metrics.py")
        pick = (range(len(events)) if size is None else
                sorted(rng.choice(len(events), size=min(size, len(events)), replace=False)))
        for j in pick:
            kb, k, gid, before, now = events[j]
            rows.append(dict(review=review, tracker=tracker, gt_id=gid, frame_before=s + kb, abs_frame=s + k,
                             **{f"b{c}": round(v, 1) for c, v in zip(("x1", "y1", "x2", "y2"), before)},
                             **{c: round(v, 1) for c, v in zip(("x1", "y1", "x2", "y2"), now)}))
            gt_boxes[s + kb], gt_boxes[s + k] = [x for _, x in gt_all[kb]], [x for _, x in gt_all[k]]
        print(f"{review}/{tracker}: {len(events)} switches, sampled {len(pick)}")
    order = rng.permutation(len(rows))
    rows = [dict(n=f"S{i + 1}", **rows[j]) for i, j in enumerate(order)]
    frames = frames_at({r["frame_before"] for r in rows} | {r["abs_frame"] for r in rows})
    gap = np.full((240, 8, 3), 255, np.uint8)
    write_sheets([np.hstack([label(crop(frames[r["frame_before"]], box(r, "b"), gt_boxes[r["frame_before"]]), r["n"]),
                             gap, crop(frames[r["abs_frame"]], box(r), gt_boxes[r["abs_frame"]])])
                  for r in rows], 2, 5, "switch_sheet")
    write_key(AUDIT / "switch_key.csv", SWITCH_FIELDS, rows)
    (AUDIT / "switch_verdicts.csv").write_text("n,verdict\n" + "".join(f"{r['n']},\n" for r in rows),
                                               encoding="utf-8")
    print(f"{len(rows)} pairs -> {AUDIT}/switch_sheet_*.png; verdicts go in {AUDIT / 'switch_verdicts.csv'}")


def box(r: dict, prefix: str = "") -> list[float]:
    return [float(r[prefix + k]) for k in ("x1", "y1", "x2", "y2")]


def frames_at(want: set[int]) -> dict:
    """The requested absolute frames, by one sequential pass (ASF seeking is inexact)."""
    frames, cap = {}, open_video(VIDEO, 0)
    for n in range(max(want) + 1):
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"{VIDEO} ended at frame {n}")
        if n in want:
            frames[n] = img
    cap.release()
    return frames


def crop(img, b: list[float], gt_boxes: list) -> np.ndarray:
    """240x240 crop around box b (red) with the frame's GT boxes (green)."""
    import cv2
    img = img.copy()
    for g in gt_boxes:
        cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 200, 0), 1)
    x1, y1, x2, y2 = (int(v) for v in b)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    half = max(60, 2 * max(x2 - x1, y2 - y1))
    img = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)   # square crops at edges
    cx, cy = (x1 + x2) // 2 + half, (y1 + y2) // 2 + half
    return cv2.resize(img[cy - half:cy + half, cx - half:cx + half], (240, 240),
                      interpolation=cv2.INTER_CUBIC)


def label(tile: np.ndarray, text: str) -> np.ndarray:
    import cv2
    cv2.putText(tile, text, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(tile, text, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return tile


def write_sheets(tiles: list, cols: int, rows: int, prefix: str) -> None:
    import cv2
    AUDIT.mkdir(parents=True, exist_ok=True)
    per = cols * rows
    for i in range(0, len(tiles), per):
        page = tiles[i:i + per] + [np.zeros_like(tiles[0])] * (per - len(tiles[i:i + per]))
        sheet = np.vstack([np.hstack(page[j:j + cols]) for j in range(0, per, cols)])
        cv2.imwrite(str(AUDIT / f"{prefix}_{i // per + 1:02d}.png"), sheet)


def write_key(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 1.0]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # clamp: at k = 0 or k = n, c -/+ h lands a rounding error outside [0, 1] (-0.0)
    return [round(float(max(0.0, c - h)), 3), round(float(min(1.0, c + h)), 3)]


def judged(key: str, verdicts: str, allowed: set[str]) -> list[dict] | None:
    """Key rows with their verdicts ('' = not judged yet), or None while the file is absent."""
    if not (AUDIT / verdicts).exists():
        return None
    v = {r["n"]: r["verdict"].strip().upper() for r in csv.DictReader((AUDIT / verdicts).open(encoding="utf-8"))}
    rows = [dict(r, verdict=v.get(r["n"], "")) for r in csv.DictReader((AUDIT / key).open(encoding="utf-8"))]
    bad = [r["n"] for r in rows if r["verdict"] not in allowed | {""}]
    if bad:
        raise SystemExit(f"{verdicts}: unknown verdicts at {', '.join(bad[:10])} (use {' '.join(sorted(allowed))})")
    return rows


def stratum(rows: list[dict], hit) -> dict:
    """One stratum in sheet order: verdict counts, share of `hit` among judged
    items (? excluded), the reported 95% interval and the stop status."""
    order = sorted(rows, key=lambda r: int(r["n"].lstrip("S")))
    filled = [bool(r["verdict"]) for r in order]
    v = [r["verdict"] for r in order if r["verdict"] not in ("", "?")]
    k, n = sum(map(hit, v)), len(v)
    lo, hi = wilson(k, n, STOP_Z)
    status = ("complete" if all(filled) else
              "stopped" if n >= STOP_N and (lo > 0.5 or hi < 0.5) else "open")
    return dict(counts={x: [r["verdict"] for r in order].count(x) for x in "YND?"}, sampled=len(order),
                judged=sum(filled), share=round(k / max(1, n), 3), wilson95=wilson(k, n), status=status,
                prefix=filled == sorted(filled, reverse=True))   # a blank before a verdict breaks randomness


def score(unblind: bool = False) -> None:
    files = {}
    rows = judged("key.csv", "verdicts.csv", {"Y", "N", "D", "?"})
    if rows:
        files["verdicts.csv"] = {
            # 'extra' boxes are GT omissions if real; 'missing' boxes are GT errors if not
            f"{review}/{tracker}/{kind} (share: GT wrong)": stratum(
                [r for r in rows if (r["review"], r["tracker"], r["kind"]) == (review, tracker, kind)],
                (lambda x: x == "Y") if kind == "extra" else (lambda x: x in ("N", "D")))
            for review, tracker in STRATA for kind in ("extra", "missing")}
    rows = judged("switch_key.csv", "switch_verdicts.csv", {"Y", "N", "?"})
    if rows:
        files["switch_verdicts.csv"] = {
            f"{review}/{tracker}/switch (share: real)": stratum(
                [r for r in rows if (r["review"], r["tracker"]) == (review, tracker)], lambda x: x == "Y")
            for review, tracker, _ in SWITCH_STRATA}
    for name, strata in files.items():          # judge-safe: no shares until a file may stop
        open_ = sum(r["status"] == "open" for r in strata.values())
        done = sum(r["judged"] for r in strata.values())
        print(f"{name}: {done} of {sum(r['sampled'] for r in strata.values())} judged; "
              + (f"continue in sheet order ({open_} strata open)" if open_ else "you may stop")
              + ("; a blank sits before a verdict: fill it (? if unsure)" if not all(
                  r["prefix"] for r in strata.values()) else ""))
    if files and (unblind or all(r["status"] != "open" for s in files.values() for r in s.values())):
        for strata in files.values():
            for k, r in strata.items():
                print(k, r)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(dict(stop_rule=dict(min_judged=STOP_N, z=STOP_Z), files=files), indent=1),
                       encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("sample", "switches", "score"))
    ap.add_argument("--unblind", action="store_true",
                    help="score: print shares before every stratum has stopped (analyst only, never the judge)")
    args = ap.parse_args()
    if args.step == "score":
        score(args.unblind)
    else:
        {"sample": sample, "switches": switches}[args.step]()


if __name__ == "__main__":
    main()
