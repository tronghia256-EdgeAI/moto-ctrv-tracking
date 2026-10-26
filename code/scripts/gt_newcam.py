"""
code/scripts/gt_newcam.py
─────────────────────────
Ground truth for held-out cameras, reusing the mixed-review workflow unchanged.
The cameras were audited clean against the detector's training images
(experiments/detector_leakage/evidence.md).

Per camera:
  select    pre-registered window rule, fixed before any tracker runs: the
            densest 600-frame window that starts after frame WARMUP, on a
            25-frame grid. Density = mean count of detections with score
            >= 0.5 (accident class excluded) from best.pt at 448 px, on every
            5th frame. (The NKKN segments were chosen by tracks/frame of a full
            C2 run; this is the same idea without a tracker in the loop.)
  cache     detections of every DETECTORS entry (0.1 confidence floor, as for
            NKKN) for frames [start - WARMUP, end), in one sequential decode
  replay    the paper's tracker variants on each detector's cache, as dumps
            with absolute frame indices under dumps/<detector>/
  prelabel  replay, then pre-labels = the union of every (detector, variant)
            output, blind to source (union_seed), as MOT and frames under
            data/annotations/prelabels/<camera>/, plus a review bundle
            (gt_mixed_review.prepare) under experiments/gt_qa/<camera>_review/
  scope     after the camera's ROI is saved: an ROI-only review bundle, so only
            boxes inside the ROI are reviewed (prints the review command)
  finalize  finalize_mixed (the ROI-only bundle if prepared) into
            experiments/gt_newcam/<camera>/reviewed, labelled held-out. Not the
            app's finish button: that writes into the mixed segment's folder
            with the mixed segment's 'development' label.

Per camera, in this order (the ROI is fixed before any metric is seen):
  review_gt_roi.cmd --bundle experiments/gt_qa/<camera>_review      draw, save the ROI
  python -m code.scripts.gt_newcam scope --cameras <camera>
  review_gt_mixed.cmd --bundle <printed path> --tracks-dir <printed dumps dir>
  python -m code.scripts.gt_newcam finalize --cameras <camera> --reviewer <name> [--detector D]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from code.scripts.compare_motion_models import VARIANTS, open_video
from code.scripts.sweep_c2_sensitivity import replay

CAMERAS = {
    "conghoa_uttich_1715": ("data/raw/videos/CongHoa-UtTich1 2017-07-17_17.15.asf", 17),
    "conghoa_truongchinh_1700": ("data/raw/videos/CongHoa-TruongChinh 2017-07-18_17_00_00_000.asf", 17),
    "duongbatrac_0800": ("data/raw/videos/DuongBaTrac-TaQuangBuu1 2017-07-18_08_00_00_000.asf", 8),
}
WINDOW, GRID, WARMUP, STEP, DENSITY_CONF = 600, 25, 1000, 5, 0.5
SELECT_WEIGHTS, SELECT_IMGSZ = "ket_qua_train/best.pt", 448
# Detection sources (weights, imgsz, task) for the union pre-labels: the manuscript's
# accuracy detector and the step-1 protocol's candidate at both of its input sizes
# (docs/PAPER_STEP1_PROTOCOL.md), so the GT is neutral whichever one is evaluated.
DETECTORS = {
    "yolo11_vn_1280": ("weights/yolo11_vn.pt", 1280, None),
    "best_640": ("ket_qua_train/best.pt", 640, "detect"),
    "best_448": ("ket_qua_train/best.pt", 448, "detect"),
}
ACCIDENT = 4
ROOT = Path("experiments/gt_newcam")
BUNDLES = Path("experiments/gt_qa")
PAPER_VARIANTS = ("baseline", "ctrv_gate_nsa", "c1_ctrv_gate", "c1c2c3")
HELD_OUT_ROLE = ("held-out test: camera never used to develop or tune the method; the training-image "
                 "overlap audit covers best.pt's export only, not yolo11_vn.pt "
                 "(experiments/detector_leakage/evidence.md)")


def detector(weights: str, imgsz: int, task: str | None = None):
    """Detector that refuses to start without its weights: Detector itself
    falls back to COCO yolo11n, which finds ~1.5 vehicles/frame on this
    footage and would silently corrupt the window choice and the cache."""
    from code.detection.detector import Detector
    if not Path(weights).is_file():
        raise SystemExit(f"weights not found: {weights}")
    return Detector(weights, img_size=imgsz, task=task)


def densest_window(counts: dict[int, int], n_frames: int) -> tuple[int, float]:
    """Start of the densest WINDOW (mean of the sampled counts inside it)."""
    best = None
    for s in range(WARMUP, n_frames - WINDOW + 1, GRID):
        inside = [c for f, c in counts.items() if s <= f < s + WINDOW]
        mean = sum(inside) / len(inside)
        if best is None or mean > best[1]:
            best = (s, mean)
    return best


def select(camera: str) -> None:
    import cv2
    video, _ = CAMERAS[camera]
    det = detector(SELECT_WEIGHTS, SELECT_IMGSZ, task="detect")
    cap, counts, n = cv2.VideoCapture(video), {}, 0
    while cap.grab():
        if n % STEP == 0:
            ok, frame = cap.retrieve()
            if ok:
                d = det.detect(frame)
                counts[n] = int(((d[:, 4] >= DENSITY_CONF) & (d[:, 5] != ACCIDENT)).sum())
        n += 1
    cap.release()
    start, mean = densest_window(counts, n)
    out = ROOT / camera
    out.mkdir(parents=True, exist_ok=True)
    (out / "window.json").write_text(json.dumps({
        "camera": camera, "video": video, "frames_decoded": n,
        "rule": f"densest {WINDOW}-frame window starting >= {WARMUP} on a {GRID}-frame grid; "
                f"density = detections with score >= {DENSITY_CONF} (no accident class), "
                f"every {STEP}th frame, {SELECT_WEIGHTS} @{SELECT_IMGSZ}",
        "start": start, "end": start + WINDOW, "mean_density": round(mean, 2),
        "profile": {str(f): c for f, c in counts.items()}}, indent=1), encoding="utf-8")
    print(f"{camera}: {n} frames, window [{start}, {start + WINDOW}) density {mean:.1f}")


def cache_path(camera: str, det: str) -> Path:
    return ROOT / camera / f"detections_{det}.npz"


def dumps_dir(camera: str, det: str) -> Path:
    return ROOT / camera / "dumps" / det


def cache(camera: str) -> None:
    """Every DETECTORS entry not cached yet, in one sequential decode (ASF seeking is inexact)."""
    from code.detection.detector import DETECTOR_CONF
    video, _ = CAMERAS[camera]
    w = json.loads((ROOT / camera / "window.json").read_text(encoding="utf-8"))
    first, end = w["start"] - WARMUP, w["end"]
    todo = [d for d in DETECTORS if not cache_path(camera, d).exists()]
    models = {d: detector(*DETECTORS[d]) for d in todo}
    dets = {d: [] for d in todo}
    cap = open_video(video, first)
    for _ in range(first, end if todo else first):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"{video} ended before frame {end}")
        for d, model in models.items():
            dets[d].append(model.detect(frame).astype(np.float32))
    cap.release()
    for d in todo:
        weights, imgsz, _ = DETECTORS[d]
        meta = {"video": video, "start_frame": first, "num_frames": len(dets[d]),
                "weights": weights, "imgsz": imgsz, "conf_floor": DETECTOR_CONF}
        np.savez_compressed(cache_path(camera, d), **{f"f{i}": x for i, x in enumerate(dets[d])},
                            meta=json.dumps(meta))
        print(f"{camera}: cached {d} for frames [{first}, {end})")


def absolute_dets(camera: str, det: str) -> tuple[list, int, int, dict]:
    """Cached detections indexed by absolute frame (None before the cache)."""
    npz = np.load(cache_path(camera, det), allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    first = meta["start_frame"]
    dets = [None] * first + [npz[f"f{i}"] for i in range(meta["num_frames"])]
    return dets, first, first + meta["num_frames"], meta


def run_variants(camera: str, det: str, names=PAPER_VARIANTS) -> Path:
    """Replay the paper's variants on one detector's cache; dumps keep absolute indices."""
    from code.pipeline import Pipeline
    from code.tracker.adaptive_bytetrack import ClassAwareSTrack
    dets, first, end, meta = absolute_dets(camera, det)
    out = dumps_dir(camera, det)
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        config, overrides = VARIANTS[name]
        pipe = Pipeline.from_config(config, dry_run_alerts=True, build_detector=False)
        pipe.reset()
        pipe.tracker.uncertainty_gating = overrides.get(
            "uncertainty_gating", pipe.tracker.uncertainty_gating)
        ClassAwareSTrack.ctrv_scale_adaptive = overrides.get(
            "scale_adaptive", ClassAwareSTrack.ctrv_scale_adaptive)
        ClassAwareSTrack.ctrv_nsa = overrides.get("nsa", ClassAwareSTrack.ctrv_nsa)
        pipe.context_estimator.hour_override = CAMERAS[camera][1]
        frames = replay(pipe, dets, first, end)
        (out / f"tracks_{name}.json").write_text(json.dumps(
            {**meta, "start_frame": 0, "num_frames": end, "detector": det, "variant": name,
             "warmup_from": first, "frames": frames}), encoding="utf-8")
        print(f"  {camera}: replayed {name} on {det}")
    return out


def union_seed(dumps: list[list], start: int, end: int) -> tuple[list, dict]:
    """Pre-labels over [start, end) from several trackers, blind to their source.

    The first tracker's boxes are kept as they are. Each later tracker adds the
    boxes no earlier one produced (no one-to-one partner at IoU >= 0.5 in that
    frame). An added box takes the seed identity its own track matched nearest
    in time, if the class agrees and that identity is not already in the frame;
    otherwise one fresh identity per (tracker, track, class). A GT seeded from
    one tracker favours it: the other's boxes never reach the reviewer
    (experiments/seed_bias); here every tracker's boxes do."""
    from scipy.optimize import linear_sum_assignment
    from code.scripts.eval_hota_ci import iou_matrix
    seed = [[(tid, cls, box) for tid, cls, box in f] for f in dumps[0][start:end]]
    next_id = 1 + max((t for f in seed for t, _, _ in f), default=0)
    stats = {"kept": sum(map(len, seed)), "linked": 0, "fresh_boxes": 0, "fresh_ids": 0}
    for frames in dumps[1:]:
        links, extra = {}, []                  # track -> {frame: (seed id, class)}
        for k, f in enumerate(frames[start:end]):
            pairs = {}
            if f and seed[k]:
                iou = iou_matrix([b for _, _, b in f], [b for _, _, b in seed[k]])
                pairs = {r: c for r, c in zip(*linear_sum_assignment(-iou)) if iou[r, c] >= 0.5}
            for r, (tid, cls, box) in enumerate(f):
                if r in pairs:
                    links.setdefault(tid, {})[k] = seed[k][pairs[r]][:2]
                else:
                    extra.append((k, tid, cls, box))
        fresh = {}
        for k, tid, cls, box in extra:
            near = min(links.get(tid, {}), key=lambda j: (abs(j - k), j > k), default=None)
            sid, scls = links[tid][near] if near is not None else (None, None)
            if sid is None or scls != cls or any(t == sid for t, _, _ in seed[k]):
                sid = fresh.setdefault((tid, cls), next_id + len(fresh))
                stats["fresh_boxes"] += 1
            else:
                stats["linked"] += 1
            seed[k].append((sid, cls, box))
        next_id += len(fresh)
        stats["fresh_ids"] += len(fresh)
    return seed, stats


def prelabel(camera: str) -> None:
    """Pre-labels = union of every (detector, variant) output, the manuscript's
    detector and baseline first, so the review adjudicates every source's boxes
    blind; seed.json records the sources and how they were merged."""
    from code.scripts.export_prelabels import extract_frames, write_mot
    from code.scripts.gt_mixed_review import prepare
    video, _ = CAMERAS[camera]
    w = json.loads((ROOT / camera / "window.json").read_text(encoding="utf-8"))
    seg = (w["start"], w["end"])
    sources = [(d, v) for d in DETECTORS for v in PAPER_VARIANTS]
    for d in DETECTORS:
        run_variants(camera, d)
    seed, merged = union_seed([json.loads((dumps_dir(camera, d) / f"tracks_{v}.json").read_text(
        encoding="utf-8"))["frames"] for d, v in sources], *seg)
    (ROOT / camera / "seed.json").write_text(json.dumps(
        {"trackers": [f"{d}/{v}" for d, v in sources],
         "detectors": {d: dict(weights=w_, imgsz=s) for d, (w_, s, _) in DETECTORS.items()},
         "rule": "first source kept; later sources add boxes with no partner at IoU >= 0.5, under "
                 "the nearest same-class matched identity of their track if free, else a fresh one",
         **merged}, indent=1), encoding="utf-8")
    out_root = Path("data/annotations/prelabels")
    stats = write_mot(seed, out_root / camera)
    if len(list((out_root / camera / "img1").glob("*.jpg"))) != seg[1] - seg[0]:
        extract_frames(video, {camera: seg}, out_root)
    bundle = prepare(source=out_root / camera, bundle=parent_bundle(camera), segment=seg)
    print(f"{camera}: union pre-labels {stats}, {merged} -> review bundle {bundle}")


def parent_bundle(camera: str) -> Path:
    return BUNDLES / f"{camera}_review"


def roi_child(camera: str) -> Path | None:
    """Where the ROI-only bundle for the camera's current roi.json lives (None: no ROI yet).
    Named by the ROI's hash, so a redrawn ROI gets a new bundle and old work is kept."""
    from code.scripts.gt_review import sha256
    roi = parent_bundle(camera) / "roi.json"
    return parent_bundle(camera) / "roi_reviews" / f"roi_{sha256(roi)[:12]}" if roi.exists() else None


def scope(camera: str) -> None:
    """After the ROI is saved in the ROI window: an ROI-only bundle (cases regenerated
    inside the ROI). activate=False: the mixed segment's roi_active.json is untouched."""
    from code.scripts.gt_roi_scope import prepare_scope
    seed = ROOT / camera / "seed.json"
    if not seed.exists() or len(json.loads(seed.read_text(encoding="utf-8"))["trackers"]) < 2:
        raise SystemExit(f"{camera}: the bundle is seeded from one tracker and would favour it "
                         "(experiments/seed_bias); rebuild it with the union prelabel first")
    out = roi_child(camera)
    if out is None:
        raise SystemExit(f"{camera}: draw and save the ROI first "
                         f"(review_gt_roi.cmd --bundle {parent_bundle(camera)})")
    prepare_scope(parent_bundle(camera), output=out, activate=False)
    print(f"{camera}: review with\n  review_gt_mixed.cmd --bundle {out} "
          f"--tracks-dir {dumps_dir(camera, next(iter(DETECTORS)))}")


def finalize(camera: str, reviewer: str, det: str = next(iter(DETECTORS))) -> None:
    """finalize_mixed into this camera's folder, scoring det's dumps, then replace
    its hard-coded 'development' role (the mixed segment's) and record window and seeding."""
    from code.detection.detector import DETECTOR_CONF
    from code.scripts.gt_mixed_review import MixedStore, finalize_mixed, json_bytes
    child = roi_child(camera)
    bundle = child if child and child.exists() else parent_bundle(camera)
    out = finalize_mixed(MixedStore(bundle), reviewer or "",
                         tracks_dir=dumps_dir(camera, det), out_dir=ROOT / camera / "reviewed")
    report = json.loads((out / "review.json").read_text(encoding="utf-8"))
    window = json.loads((ROOT / camera / "window.json").read_text(encoding="utf-8"))
    old, report["evaluation_role"] = report["evaluation_role"], HELD_OUT_ROLE
    report["window"] = {k: v for k, v in window.items() if k != "profile"}
    seed = ROOT / camera / "seed.json"      # absent: the original baseline-only pre-labels
    report["prelabel_seed"] = dict(**(json.loads(seed.read_text(encoding="utf-8")) if seed.exists()
                                      else {"trackers": ["yolo11_vn_1280/baseline"]}),
                                   conf_floor=DETECTOR_CONF)
    report["scored_detector"] = dict(name=det, weights=DETECTORS[det][0], imgsz=DETECTORS[det][1])
    (out / "review.json").write_bytes(json_bytes(report))
    summary = out / "summary.md"
    summary.write_text(summary.read_text(encoding="utf-8").replace(old, HELD_OUT_ROLE), encoding="utf-8")
    print(f"{camera}: reviewed GT -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("select", "cache", "prelabel", "replay", "scope", "finalize"))
    ap.add_argument("--cameras", nargs="*", default=list(CAMERAS), choices=list(CAMERAS))
    ap.add_argument("--reviewer", help="finalize: name recorded in review.json")
    ap.add_argument("--detector", choices=list(DETECTORS), default=next(iter(DETECTORS)),
                    help="finalize: whose dumps to score")
    args = ap.parse_args()
    for camera in args.cameras:
        if args.step == "finalize":
            finalize(camera, args.reviewer, args.detector)
        elif args.step == "replay":
            for d in DETECTORS:
                run_variants(camera, d)
        else:
            {"select": select, "cache": cache, "prelabel": prelabel, "scope": scope}[args.step](camera)


if __name__ == "__main__":
    main()
