"""
code/scripts/compare_motion_models.py
──────────────────────────────────────
C2 evidence on real footage: run baseline (linear KF) and exp1_C2 (CTRV)
side-by-side on the SAME detections and compare track stability.

Single pass over the video: the detector runs once per frame and both
tracker pipelines consume identical detections, so any difference in
track behaviour is attributable to the motion model alone.

No ground truth exists on raw footage, so we report no-GT proxies:
  - unique track IDs over the segment (fewer = less fragmentation,
    given identical detections)
  - track lifespan distribution + short-track count
  - "re-identification events": ID a last seen at frame f, ID b born at
    f+g (g <= GAP_MAX) with IoU(a.last_box, b.first_box) >= IOU_MIN and
    the same class — the signature of a fragmented track. Each event is
    saved with its frame index for qualitative clip rendering.

Usage (repo root, venv active):
  python -m code.scripts.compare_motion_models \
      --video "data/raw/videos/clip.asf" --start-frame 2500 --max-frames 2000
  python -m code.scripts.compare_motion_models \
      --video "data/raw/videos/clip.asf" --render 3100 3350 --out-dir experiments/c2_real

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.pipeline import Pipeline

CONFIG_BASELINE = "configs/baseline.yaml"
CONFIG_CTRV = "configs/exp1_C2.yaml"

CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "accident"}

# One fixed color per class (BGR), per co-author's convention:
# motorcycle blue, car sky blue, bus yellow, truck orange, accident red.
CLASS_COLORS = {
    0: (235, 206, 135),   # car        - sky blue
    1: (255, 0, 0),       # motorcycle - blue
    2: (0, 255, 255),     # bus        - yellow
    3: (0, 165, 255),     # truck      - orange
    4: (0, 0, 255),       # accident   - red
}
NEWBORN_FLASH = 5   # frames a newly born track is drawn thick + starred

# Tracker variants for --replay: config + attribute overrides applied to
# the built pipeline (tracker-only replay from cached detections, so a
# variant sweep costs seconds, not detector-hours).
#
# EVERY toggled feature must be listed explicitly on EVERY variant.
# Overrides that are omitted fall through to the config file, and the exp
# configs turn features on by default — which silently collapses an
# ablation arm into its own control. That is exactly what happened to
# `nsa`: configs/exp1_C2.yaml gained `ctrv.nsa_noise: true`, so `ctrv`
# and `ctrv_gate` inherited NSA, `ctrv_gate` and `ctrv_gate_nsa` became
# the same configuration, and their dumps came out identical on all
# 13726 frames. Fall-through is never the right default here.
CONFIG_C1C2 = "configs/exp2_C1C2.yaml"
CONFIG_C1C2C3 = "configs/exp3_C1C2C3.yaml"

VARIANTS = {
    "baseline":     (CONFIG_BASELINE, {}),
    # Gating x NSA is a full 2x2: on real GT the two are not additive —
    # gating without NSA raises ID switches (over-smoothed predictions get
    # rescued onto the wrong detection), which is the same wrong-merge
    # mechanism the gate sweep found through the re-id proxy.
    "ctrv":         (CONFIG_CTRV, {"uncertainty_gating": False,
                                   "nsa": False}),
    "ctrv_nsa":     (CONFIG_CTRV, {"uncertainty_gating": False,
                                   "nsa": True}),
    "ctrv_gate":    (CONFIG_CTRV, {"uncertainty_gating": True,
                                   "nsa": False}),
    "ctrv_gate_snoise": (CONFIG_CTRV, {"uncertainty_gating": True,
                                       "nsa": False,
                                       "scale_adaptive": True}),
    "c1_ctrv_gate": (CONFIG_C1C2, {}),   # Exp2: adaptive params + C2
    "c1c2c3":       (CONFIG_C1C2C3, {}),  # Exp3: proposed method
    "ctrv_gate_nsa": (CONFIG_CTRV, {"uncertainty_gating": True,
                                    "nsa": True}),
}

# Worst-case window for latency reporting: ~47 tracks/frame vs a global
# mean of 25, and the segment with human GT.
DENSE_SEG = (10700, 11300)

GAP_MAX = 40        # frames: max death→birth gap for a re-id event
IOU_MIN = 0.3       # min IoU(last box, first box) for a re-id event
SHORT_TRACK = 10    # lifespan below this counts as churn


# ── Helpers ──────────────────────────────────────────────────────────

def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def open_video(path: str, start_frame: int):
    """Open the video positioned at start_frame by SEQUENTIAL grab.

    Never use cap.set(CAP_PROP_POS_FRAMES) here: on ASF/WMV surveillance
    exports OpenCV seeks to the nearest keyframe, not the exact frame, so
    seeked positions land tens of frames off. Track dumps are collected
    sequentially — a seeked render draws boxes on the wrong frames and
    real vehicles appear as 'ghost boxes' ahead of themselves.
    """
    import cv2
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"cannot open video: {path}"
    for _ in range(start_frame):
        cap.grab()
    return cap


# ── Phase 1: single-pass dual-tracker run ────────────────────────────

def run_pair(video: str, start_frame: int, max_frames: int | None,
             out_dir: Path, weights: str | None = None,
             imgsz: int | None = None) -> None:
    pipe_base = Pipeline.from_config(CONFIG_BASELINE, dry_run_alerts=True)
    pipe_ctrv = Pipeline.from_config(CONFIG_CTRV, dry_run_alerts=True,
                                     build_detector=False)
    if weights or imgsz:
        from code.detection.detector import Detector
        pipe_base.detector = Detector(
            weights or "weights/yolo11_vn.pt",
            img_size=imgsz or 640,
        )
    pipe_base.reset()
    pipe_ctrv.reset()

    cap = open_video(video, start_frame)
    dumps = {"baseline": [], "ctrv": []}
    det_cache: list[np.ndarray] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(n_done: int) -> None:
        # Checkpoint everything written so far — a killed run keeps its
        # progress and analyze()/render() work on the partial dumps.
        meta = {"video": video, "start_frame": start_frame,
                "num_frames": n_done}
        for name in dumps:
            with open(out_dir / f"tracks_{name}.json", "w",
                      encoding="utf-8") as f:
                json.dump({**meta, "frames": dumps[name]}, f)
        np.savez_compressed(out_dir / "detections.npz",
                            **{f"f{i}": d for i, d in enumerate(det_cache)},
                            meta=json.dumps(meta))

    n, t0 = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        dets = pipe_base.detector.detect(frame)
        det_cache.append(dets.astype(np.float32))
        for name, pipe in (("baseline", pipe_base), ("ctrv", pipe_ctrv)):
            result = pipe.process(frame, detections=dets)
            dumps[name].append([
                (int(t.track_id), int(t.class_id),
                 [round(float(v), 1) for v in t.tlbr])
                for t in result["tracks_mappo"]
            ])
        n += 1
        if n % 500 == 0:
            fps = n / (time.perf_counter() - t0)
            print(f"  frame {n} ({fps:.1f} fps incl. detection)", flush=True)
        if n % 1000 == 0:
            save(n)
    cap.release()

    save(n)
    print(f"Saved track dumps + detections -> {out_dir} ({n} frames)")


# ── Phase 1b: tracker-only replay from cached detections ────────────

def run_replay(det_npz: str, out_dir: Path, variants: list[str]) -> list[str]:
    """Replay cached detections through each tracker variant.

    No detector, no video decode: a dummy frame feeds the context
    estimator (C1 is off in these configs, so only brightness EMA sees
    it). Detections are identical across variants BY CONSTRUCTION.
    """
    from code.tracker.adaptive_bytetrack import ClassAwareSTrack

    npz = np.load(det_npz, allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    n = meta["num_frames"]
    dets = [npz[f"f{i}"] for i in range(n)]
    dummy = np.full((450, 800, 3), 128, dtype=np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    latency: dict[str, dict] = {}
    if len(variants) > 1:
        # Measured the hard way: running all seven variants in one process
        # put `baseline` at 13.19 ms in position 1 and 4.07 ms in position 7
        # — same code, 3.2x apart — because of cold-start warm-up plus any
        # external CPU load. The gradient is larger than the differences
        # being measured, and it flatters whatever runs last. Tracking
        # output is unaffected (replay is deterministic); only timings are.
        print("  NOTE: latency from a multi-variant run is position-biased "
              "and must not be quoted.\n"
              "        For real numbers run one variant per process — see "
              "experiments/latency/evidence.md", flush=True)
    for name in variants:
        config, overrides = VARIANTS[name]
        pipe = Pipeline.from_config(config, dry_run_alerts=True,
                                    build_detector=False)
        pipe.reset()
        # Overrides beat the config; absent keys keep the config's value.
        pipe.tracker.uncertainty_gating = overrides.get(
            "uncertainty_gating", pipe.tracker.uncertainty_gating)
        ClassAwareSTrack.ctrv_scale_adaptive = overrides.get(
            "scale_adaptive", ClassAwareSTrack.ctrv_scale_adaptive)
        ClassAwareSTrack.ctrv_nsa = overrides.get(
            "nsa", ClassAwareSTrack.ctrv_nsa)
        # Replay has no valid wall clock; the footage is 08:00 rush hour.
        # (Dummy frames give brightness 128 → never NIGHT, correct for
        # this daytime clip; density comes from real tracks, so C1's
        # DENSE_PEAK row engages during platoon surges.)
        pipe.context_estimator.hour_override = 8

        frames_out, lat, t0 = [], [], time.perf_counter()
        for d in dets:
            result = pipe.process(dummy, detections=d)
            lat.append(float(result["latency_ms"]))
            frames_out.append([
                (int(t.track_id), int(t.class_id),
                 [round(float(v), 1) for v in t.tlbr])
                for t in result["tracks_mappo"]
            ])
        fps = n / (time.perf_counter() - t0)
        # Latency here EXCLUDES detection (replay feeds cached boxes), so
        # it isolates the cost this paper's contributions actually add.
        # The dense window is reported separately because it is the worst
        # case: ~47 tracks/frame against a global mean of 25.
        # Trustworthy only in a single-variant run on an idle machine.
        lat_a = np.sort(np.asarray(lat))
        dense_a = np.sort(np.asarray(lat[DENSE_SEG[0]:DENSE_SEG[1]])) \
            if len(lat) >= DENSE_SEG[1] else lat_a
        stats = {
            "fps_replay": round(fps, 1),
            "latency_ms_mean": round(float(lat_a.mean()), 3),
            "latency_ms_p50": round(float(np.percentile(lat_a, 50)), 3),
            "latency_ms_p95": round(float(np.percentile(lat_a, 95)), 3),
            "latency_ms_p95_dense": round(float(np.percentile(dense_a, 95)), 3),
        }
        latency[name] = stats
        with open(out_dir / f"tracks_{name}.json", "w",
                  encoding="utf-8") as f:
            json.dump({**meta, "variant": name, "frames": frames_out,
                       "latency": stats}, f)
        print(f"  {name}: {n} frames replayed ({fps:.0f} fps) | "
              f"tracker latency mean {stats['latency_ms_mean']:.2f} ms, "
              f"p95 {stats['latency_ms_p95']:.2f} ms "
              f"(dense p95 {stats['latency_ms_p95_dense']:.2f} ms)",
              flush=True)

    with open(out_dir / "latency.json", "w", encoding="utf-8") as f:
        json.dump(latency, f, indent=2)
    return variants


# ── Phase 2: analysis ────────────────────────────────────────────────

def track_table(frames: list) -> dict:
    """Per-ID: first/last frame, first/last box, class mode."""
    tab: dict[int, dict] = {}
    for k, entries in enumerate(frames):
        for tid, cls, box in entries:
            if tid not in tab:
                tab[tid] = {"first": k, "first_box": box, "classes": []}
            tab[tid]["last"] = k
            tab[tid]["last_box"] = box
            tab[tid]["classes"].append(cls)
    for rec in tab.values():
        vals, cnts = np.unique(rec.pop("classes"), return_counts=True)
        rec["cls"] = int(vals[np.argmax(cnts)])
        rec["lifespan"] = rec["last"] - rec["first"] + 1
    return tab


def find_reid_events(tab: dict) -> list[dict]:
    """Death→birth pairs that look like one physical object splitting
    into two track IDs."""
    events = []
    for a, ra in tab.items():
        for b, rb in tab.items():
            if b == a or rb["first"] <= ra["last"]:
                continue
            gap = rb["first"] - ra["last"]
            if gap > GAP_MAX or ra["cls"] != rb["cls"]:
                continue
            ov = iou(np.array(ra["last_box"]), np.array(rb["first_box"]))
            if ov >= IOU_MIN:
                events.append({"old_id": a, "new_id": b,
                               "frame_lost": ra["last"],
                               "frame_reborn": rb["first"],
                               "gap": gap, "iou": round(ov, 3),
                               "cls": CLASS_NAMES.get(ra["cls"], ra["cls"])})
    return sorted(events, key=lambda e: e["frame_lost"])


def analyze(out_dir: Path, names: list[str] | None = None) -> None:
    if names is None:
        names = sorted(p.stem.removeprefix("tracks_")
                       for p in out_dir.glob("tracks_*.json"))
    summary = {}
    for name in names:
        with open(out_dir / f"tracks_{name}.json", encoding="utf-8") as f:
            data = json.load(f)
        tab = track_table(data["frames"])
        moto = {t: r for t, r in tab.items() if r["cls"] == 1}
        spans = [r["lifespan"] for r in tab.values()]
        events = find_reid_events(tab)
        summary[name] = {
            "unique_ids": len(tab),
            "unique_ids_motorcycle": len(moto),
            "mean_lifespan": round(float(np.mean(spans)), 1) if spans else 0,
            "median_lifespan": float(np.median(spans)) if spans else 0,
            "short_tracks(<%d)" % SHORT_TRACK:
                sum(1 for s in spans if s < SHORT_TRACK),
            "reid_events": len(events),
            "reid_events_motorcycle":
                sum(1 for e in events if e["cls"] == "motorcycle"),
            "events": events,
        }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    header = "".join(f"{n:>18}" for n in names)
    print(f"\n{'metric':<28}{header}")
    keys = ["unique_ids", "unique_ids_motorcycle", "mean_lifespan",
            "median_lifespan", "short_tracks(<%d)" % SHORT_TRACK,
            "reid_events", "reid_events_motorcycle"]
    for k in keys:
        row = "".join(f"{summary[n][k]:>18}" for n in names)
        print(f"{k:<28}{row}")
    print(f"\nSaved -> {out_dir / 'summary.json'}"
          f" (re-id events listed per run for clip rendering)")


# ── Phase 3: qualitative rendering ───────────────────────────────────

def render(video: str, start_frame: int, out_dir: Path,
           win_from: int, win_to: int,
           names: tuple[str, str] = ("baseline", "ctrv"),
           crop: tuple[int, int, int, int] | None = None,
           zoom: int = 2) -> None:
    """Side-by-side clip (names[0] | names[1]) for segment-relative
    frames [win_from, win_to). Optional crop=(cx, cy, w, h): each panel
    shows only that region, magnified `zoom`× — use it to make a single
    re-id event human-visible instead of a full-frame wall of boxes."""
    import cv2
    runs = {}
    for name in names:
        with open(out_dir / f"tracks_{name}.json", encoding="utf-8") as f:
            runs[name] = json.load(f)["frames"]

    # Birth frame per track id (whole run) — a track drawn within
    # NEWBORN_FLASH frames of its birth gets a thick, starred box, so
    # fragmentation shows up as "flashing" new boxes.
    births = {}
    for name in names:
        births[name] = {}
        for k, entries in enumerate(runs[name]):
            for tid, _, _ in entries:
                births[name].setdefault(tid, k)

    cap = open_video(video, start_frame + win_from)
    out_path = out_dir / f"clip_{names[0]}_vs_{names[1]}_{win_from}_{win_to}.mp4"
    if crop is not None:
        ccx, ccy, cw, ch = crop
        cx1 = int(np.clip(ccx - cw // 2, 0, 800 - cw))
        cy1 = int(np.clip(ccy - ch // 2, 0, 450 - ch))
        panel_wh = (cw * zoom, ch * zoom)
    else:
        panel_wh = (800, 450)
    w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        12.5, (panel_wh[0] * 2, panel_wh[1]))
    for k in range(win_from, min(win_to, len(runs[names[0]]))):
        ok, frame = cap.read()
        if not ok:
            break
        panels = []
        for name in names:
            panel = frame.copy()
            for tid, cls, box in runs[name][k]:
                x1, y1, x2, y2 = (int(v) for v in box)
                c = CLASS_COLORS.get(cls, (200, 200, 200))
                newborn = k - births[name].get(tid, 0) < NEWBORN_FLASH
                cv2.rectangle(panel, (x1, y1), (x2, y2), c,
                              4 if newborn else 2)
                label = f"*{tid}" if newborn else str(tid)
                cv2.putText(panel, label, (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)
            if crop is not None:
                panel = panel[cy1:cy1 + ch, cx1:cx1 + cw]
                panel = cv2.resize(panel, panel_wh,
                                   interpolation=cv2.INTER_LINEAR)
            cv2.putText(panel, name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            panels.append(panel)
        w.write(np.hstack(panels))
    w.release()
    cap.release()
    print(f"Saved clip -> {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="C2 motion-model comparison "
                                            "on real footage.")
    p.add_argument("--video", default=None)
    p.add_argument("--replay", type=str, default=None, metavar="DET_NPZ",
                   help="Replay cached detections.npz through tracker "
                        "variants (no detector/video needed).")
    p.add_argument("--variants", type=str,
                   default="baseline,ctrv,ctrv_gate,ctrv_gate_snoise",
                   help="Comma list of variants for --replay "
                        f"(available: {', '.join(VARIANTS)}).")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="experiments/c2_real")
    p.add_argument("--weights", type=str, default=None,
                   help="Detector weights override (e.g. weights/yolo11s.pt; "
                        "80-class COCO models are remapped automatically).")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Detector inference size override (e.g. 1280).")
    p.add_argument("--analyze-only", action="store_true",
                   help="Skip the run, re-analyze existing dumps.")
    p.add_argument("--render", nargs=2, type=int, metavar=("FROM", "TO"),
                   help="Render side-by-side clip for segment-relative "
                        "frame window [FROM, TO).")
    args = p.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    if args.render:
        assert args.video, "--render needs --video"
        render(args.video, args.start_frame, out_dir, *args.render)
        return
    if args.replay:
        names = run_replay(args.replay, out_dir,
                           [v.strip() for v in args.variants.split(",")])
        analyze(out_dir, names)
        return
    if not args.analyze_only:
        assert args.video, "--video is required (or use --replay)"
        run_pair(args.video, args.start_frame, args.max_frames, out_dir,
                 weights=args.weights, imgsz=args.imgsz)
    analyze(out_dir)


if __name__ == "__main__":
    main()
