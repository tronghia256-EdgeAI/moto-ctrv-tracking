"""
code/scripts/run_experiment.py
───────────────────────────────
CLI entry point for running a single ablation experiment.

Usage (from the repo root, venv active):
    # real footage (needs data/raw/videos/ + YOLO weights or COCO fallback)
    python -m code.scripts.run_experiment --config configs/baseline.yaml --video data/raw/videos/cam01.mp4 --save-metrics

    # synthetic benchmark (no footage / weights / torch needed)
    python -m code.scripts.run_experiment --config configs/exp1_C2.yaml --synthetic --save-metrics

Run as a module (`python -m code.scripts.run_experiment`) so the project
package `code` wins over the stdlib module of the same name.

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.evaluation.metrics import compute_tracking_metrics, SystemMetrics
from code.pipeline import Pipeline
from code.utils.synthetic import (
    CLASS_CAR,
    CLASS_MOTORCYCLE,
    straight,
    weaving,
    constant_turn,
    to_detections,
    centers_to_bboxes,
)

FRAME_HW = (720, 1280)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Paper 2 ablation experiment."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (e.g. configs/baseline.yaml)")
    parser.add_argument("--video", type=str, default=None,
                        help="Video source path (overrides config).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run the synthetic mixed-traffic benchmark instead "
                             "of a video (no weights/footage required).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (smoke tests).")
    parser.add_argument("--save-metrics", action="store_true",
                        help="Save metrics JSON to experiments/<config-stem>/.")
    return parser.parse_args()


# ── Synthetic benchmark ──────────────────────────────────────────────

def synthetic_benchmark(n_frames: int = 300, seed: int = 42):
    """Deterministic mixed-traffic scene: 6 weaving motorcycles (with
    occlusion gaps), 2 straight cars, 1 turning motorcycle.

    Returns (frames_detections, gt_frames) where gt_frames follows the
    metrics FrameList format: per frame, list of (gt_id, box).
    """
    rng = np.random.default_rng(seed)
    trajs, classes = [], []
    for i in range(6):
        trajs.append(weaving(v=float(rng.uniform(4, 7)),
                             amp=float(rng.uniform(0.3, 0.7)),
                             period=float(rng.uniform(40, 80)),
                             n=n_frames, noise=0.0,
                             start=(float(rng.uniform(30, 200)),
                                    float(rng.uniform(60, 640))), seed=i))
        classes.append(CLASS_MOTORCYCLE)
    for i in range(2):
        trajs.append(straight(v=float(rng.uniform(4, 6)), n=n_frames, noise=0.0,
                              start=(30.0, 150.0 + 350.0 * i), heading=0.05))
        classes.append(CLASS_CAR)
    trajs.append(constant_turn(v=5.0, omega=0.02, n=n_frames, noise=0.0,
                               start=(100.0, 350.0)))
    classes.append(CLASS_MOTORCYCLE)

    # 14-frame occlusion gaps on three motorcycles — long enough that the
    # coasted prediction quality decides whether the ID survives.
    gaps = {0: list(range(100, 114)),
            3: list(range(180, 194)),
            8: list(range(140, 154))}

    noisy = [t + rng.normal(0, 3.0, size=t.shape) for t in trajs]
    frames = to_detections(noisy, class_ids=classes, drop_frames=gaps)

    gt_frames = []
    for k in range(n_frames):
        entries = []
        for gid, traj in enumerate(trajs):
            if k in gaps.get(gid, ()):   # GT hidden while occluded
                continue
            box = centers_to_bboxes(traj[k:k + 1])[0]
            entries.append((gid, box.tolist()))
        gt_frames.append(entries)
    return frames, gt_frames


def run_synthetic(pipeline: Pipeline, max_frames: int | None):
    frames, gt_frames = synthetic_benchmark()
    if max_frames:
        frames, gt_frames = frames[:max_frames], gt_frames[:max_frames]

    dummy = np.full((*FRAME_HW, 3), 128, dtype=np.uint8)
    pred_frames, latencies = [], []
    for dets in frames:
        result = pipeline.process(dummy, detections=dets)
        latencies.append(result["latency_ms"])
        pred_frames.append(
            [(t.track_id, t.tlbr.tolist()) for t in result["tracks_mappo"]]
        )

    tracking = compute_tracking_metrics(pred_frames, gt_frames)
    lat = float(np.mean(latencies))
    system = SystemMetrics(fps=1000.0 / max(lat, 1e-9), latency_ms=lat)
    return {"tracking": asdict(tracking), "system": asdict(system),
            "num_frames": len(frames), "mode": "synthetic"}


# ── Video run ────────────────────────────────────────────────────────

def run_video(pipeline: Pipeline, video_path: str, max_frames: int | None):
    import cv2

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"cannot open video: {video_path}"

    latencies, n = [], 0
    track_counts = []
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        result = pipeline.process(frame)
        latencies.append(result["latency_ms"])
        track_counts.append(len(result["tracks_mappo"]))
        n += 1
    cap.release()

    wall = time.perf_counter() - t_start
    lat = float(np.mean(latencies)) if latencies else 0.0
    # No GT on raw footage — tracking metrics need annotations (TODO once
    # the VN dataset is labeled). Report system metrics + track stats.
    return {
        "system": asdict(SystemMetrics(fps=n / max(wall, 1e-9), latency_ms=lat)),
        "mean_active_tracks": float(np.mean(track_counts)) if track_counts else 0.0,
        "num_frames": n,
        "mode": "video",
        "video": video_path,
    }


def main() -> None:
    args = parse_args()

    pipeline = Pipeline.from_config(
        args.config,
        dry_run_alerts=True,                  # experiments never spam Telegram
        build_detector=not args.synthetic,    # synthetic runs skip torch
    )
    pipeline.reset()

    if args.synthetic:
        results = run_synthetic(pipeline, args.max_frames)
    else:
        video = args.video
        assert video, "--video is required unless --synthetic is set"
        results = run_video(pipeline, video, args.max_frames)

    results["config"] = args.config
    print(json.dumps(results, indent=2))

    if args.save_metrics:
        exp_name = Path(args.config).stem
        out_dir = REPO_ROOT / "experiments" / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / ("results_synthetic.json" if args.synthetic
                              else "results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    main()
