"""
code/scripts/benchmark_realtime.py
───────────────────────────────────
End-to-end real-time benchmark: detector INCLUDED.

Why this exists separately from run_experiment.py
-------------------------------------------------
Every latency number reported so far came from a replay harness that fed
cached detections to the tracker, so it measured tracker-side cost only
(experiments/latency/evidence.md). That is the right number for "what does
C2 cost" and the wrong number for "does this run in real time" — the
detector is the dominant term and was excluded from it. This script measures
the whole path: decode -> context -> detect -> track -> C3 -> accident.

Two rules are baked in rather than left to the operator, because both were
learned the expensive way:

1. ONE VARIANT PER PROCESS. Timing seven variants in one process makes the
   measurement track position rather than variant — the same baseline code
   measured 13.19 ms first and 4.07 ms last, a 3.2x spread that exceeds the
   differences being measured and flatters whichever variant runs last.
   --sweep therefore re-execs this script once per config as a subprocess;
   it never loops in-process.
2. WARM-UP IS DISCARDED, NOT AVERAGED IN. The first frames pay for lazy
   imports, oneDNN autotuning and page faults. They are measured and
   reported separately so the cost stays visible, but they never enter the
   steady-state percentiles.

Usage (from the repo root, venv active):
    # one variant
    python -m code.scripts.benchmark_realtime --config configs/exp3_C1C2C3.yaml --video <path> --frames 600 --save

    # every ablation config, one subprocess each
    python -m code.scripts.benchmark_realtime --sweep --video <path> --frames 600 --save

    # operating-point sweep for the Pareto figure
    python -m code.scripts.benchmark_realtime --config configs/exp3_C1C2C3.yaml --video <path> --imgsz 320 --weights weights/yolo11n.pt --label n@320 --save

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.pipeline import Pipeline

OUT_DIR = REPO_ROOT / "experiments" / "realtime"

# Ablation configs, in the order the paper tables use them.
SWEEP_CONFIGS = [
    "configs/baseline.yaml",
    "configs/exp1_C2.yaml",
    "configs/exp2_C1C2.yaml",
    "configs/exp3_C1C2C3.yaml",
]

STAGES = ["decode", "context", "detect", "track", "c3", "accident"]


# ── Stats ────────────────────────────────────────────────────────────

def summarize(samples: list[float]) -> dict:
    """mean/p50/p95/p99/max over a millisecond series."""
    if not samples:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    a = np.asarray(samples, dtype=float)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def environment() -> dict:
    """Everything a reader needs to know whether a latency number is comparable."""
    env = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_threads"] = torch.get_num_threads()
        env["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:                       # torch optional at import time
        env["torch"] = f"unavailable ({exc})"
    return env


# ── Single-variant benchmark ─────────────────────────────────────────

def benchmark(config: str, video: str, frames: int, warmup: int,
              imgsz: int | None, weights: str | None,
              task: str | None = None) -> dict:
    import cv2

    pipeline = Pipeline.from_config(config, dry_run_alerts=True,
                                    build_detector=True)
    pipeline.reset()

    # Detector overrides for the Pareto sweep. Applied after construction so
    # Pipeline.from_config keeps one code path; both attributes are read per
    # call inside Detector.detect().
    if pipeline.detector is not None:
        if imgsz:
            pipeline.detector.img_size = imgsz
        if weights:
            from code.detection.detector import Detector
            pipeline.detector = Detector(weights,
                                         img_size=imgsz or pipeline.detector.img_size,
                                         task=task)

    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f"cannot open video: {video}"

    # The real-time budget is set by the SOURCE, not by a round number. This
    # footage is 25 fps, so the deadline is 40 ms/frame; quoting a 30 fps
    # budget here would silently grade against the wrong target.
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    budget_ms = 1000.0 / src_fps if src_fps > 0 else 1000.0 / 30.0

    per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
    total_ms: list[float] = []
    warm_total: list[float] = []
    track_counts: list[int] = []
    n = 0
    t_wall0 = None

    while n < warmup + frames:
        t_dec0 = time.perf_counter()
        ok, frame = cap.read()
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0
        if not ok:
            break

        result = pipeline.process(frame)
        frame_ms = decode_ms + result["latency_ms"]

        if n < warmup:
            warm_total.append(frame_ms)
        else:
            if t_wall0 is None:                    # clock starts after warm-up
                t_wall0 = t_dec0
            per_stage["decode"].append(decode_ms)
            for stage, ms in result["stages_ms"].items():
                per_stage[stage].append(ms)
            total_ms.append(frame_ms)
            track_counts.append(len(result["tracks_mappo"]))
        n += 1

    cap.release()
    measured = len(total_ms)
    assert measured, f"no frames measured (video yielded {n}, warmup {warmup})"
    wall = time.perf_counter() - t_wall0

    stage_summary = {s: summarize(v) for s, v in per_stage.items() if v}
    total = summarize(total_ms)

    return {
        "label": None,                             # filled by main()
        "config": config,
        "video": Path(video).name,
        "frames_measured": measured,
        "frames_warmup": len(warm_total),
        "imgsz": imgsz,
        "weights": weights,
        "task": task,
        "stages_ms": stage_summary,
        "total_ms": total,
        # Two different FPS numbers, and the difference matters:
        #   fps_pipeline — 1000/mean latency, the throughput of the code
        #   fps_wall     — frames / wall-clock, which also pays for whatever
        #                  the OS did in between. The honest headline.
        "fps_pipeline": 1000.0 / max(total["mean"], 1e-9),
        "fps_wall": measured / max(wall, 1e-9),
        "source_fps": src_fps,
        "budget_ms": budget_ms,
        "realtime_source": bool(total["p95"] <= budget_ms),
        "realtime_30fps": bool(total["p95"] <= 33.3),
        "warmup_ms": summarize(warm_total),
        "mean_active_tracks": float(np.mean(track_counts)) if track_counts else 0.0,
        "environment": environment(),
    }


# ── Sweep driver (subprocess per variant) ────────────────────────────

def run_sweep(args: argparse.Namespace) -> list[dict]:
    """Re-exec this script once per config. See rule 1 in the module docstring."""
    results = []
    for config in SWEEP_CONFIGS:
        cmd = [sys.executable, "-m", "code.scripts.benchmark_realtime",
               "--config", config, "--video", args.video,
               "--frames", str(args.frames), "--warmup", str(args.warmup),
               "--label", Path(config).stem, "--json-only"]
        if args.imgsz:
            cmd += ["--imgsz", str(args.imgsz)]
        if args.weights:
            cmd += ["--weights", args.weights]
        if args.task:
            cmd += ["--task", args.task]
        if args.save:
            cmd += ["--save"]
        print(f"-- {config} --", flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"{config} failed (exit {proc.returncode})")
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    return results


# ── Reporting ────────────────────────────────────────────────────────

def print_report(r: dict) -> None:
    print(f"\n{r['label']}  -  {r['frames_measured']} frames "
          f"({r['frames_warmup']} warm-up discarded)")
    print(f"{'stage':<10} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
    print("-" * 46)
    for stage in STAGES:
        s = r["stages_ms"].get(stage)
        if not s:
            continue
        print(f"{stage:<10} {s['mean']:>8.2f} {s['p50']:>8.2f} "
              f"{s['p95']:>8.2f} {s['p99']:>8.2f}")
    t = r["total_ms"]
    print("-" * 46)
    print(f"{'TOTAL':<10} {t['mean']:>8.2f} {t['p50']:>8.2f} "
          f"{t['p95']:>8.2f} {t['p99']:>8.2f}")
    print(f"\nFPS  pipeline {r['fps_pipeline']:.1f} | wall {r['fps_wall']:.1f} "
          f"| source {r['source_fps']:.0f} fps -> {r['budget_ms']:.1f} ms budget "
          f"at p95: {'MET' if r['realtime_source'] else 'MISSED'}")
    print(f"warm-up mean {r['warmup_ms']['mean']:.1f} ms "
          f"(max {r['warmup_ms']['max']:.1f}) - excluded above")


def print_sweep_table(results: list[dict]) -> None:
    print(f"\n{'variant':<16} {'detect':>8} {'track':>8} {'total':>8} "
          f"{'p95':>8} {'fps':>7}")
    print("-" * 60)
    for r in results:
        det = r["stages_ms"].get("detect", {}).get("mean", 0.0)
        trk = r["stages_ms"].get("track", {}).get("mean", 0.0)
        print(f"{r['label']:<16} {det:>8.2f} {trk:>8.2f} "
              f"{r['total_ms']['mean']:>8.2f} {r['total_ms']['p95']:>8.2f} "
              f"{r['fps_wall']:>7.1f}")


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end real-time benchmark (detector included).")
    p.add_argument("--config", type=str, default="configs/exp3_C1C2C3.yaml")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--frames", type=int, default=600,
                   help="Steady-state frames to measure (after warm-up).")
    p.add_argument("--warmup", type=int, default=30,
                   help="Frames to run and discard before measuring.")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Detector input size override (Pareto sweep).")
    p.add_argument("--weights", type=str, default=None,
                   help="Detector weights override (Pareto sweep).")
    p.add_argument("--task", type=str, default=None,
                   choices=["detect", "segment"],
                   help="Required for exported weights (.onnx / openvino dir): "
                        "they do not always carry the task, and a wrong guess "
                        "silently yields zero detections.")
    p.add_argument("--label", type=str, default=None,
                   help="Name for this run; defaults to the config stem.")
    p.add_argument("--sweep", action="store_true",
                   help="Run every ablation config, one subprocess each.")
    p.add_argument("--save", action="store_true",
                   help="Write JSON under experiments/realtime/.")
    p.add_argument("--json-only", action="store_true",
                   help="Print only the JSON result (used by --sweep).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sweep:
        results = run_sweep(args)
        print_sweep_table(results)
        return

    result = benchmark(args.config, args.video, args.frames, args.warmup,
                       args.imgsz, args.weights, args.task)
    result["label"] = args.label or Path(args.config).stem

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{result['label']}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not args.json_only:
            print(f"Saved -> {out.relative_to(REPO_ROOT)}")

    if args.json_only:
        print(json.dumps(result))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
