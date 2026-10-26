"""Detector-only latency under one machine state, comparable across models and input sizes.

Every (weights, input size) times the same 200 decoded frames of the dense segment
[10700, 10900) after 20 warm-up frames, with the paper run's ROI pixel mask, torch 8
threads and cv2 1 thread, each configuration in its own process. Decode time is
excluded (frames are decoded once by the parent). The detector is about 91% of the
end-to-end frame time (experiments/realtime/evidence.md); run_roi_paper's benchmark
gives the full pipeline for best.pt.
    PYTHONPATH=. python experiments/operating_point/time_detectors.py <paper run dir>
Kept outside code/ because the paper runs hash every .py there.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CONFIGS = [("ket_qua_train/best.pt", 448, "detect"), ("ket_qua_train/best.pt", 640, "detect"),
           ("ket_qua_train/best.pt", 960, "detect"), ("ket_qua_train/best.pt", 1280, "detect"),
           ("weights/yolo11_vn.pt", 640, None), ("weights/yolo11_vn.pt", 1280, None)]
FIRST, WARMUP, MEASURED = 10680, 20, 200
FRAMES = Path("data/processed/latency_frames/dense_frames.npy")   # decoded footage: gitignored


def counter() -> int:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          '[int](Get-Counter "\\Processor Information(_Total)\\% Processor Performance").CounterSamples.CookedValue'],
                         capture_output=True, text=True)
    return int(out.stdout.strip() or -1)


def child(weights: str, size: int, task: str, roi_dir: str) -> None:
    import cv2
    import torch
    from code.detection.detector import Detector
    from code.scripts.gt_roi import load_roi
    frames = np.load(FRAMES)
    det = Detector(weights, img_size=size, task=task or None, roi=load_roi(Path(roi_dir)))
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    times = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        det.detect(frame)
        if i >= WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    print(json.dumps(dict(weights=weights, imgsz=size, mean_ms=round(float(t.mean()), 1),
                          p50_ms=round(float(np.median(t)), 1), p95_ms=round(float(np.percentile(t, 95)), 1),
                          fps=round(1000 / float(t.mean()), 1), frames=len(t))))


def main() -> None:
    from code.scripts.compare_motion_models import open_video
    run = Path(sys.argv[1])
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    roi_dir = str(Path(protocol["segments"]["dense"]["gt"]).parent)
    if not FRAMES.exists():
        FRAMES.parent.mkdir(parents=True, exist_ok=True)
        cap, keep = open_video(protocol["video"], 0), []
        for n in range(FIRST + WARMUP + MEASURED):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"video ended at {n}")
            if n >= FIRST:
                keep.append(frame)
        cap.release()
        np.save(FRAMES, np.stack(keep))
    rows = []
    for weights, size, task in CONFIGS:
        before = counter()
        out = subprocess.run([sys.executable, "-X", "utf8", __file__, "child", weights, str(size), task or "", roi_dir],
                             capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "."})
        row = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else dict(
            weights=weights, imgsz=size, error=out.stderr[-500:])
        row.update(counter_before=before, counter_after=counter())
        rows.append(row)
        print(row, flush=True)
    dest = Path("experiments/operating_point/detector_latency.json")
    dest.write_text(json.dumps(dict(method=__doc__.split("\n\n")[1].strip(), rows=rows), indent=1), encoding="utf-8")
    print(f"-> {dest}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        main()
