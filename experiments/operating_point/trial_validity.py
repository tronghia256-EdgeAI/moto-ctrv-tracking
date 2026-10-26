"""Pre-registered latency validity (experiments/operating_point/evidence.md, ce8f3d5).

Per benchmark trial of a run size dir: the measured window is [t_end - sum(frame_ms), t_end],
t_end being the trial file's mtime; the trial is valid if the median of the 1 Hz
% Processor Performance samples inside the window is >= 90. Records accumulate in
<size>/benchmark_validity.json (keyed by trial name and t_end, so a re-measured trial keeps
the old verdict next to the new one). With --move, an invalid trial is moved with its record
to <size>/benchmark_trials_invalid/, so the runner re-measures it; nothing is deleted.
Kept outside code/ while paper runs are pending; move to code/scripts afterwards.
    PYTHONPATH=. python experiments/operating_point/trial_validity.py <samples.csv> <run>/<size> [...] [--move]
"""
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

THRESHOLD = 90.0
CLOCK = "trial file mtime and sampler stamps both read this machine's system clock"

args = [a for a in sys.argv[1:] if a != "--move"]
with open(args[0], newline="", encoding="ascii") as f:
    samples = [(int(r["unix_ms"]) / 1000, float(r["processor_performance"])) for r in csv.DictReader(f)]
for size_dir in map(Path, args[1:]):
    out = size_dir / "benchmark_validity.json"
    records = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for trial in sorted((size_dir / "benchmark_trials").glob("*.json")):
        t_end = trial.stat().st_mtime
        frame_ms = json.loads(trial.read_text(encoding="utf-8"))["frame_ms"]
        window = [t_end - sum(frame_ms) / 1000, t_end]
        inside = [v for t, v in samples if window[0] <= t <= window[1]]
        median = statistics.median(inside) if inside else None
        rec = dict(trial=trial.name, t_end_unix=t_end, sum_frame_ms=sum(frame_ms), window_unix=window,
                   samples_in_window=inside, median=median, threshold=THRESHOLD,
                   valid=median is not None and median >= THRESHOLD, clock=CLOCK)
        records[f"{trial.name}@{t_end:.3f}"] = rec
        if "--move" in sys.argv and not rec["valid"]:
            dest = size_dir / "benchmark_trials_invalid"
            dest.mkdir(exist_ok=True)
            stamp = f"{trial.stem}@{int(t_end)}"
            shutil.move(trial, dest / f"{stamp}.json")
            (dest / f"{stamp}_validity.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{size_dir.name}/{trial.name}: {len(inside)} samples, median {median}, valid {rec['valid']}")
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")
