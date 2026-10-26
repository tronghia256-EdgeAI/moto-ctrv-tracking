"""Pre-registered C1 density threshold: per input size, the 75th percentile of the C1 arm's
own density EMA (the value AdaptiveController compares) over NKKN frames OUTSIDE the
three GT segments, read from a run_roi_paper context log. Fixed before any C1 result
under the new threshold exists. Kept outside code/ because the paper runs hash every .py
there; move to code/scripts once no run is pending.
    PYTHONPATH=. python experiments/c1_recalibration/c1_calibration.py <run dir> <size> [<size> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

PERCENTILE = 75

run = Path(sys.argv[1])
protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
segments = {k: v["segment"] for k, v in protocol["segments"].items()}
out = Path("experiments/c1_recalibration/calibration.json")
result = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {
    "rule": f"p{PERCENTILE} of the c1_ctrv_gate arm's density EMA over frames outside the GT segments, per input size",
    "gt_segments": segments, "old_threshold": 30, "sizes": {}}
for size in sys.argv[2:]:
    frames = json.loads((run / size / "context_c1_ctrv_gate.json").read_text(encoding="utf-8"))["frames"]
    inside = np.zeros(len(frames), bool)
    for s, e in segments.values():
        inside[s:e] = True
    dens = np.array([f["density"] for f in frames], float)
    calib = dens[~inside]
    thr = float(np.percentile(calib, PERCENTILE))
    result["sizes"][size] = dict(
        source=str(run / size / "context_c1_ctrv_gate.json"), calibration_frames=int(calib.size),
        percentiles={p: float(np.percentile(calib, p)) for p in (50, 75, 90, 99)}, threshold=thr,
        # what the rule implies for the evaluation segments (context only, no GT or metric involved)
        share_frames_at_or_above={seg: round(float((dens[s:e] >= thr).mean()), 3) for seg, (s, e) in segments.items()})
    print(size, json.dumps(result["sizes"][size]["percentiles"]), "threshold", thr,
          result["sizes"][size]["share_frames_at_or_above"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1), encoding="utf-8")
