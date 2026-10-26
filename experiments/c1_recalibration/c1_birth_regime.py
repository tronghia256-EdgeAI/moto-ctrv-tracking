"""Share of dense motorcycle track-frames of the C1 arm carried by tracks created while the
controller was in DEFAULT (linear model for the track's lifetime, adaptive_bytetrack
_should_use_ctrv). Creation frame = first output frame - 1 (a track is created unconfirmed
and output from its second match). Post hoc from the dump and the per-frame regime log.
    PYTHONPATH=. python c1_birth_regime.py <run dir> <size> [...]"""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
for size in sys.argv[2:]:
    d = run / size
    frames = json.loads((d / "tracks_c1_ctrv_gate.json").read_text(encoding="utf-8"))["frames"]
    regime = [f["regime"] for f in json.loads((d / "context_c1_ctrv_gate.json").read_text(encoding="utf-8"))["frames"]]
    first = {}
    for i, rows in enumerate(frames):
        for tid, cls, _ in rows:
            first.setdefault(tid, i)
    a, b = protocol["segments"]["dense"]["segment"]
    tf = {"DEFAULT": 0, "DENSE_PEAK": 0, "NIGHT": 0}
    tracks = {"DEFAULT": set(), "DENSE_PEAK": set(), "NIGHT": set()}
    for i in range(a, b):
        for tid, cls, _ in frames[i]:
            if cls == 1:
                born = regime[max(0, first[tid] - 1)]
                tf[born] += 1
                tracks[born].add(tid)
    seg_default = sum(r == "DEFAULT" for r in regime[a:b])
    lead = next((k for k, r in enumerate(regime[a:b]) if r == "DENSE_PEAK"), b - a)
    n = sum(tf.values())
    print(f"{run.name}/{size}: dense moto track-frames {n}; born in DEFAULT {tf['DEFAULT'] / n:.2f} "
          f"({len(tracks['DEFAULT'])} of {sum(map(len, tracks.values()))} tracks); "
          f"DEFAULT frames in dense {seg_default} (first {lead} before the first DENSE_PEAK frame)")
