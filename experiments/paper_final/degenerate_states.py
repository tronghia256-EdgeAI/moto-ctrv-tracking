"""Per-arm counts of the degenerate-state rule of Sect. 4.3: a tracker state whose
box is non-finite or inverted counts as having left the ROI.

Read-only replay of one final run's cache, run from the tree that produced it
(main for best85_roi_final, ../CAAP-fix for yolo11vn_roi_final2), so the code is
the run's own. The ROI-membership functions are wrapped: an invalid box is
recorded and counted as outside, which is what the run did (externals, and our
arms on CAAP-fix) or would have failed on (our arms on main, which fail closed).
Every replayed dump is compared with the run's own dump.

    PYTHONPATH=. python experiments/paper_final/degenerate_states.py <run>/<size> <out.json>
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import code.scripts.gt_roi as gt_roi
from code.scripts import run_roi_paper as run

directory, out = Path(sys.argv[1]), Path(sys.argv[2])
protocol = json.loads((directory.parent / 'protocol.json').read_text(encoding='utf-8'))
_, detections, brightness = run.read_cache(directory, protocol)
spans = {k: v['segment'] for k, v in protocol['segments'].items()}
now, events = {}, []


def segment(i):
    f = protocol['start_frame'] + i
    return next((k for k, (s, e) in spans.items() if s <= f < e), 'outside')


def same(a, b):
    return all(x == y or (math.isnan(x) and math.isnan(y)) for x, y in zip(a, b))


def guard(original):
    def wrapped(box, roi):
        b = [float(v) for v in box]
        finite = len(b) == 4 and all(math.isfinite(v) for v in b)
        if finite and b[2] >= b[0] and b[3] >= b[1]:
            return original(box, roi)
        e = dict(arm=now['arm'], frame=now['i'], segment=segment(now['i']),
                 box=[round(v, 2) if math.isfinite(v) else str(v) for v in b],
                 kind='inverted' if finite else 'non-finite')
        pipe = now['pipe']
        if pipe is None:   # externals: states are arrays, output rows are lists
            e['site'] = 'output' if isinstance(box, list) else 'state'
        else:              # ours: a pool track during expiry, else the output filter
            t = next((t for t in pipe.tracker.tracked_stracks + pipe.tracker.lost_stracks
                      if same([float(v) for v in t.tlbr], b)), None)
            e['site'] = 'output' if t is None else 'state'
            if t is not None:
                e.update(track_id=int(t.track_id), class_id=int(t.class_id), ctrv=t.ctrv is not None)
        events.append(e)
        return False
    return wrapped


for module, name in ((gt_roi, 'contains_box'), (gt_roi, 'contains_state'), (run, 'inside_roi'), (run, 'contains_state')):
    if hasattr(module, name):
        setattr(module, name, guard(getattr(module, name)))
member = getattr(run, 'inside_roi', None) or run.contains_state   # the tree's own output filter, now wrapped

w, h = protocol['roi']['image_size']
dummy = np.zeros((h, w, 3), np.uint8)
identical = {}
for name in (*run.ARMS, *run.EXTERNAL):
    now.update(arm=name, pipe=None)
    frames = []
    if name in run.ARMS:
        pipe = now['pipe'] = run.paper_pipeline(directory, name, brightness)
        for i, d in enumerate(detections):
            now['i'] = i
            frames.append(run.pack(pipe.process(dummy, detections=d)['tracks_mappo']))
    else:
        tracker = run.external_tracker(name, protocol['source_fps'])
        for i, d in enumerate(detections):
            now['i'] = i
            o = tracker.update(d)
            run.expire_external(tracker, name, protocol['roi'])
            rows = [(int(r[4]), int(r[5]), [round(float(x), 1) for x in r[:4]]) for r in o] \
                if name.startswith('ocsort') else run.pack(o)
            frames.append([row for row in rows if member(row[2], protocol['roi'])])
    dump = json.loads((directory / f'tracks_{name}.json').read_text(encoding='utf-8'))['frames']
    identical[name] = json.loads(json.dumps(frames)) == dump
    print(name, 'identical' if identical[name] else 'DIFFERS', sum(e['arm'] == name for e in events), flush=True)

counts = {name: dict(Counter(f"{e['site']}/{e['kind']}/{'gt' if e['segment'] != 'outside' else 'outside'}"
                             for e in events if e['arm'] == name))
          for name in (*run.ARMS, *run.EXTERNAL)}
out.write_text(json.dumps(dict(run=str(directory), identical_to_run=identical, counts=counts, events=events),
                          indent=1), encoding='utf-8')
