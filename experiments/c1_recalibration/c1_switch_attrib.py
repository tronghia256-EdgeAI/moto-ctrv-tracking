"""Attribute the C1 arm's dense motorcycle ID switches to the birth regime of the predicted
tracks involved (CLEAR matching replicated from audit_disagreements.switch_events, which
does not return pred ids). A switch counts as DEFAULT-involved if the track it leaves or the
track it moves to was created while the controller was in DEFAULT.
    PYTHONPATH=. python c1_switch_attrib.py <run dir> <size> [...]"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from code.scripts.audit_disagreements import IOU, _iou_matrix
from code.scripts.import_gt import load_mot, to_framelist, tracks_json_slice

run = Path(sys.argv[1])
protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))


def switches(gt_fl, pr_fl):
    last, out = {}, []
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
            gid, pid = gts[i][0], preds[j][0]
            if gid in last and last[gid][0] != pid:
                out.append((k, gid, last[gid][0], pid))
            last[gid] = (pid, k)
    return out


for size in sys.argv[2:]:
    d = run / size
    frames = json.loads((d / "tracks_c1_ctrv_gate.json").read_text(encoding="utf-8"))["frames"]
    regime = [f["regime"] for f in json.loads((d / "context_c1_ctrv_gate.json").read_text(encoding="utf-8"))["frames"]]
    first = {}
    for i, rows in enumerate(frames):
        for tid, _, _ in rows:
            first.setdefault(tid, i)
    born = {tid: regime[max(0, f - 1)] for tid, f in first.items()}
    a, b = protocol["segments"]["dense"]["segment"]
    n = b - a
    gt = to_framelist(load_mot(protocol["segments"]["dense"]["gt"]), n, 1)
    pr = to_framelist(tracks_json_slice(d / "tracks_c1_ctrv_gate.json", a, b), n, 1)
    ev = switches(gt, pr)
    dflt = [e for e in ev if "DEFAULT" in (born[e[2]], born[e[3]])]
    early = [e for e in ev if e[0] < next((k for k, r in enumerate(regime[a:b]) if r == "DENSE_PEAK"), n)]
    print(f"{run.name}/{size}: dense moto IDSW {len(ev)}; involving a DEFAULT-born track {len(dflt)} "
          f"({len(dflt) / max(1, len(ev)):.2f}); during the leading DEFAULT stretch {len(early)}")
