"""Why does dropping the ROI pixel mask add motorcycle misses on dense at 960/1280 px?

Per GT motorcycle box on the dense segment, masked (best85_roi_final) vs unmasked
(best85_roi_nomask2), at each size:
  det  matched one-to-one (IoU >= 0.5) by a cached motorcycle detection whose
       bottom-centre is in the ROI, at conf >= 0.5 (first stage) or >= 0.1 (any)
  trk  matched one-to-one (IoU >= 0.5) by C2's output (tracks_ctrv_gate_nsa)
Boxes C2 matches with the mask but misses without it ("lost"), and the reverse
("gained"), are split by detection status, GT height, whether the mask cuts the
object (top-centre outside the ROI) and the bottom-centre's distance to the ROI edge.
For GT boxes detected in both runs, the detector-confidence shift (unmasked - masked)
is reported for the lost boxes and for all whole boxes >= 20 px from the ROI edge.
    PYTHONPATH=. python experiments/operating_point/dense_nomask_misses.py
"""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from code.scripts import run_roi_paper as run
from code.scripts.eval_hota_ci import iou_matrix
from code.scripts.gt_roi import contains_box
from code.scripts.import_gt import load_mot

R = Path('data/processed/paper_runs')
RUNS = {'masked': 'best85_roi_final', 'unmasked': 'best85_roi_nomask2'}
ARM = 'ctrv_gate_nsa'


def matched(gt, boxes):
    """GT index -> matched box index, one-to-one, IoU >= 0.5."""
    if not gt or not boxes:
        return {}
    iou = iou_matrix(gt, boxes)
    return {r: c for r, c in zip(*linear_sum_assignment(-iou)) if iou[r, c] >= 0.5}


def height_bin(h):
    return '<20' if h < 20 else '20-40' if h < 40 else '>=40'


def edge_bin(d):
    return '<5px' if d < 5 else '5-20px' if d < 20 else '>=20px'


protocol = json.loads((R / RUNS['masked'] / 'protocol.json').read_text(encoding='utf-8'))
roi = protocol['roi']
poly = np.array(roi['polygon'], dtype=np.float32)
start, end = protocol['segments']['dense']['segment']
truth = load_mot(protocol['segments']['dense']['gt'])
result = {}
for size in (448, 640, 960, 1280):
    hits = {}
    for name, run_name in RUNS.items():
        d = R / run_name / str(size)
        _, dets, _ = run.read_cache(d, json.loads((d.parent / 'protocol.json').read_text(encoding='utf-8')))
        frames = json.loads((d / f'tracks_{ARM}.json').read_text(encoding='utf-8'))['frames']
        h = hits[name] = {'det50': [], 'det10': [], 'conf10': [], 'trk': [], 'n_det50': 0}
        for k in range(1, end - start + 1):
            i = start + k - 1
            gt = [b for _, b, cls in truth.get(k, []) if cls == 1]
            rows = [r for r in dets[i] if int(r[5]) == 1 and contains_box(list(r[:4]), roi)]
            d50 = [list(r[:4]) for r in rows if r[4] >= 0.5]
            h['n_det50'] += len(d50)
            h['det50'].append(matched(gt, d50))
            d10 = [r for r in rows if r[4] >= 0.1]
            m10 = matched(gt, [list(r[:4]) for r in d10])
            h['det10'].append(m10)
            h['conf10'].append({j: float(d10[c][4]) for j, c in m10.items()})
            h['trk'].append(matched(gt, [box for _, cls, box in frames[i] if cls == 1]))
    summary = {name: dict(gt_matched_by_c2=sum(map(len, h['trk'])), gt_matched_by_det50=sum(map(len, h['det50'])),
                          gt_matched_by_det10=sum(map(len, h['det10'])), detections_conf50_in_roi=h['n_det50'])
               for name, h in hits.items()}
    groups = {'lost': Counter(), 'gained': Counter()}
    shift = {'lost': [], 'all whole, >=20px from the edge': []}   # unmasked - masked confidence of the same GT box
    crossing = Counter()
    for k in range(1, end - start + 1):
        gt = [b for _, b, cls in truth.get(k, []) if cls == 1]
        for j, b in enumerate(gt):
            cm, cu = hits['masked']['conf10'][k - 1].get(j), hits['unmasked']['conf10'][k - 1].get(j)
            interior = cv2.pointPolygonTest(poly, ((b[0] + b[2]) / 2, b[1]), False) >= 0 and                 abs(cv2.pointPolygonTest(poly, ((b[0] + b[2]) / 2, b[3]), True)) >= 20
            if cm is not None and cu is not None and interior:
                shift['all whole, >=20px from the edge'].append(cu - cm)
                crossing['>=0.5 masked, <0.5 unmasked' if cm >= 0.5 > cu else '<0.5 masked, >=0.5 unmasked' if cu >= 0.5 > cm else 'same side'] += 1
            m, u = j in hits['masked']['trk'][k - 1], j in hits['unmasked']['trk'][k - 1]
            if m == u:
                continue
            group, other = ('lost', 'unmasked') if m else ('gained', 'masked')
            det = 'det50' if j in hits[other]['det50'][k - 1] else 'det10 only' if j in hits[other]['det10'][k - 1] else 'no det'
            cut = cv2.pointPolygonTest(poly, ((b[0] + b[2]) / 2, b[1]), False) < 0
            edge = cv2.pointPolygonTest(poly, ((b[0] + b[2]) / 2, b[3]), True)
            if group == 'lost' and cm is not None and cu is not None:
                shift['lost'].append(cu - cm)
            g = groups[group]
            g['total'] += 1
            g[f'detection in the {other} run: {det}'] += 1
            g[f'height {height_bin(b[3] - b[1])}'] += 1
            g['mask cuts the object' if cut else 'object whole inside'] += 1
            g[f'bottom-centre to ROI edge {edge_bin(abs(edge))}'] += 1
    conf = {k: dict(n=len(v), median=round(float(np.median(v)), 3), share_lower=round(float(np.mean(np.array(v) < 0)), 3))
            for k, v in shift.items() if v}
    result[str(size)] = dict(summary=summary, confidence_shift_unmasked_minus_masked=conf,
                             interior_threshold_crossings=dict(crossing),
                             **{g: dict(sorted(c.items())) for g, c in groups.items()})
    print(size, json.dumps(result[str(size)], indent=1))
Path('experiments/operating_point/dense_nomask_misses.json').write_text(json.dumps(dict(
    rule='dense segment, GT motorcycles; one-to-one IoU >= 0.5 per frame; detections: class motorcycle, bottom-centre in the ROI; '
         'C2 = tracks_ctrv_gate_nsa; lost = matched by C2 with the mask, missed without; gained = the reverse; '
         '"detection in the X run" = the GT box was matched by a detection of the run where C2 missed it',
    runs=RUNS, results=result), indent=1), encoding='utf-8')
