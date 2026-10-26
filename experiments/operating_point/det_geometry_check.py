"""Small-object 'misses' under the ROI protocol: absent detections or box geometry?
Recall of GT motorcycles by height at IoU >= 0.5 / 0.3 / 0.1 (one-to-one per frame),
for the old full-frame yolo11_vn@1280 cache and the ROI-masked caches; plus the share
of GT boxes whose top-centre lies outside the ROI (the mask cuts the object).
    PYTHONPATH=. python det_geometry_check.py"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from code.scripts import run_roi_paper as run
from code.scripts.eval_hota_ci import iou_matrix
from code.scripts.gt_roi import contains_box
from code.scripts.import_gt import load_mot

R = Path('data/processed/paper_runs')
HEIGHTS = ((0, 20), (20, 40), (40, 10_000))
protocol = json.loads((R / 'yolo11vn_roi_final2/protocol.json').read_text(encoding='utf-8'))
roi = protocol['roi']
poly = np.array(roi['polygon'], dtype=np.float32)
old = np.load('experiments/c2_real_vnweights_full/detections_full13726.npz', allow_pickle=False)
sources = {'yolo_full_old': lambda i: old[f'f{i}'].astype(float)}
for label, d in (('yolo_roi', R / 'yolo11vn_roi_final2/1280'), ('best_roi_1280', R / 'best85_roi_final/1280')):
    _, dets, _ = run.read_cache(d, json.loads((d.parent / 'protocol.json').read_text(encoding='utf-8')))
    sources[label] = (lambda dets: lambda i: dets[i])(dets)


def matched(gt, det, thr):
    if not gt or not det:
        return set()
    iou = iou_matrix(gt, det)
    return {r for r, c in zip(*linear_sum_assignment(-iou)) if iou[r, c] >= thr}


for seg, item in protocol['segments'].items():
    start, end = item['segment']
    truth = load_mot(item['gt'])
    cut = {f'{lo}-{hi}': [0, 0] for lo, hi in HEIGHTS}
    for k in range(1, end - start + 1):
        for _, b, cls in truth.get(k, []):
            if cls == 1:
                key = next(f'{lo}-{hi}' for lo, hi in HEIGHTS if lo <= b[3] - b[1] < hi)
                cut[key][0] += 1
                cut[key][1] += cv2.pointPolygonTest(poly, ((b[0] + b[2]) / 2, b[1]), False) < 0
    print(f'{seg}: GT boxes whose top-centre is outside the ROI: ' +
          ' '.join(f'{h}:{v[1] / max(1, v[0]):.2f}' for h, v in cut.items()))
    for label, get in sources.items():
        for conf in (0.5, 0.1):
            hits = {t: {f'{lo}-{hi}': [0, 0] for lo, hi in HEIGHTS} for t in (0.5, 0.3, 0.1)}
            for k in range(1, end - start + 1):
                gt = [b for _, b, cls in truth.get(k, []) if cls == 1]
                d = get(start + k - 1)
                det = [list(r[:4]) for r in d if int(r[5]) == 1 and r[4] >= conf and contains_box(list(r[:4]), roi)]
                for t in hits:
                    m = matched(gt, det, t)
                    for i, b in enumerate(gt):
                        key = next(f'{lo}-{hi}' for lo, hi in HEIGHTS if lo <= b[3] - b[1] < hi)
                        hits[t][key][0] += 1
                        hits[t][key][1] += i in m
            print(f'  {label:<14} conf>={conf}: ' + ' | '.join(
                f'IoU>={t}: ' + ' '.join(f'{h}:{v[1] / max(1, v[0]):.2f}' for h, v in hb.items() if h != '40-10000')
                for t, hb in hits.items()))
