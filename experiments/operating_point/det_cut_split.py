"""Recall of GT motorcycles under 20 px at conf >= 0.5, IoU >= 0.5, split by whether the
ROI mask cuts the object (its top-centre lies outside the ROI polygon); objects >= 20 px
for reference. Writes experiments/operating_point/mask_cut_split.json.
    PYTHONPATH=. python experiments/operating_point/det_cut_split.py"""
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
protocol = json.loads((R / 'yolo11vn_roi_final2/protocol.json').read_text(encoding='utf-8'))
roi = protocol['roi']; poly = np.array(roi['polygon'], dtype=np.float32)
old = np.load('experiments/c2_real_vnweights_full/detections_full13726.npz', allow_pickle=False)
src = {'yolo_full_old': lambda i: old[f'f{i}'].astype(float)}
for label, d in (('yolo_roi', R / 'yolo11vn_roi_final2/1280'), ('best_roi_1280', R / 'best85_roi_final/1280'),
                 ('best_roi_960', R / 'best85_roi_final/960'), ('best_roi_640', R / 'best85_roi_final/640')):
    _, dets, _ = run.read_cache(d, json.loads((d.parent / 'protocol.json').read_text(encoding='utf-8')))
    src[label] = (lambda dets: lambda i: dets[i])(dets)
out = {}
for seg, item in protocol['segments'].items():
    start, end = item['segment']; truth = load_mot(item['gt'])
    for label, get in src.items():
        c = {'cut': [0, 0], 'whole': [0, 0], 'ge20': [0, 0]}
        for k in range(1, end - start + 1):
            gt = [b for _, b, cls in truth.get(k, []) if cls == 1]
            det = [list(r[:4]) for r in get(start + k - 1) if int(r[5]) == 1 and r[4] >= 0.5 and contains_box(list(r[:4]), roi)]
            m = set()
            if gt and det:
                iou = iou_matrix(gt, det)
                m = {r for r, cc in zip(*linear_sum_assignment(-iou)) if iou[r, cc] >= 0.5}
            for i, b in enumerate(gt):
                key = 'ge20' if b[3] - b[1] >= 20 else ('cut' if cv2.pointPolygonTest(poly, ((b[0] + b[2]) / 2, b[1]), False) < 0 else 'whole')
                c[key][0] += 1; c[key][1] += i in m
        print(f'{seg:<9}{label:<15}' + '  '.join(f'{k} n={v[0]} recall {v[1] / max(1, v[0]):.2f}' for k, v in c.items()))
        out[f'{seg}|{label}'] = {k: dict(n=v[0], recall=round(v[1] / max(1, v[0]), 3)) for k, v in c.items()}
Path('experiments/operating_point/mask_cut_split.json').write_text(json.dumps(dict(
    rule='GT motorcycles; detections of class motorcycle, conf >= 0.5, bottom-centre in the ROI; one-to-one IoU >= 0.5 per frame; cut = top-centre of the GT box outside the ROI polygon',
    sources={'yolo_full_old': 'experiments/c2_real_vnweights_full/detections_full13726.npz (unmasked, seeded the GT)', 'yolo_roi': 'yolo11vn_roi_final2/1280', 'best_roi_1280': 'best85_roi_final/1280', 'best_roi_960': 'best85_roi_final/960', 'best_roi_640': 'best85_roi_final/640'},
    results=out), indent=1), encoding='utf-8')
