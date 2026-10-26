"""Run a local detector sanity check and open GT/prediction/video previews."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from code.scripts.clean_detection_dataset import REPO, stamp, payload
from code.scripts.gt_review import sha256
from code.scripts.kaggle_detection_helpers import NAMES

DATASET = REPO / 'data/processed/kaggle/caap_accident_vehicle_dev_20261011'
VIDEO = REPO / 'data/raw/videos/NKKN-VoThiSau 2017-07-18_08_00_00_000.asf'
OUTPUT = REPO / 'data/processed/detector_previews/best_20261012_v2'


def draw_gt(path, label):
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for line in label.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        cid, x, y, bw, bh = map(float, line.split())
        box = ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)
        draw.rectangle(box, outline='lime', width=2)
        draw.text((box[0], max(0, box[1]-12)), NAMES[int(cid)], fill='lime')
    return image


def run(weights, dataset, video, output, start=2500, frames=150):
    from code.detection.detector import Detector
    import torch
    import ultralytics
    weights, dataset, video, output = map(lambda p: Path(p).resolve(), (weights, dataset, video, output))
    if output.exists():
        raise FileExistsError('Use a new output folder; previous previews are preserved')
    if frames < 1 or start < 0:
        raise ValueError('Positive frame count and nonnegative start required')
    with (dataset / 'manifest.csv').open(encoding='utf-8', newline='') as stream:
        candidates = [r for r in csv.DictReader(stream) if r['split'] == 'val']
    candidates.sort(key=lambda r: hashlib.sha256(('42' + r['source_path']).encode()).hexdigest())
    label_classes = {r['source_path']: {int(line.split()[0])
                     for line in (dataset / r['label']).read_text().splitlines() if line.strip()}
                     for r in candidates}
    # Fixed sample independent of model predictions, with class coverage first.
    covered, sample = set(), []
    while candidates and len(sample) < 12:
        def classes(row):
            return label_classes[row['source_path']]
        row = max(candidates, key=lambda r: len(classes(r) - covered))
        covered.update(classes(row)); sample.append(row); candidates.remove(row)
    if not sample:
        raise ValueError('No validation images')
    detector = Detector(str(weights), img_size=640)
    assert detector.model.task == 'detect'
    assert [detector.model.names[i] for i in range(5)] == NAMES
    assert detector.remap == {0: 4, 1: 2, 2: 0, 3: 1, 4: 3}
    output.mkdir(parents=True)
    report = {'mode': 'DEVELOPMENT_QUALITATIVE_SANITY_CHECK', 'created': stamp(),
              'checkpoint_sha256': sha256(weights), 'dataset_manifest_sha256': sha256(dataset / 'manifest.csv'),
              'task': detector.model.task, 'names': NAMES, 'project_remap': detector.remap,
              'software': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__},
              'device': 'cpu', 'imgsz': 640, 'display_conf': .25, 'images': [],
              'limitations': ['Development val, not an independent test.',
                             'Camera/training overlap has not been certified.',
                             'Detector boxes only: no tracking IDs or accident-event alert performance.',
                             'Preview timing is not a controlled paper latency benchmark.']}
    first = cv2.imread(str(dataset / sample[0]['image']))
    detector.model.predict(first, imgsz=640, conf=.25, device='cpu', verbose=False)
    mapped = detector.detect(first)
    assert mapped.ndim == 2 and mapped.shape[1] == 6 and np.isfinite(mapped).all()
    assert all(int(v) in range(5) for v in mapped[:, 5])
    report['wrapper_check'] = {'passed': True, 'conf': .1, 'detections': len(mapped)}
    for index, row in enumerate(sample, 1):
        image = dataset / row['image']; label = dataset / row['label']
        if sha256(image) != row['image_sha256'] or sha256(label) != row['label_sha256']:
            raise ValueError('Preview sample changed')
        result = detector.model.predict(str(image), imgsz=640, conf=.25, device='cpu', verbose=False)[0]
        pred = Image.fromarray(result.plot(line_width=1, font_size=8)[..., ::-1])
        gt = draw_gt(image, label)
        pair = Image.new('RGB', (1280, 676), 'white')
        pair.paste(gt.resize((640, 640)), (0, 36)); pair.paste(pred.resize((640, 640)), (640, 36))
        ImageDraw.Draw(pair).text((8, 10), 'GROUND TRUTH | development val', fill='black')
        ImageDraw.Draw(pair).text((648, 10), 'PREDICTION | best.pt | conf >= 0.25', fill='black')
        name = f'val_{index:02d}.jpg'; pair.save(output / name, quality=92)
        counts = Counter(NAMES[int(c)] for c in result.boxes.cls.cpu().tolist())
        report['images'].append({'file': name, 'source_path': row['source_path'], 'predictions': dict(counts),
                                 'speed_ms': result.speed})
        print(f'Val preview {index}/{len(sample)}: {dict(counts)}', flush=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError('Cannot open preview video')
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.
    writer = None
    timing, counts, representative = [], Counter(), None
    try:
        for index in range(frames):
            ok, frame = capture.read()
            if not ok:
                break
            t0 = time.perf_counter()
            result = detector.model.predict(frame, imgsz=640, conf=.25, device='cpu', verbose=False)[0]
            timing.append((time.perf_counter()-t0)*1000)
            counts.update(NAMES[int(c)] for c in result.boxes.cls.cpu().tolist())
            rendered = result.plot(line_width=1, font_size=8)
            scale = min(1., 1200 / rendered.shape[1])
            rendered = cv2.resize(rendered, (int(rendered.shape[1]*scale), int(rendered.shape[0]*scale)))
            cv2.putText(rendered, f'best.pt | frame {start+index} | conf .25 | CPU preview, no tracker',
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            if writer is None:
                writer = cv2.VideoWriter(str(output / 'camera_preview.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps,
                                         (rendered.shape[1], rendered.shape[0]))
                if not writer.isOpened():
                    raise ValueError('Cannot create preview video')
            writer.write(rendered)
            if index == frames//2:
                representative = rendered.copy()
            if (index+1) % 25 == 0:
                print(f'Camera preview {index+1}/{frames}', flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not timing:
        raise ValueError('No video frames decoded')
    report['video'] = {'source': video.name, 'start_frame_zero_based': start, 'frames': len(timing), 'fps': fps,
                       'median_predict_ms': float(np.median(timing)), 'p95_predict_ms': float(np.percentile(timing, 95)),
                       'detections_across_frames_not_unique_objects': dict(counts), 'file': 'camera_preview.avi'}
    if representative is not None:
        cv2.imwrite(str(output / 'camera_frame.jpg'), representative)
    overview = Image.new('RGB', (1280, 676), 'white')
    for index, row in enumerate(report['images'][:4]):
        with Image.open(output / row['file']) as im:
            overview.paste(im.resize((640, 338)), ((index%2)*640, (index//2)*338))
    overview.save(output / 'overview.jpg', quality=92)
    (output / 'report.json').write_bytes(payload(report))
    print(json.dumps(report['video'], indent=2), flush=True)
    print(f'Preview saved: {output}', flush=True)


def view(output):
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk
    output = Path(output).resolve()
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    root = tk.Tk(); root.title('CAAP — best.pt · GT / predictions / camera preview')
    root.geometry('1300x850')
    ttk.Label(root, text='Left: GT | Right: predictions. Development preview; no independent test or event alert score.').pack()
    options = ['Camera video'] + [row['file'] for row in report['images']]
    selected = tk.StringVar(value=options[0])
    toolbar = ttk.Frame(root); toolbar.pack(fill='x')
    combo = ttk.Combobox(toolbar, values=options, textvariable=selected, state='readonly', width=35)
    combo.pack(side='left', padx=8)
    canvas = tk.Label(root, bg='#202020'); canvas.pack(fill='both', expand=True)
    status = tk.StringVar(); ttk.Label(root, textvariable=status).pack()
    state = {'capture': None, 'playing': True, 'job': None, 'photo': None}
    def render(image):
        image.thumbnail((max(100, canvas.winfo_width()), max(100, canvas.winfo_height())))
        state['photo'] = ImageTk.PhotoImage(image); canvas.configure(image=state['photo'])
    def tick():
        if state['capture'] is not None and state['playing']:
            ok, frame = state['capture'].read()
            if not ok:
                state['capture'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = state['capture'].read()
            if ok:
                render(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        state['job'] = root.after(max(1, int(1000/report['video']['fps'])), tick)
    def change(_=None):
        if state['capture'] is not None:
            state['capture'].release(); state['capture'] = None
        if selected.get() == 'Camera video':
            state['capture'] = cv2.VideoCapture(str(output / report['video']['file']))
            state['playing'] = True
            status.set(f"Camera preview | median predict {report['video']['median_predict_ms']:.1f} ms on CPU | conf 0.25 | boxes only")
        else:
            row = next(r for r in report['images'] if r['file'] == selected.get())
            with Image.open(output / row['file']) as im:
                render(im.convert('RGB'))
            status.set(row['source_path'] + ' | ' + str(row['predictions']))
    def toggle():
        state['playing'] = not state['playing']
    def close():
        if state['job'] is not None: root.after_cancel(state['job'])
        if state['capture'] is not None: state['capture'].release()
        root.destroy()
    ttk.Button(toolbar, text='Play / Pause', command=toggle).pack(side='left')
    combo.bind('<<ComboboxSelected>>', change)
    root.protocol('WM_DELETE_WINDOW', close)
    root.after(200, change); root.after(250, tick); root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=REPO / 'best.pt')
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--video', type=Path, default=VIDEO)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--start', type=int, default=2500)
    parser.add_argument('--frames', type=int, default=150)
    parser.add_argument('--view', action='store_true')
    args = parser.parse_args()
    if args.view: view(args.output)
    else: run(args.weights, args.dataset, args.video, args.output, args.start, args.frames)


if __name__ == '__main__':
    main()
