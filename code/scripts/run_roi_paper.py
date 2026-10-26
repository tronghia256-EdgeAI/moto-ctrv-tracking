"""Frozen development experiment: current detector, reconciled ROI GT, shared inputs.

Run from the repository root with ``python -m code.scripts.run_roi_paper``.
Outputs stay under data/processed; no footage or GT is published. This is a
development experiment, never an independent-test or accident-alert result.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from code.models.context_estimator import ContextEstimator
from code.pipeline import Pipeline, load_config
from code.scripts.compare_motion_models import VARIANTS, open_video
from code.scripts.gt_roi import contains_box, load_roi, pixel_mask
from code.scripts.import_gt import load_mot, to_framelist, tracks_json_slice

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / 'data/annotations/paper_gt_reconciliation/reconcile_90026162a7963db7/reconciled_e880ead8.json'
DEFAULT_VIDEO = ROOT / 'data/raw/videos/NKKN-VoThiSau 2017-07-18_08_00_00_000.asf'
ARMS = ('baseline', 'ctrv', 'ctrv_nsa', 'ctrv_gate', 'ctrv_gate_nsa',
        'c1_ctrv_gate', 'c1c2c3')
EXTERNAL = ('ocsort', 'ocsort_matched', 'ocsort_byte', 'botsort', 'botsort_matched')
SCORED = (*ARMS, *EXTERNAL, 'c1c2c3_raw')
CLASSES = {'motorcycle': 1, 'car': 0, 'bus': 2, 'truck': 3, 'ALL': None}
C1_PERCENTILE = 75  # pre-registered, experiments/c1_recalibration


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


@contextmanager
def run_lock(path):
    """OS lock releases on exit/crash; two runners cannot rewrite one cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('Another experiment is running in this output directory') from error
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError('Another experiment is running in this output directory') from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class CachedContext(ContextEstimator):
    """Replay exact raw ROI means, with the production EMA and track history."""
    def __init__(self, brightness):
        super().__init__(hour_override=8)
        self.values = brightness
        self.position = 0

    def _compute_brightness(self, frame, pixel_mask=None):
        value = float(self.values[self.position])
        self.position += 1
        self._brightness_ema = self._ema(value, self._brightness_ema)
        return self._brightness_ema


def resolved_config(name, roi_file, weights, size, fps):
    config_file, overrides = VARIANTS[name]
    cfg = load_config(str(ROOT / config_file))
    cfg['weights'] = {'yolo11': str(weights)}
    cfg['roi'] = {'file': str(roi_file)}
    cfg['detector']['img_size'] = size
    cfg['tracker']['frame_rate'] = fps
    cfg['tracker']['uncertainty_gating'] = bool(overrides.get(
        'uncertainty_gating', cfg['tracker'].get('uncertainty_gating', False)))
    cfg['ctrv']['nsa_noise'] = bool(overrides.get('nsa', cfg['ctrv'].get('nsa_noise', False)))
    cfg['ctrv']['scale_adaptive'] = bool(overrides.get('scale_adaptive', False))
    if cfg.get('ablation', {}).get('use_C1_adaptive', True):  # Pipeline's default
        # Set per size after caching by the pre-registered rule (c1_calibrate);
        # null fails loudly if a C1 arm is ever built without it.
        cfg['adaptive_controller']['density_threshold'] = None
    return cfg


def prepare(args):
    import torch
    import ultralytics
    from code.scripts.run_external_trackers import PARAMS, SOURCES
    receipt = json.loads(args.receipt.read_text(encoding='utf-8'))
    inputs = {}
    def bind(path):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f'Missing required input: {path}')
        inputs[str(path)] = digest(path)
        return path
    weights, video = bind(args.weights), bind(args.video)
    bind(args.receipt)
    trackeval = args.trackeval
    if trackeval is None:
        previous = ROOT / 'experiments/external_trackers/metrics.json'
        if previous.exists():
            prior = json.loads(previous.read_text(encoding='utf-8')).get('trackeval_crosscheck', {})
            # Older receipt keys can differ; a missing installation is explicit.
            if not prior:
                prior = json.loads(previous.read_text(encoding='utf-8')).get('trackeval', {})
            candidate = Path(prior.get('trackeval_file', '')).parent.parent
            if (candidate / 'trackeval/__init__.py').is_file():
                trackeval = candidate
    if trackeval is not None:
        if not (trackeval / 'trackeval/__init__.py').is_file():
            raise ValueError('TrackEval installation not found')
        for source in sorted((trackeval / 'trackeval').rglob('*.py')):
            bind(source)
    segments, roi = {}, None
    for name, item in receipt['segments'].items():
        release = Path(item['release'])
        current = load_roi(release)
        if not current or not current['enabled']:
            raise ValueError('A frozen, enabled ROI is required')
        if roi and any(current[k] != roi[k] for k in ('image_size', 'polygon', 'rule')):
            raise ValueError('Segments must share one camera ROI')
        roi = current
        gt = bind(release / 'gt_roi.txt')
        roi_file = bind(release / 'roi.json')
        bind(release / 'manifest.json')
        span = item['report']['segment_zero_based_end_exclusive']
        truth = load_mot(gt)
        if any(not contains_box(b, roi) for rows in truth.values() for _, b, _ in rows):
            raise ValueError('GT outside frozen ROI')
        if min(truth, default=1) < 1 or max(truth, default=0) > span[1] - span[0]:
            raise ValueError('GT frame index outside segment')
        segments[name] = {'gt': str(gt), 'roi_file': str(roi_file), 'segment': span,
                          'boxes': sum(map(len, truth.values()))}
    cap = open_video(str(video), 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    end = max(v['segment'][1] for v in segments.values())
    if fps <= 0 or count < end:
        raise ValueError('Invalid source FPS/frame count')
    # Freeze all implementation/config inputs, including adapters and evaluator.
    for folder in ('code', 'configs'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.suffix in ('.py', '.yaml') and path.name != 'tele_api.yaml':
                bind(path)
    protocol = dict(schema_version=1, role='DEVELOPMENT_ONLY_SAME_CAMERA',
                    receipt=str(args.receipt.resolve()), weights=str(weights), video=str(video),
                    inputs_sha256=inputs, segments=segments, roi=roi,
                    start_frame=0, num_frames=end, source_fps=fps, hour_override=8,
                    sizes=args.sizes, variants=list(ARMS), external=list(EXTERNAL),
                    external_params=PARAMS, external_sources=SOURCES,
                    trackeval_dir=str(trackeval.resolve()) if trackeval else None,
                    confidence=0.1, chunk_frames=200, torch_threads=args.threads,
                    cv2_threads=1, severity_enabled=False, alerts_enabled=False,
                    tracking_output='tracks_mappo (C3 task threshold); also save raw tracks',
                    roi_exit_policy='expire tracked and lost states outside ROI after each update; '
                                    'a non-finite or inverted external state counts as outside',
                    c1_density_threshold=f'per size, fixed after caching and before any C1 arm: p{C1_PERCENTILE} '
                                         'of the density EMA of c1_ctrv_gate replayed with an infinite density '
                                         'threshold, over source frames outside every GT segment '
                                         '(pre-registered in experiments/c1_recalibration); {size}/c1_threshold.json',
                    runtime=dict(python=platform.python_version(), torch=torch.__version__,
                                 ultralytics=ultralytics.__version__, numpy=np.__version__,
                                 opencv=cv2.__version__, device='cpu',
                                 platform=platform.platform(), processor=platform.processor()),
                    bootstrap=dict(seed=20261013, repeats=10000, block_lengths=[15, 30, 60],
                                   scope='within-segment block statistics; independent IDs per block; not camera-level uncertainty'))
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Frozen protocol differs. Use a new output directory; never overwrite a run.')
    else:
        write_json(protocol_path, protocol)
    for size in args.sizes:
        directory = args.out / str(size)
        for name in ARMS:
            cfg = resolved_config(name, segments['dense']['roi_file'], weights, size, fps)
            path = directory / 'configs' / f'{name}.yaml'
            raw = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text(encoding='utf-8') != raw:
                raise ValueError(f'Frozen config differs: {path}')
            path.write_text(raw, encoding='utf-8')
    return protocol


def verify_inputs(protocol):
    for path, expected in protocol['inputs_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def read_cache(directory, protocol):
    index = json.loads((directory / 'cache_index.json').read_text(encoding='utf-8'))
    if index['protocol_sha256'] != digest(directory.parent / 'protocol.json'):
        raise ValueError('Cache belongs to another protocol')
    detections, brightness = [], []
    for part in index['chunks']:
        path = directory / part['file']
        if part['start'] != len(detections) or digest(path) != part['sha256']:
            raise ValueError(f'Invalid cache chunk: {path}')
        with np.load(path, allow_pickle=False) as z:
            brightness.extend(z['brightness'].tolist())
            detections.extend(z[f'f{i}'] for i in range(part['end'] - part['start']))
    if len(detections) != index['num_frames'] or len(brightness) != len(detections):
        raise ValueError('Cache length mismatch')
    if len(detections) > protocol['num_frames']:
        raise ValueError('Cache exceeds frozen frame range')
    return index, detections, brightness


def cache(directory, protocol, size, stop_after=None):
    from code.detection.detector import Detector
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'cache_index.json').exists():
        index, _, _ = read_cache(directory, protocol)
    else:
        index = dict(protocol_sha256=digest(directory.parent / 'protocol.json'),
                     num_frames=0, complete=False, chunks=[])
    target = min(protocol['num_frames'], stop_after or protocol['num_frames'])
    if index['num_frames'] >= target:
        return
    detector = Detector(protocol['weights'], img_size=size, roi=protocol['roi'])
    # A segment checkpoint (yolo11_vn) is read for its boxes only.
    if detector.using_coco_fallback or detector.model.task not in ('detect', 'segment'):
        raise ValueError('Expected the exact custom detection checkpoint')
    # Ultralytics selects a CPU thread count during its first prediction.
    # Warm up before setting the recorded value, rather than silently measuring
    # eight threads while claiming the four configured before model loading.
    import torch
    w, h = protocol['roi']['image_size']
    detector.detect(np.zeros((h, w, 3), np.uint8))
    torch.set_num_threads(protocol['torch_threads'])
    cap = open_video(protocol['video'], index['num_frames'])
    start, t0 = index['num_frames'], time.perf_counter()
    pending, light = [], []
    try:
        for frame_no in range(start, target):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Source ended at {frame_no}')
            mask = pixel_mask(frame, protocol['roi'])
            light.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[mask > 0].mean()))
            detections = detector.detect(frame).astype(np.float64)
            if not np.isfinite(detections).all():
                raise ValueError('Non-finite detections')
            pending.append(detections)
            if len(pending) == protocol['chunk_frames'] or frame_no + 1 == target:
                lo, hi = index['num_frames'], frame_no + 1
                path = directory / f'detections_{lo:05d}_{hi:05d}.npz'
                temp = path.with_suffix('.tmp.npz')
                np.savez_compressed(temp, brightness=np.asarray(light),
                                    **{f'f{i}': d for i, d in enumerate(pending)})
                os.replace(temp, path)
                index['chunks'].append(dict(file=path.name, start=lo, end=hi, sha256=digest(path)))
                index['num_frames'] = hi
                index['complete'] = hi == protocol['num_frames']
                write_json(directory / 'cache_index.json', index)
                pending, light = [], []
                elapsed = time.perf_counter() - t0
                print(f'cache {size}: {hi}/{protocol["num_frames"]}, {(hi-start)/elapsed:.1f} fps', flush=True)
    finally:
        cap.release()


def paper_pipeline(directory, name, brightness=None, density_threshold=None):
    pipe = Pipeline.from_config(str(directory / 'configs' / f'{name}.yaml'),
                                build_detector=False, dry_run_alerts=True)
    pipe.alert = None
    pipe.accident_classifier = None
    pipe.reset()
    if brightness is not None:
        pipe.context_estimator = CachedContext(brightness)
    pipe.context_estimator.hour_override = 8
    if pipe.use_c1:
        if density_threshold is None:
            record = json.loads((directory / 'c1_threshold.json').read_text(encoding='utf-8'))
            if record['protocol_sha256'] != digest(directory.parent / 'protocol.json'):
                raise ValueError('C1 threshold belongs to another protocol')
            density_threshold = record['threshold']
        pipe.adaptive_controller.density_threshold = density_threshold
    return pipe


def c1_rule(density, segments):
    """Pre-registered C1 threshold: the C1_PERCENTILE-th percentile of the density
    EMA over frames outside every GT segment, so no evaluated frame sets it."""
    density = np.asarray(density, float)
    outside = np.ones(len(density), bool)
    for start, end in segments.values():
        outside[start:end] = False
    if not outside.any():
        raise ValueError('The C1 rule needs frames outside the GT segments')
    threshold = float(np.percentile(density[outside], C1_PERCENTILE))
    return dict(threshold=threshold, calibration_frames=int(outside.sum()),
                percentiles={str(p): float(np.percentile(density[outside], p)) for p in (50, 75, 90, 99)},
                # what the threshold implies on the evaluated segments (density only, no GT)
                share_frames_at_or_above={k: round(float((density[s:e] >= threshold).mean()), 3)
                                          for k, (s, e) in segments.items()})


def c1_calibrate(directory, protocol, detections, brightness):
    """Probe: c1_ctrv_gate with an infinite density threshold never enters DENSE_PEAK,
    so its density is the DEFAULT-regime signal the controller would compare."""
    path = directory / 'c1_threshold.json'
    if path.exists():
        return
    w, h = protocol['roi']['image_size']
    dummy = np.zeros((h, w, 3), np.uint8)
    pipe = paper_pipeline(directory, 'c1_ctrv_gate', brightness, density_threshold=float('inf'))
    density = [pipe.process(dummy, detections=d)['state'].density for d in detections]
    record = c1_rule(density, {k: v['segment'] for k, v in protocol['segments'].items()})
    write_json(path, dict(protocol_sha256=digest(directory.parent / 'protocol.json'),
                          rule=protocol['c1_density_threshold'], **record))
    print(f'c1 threshold {directory.name}: {record["threshold"]}', flush=True)


def pack(tracks):
    return [(int(t.track_id), int(t.class_id), [round(float(x), 1) for x in t.tlbr]) for t in tracks]


def inside_roi(box, roi):
    """contains_box for tracker states. OC-SORT's Kalman filter can reach a
    negative area, whose box is NaN; upstream OC-SORT deletes such a tracker at
    its next predict, so a non-finite or inverted box counts as outside here."""
    b = np.asarray(box, dtype=float)
    return bool(np.isfinite(b).all() and b[2] >= b[0] and b[3] >= b[1]) and contains_box(b.tolist(), roi)


def expire_external(tracker, name, roi):
    if name.startswith('ocsort'):
        tracker.trackers = [t for t in tracker.trackers if inside_roi(t.get_state()[0], roi)]
    else:
        for attr in ('tracked_stracks', 'lost_stracks'):
            kept = []
            for t in getattr(tracker, attr):
                if inside_roi(t.tlbr, roi):
                    kept.append(t)
                else:
                    t.mark_removed()
                    tracker.removed_stracks.append(t)
            setattr(tracker, attr, kept)


def external_tracker(name, fps):
    from code.scripts.run_external_trackers import PARAMS
    if name.startswith('ocsort'):
        from code.tracker.external.ocsort import OCSort
        from code.tracker.external.ocsort.ocsort import KalmanBoxTracker
        KalmanBoxTracker.count = 0
        return OCSort(**PARAMS[name])
    from code.tracker.external.botsort import BoTSORT
    from code.tracker.external.botsort.basetrack import BaseTrack
    BaseTrack._count = 0
    return BoTSORT(SimpleNamespace(**PARAMS[name]), frame_rate=fps)


def replay(directory, protocol):
    index, detections, brightness = read_cache(directory, protocol)
    if not index['complete']:
        raise ValueError('Replay/scoring requires a complete cache')
    w, h = protocol['roi']['image_size']
    dummy = np.zeros((h, w, 3), np.uint8)  # C1 brightness is injected from real frames.
    c1_calibrate(directory, protocol, detections, brightness)
    for name in (*ARMS, *EXTERNAL):
        path = directory / f'tracks_{name}.json'
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old.get('protocol_sha256') != index['protocol_sha256'] or len(old['frames']) != len(detections):
                raise ValueError(f'Invalid existing dump: {path}')
            continue
        frames, raw_frames, states = [], [], []
        if name in ARMS:
            pipe = paper_pipeline(directory, name, brightness)
            for i, d in enumerate(detections):
                result = pipe.process(dummy, detections=d)
                frames.append(pack(result['tracks_mappo']))
                raw_frames.append(pack(pipe._last_tracks))
                state = asdict(result['state'])
                if pipe.use_c1:
                    state['regime'] = pipe.adaptive_controller._active.name  # applied this frame
                states.append(state)
                if (i+1) % 2000 == 0:
                    print(f'replay {directory.name}/{name}: {i+1}/{len(detections)}', flush=True)
        else:
            tracker = external_tracker(name, protocol['source_fps'])
            for d in detections:
                out = tracker.update(d)
                expire_external(tracker, name, protocol['roi'])
                rows = [(int(r[4]), int(r[5]), [round(float(x), 1) for x in r[:4]]) for r in out] if name.startswith('ocsort') else pack(out)
                frames.append([row for row in rows if inside_roi(row[2], protocol['roi'])])
        if name in ARMS:
            write_json(directory / f'raw_tracks_{name}.json', dict(start_frame=0, num_frames=len(raw_frames),
                        protocol_sha256=index['protocol_sha256'], frames=raw_frames))
            write_json(directory / f'context_{name}.json', dict(frames=states))
        # Main dump is the completion marker: write ancillary outputs first so
        # a crash cannot cause resume to skip an arm with its raw C3 control missing.
        write_json(path, dict(start_frame=0, num_frames=len(frames), variant=name,
                             protocol_sha256=index['protocol_sha256'], frames=frames))
        print(f'replay {directory.name}/{name}: saved', flush=True)


def paired_ci(a, b, length, n, repeats=10000):
    from code.scripts.eval_hota_ci import metrics_from_rows
    starts = np.random.default_rng([20261013, length]).integers(0, n-length+1, size=(repeats, n//length))
    ma, ia = metrics_from_rows(a[starts].sum(axis=1))
    mb, ib = metrics_from_rows(b[starts].sum(axis=1))
    return {key: dict(mean=float(delta.mean()), ci95=np.percentile(delta, [2.5, 97.5]).tolist())
            for key, delta in (('mota', ma-mb), ('idf1', ia-ib))}


def score(directory, protocol):
    from code.evaluation.metrics import compute_tracking_metrics
    from code.scripts.eval_hota_ci import frame_data, hota, hota_data, hota_summary, exact_box_share, block_table
    index, _, _ = read_cache(directory, protocol)
    if not index['complete']:
        raise ValueError('Scoring requires complete cache')
    output = dict(role=protocol['role'], protocol_sha256=digest(directory.parent / 'protocol.json'),
                  size=int(directory.name), segments={}, dump_sha256={})
    def dump_path(name):
        return directory / ('raw_tracks_c1c2c3.json' if name == 'c1c2c3_raw' else f'tracks_{name}.json')
    for name in SCORED:
        path = dump_path(name)
        dump = json.loads(path.read_text(encoding='utf-8'))
        if dump['protocol_sha256'] != output['protocol_sha256'] or len(dump['frames']) != protocol['num_frames']:
            raise ValueError('Dump protocol/length mismatch')
        if any(not contains_box(box, protocol['roi']) for rows in dump['frames'] for _, _, box in rows):
            raise ValueError('Prediction outside frozen ROI')
        output['dump_sha256'][name] = digest(path)
    markdown = ['# Current detector + reconciled ROI GT (development)', '',
                f'Input size: {directory.name}. Same-camera development; fixed cache for all trackers.',
                'MOTA/IDF1: repository evaluator, IoU=0.5. HOTA averaged over 0.05:0.05:0.95.',
                'ALL is class-agnostic; other rows filter both GT and predictions by class.',
                'Block confidence intervals describe within-segment statistics, with identities reset per block.', '']
    for segment_name, item in protocol['segments'].items():
        start, end = item['segment']
        n = end-start
        truth = load_mot(item['gt'])
        scores, data, crosscheck_data = {}, {}, {}
        for name in SCORED:
            preds = tracks_json_slice(dump_path(name), start, end)
            scores[name] = {}
            for cls, cid in CLASSES.items():
                gt_fl, pr_fl = to_framelist(truth, n, cid), to_framelist(preds, n, cid)
                if not any(gt_fl):
                    continue
                d = frame_data(gt_fl, pr_fl)
                crosscheck_data[f'{name}|{cls}'] = d
                scores[name][cls] = {**asdict(compute_tracking_metrics(pr_fl, gt_fl)),
                                      **hota_summary(hota(hota_data(d))), 'exact_box_share': exact_box_share(d)}
                if cls == 'motorcycle' and name in ('baseline', 'ctrv_gate_nsa', 'c1c2c3'):
                    data[name] = d
            print(f'score {directory.name}/{segment_name}/{name}', flush=True)
        crosscheck = None
        if protocol.get('trackeval_dir'):
            from code.scripts.eval_hota_ci import trackeval_crosscheck
            crosscheck = trackeval_crosscheck(protocol['trackeval_dir'], crosscheck_data)
            for key, check in crosscheck['results'].items():
                if max(check[k] for k in ('max_abs_diff_HOTA_array', 'max_abs_diff_DetA_array', 'max_abs_diff_AssA_array', 'max_abs_diff_LocA_array')) > 1e-10:
                    raise ValueError('HOTA differs from official TrackEval')
                name, cls = key.split('|')
                scores[name][cls].update({k: check[k] for k in ('trackeval_MOTA', 'trackeval_IDF1', 'trackeval_IDSW', 'trackeval_Frag')})
        ci = {}
        if data:
            for length in protocol['bootstrap']['block_lengths']:
                tables = {name: block_table(d, length) for name, d in data.items()}
                ci[str(length)] = {name+'-baseline': paired_ci(tables[name], tables['baseline'], length, n)
                                   for name in ('ctrv_gate_nsa', 'c1c2c3')}
        output['segments'][segment_name] = dict(segment=[start, end], gt_sha256=digest(item['gt']), metrics=scores,
                                                motorcycle_block_ci=ci, trackeval_crosscheck=crosscheck)
        markdown.extend([f'## {segment_name} — motorcycle', '',
                         '| Tracker | MOTA | IDF1 | HOTA | DetA | AssA | IDSW | FN | FP |',
                         '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for name in SCORED:
            r = scores[name].get('motorcycle')
            if r:
                markdown.append(f'| {name} | {r["mota"]:.4f} | {r["idf1"]:.4f} | {r["HOTA"]:.4f} | {r["DetA"]:.4f} | {r["AssA"]:.4f} | {r["num_switches"]} | {r["num_misses"]} | {r["num_false_positives"]} |')
        markdown.append('')
        write_json(directory / 'metrics.json', output)
    (directory / 'summary.md').write_text('\n'.join(markdown)+'\n', encoding='utf-8')
    print(f'scores saved: {directory}', flush=True)


def benchmark_trial(directory, protocol, name, measured=200, warmup=20):
    """One trial; the orchestrator starts a fresh process for every trial."""
    from code.detection.detector import Detector
    index, detections, brightness = read_cache(directory, protocol)
    if not index['complete']:
        raise ValueError('Benchmark requires complete cache for identical prehistory')
    start = protocol['segments']['dense']['segment'][0]
    if start < warmup or start+measured > protocol['segments']['dense']['segment'][1]:
        raise ValueError('Benchmark exceeds dense GT window')
    w, h = protocol['roi']['image_size']
    dummy = np.zeros((h, w, 3), np.uint8)
    pipe = paper_pipeline(directory, name, brightness)
    for d in detections[:start-warmup]:
        pipe.process(dummy, detections=d)
    cached = pipe.context_estimator
    real = ContextEstimator(hour_override=8)
    real._brightness_ema, real._density_ema = cached._brightness_ema, cached._density_ema
    pipe.context_estimator = real
    pipe.detector = Detector(protocol['weights'], img_size=int(directory.name), roi=protocol['roi'])
    import torch
    pipe.detector.detect(dummy)
    torch.set_num_threads(protocol['torch_threads'])
    cap = open_video(protocol['video'], start-warmup)
    elapsed, stages = [], []
    try:
        for i in range(warmup+measured):
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Short benchmark input')
            td = time.perf_counter()
            r = pipe.process(frame)
            total = (time.perf_counter()-t0)*1000
            if i >= warmup:
                elapsed.append(total)
                stages.append({'decode': (td-t0)*1000, **r['stages_ms']})
    finally:
        cap.release()
    return dict(mean_ms=float(np.mean(elapsed)), p50_ms=float(np.percentile(elapsed,50)),
                p95_ms=float(np.percentile(elapsed,95)), fps=1000/float(np.mean(elapsed)),
                stages_mean_ms={k: float(np.mean([s[k] for s in stages])) for k in stages[0]},
                frame_ms=elapsed, torch_threads=torch.get_num_threads())


def benchmark(directory, protocol, repeats=3, measured=200, warmup=20):
    """Fresh process per arm/repeat; full prehistory, actual inference and decode."""
    from code.scripts.benchmark_realtime import environment
    start = protocol['segments']['dense']['segment'][0]
    result = dict(protocol_sha256=digest(directory.parent / 'protocol.json'), environment=environment(), size=int(directory.name),
                  scope='decode + ROI/context + detect + track + C3; excludes legacy severity and network alerts',
                  measured_segment=[start, start+measured], warmup_frames=warmup, repeats=repeats,
                  fresh_process_per_trial=True, variants={})
    trial_code = ('import cv2,json,sys,torch; from pathlib import Path; '
                  'from code.scripts.run_roi_paper import benchmark_trial,write_json,digest; '
                  'd=Path(sys.argv[1]); p=json.loads((d.parent/"protocol.json").read_text(encoding="utf-8")); '
                  'cv2.setNumThreads(1); torch.set_num_threads(p["torch_threads"]); '
                  'r=benchmark_trial(d,p,sys.argv[2],int(sys.argv[4]),int(sys.argv[5])); '
                  'r["protocol_sha256"]=digest(d.parent/"protocol.json"); write_json(sys.argv[3],r)')
    for name in ('baseline', 'ctrv_gate_nsa', 'c1c2c3'):
        samples = []
        for repeat in range(repeats):
            trial = directory / 'benchmark_trials' / f'{name}_{repeat}.json'
            trial.parent.mkdir(exist_ok=True)
            if not trial.exists():
                subprocess.run([sys.executable, '-c', trial_code, str(directory), name, str(trial),
                                str(measured), str(warmup)], check=True, cwd=ROOT)
            sample = json.loads(trial.read_text(encoding='utf-8'))
            if sample['protocol_sha256'] != result['protocol_sha256'] or len(sample['frame_ms']) != measured:
                raise ValueError('Benchmark trial protocol/length mismatch')
            samples.append(sample)
            print(f'benchmark {directory.name}/{name}/{repeat+1}: {samples[-1]["fps"]:.1f} fps, p95 {samples[-1]["p95_ms"]:.1f} ms', flush=True)
        result['variants'][name] = samples
        write_json(directory / 'benchmark.json', result)


def execute(args):
    import torch
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    protocol = prepare(args)
    verify_inputs(protocol)
    if args.phase == 'prepare':
        print(f'Protocol frozen: {args.out / "protocol.json"}', flush=True)
        return
    status = dict(status='RUNNING', requested_phase=args.phase, protocol_sha256=digest(args.out / 'protocol.json'))
    try:
        for size in args.sizes:
            directory = args.out / str(size)
            for phase, fn in (('cache', cache), ('replay', replay), ('score', score), ('benchmark', benchmark)):
                if args.phase not in ('all', phase):
                    continue
                status.update(size=size, phase=phase)
                write_json(args.out / 'status.json', status)
                verify_inputs(protocol)
                if phase == 'cache':
                    fn(directory, protocol, size, args.stop_after)
                else:
                    fn(directory, protocol)
        status['status'] = 'COMPLETED' if args.stop_after is None else 'PARTIAL_CACHE_SMOKE_TEST'
        write_json(args.out / 'status.json', status)
    except BaseException as error:
        status.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=str(error))
        write_json(args.out / 'status.json', status)
        raise
    print('Requested phases complete. Results are development-only.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['prepare', 'cache', 'replay', 'score', 'benchmark', 'all'], default='all')
    ap.add_argument('--receipt', type=Path, default=DEFAULT_RECEIPT)
    ap.add_argument('--video', type=Path, default=DEFAULT_VIDEO)
    ap.add_argument('--weights', type=Path, default=ROOT / 'ket_qua_train/best.pt')
    ap.add_argument('--out', type=Path, default=ROOT / 'data/processed/paper_runs/best85_roi_20261013_v4')
    ap.add_argument('--sizes', type=int, nargs='+', default=[640, 448])
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--trackeval', type=Path, help='Directory containing the official trackeval package (optional cross-check)')
    ap.add_argument('--stop-after', type=int, help='Cache smoke test only; partial cache cannot be scored')
    args = ap.parse_args()
    if args.threads < 1 or any(s < 32 or s % 32 for s in args.sizes) or len(set(args.sizes)) != len(args.sizes):
        ap.error('Positive threads, unique image sizes divisible by 32 are required')
    if args.stop_after is not None and (args.phase != 'cache' or args.stop_after < 1):
        ap.error('--stop-after is positive and only available for cache')
    with run_lock(args.out / 'run.lock'):
        execute(args)


if __name__ == '__main__':
    main()
