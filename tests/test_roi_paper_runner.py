import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from code.models.context_estimator import ContextEstimator
from code.scripts.gt_roi import RULE, pixel_mask
from code.scripts import run_roi_paper as run


def test_cached_context_matches_real_roi_ema_and_history():
    rng = np.random.default_rng(42)
    roi = dict(enabled=True, image_size=[24, 16], polygon=[[0, 8], [24, 8], [24, 16], [0, 16]])
    frames = [rng.integers(0, 256, (16, 24, 3), dtype=np.uint8) for _ in range(10)]
    means = [float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[pixel_mask(f, roi) > 0].mean()) for f in frames]
    real, cached = ContextEstimator(hour_override=8), run.CachedContext(means)
    for i, frame in enumerate(frames):
        tracks = [SimpleNamespace(age=j) for j in range(i % 5)]
        assert real.estimate(frame, tracks, pixel_mask(frame, roi)) == cached.estimate(frame, tracks)


def small_run(tmp_path):
    root, directory = tmp_path, tmp_path / '64'
    bundle = tmp_path / 'roi'
    bundle.mkdir()
    run.write_json(bundle / 'manifest.json', dict(image_size=[64, 48]))
    roi = dict(schema_version=1, rule=RULE, enabled=True, image_size=[64, 48],
               polygon=[[0, 16], [64, 16], [64, 48], [0, 48]], reviewer='test', reason='scope',
               manifest_sha256=run.digest(bundle / 'manifest.json'))
    run.write_json(bundle / 'roi.json', roi)
    gt = tmp_path / 'gt.txt'
    gt.write_text(''.join(f'{i+1},1,{10+i},20,10,10,1,2,1\n' for i in range(6)), encoding='utf-8')
    # frames 6-7 lie outside the GT segment: the C1 rule calibrates there
    protocol = dict(num_frames=8, roi=roi, source_fps=25, role='DEVELOPMENT_ONLY_SAME_CAMERA',
                    segments={'dense': {'gt': str(gt), 'segment': [0, 6]}},
                    bootstrap={'block_lengths': [2, 4]}, c1_density_threshold='test rule')
    run.write_json(root / 'protocol.json', protocol)
    directory.mkdir()
    path = directory / 'detections_00000_00008.npz'
    np.savez_compressed(path, brightness=np.full(8, 120.),
                        **{f'f{i}': np.array([[10+i, 20, 20+i, 30, .9, 1.]]) for i in range(8)})
    run.write_json(directory / 'cache_index.json', dict(protocol_sha256=run.digest(root / 'protocol.json'),
        num_frames=8, complete=True, chunks=[dict(file=path.name, start=0, end=8, sha256=run.digest(path))]))
    for name in run.ARMS:
        cfg = run.resolved_config(name, bundle / 'roi.json', tmp_path / 'does_not_exist.pt', 64, 25)
        config = directory / 'configs' / f'{name}.yaml'
        config.parent.mkdir(exist_ok=True)
        config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    return directory, protocol


def test_replay_and_score_all_arms_without_detector_or_alerts(tmp_path):
    directory, protocol = small_run(tmp_path)
    run.replay(directory, protocol)
    run.score(directory, protocol)
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    assert len(metrics['segments']['dense']['metrics']) == len(run.SCORED)
    assert metrics['segments']['dense']['metrics']['baseline']['motorcycle']['idf1'] > .9
    assert (directory / 'raw_tracks_c1c2c3.json').exists()
    assert (directory / 'summary.md').exists()
    pipe = run.paper_pipeline(directory, 'baseline', [120.] * 8)
    assert pipe.detector is None and pipe.accident_classifier is None and pipe.alert is None
    record = json.loads((directory / 'c1_threshold.json').read_text(encoding='utf-8'))
    assert record['calibration_frames'] == 2 and record['rule'] == 'test rule'
    assert run.paper_pipeline(directory, 'c1c2c3').adaptive_controller.density_threshold == record['threshold']
    states = json.loads((directory / 'context_c1c2c3.json').read_text(encoding='utf-8'))['frames']
    assert all('regime' in s for s in states)
    assert not any('regime' in s for s in json.loads((directory / 'context_baseline.json').read_text(encoding='utf-8'))['frames'])


def test_c1_rule_uses_only_frames_outside_gt():
    record = run.c1_rule([100] * 6 + [1, 2, 3, 4], {'dense': (0, 6)})
    assert record['threshold'] == 3.25 and record['calibration_frames'] == 4
    assert record['share_frames_at_or_above'] == {'dense': 1.0}
    with pytest.raises(ValueError, match='outside the GT'):
        run.c1_rule([1, 2], {'dense': (0, 2)})


def test_changed_cache_fails_closed(tmp_path):
    directory, protocol = small_run(tmp_path)
    path = directory / 'detections_00000_00008.npz'
    with path.open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='Invalid cache chunk'):
        run.read_cache(directory, protocol)


def test_ablations_are_explicit_and_do_not_inherit_nsa(tmp_path):
    configs = {name: run.resolved_config(name, tmp_path / 'roi.json', tmp_path / 'best.pt', 640, 25)
               for name in run.ARMS}
    for name, gate, nsa in [('ctrv', False, False), ('ctrv_nsa', False, True),
                            ('ctrv_gate', True, False), ('ctrv_gate_nsa', True, True)]:
        assert configs[name]['tracker']['uncertainty_gating'] is gate
        assert configs[name]['ctrv']['nsa_noise'] is nsa
        assert configs[name]['tracker']['frame_rate'] == 25
        assert not configs[name]['ctrv']['scale_adaptive']
    assert not configs['baseline']['ablation']['use_C2_ctrv']
    # C1 arms carry no threshold until c1_calibrate fixes it per size
    assert configs['c1_ctrv_gate']['adaptive_controller']['density_threshold'] is None
    assert configs['c1c2c3']['adaptive_controller']['density_threshold'] is None


def test_partial_cache_cannot_be_scored(tmp_path):
    directory, protocol = small_run(tmp_path)
    path = directory / 'cache_index.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['complete'] = False
    run.write_json(path, index)
    with pytest.raises(ValueError, match='complete cache'):
        run.score(directory, protocol)


def synthetic_video(tmp_path):
    video = tmp_path / 'frames.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 25, (64, 48))
    assert writer.isOpened()
    for i in range(8):
        writer.write(np.full((48, 64, 3), 100+i, np.uint8))
    writer.release()
    return video


class FakeDetector:
    using_coco_fallback = False
    model = SimpleNamespace(task='detect')

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame):
        return np.array([[10., 20, 20, 30, .9, 1]])


def test_cache_resume_preserves_frame_order_and_brightness(tmp_path, monkeypatch):
    directory, protocol = small_run(tmp_path)
    directory = tmp_path / '96'
    protocol.update(video=str(synthetic_video(tmp_path)), weights='fake.pt', chunk_frames=3, torch_threads=1)
    run.write_json(tmp_path / 'protocol.json', protocol)
    monkeypatch.setattr('code.detection.detector.Detector', FakeDetector)
    run.cache(directory, protocol, 96, stop_after=3)
    index, _, means = run.read_cache(directory, protocol)
    assert not index['complete'] and means == [100, 101, 102]
    run.cache(directory, protocol, 96)
    index, dets, means = run.read_cache(directory, protocol)
    assert index['complete'] and len(dets) == 8
    assert means == list(range(100, 108))


def test_live_benchmark_trial_without_severity(tmp_path, monkeypatch):
    directory, protocol = small_run(tmp_path)
    protocol.update(video=str(synthetic_video(tmp_path)), weights='fake.pt', torch_threads=1)
    protocol['segments']['dense']['segment'] = [4, 8]
    run.write_json(tmp_path / 'protocol.json', protocol)
    index_path = directory / 'cache_index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    index['protocol_sha256'] = run.digest(tmp_path / 'protocol.json')
    run.write_json(index_path, index)
    monkeypatch.setattr('code.detection.detector.Detector', FakeDetector)
    result = run.benchmark_trial(directory, protocol, 'ctrv_gate_nsa', measured=2, warmup=2)
    assert len(result['frame_ms']) == 2
    assert result['torch_threads'] == 1


def test_runner_lock_blocks_concurrent_writer_and_releases(tmp_path):
    path = tmp_path / 'run.lock'
    with run.run_lock(path):
        with pytest.raises(RuntimeError, match='Another experiment'):
            with run.run_lock(path):
                pass
    with run.run_lock(path):
        pass  # A remaining lock file does not block a new process.


def test_benchmark_dispatches_fresh_trials_and_resumes(tmp_path, monkeypatch):
    directory, protocol = small_run(tmp_path)
    calls = []
    def trial_process(command, check, cwd):
        assert check and cwd == run.ROOT
        compile(command[2], '<trial>', 'exec')
        calls.append(command[4])
        run.write_json(command[5], dict(protocol_sha256=run.digest(tmp_path / 'protocol.json'),
                                      frame_ms=[1, 1], fps=1000., p95_ms=1.))
    monkeypatch.setattr('code.scripts.benchmark_realtime.environment', lambda: {'test': True})
    monkeypatch.setattr(run.subprocess, 'run', trial_process)
    run.benchmark(directory, protocol, repeats=2, measured=2, warmup=2)
    assert calls == ['baseline']*2 + ['ctrv_gate_nsa']*2 + ['c1c2c3']*2
    result = json.loads((directory / 'benchmark.json').read_text(encoding='utf-8'))
    assert result['fresh_process_per_trial'] and len(result['variants']) == 3
    run.benchmark(directory, protocol, repeats=2, measured=2, warmup=2)
    assert len(calls) == 6


def test_non_finite_external_state_counts_as_outside_roi():
    roi = {'enabled': True, 'polygon': [[0, 0], [800, 0], [800, 450], [0, 450]]}
    assert run.inside_roi([10, 10, 50, 50], roi)
    assert not run.inside_roi([np.nan, 10, np.nan, 50], roi)   # OC-SORT KF with a negative area
    assert not run.inside_roi([50, 10, 10, 50], roi)           # inverted
    tracker = SimpleNamespace(trackers=[SimpleNamespace(get_state=lambda: [[10, 10, 50, 50]]),
                                        SimpleNamespace(get_state=lambda: [[np.nan, 1, np.nan, 2]])])
    run.expire_external(tracker, 'ocsort', roi)                # used to raise ValueError
    assert len(tracker.trackers) == 1
