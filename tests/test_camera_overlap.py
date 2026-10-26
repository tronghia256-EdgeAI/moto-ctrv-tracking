"""The camera-overlap audit's hash must survive Roboflow's letterboxing and tell
different scenes apart (code/scripts/audit_camera_overlap.py)."""

import cv2
import numpy as np

from code.scripts.audit_camera_overlap import content_crop, min_distance, phash


def _scene(seed):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 200, (9, 16, 3), dtype=np.uint8)   # low-frequency "scene"
    return cv2.resize(small, (800, 450), interpolation=cv2.INTER_CUBIC)


def _letterbox_white(img, size=640):
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y = (size - small.shape[0]) // 2
    canvas[y:y + small.shape[0], :small.shape[1]] = small
    return canvas


def _hash(img):
    return np.array([phash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))], dtype=np.uint64)


def test_letterboxed_copy_matches_and_other_scene_does_not():
    frame = _scene(0)
    boxed = content_crop(_letterbox_white(frame))
    assert boxed.shape[:2] == (360, 640)                     # white bands removed
    assert min_distance(_hash(boxed), _hash(frame))[0] <= 4
    assert min_distance(_hash(_scene(1)), _hash(frame))[0] > 16
