"""
code/utils/synthetic.py
───────────────────────
Synthetic trajectory generators shared by the tests and the experiment
harness (run_experiment.py --synthetic).

All trajectories are sequences of bbox centers (n, 2) in pixels. Detections
are N×6 arrays [x1, y1, x2, y2, score, class_id] — the same format the
pipeline's Detector produces from YOLO11.
"""

from __future__ import annotations

import numpy as np

# Project class ids (match code/detection contract)
CLASS_CAR = 0
CLASS_MOTORCYCLE = 1
CLASS_BUS = 2
CLASS_TRUCK = 3


def straight(
    v: float = 5.0,
    n: int = 100,
    noise: float = 0.0,
    start: tuple[float, float] = (100.0, 100.0),
    heading: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Constant-velocity straight line. Returns centers (n, 2)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    cx = start[0] + v * np.cos(heading) * t
    cy = start[1] + v * np.sin(heading) * t
    centers = np.stack([cx, cy], axis=1)
    if noise > 0:
        centers = centers + rng.normal(0.0, noise, size=centers.shape)
    return centers


def constant_turn(
    v: float = 5.0,
    omega: float = 0.05,
    n: int = 100,
    noise: float = 0.0,
    start: tuple[float, float] = (300.0, 300.0),
    heading0: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Circular arc: constant speed v, constant turn rate omega (rad/frame)."""
    rng = np.random.default_rng(seed)
    centers = np.empty((n, 2))
    px, py, theta = start[0], start[1], heading0
    for k in range(n):
        centers[k] = (px, py)
        px += v * np.cos(theta)
        py += v * np.sin(theta)
        theta += omega
    if noise > 0:
        centers = centers + rng.normal(0.0, noise, size=centers.shape)
    return centers


def weaving(
    v: float = 5.0,
    amp: float = 0.6,
    period: float = 60.0,
    n: int = 300,
    noise: float = 0.0,
    start: tuple[float, float] = (50.0, 400.0),
    seed: int = 0,
) -> np.ndarray:
    """
    Sinusoidal heading — the motorcycle lane-weaving failure case.

    heading(k) = amp * sin(2*pi*k / period), speed constant.
    """
    rng = np.random.default_rng(seed)
    centers = np.empty((n, 2))
    px, py = start
    for k in range(n):
        centers[k] = (px, py)
        theta = amp * np.sin(2.0 * np.pi * k / period)
        px += v * np.cos(theta)
        py += v * np.sin(theta)
    if noise > 0:
        centers = centers + rng.normal(0.0, noise, size=centers.shape)
    return centers


def centers_to_bboxes(centers: np.ndarray, wh: tuple[float, float] = (40.0, 60.0)) -> np.ndarray:
    """(n, 2) centers → (n, 4) [x1, y1, x2, y2] with fixed width/height."""
    w, h = wh
    x1 = centers[:, 0] - w / 2.0
    y1 = centers[:, 1] - h / 2.0
    return np.stack([x1, y1, x1 + w, y1 + h], axis=1)


def to_detections(
    trajectories: list[np.ndarray],
    wh: tuple[float, float] = (40.0, 60.0),
    scores: float | list[float] = 0.9,
    class_ids: int | list[int] = CLASS_MOTORCYCLE,
    drop_frames: dict[int, list[int]] | None = None,
) -> list[np.ndarray]:
    """
    Convert per-object center trajectories into per-frame detection arrays.

    Args:
        trajectories: list of (n, 2) center arrays (same n for all objects).
        wh          : bbox width/height applied to every object.
        scores      : detection score, scalar or per-object list.
        class_ids   : class id, scalar or per-object list.
        drop_frames : {object_index: [frame indices to drop]} — simulates
                      missed detections / occlusion.

    Returns:
        List of length n; element k is an (M_k, 6) array
        [x1, y1, x2, y2, score, class_id] of detections visible at frame k.
    """
    n_obj = len(trajectories)
    n = len(trajectories[0])
    if np.isscalar(scores):
        scores = [float(scores)] * n_obj
    if np.isscalar(class_ids):
        class_ids = [int(class_ids)] * n_obj
    drop_frames = drop_frames or {}
    drop_sets = {i: set(f) for i, f in drop_frames.items()}

    bboxes = [centers_to_bboxes(tr, wh) for tr in trajectories]
    frames: list[np.ndarray] = []
    for k in range(n):
        rows = []
        for i in range(n_obj):
            if k in drop_sets.get(i, ()):
                continue
            x1, y1, x2, y2 = bboxes[i][k]
            rows.append([x1, y1, x2, y2, scores[i], class_ids[i]])
        frames.append(np.array(rows, dtype=float).reshape(-1, 6))
    return frames
