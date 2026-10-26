"""
code/scripts/audit_camera_overlap.py
─────────────────────────────────────
Does the detector's training export contain frames from any raw CCTV clip?

The manuscript claims the training set "contains no frames from the evaluation
camera"; until this audit that rested on a filename search alone.

Method: 64-bit DCT perceptual hash (pHash) of every training image and of one
frame per second of every clip. Roboflow letterboxed each image into 640x640
with white edges ("Fit (white edges)"), so training images are cropped to their
non-white content first; horizontal flips are hashed too (augmentation). A fixed
camera's background dominates the hash, so an image taken from a clip's camera
lands within a few bits of that clip's frames, while other scenes land far away.
Crops and rotations are not covered.

Positive control: the export holds images named after the two 7_20_2017 clips.
If those do not match their clips, the instrument is not trusted.

Usage (from the repo root):
    python -m code.scripts.audit_camera_overlap
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "valid", "test")
VIDEO_EXT = {".asf", ".mkv", ".mov", ".mp4", ".avi"}
# Training-image name prefix -> clip it must match (the positive control).
CONTROL = {"7_20_2017-11_59_59": "7_20_2017 11_59_59 AM",
           "7_20_2017-4_59_59": "7_20_2017 4_59_59 PM"}


def phash(gray: np.ndarray) -> np.uint64:
    """64-bit pHash: 32x32 grey -> top-left 8x8 DCT block thresholded at its median."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    block = cv2.dct(small)[:8, :8].ravel()
    return np.packbits(block > np.median(block)).view(np.uint64)[0]


def content_crop(img: np.ndarray, white: int = 245) -> np.ndarray:
    """Drop the white letterbox bands: keep rows/columns with any non-white pixel."""
    mask = (img < white).any(axis=2)
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def image_hashes(path: Path) -> tuple[np.uint64, np.uint64]:
    """(hash, hash of the horizontal flip) of one training image."""
    grey = cv2.cvtColor(content_crop(cv2.imread(str(path))), cv2.COLOR_BGR2GRAY)
    return phash(grey), phash(np.ascontiguousarray(grey[:, ::-1]))


def clip_hashes(path: Path, step_sec: float) -> tuple[np.ndarray, int]:
    """Hash one frame every `step_sec` seconds; decode sequentially (ASF seeking
    is unreliable, see compare_motion_models.py). Returns (hashes, frames read)."""
    cap = cv2.VideoCapture(str(path))
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 25.0) * step_sec))
    hashes, i = [], 0
    while cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                hashes.append(phash(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        i += 1
    cap.release()
    return np.array(hashes, dtype=np.uint64), i


def min_distance(queries: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Smallest Hamming distance from each query hash to any reference hash."""
    return np.bitwise_count(queries[:, None] ^ refs[None, :]).min(axis=1)


def write_sheets(rows: list, clips: list[Path], images_root: Path, out: Path,
                 max_bits: int, tile=(320, 180), cols: int = 4) -> None:
    """Contact sheet per clip: one clip frame (tile 0) next to every training
    image within `max_bits` of that clip, for the visual verdict. The sheets
    show dataset images, so they go under gitignored data/, never experiments/."""
    out.mkdir(parents=True, exist_ok=True)
    tw, th = tile
    for j, clip in enumerate(clips):
        cands = sorted((int(r[4 + j]), r[0], r[1]) for r in rows if int(r[4 + j]) <= max_bits)
        if not cands:
            continue
        cap = cv2.VideoCapture(str(clip))
        for _ in range(250):
            cap.grab()
        ok, ref = cap.retrieve()
        cap.release()
        tiles = [(ref, f"CLIP {clip.name[:30]}")] + [
            (content_crop(cv2.imread(str(images_root / s / "images" / n))), f"d={d} {s} {n[:22]}")
            for d, s, n in cands]
        sheet = np.full((-(-len(tiles) // cols) * (th + 18), cols * tw, 3), 255, np.uint8)
        for i, (im, label) in enumerate(tiles):
            y, x = (i // cols) * (th + 18), (i % cols) * tw
            sheet[y + 18:y + 18 + th, x:x + tw] = cv2.resize(im, (tw, th))
            cv2.putText(sheet, label, (x + 2, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 200) if i == 0 else (0, 0, 0), 1)
        cv2.imwrite(str(out / f"sheet_{clip.stem.replace(' ', '_')}.png"), sheet)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--videos", default="data/raw/videos")
    ap.add_argument("--images", default="data/annotations/accident_vehicle")
    ap.add_argument("--step-sec", type=float, default=1.0)
    ap.add_argument("--threshold", type=int, default=10,
                    help="match = best distance <= this many of 64 bits")
    ap.add_argument("--out", default="experiments/detector_leakage")
    ap.add_argument("--sheets", default=None, metavar="DIR",
                    help="only render contact sheets from an existing overlap.csv "
                         "(keep DIR under gitignored data/)")
    ap.add_argument("--sheet-bits", type=int, default=12)
    args = ap.parse_args()

    if args.sheets:
        with open(Path(args.out) / "overlap.csv", encoding="utf-8") as f:
            header, *rows = list(csv.reader(f))
        clips = [Path(args.videos) / name for name in header[4:]]
        write_sheets(rows, clips, Path(args.images), Path(args.sheets), args.sheet_bits)
        return

    images = [(s, p) for s in SPLITS
              for p in sorted((Path(args.images) / s / "images").glob("*"))]
    hashes = np.array([image_hashes(p) for _, p in images], dtype=np.uint64)  # (N, 2)
    print(f"hashed {len(images)} training images")

    clips = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in VIDEO_EXT)
    dist = np.empty((len(images), len(clips)), dtype=np.int64)
    clip_info = {}
    for j, clip in enumerate(clips):
        refs, n_read = clip_hashes(clip, args.step_sec)
        dist[:, j] = np.minimum(min_distance(hashes[:, 0], refs),
                                min_distance(hashes[:, 1], refs))
        clip_info[clip.name] = {"frames_read": n_read, "frames_hashed": len(refs)}
        print(f"{clip.name}: {n_read} frames read, {len(refs)} hashed")

    best = dist.argmin(axis=1)
    best_d = dist[np.arange(len(images)), best]
    matched = best_d <= args.threshold

    for j, clip in enumerate(clips):
        hit = matched & (best == j)
        clip_info[clip.name]["matched_images"] = {
            s: int((hit & np.array([sp == s for sp, _ in images])).sum()) for s in SPLITS}

    control = {}
    for prefix, clip_key in CONTROL.items():
        rows = [i for i, (_, p) in enumerate(images) if p.name.startswith(prefix)]
        ok = [i for i in rows if matched[i] and clip_key in clips[best[i]].name]
        control[prefix] = {"images": len(rows), "matched_own_clip": len(ok),
                           "max_distance_to_own_clip": int(max(
                               (dist[i, [j for j, c in enumerate(clips) if clip_key in c.name]].min()
                                for i in rows), default=-1))}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "overlap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "best_clip", "best_distance"] + [c.name for c in clips])
        for (s, p), b, d, row in zip(images, best, best_d, dist):
            w.writerow([s, p.name, clips[b].name, int(d)] + [int(x) for x in row])
    summary = {
        "threshold_bits": args.threshold, "step_sec": args.step_sec,
        "training_images": len(images),
        "best_distance_histogram": np.bincount(best_d, minlength=65).tolist(),
        "clips": clip_info, "positive_control": control,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"clips": clip_info, "positive_control": control}, indent=2))


if __name__ == "__main__":
    main()
