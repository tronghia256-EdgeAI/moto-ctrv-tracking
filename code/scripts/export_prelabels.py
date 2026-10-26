"""
code/scripts/export_prelabels.py
─────────────────────────────────
Phase 2 GT infrastructure: export tracker pre-labels + frame images for
CVAT annotation ("tools build infra, humans fix errors" workflow).

Takes the best tracker's full-video dump (ctrv_gate_nsa) and, for each
chosen segment, produces:

  <out>/<segment>/gt/gt.txt       MOT 1.1 annotations (CVAT-importable)
  <out>/<segment>/gt/labels.txt   class names, order fixed by PROJECT map
  <out>/<segment>/img1/*.jpg      frames (sequential grab — no ASF seek)
  <out>/<segment>_annotations.zip archive to upload into CVAT

Segments were chosen from the measured per-frame track-count profile of
the NKKN full-video run (global mean 25.2 tracks/frame):

  freeflow  [ 6400: 7000]  ~12 tracks/frame  (sparsest 600-frame window)
  mixed     [ 2500: 3100]  ~26 tracks/frame  (≈ global mean)
  dense     [10700:11300]  ~47 tracks/frame  (densest window; also lies
                            in the frames-9000+ held-out region)

MOT 1.1 line format (what CVAT's "MOT 1.1" importer reads):
  frame_id, track_id, x, y, w, h, not_ignored, class_id, visibility
with frame_id 1-based *within the segment*, class_id a 1-based index
into labels.txt, and (x, y) the box top-left in pixels.

Usage (repo root, venv active):
  python -m code.scripts.export_prelabels
  python -m code.scripts.export_prelabels --segments dense --no-images

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.scripts.compare_motion_models import open_video

# Order defines MOT class ids: class_id = project_class + 1 (1-based).
LABELS = ["car", "motorcycle", "bus", "truck", "accident"]

DEFAULT_TRACKS = "experiments/c2_final/tracks_ctrv_gate_nsa.json"

# segment name -> [start, end) in absolute frame indices of the dump.
SEGMENTS = {
    "freeflow": (6400, 7000),
    "mixed": (2500, 3100),
    "dense": (10700, 11300),
}


def write_mot(frames: list, out_dir: Path) -> dict:
    """Write a segment's per-frame track lists as MOT 1.1 gt/gt.txt.

    `frames` is the tracks-dump slice: per frame, a list of
    (track_id, class_id0, [x1, y1, x2, y2]). Returns simple stats.
    """
    gt_dir = out_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    ids = set()
    n_boxes = 0
    with open(gt_dir / "gt.txt", "w", encoding="utf-8") as f:
        for k, entries in enumerate(frames, start=1):
            for tid, cls, (x1, y1, x2, y2) in entries:
                f.write(f"{k},{tid},{x1:.1f},{y1:.1f},"
                        f"{x2 - x1:.1f},{y2 - y1:.1f},1,{cls + 1},1.0\n")
                ids.add(tid)
                n_boxes += 1
    (gt_dir / "labels.txt").write_text("\n".join(LABELS) + "\n",
                                       encoding="utf-8")
    return {"frames": len(frames), "boxes": n_boxes, "track_ids": len(ids)}


def zip_annotations(out_dir: Path, name: str) -> Path:
    """CVAT wants the MOT archive with gt/gt.txt at the top level."""
    zpath = out_dir.parent / f"{name}_annotations.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "gt" / "gt.txt", "gt/gt.txt")
        z.write(out_dir / "gt" / "labels.txt", "gt/labels.txt")
    return zpath


def extract_frames(video: str, segments: dict[str, tuple[int, int]],
                   out_root: Path, quality: int = 90) -> None:
    """One sequential pass over the video, saving JPEGs for every frame
    inside any requested segment. Never seeks (ASF keyframe quirk)."""
    import cv2
    last = max(end for _, end in segments.values())
    cap = open_video(video, 0)
    n = 0
    while n < last:
        ok, frame = cap.read()
        if not ok:
            break
        for name, (a, b) in segments.items():
            if a <= n < b:
                img_dir = out_root / name / "img1"
                img_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_dir / f"{n - a + 1:06d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
        n += 1
        if n % 2000 == 0:
            print(f"  frame {n}/{last}", flush=True)
    cap.release()
    if n < last:
        raise RuntimeError(f"video ended at frame {n}, needed {last}")
    # One zip per segment: CVAT (cloud or local) takes it as task data
    # directly — much easier than drag-dropping 600 individual files.
    for name in segments:
        img_dir = out_root / name / "img1"
        zpath = out_root / f"{name}_images.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for jpg in sorted(img_dir.glob("*.jpg")):
                z.write(jpg, jpg.name)
        print(f"  {name:9s} images zipped -> {zpath}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=DEFAULT_TRACKS,
                    help="full-video tracks_*.json dump to slice")
    ap.add_argument("--video", default=None,
                    help="video path (default: taken from the dump's meta)")
    ap.add_argument("--out", default="data/annotations/prelabels")
    ap.add_argument("--segments", nargs="*", default=list(SEGMENTS),
                    choices=list(SEGMENTS))
    ap.add_argument("--no-images", action="store_true",
                    help="skip frame extraction (annotations only)")
    args = ap.parse_args()

    with open(args.tracks, encoding="utf-8") as f:
        dump = json.load(f)
    assert dump.get("start_frame", 0) == 0, \
        "dump must start at frame 0 for absolute segment indices"
    frames = dump["frames"]
    out_root = Path(args.out)
    segs = {name: SEGMENTS[name] for name in args.segments}

    print(f"Slicing {args.tracks} ({len(frames)} frames)")
    for name, (a, b) in segs.items():
        assert b <= len(frames), f"{name} ends past the dump"
        seg_dir = out_root / name
        stats = write_mot(frames[a:b], seg_dir)
        zpath = zip_annotations(seg_dir, name)
        print(f"  {name:9s} [{a}:{b}] {stats['boxes']} boxes, "
              f"{stats['track_ids']} track IDs -> {zpath}")

    if not args.no_images:
        video = args.video or dump["video"]
        print(f"Extracting frames from {video} (sequential, no seek)")
        extract_frames(video, segs, out_root)
        for name in segs:
            n_img = len(list((out_root / name / "img1").glob("*.jpg")))
            print(f"  {name:9s} {n_img} frames -> {out_root / name / 'img1'}")

    print("\nNext: follow docs/GT_ANNOTATION_GUIDE.md to load these into "
          "CVAT and fix the pre-labels.")


if __name__ == "__main__":
    main()
