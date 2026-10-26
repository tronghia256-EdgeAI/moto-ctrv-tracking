"""
code/scripts/gt_review_queue.py
────────────────────────────────
Phase 2 GT infrastructure: turn CVAT pre-labels into a *ranked list of
places to look*, so producing GT means visiting ~100 flagged spots instead
of scrubbing 600 frames of playback.

Why this script and not "auto-GT": the pre-labels ARE the output of the
tracker under evaluation, so every error left in them is an error the
tracker made. Anything that finishes GT automatically leaves those errors
in place and then scores the tracker against them — circular, and inflated
exactly on the identity metrics C2 claims. So this script never decides
what is correct. It only finds the *structural signatures* of likely
errors and routes human attention; the human makes every GT call in CVAT.

Six checks, ordered by how much they matter to the C2 claim:

  MERGE      a track dies mid-scene and another is born nearby moments
             later -> one vehicle, two IDs. The `M` fix, and the exact
             failure C2 claims to reduce.
  SPLIT      the opposite error: one ID whose box teleports or rescales
             across a gap -> several vehicles chained into one identity.
             Costly to miss, because an over-merged ground truth rewards
             whichever tracker bridges gaps most aggressively.
  JUMP       a track's box moves or resizes far outside that segment's
             own distribution -> a bad box, or an ID that stepped onto a
             different vehicle. Neither leaves a birth or a death, so
             playback almost never reveals them.
  DUP        two IDs on the same vehicle in one frame.
  GHOST      a track only a few frames long -> likely a false positive.
  CLASSFLIP  a track's class changes mid-life -> GT needs one class.
  ORPHAN     interior birth/death with no partner -> either a genuine
             occlusion (leave the break, it is correct per the guide) or
             a vehicle the detector missed (draw it).

Detection geometry: MERGE compares a dead track's *motion-extrapolated*
position against a newborn's first box, normalised by box size, instead of
raw IoU. A weaving motorcycle that reappears one body-length downstream
has IoU 0 with its own last box; the re-id proxy in
compare_motion_models.py would miss it, a review queue must not.

JUMP thresholds are calibrated per segment from the data itself rather
than fixed: on `dense` the centre-jump distribution tops out at 0.67
box-diagonals/frame (median 0.07), so any limit picked by intuition
either fires constantly or never. See robust_threshold for why the
calibration is median+MAD and not a percentile.

Usage (repo root, venv active):
  python -m code.scripts.gt_review_queue --segment dense
  python -m code.scripts.gt_review_queue \
      --prelabels data/annotations/prelabels/dense/gt/gt.txt \
      --frame-size 800 450 --offset 10700

Writes, next to the pre-labels:
  review_queue.md    checklist to work through in CVAT
  review_queue.csv   same rows, machine readable

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.scripts.import_gt import CLASS_NAMES, load_mot

# Keep in sync with export_prelabels.SEGMENTS (imported lazily would drag
# in cv2 + ultralytics through compare_motion_models; these are fixed by
# the already-exported zips).
SEGMENTS = {
    "freeflow": (6400, 7000),
    "mixed": (2500, 3100),
    "dense": (10700, 11300),
}

# ── Thresholds ───────────────────────────────────────────────────────
# GAP_MAX/SHORT_TRACK mirror compare_motion_models.py so the queue and
# the re-id proxy talk about the same phenomena.
GAP_MAX = 40          # frames: widest death->birth gap still worth pairing
SHORT_TRACK = 10      # lifespan below this is churn (GHOST candidate)
MERGE_DIST_MAX = 3.0  # predicted-vs-actual centre gap, in box diagonals
MERGE_AREA_RATIO = 3.0    # larger area mismatch kills a merge candidate
MERGE_CONF_MIN = 0.10     # drop candidates below this confidence
DUP_IOU = 0.65        # two IDs this overlapped are probably one vehicle
BORDER_MARGIN = 24    # px: births/deaths this close to an edge are normal
VEL_WINDOW = 5        # frames used to estimate a track's exit velocity

# JUMP is calibrated off the segment's own distribution (see
# robust_threshold), floored so that a very clean segment does not start
# flagging perfectly good boxes.
JUMP_MAD_K = 10.0
JUMP_DIST_FLOOR = 0.35    # box diagonals per frame
JUMP_AREA_FLOOR = 2.30    # area ratio across <=2 frames

# Most interior breaks in dense traffic are occlusions, which the guide
# says to leave alone, and sub-8px vehicles are not labelled at all. Only
# surface breaks on boxes big enough that a miss would be visible.
ORPHAN_MIN_DIAG = 40.0    # px

# SPLIT: raw (not per-frame) displacement across an occlusion gap.
# A real vehicle reappears near where its motion would put it and at a
# similar scale; a different vehicle does not.
SPLIT_MIN_GAP = 5         # frames: shorter gaps are ordinary misses
SPLIT_MOVE_MAX = 2.0      # box diagonals of raw displacement
SPLIT_AREA_MAX = 2.0      # area ratio across the gap

# Ordering of the checklist. Lower sorts first.
PRIORITY = {"MERGE": 0, "SPLIT": 1, "JUMP": 2, "DUP": 3,
            "GHOST": 4, "CLASSFLIP": 5, "ORPHAN": 6}

# Rough seconds a human spends per stop, for the time estimate.
SECONDS_PER_STOP = {"MERGE": 30, "SPLIT": 35, "JUMP": 35, "DUP": 20,
                    "GHOST": 10, "CLASSFLIP": 15, "ORPHAN": 25}


def robust_threshold(values: list[float], k: float, floor: float) -> float:
    """median + k · MAD, floored — "unusual for this segment".

    Deliberately not a high percentile. A percentile is computed *from*
    the outliers, so with few samples it lands on the very thing it
    should flag: a 30-step track containing one teleport has its p99.9
    equal to that teleport, and the check goes silent exactly when it
    matters. Median and MAD describe the bulk of the distribution, so no
    outlier can raise the bar that catches it.

    On the real `dense` segment the two agree anyway (0.42 here vs 0.39
    for p99.9, 23 vs 24 hits) — the difference is robustness on sparse
    segments like `freeflow`, not on this one.
    """
    if not values:
        return floor
    s = sorted(values)
    mid = len(s) // 2
    med = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0
    dev = sorted(abs(v - med) for v in s)
    mad = dev[mid] if len(dev) % 2 else (dev[mid - 1] + dev[mid]) / 2.0
    return max(floor, med + k * mad)


# ── Track container ──────────────────────────────────────────────────

@dataclass
class Track:
    """One pre-label track: parallel per-frame boxes and classes."""
    tid: int
    frames: list[int] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    classes: list[int] = field(default_factory=list)

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]

    @property
    def span(self) -> int:
        return self.frames[-1] - self.frames[0] + 1

    @property
    def cls(self) -> int:
        """Majority class — what GT should collapse the track to."""
        return Counter(self.classes).most_common(1)[0][0]

    def velocity(self) -> tuple[float, float]:
        """Centre velocity in px/frame from the last VEL_WINDOW samples.

        Used to extrapolate where a dying track *would* be when a
        candidate successor is born, which is what makes the pairing work
        across gaps where the boxes no longer overlap at all.
        """
        if len(self.frames) < 2:
            return 0.0, 0.0
        i = max(0, len(self.frames) - VEL_WINDOW)
        dt = self.frames[-1] - self.frames[i]
        if dt <= 0:
            return 0.0, 0.0
        (x0, y0), (x1, y1) = _centre(self.boxes[i]), _centre(self.boxes[-1])
        return (x1 - x0) / dt, (y1 - y0) / dt


def _centre(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _diag(box) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1)


def _area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a, b) -> float:
    """Local IoU on plain tuples — the DUP check runs it O(n^2) per frame
    (~660k calls over the segment), where np.asarray round-trips dominate."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _touches_border(box, w: int, h: int, margin: int = BORDER_MARGIN) -> bool:
    """A track appearing/vanishing at an image edge is ordinary traffic
    entering or leaving frame — not an annotation error."""
    x1, y1, x2, y2 = box
    return (x1 <= margin or y1 <= margin
            or x2 >= w - margin or y2 >= h - margin)


def build_tracks(frames: dict[int, list]) -> dict[int, Track]:
    tracks: dict[int, Track] = {}
    for fid in sorted(frames):
        for tid, box, cls in frames[fid]:
            t = tracks.setdefault(tid, Track(tid))
            t.frames.append(fid)
            t.boxes.append(tuple(box))
            t.classes.append(cls)
    return tracks


# ── Findings ─────────────────────────────────────────────────────────

@dataclass
class Finding:
    kind: str
    frame: int              # MOT frame (1-based within segment)
    ids: tuple[int, ...]
    cls: str
    conf: float             # 0..1, how likely this is a real error
    detail: str             # frame numbers written as @N (MOT); see to_cvat
    action: str


def to_cvat(text: str) -> str:
    """Rewrite every `@N` frame reference from MOT to CVAT numbering.

    Detectors work in MOT frame numbers (1-based, matching gt.txt), but
    every number in the report is something the annotator types into
    CVAT's frame box, which counts from 0. Tagging references at the
    point they are written keeps the two conventions from mixing in the
    prose — sending someone to the wrong frame is the one failure that
    makes the whole queue untrustworthy.
    """
    return re.sub(r"@(\d+)", lambda m: str(int(m.group(1)) - 1), text)


def find_merges(tracks: dict[int, Track], n_frames: int,
                w: int, h: int) -> list[Finding]:
    """Pair interior deaths with interior births: one vehicle, two IDs.

    For each dead track we extrapolate its last known motion forward to a
    candidate's birth frame and measure the miss distance in box
    diagonals, so the test is scale-free (a 15 px motorcycle in the
    mid-field and a 90 px car in the foreground are judged alike).
    Confidence decays with distance and with gap length, and is halved
    when the two tracks disagree on class.
    """
    deaths = [t for t in tracks.values()
              if t.last < n_frames and not _touches_border(t.boxes[-1], w, h)]
    births = [t for t in tracks.values()
              if t.first > 1 and not _touches_border(t.boxes[0], w, h)]
    births_by_frame: dict[int, list[Track]] = {}
    for t in births:
        births_by_frame.setdefault(t.first, []).append(t)

    cands: list[tuple[float, Track, Track, int, float]] = []
    for d in deaths:
        vx, vy = d.velocity()
        cx, cy = _centre(d.boxes[-1])
        scale = max(_diag(d.boxes[-1]), 1e-6)
        for g in range(1, GAP_MAX + 1):
            for b in births_by_frame.get(d.last + g, ()):
                if b.tid == d.tid:
                    continue
                bx, by = _centre(b.boxes[0])
                dist = math.hypot(cx + vx * g - bx, cy + vy * g - by)
                norm = dist / max(scale, _diag(b.boxes[0]), 1e-6)
                if norm > MERGE_DIST_MAX:
                    continue
                a_d, a_b = _area(d.boxes[-1]), _area(b.boxes[0])
                ratio = max(a_d, a_b) / max(min(a_d, a_b), 1e-6)
                if ratio > MERGE_AREA_RATIO:
                    continue
                conf = (math.exp(-norm / 1.5) * math.exp(-g / 20.0)
                        * (1.0 if d.cls == b.cls else 0.5)
                        * (1.0 / (1.0 + 0.5 * (ratio - 1.0))))
                if conf >= MERGE_CONF_MIN:
                    cands.append((conf, d, b, g, norm))

    # Greedy one-to-one: a dead track has at most one true successor, and
    # a newborn at most one predecessor. Taking the most confident pair
    # first avoids proposing a chain of contradictory merges.
    cands.sort(key=lambda c: -c[0])
    used_d: set[int] = set()
    used_b: set[int] = set()
    out: list[Finding] = []
    for conf, d, b, g, norm in cands:
        if d.tid in used_d or b.tid in used_b:
            continue
        used_d.add(d.tid)
        used_b.add(b.tid)
        same = "cung class" if d.cls == b.cls else \
            f"KHAC class ({CLASS_NAMES.get(d.cls)}/{CLASS_NAMES.get(b.cls)})"
        out.append(Finding(
            "MERGE", d.last, (d.tid, b.tid), CLASS_NAMES.get(d.cls, "?"), conf,
            f"ID {d.tid} chet o frame @{d.last}, ID {b.tid} sinh o frame "
            f"@{b.first} (gap {g}f, lech {norm:.1f} box, {same})",
            f"click box ID {d.tid} -> M -> tua toi frame @{b.first} -> "
            f"click box ID {b.tid} -> M",
        ))
    return out


def _steps(tracks: dict[int, Track], w: int, h: int):
    """Yield (track, frame, jump, area_ratio, dt) for every consecutive
    pair, jump in box-diagonals per frame.

    Steps where either box touches an image edge are skipped: a vehicle
    driving out of frame gets clipped, so its area collapses over a
    couple of frames. Both of the largest area changes in `dense` were
    exactly that — cars leaving the bottom edge, not identity errors.
    """
    for t in tracks.values():
        for i in range(len(t.frames) - 1):
            dt = t.frames[i + 1] - t.frames[i]
            if dt <= 0:
                continue
            if (_touches_border(t.boxes[i], w, h, margin=2)
                    or _touches_border(t.boxes[i + 1], w, h, margin=2)):
                continue
            (x0, y0), (x1, y1) = _centre(t.boxes[i]), _centre(t.boxes[i + 1])
            jump = (math.hypot(x1 - x0, y1 - y0)
                    / (max(_diag(t.boxes[i]), 1e-6) * dt))
            a0, a1 = _area(t.boxes[i]), _area(t.boxes[i + 1])
            ratio = max(a0, a1) / max(min(a0, a1), 1e-6)
            yield t, t.frames[i + 1], jump, ratio, dt


def find_jumps(tracks: dict[int, Track], w: int,
               h: int) -> tuple[list[Finding], dict]:
    """Within-track anomalies: the ID stayed, the box did something odd.

    Two passes. The first measures how this segment actually moves; the
    second flags only what is unusual for it. That matters because the
    honest interpretation of a hit is "most unusual step in the segment",
    not "definitely an ID switch" — a box can also shrink 3x because a bus
    passed in front of it, which is a box to fix rather than an identity
    error. Both are worth 30 seconds of a human's eye; neither is decided
    here.

    Only the worst step per track is reported, so a drifting track costs
    one stop rather than a run of near-duplicate rows.
    """
    all_jumps = [j for _, _, j, _, _ in _steps(tracks, w, h)]
    all_areas = [r for _, _, _, r, dt in _steps(tracks, w, h) if dt <= 2]
    thr_jump = robust_threshold(all_jumps, JUMP_MAD_K, JUMP_DIST_FLOOR)
    thr_area = robust_threshold(all_areas, JUMP_MAD_K, JUMP_AREA_FLOOR)

    worst: dict[int, tuple] = {}
    for t, frame, jump, ratio, dt in _steps(tracks, w, h):
        hit_jump = jump > thr_jump
        hit_area = dt <= 2 and ratio > thr_area
        if not (hit_jump or hit_area):
            continue
        # Only let a signal contribute to severity if it actually fired:
        # across a long gap the area ratio means nothing (the vehicle has
        # genuinely changed size), and folding it in inflated confidence
        # on rows whose only real trigger was a marginal jump.
        sev = max(jump / thr_jump if hit_jump else 0.0,
                  ratio / thr_area if hit_area else 0.0)
        if t.tid not in worst or sev > worst[t.tid][0]:
            worst[t.tid] = (sev, t, frame, jump, ratio, dt)

    out: list[Finding] = []
    for sev, t, frame, jump, ratio, dt in worst.values():
        reason = []
        if jump > thr_jump:
            reason.append(f"nhay {jump:.2f} box/frame (nguong {thr_jump:.2f})")
        if dt <= 2 and ratio > thr_area:
            reason.append(f"dien tich doi {ratio:.1f}x (nguong {thr_area:.1f})")
        out.append(Finding(
            "JUMP", frame, (t.tid,), CLASS_NAMES.get(t.cls, "?"),
            min(1.0, (sev - 1.0) / 2.0),
            f"ID {t.tid} tai frame @{frame}: " + ", ".join(reason)
            + (f" (cach frame truoc {dt}f)" if dt > 1 else ""),
            f"xem ID {t.tid} truoc/sau frame @{frame}: box nhay sang xe khac "
            f"-> Split roi merge lai; box chi bi che/lech -> keo lai cho khit",
        ))
    return out, {"jump": thr_jump, "area": thr_area}


def find_splits(tracks: dict[int, Track]) -> list[Finding]:
    """One identity that is really several vehicles chained together.

    JUMP cannot catch these: it divides displacement by dt, so a box that
    teleports across a 29-frame occlusion looks unremarkable per frame.
    Here we deliberately use the *raw* displacement across the gap,
    measured in box diagonals, plus the area ratio.

    This is the error that matters most for an ablation. A ground-truth
    identity that wrongly spans two vehicles rewards any tracker willing
    to bridge the gap between them, and penalises a tracker that
    correctly refuses — so it flatters exactly the mechanisms designed to
    re-associate across occlusion, in the direction of overstating them.
    """
    out: list[Finding] = []
    for t in tracks.values():
        worst = None
        for i in range(len(t.frames) - 1):
            gap = t.frames[i + 1] - t.frames[i]
            if gap <= SPLIT_MIN_GAP:
                continue
            (x0, y0), (x1, y1) = _centre(t.boxes[i]), _centre(t.boxes[i + 1])
            d0 = max(_diag(t.boxes[i]), 1e-6)
            move = math.hypot(x1 - x0, y1 - y0) / d0
            a0, a1 = _area(t.boxes[i]), _area(t.boxes[i + 1])
            ratio = max(a0, a1) / max(min(a0, a1), 1e-6)
            if move <= SPLIT_MOVE_MAX and ratio <= SPLIT_AREA_MAX:
                continue
            sev = max(move / SPLIT_MOVE_MAX, ratio / SPLIT_AREA_MAX)
            if worst is None or sev > worst[0]:
                worst = (sev, t.frames[i], t.frames[i + 1], move, ratio, gap)
        if worst is None:
            continue
        sev, f0, f1, move, ratio, gap = worst
        why = []
        if move > SPLIT_MOVE_MAX:
            why.append(f"nhay {move:.1f} box qua khoang trong")
        if ratio > SPLIT_AREA_MAX:
            why.append(f"kich thuoc doi {ratio:.1f}x")
        out.append(Finding(
            "SPLIT", f1, (t.tid,), CLASS_NAMES.get(t.cls, "?"),
            min(1.0, (sev - 1.0) / 2.0),
            f"ID {t.tid} bien mat o frame @{f0} roi hien lai o @{f1} "
            f"(cach {gap}f): " + ", ".join(why),
            f"so sanh xe o @{f0} va @{f1}: neu la HAI xe khac nhau -> "
            f"Split track tai @{f1} thanh 2 ID",
        ))
    return out


def find_dups(frames: dict[int, list], tracks: dict[int, Track]) -> list[Finding]:
    """Two IDs on one vehicle. Reported once per ID pair, at first sight."""
    seen: dict[tuple[int, int], list] = {}
    for fid in sorted(frames):
        entries = frames[fid]
        for i in range(len(entries)):
            tid_a, box_a, _ = entries[i]
            for j in range(i + 1, len(entries)):
                tid_b, box_b, _ = entries[j]
                ov = _iou(tuple(box_a), tuple(box_b))
                if ov < DUP_IOU:
                    continue
                key = (min(tid_a, tid_b), max(tid_a, tid_b))
                if key in seen:
                    seen[key][1] += 1
                    seen[key][2] = max(seen[key][2], ov)
                else:
                    seen[key] = [fid, 1, ov]
    out: list[Finding] = []
    for (a, b), (fid, count, ov) in seen.items():
        cls = CLASS_NAMES.get(tracks[a].cls, "?") if a in tracks else "?"
        out.append(Finding(
            "DUP", fid, (a, b), cls, min(1.0, ov),
            f"ID {a} va ID {b} chong nhau IoU {ov:.2f} trong {count} frame, "
            f"tu frame @{fid}",
            f"neu la 1 xe: xoa track thua (hoac merge); neu that su 2 xe "
            f"che nhau: giu nguyen",
        ))
    return out


def find_ghosts(tracks: dict[int, Track], n_frames: int,
                w: int, h: int) -> list[Finding]:
    """Very short tracks — the churn that inflates false positives."""
    out: list[Finding] = []
    for t in tracks.values():
        if t.span >= SHORT_TRACK:
            continue
        # A short track that both enters and leaves at an edge is a
        # vehicle clipping the frame corner, not a phantom detection.
        if (_touches_border(t.boxes[0], w, h)
                and _touches_border(t.boxes[-1], w, h)):
            continue
        out.append(Finding(
            "GHOST", t.first, (t.tid,), CLASS_NAMES.get(t.cls, "?"),
            min(1.0, (SHORT_TRACK - t.span) / SHORT_TRACK),
            f"ID {t.tid} chi song {t.span} frame (@{t.first}-@{t.last}), "
            f"box ~{_diag(t.boxes[0]):.0f}px",
            f"neu khong co xe that o day -> xoa ca track (FP). Neu co xe "
            f"-> keo dai track cho du",
        ))
    return out


def find_classflips(tracks: dict[int, Track]) -> list[Finding]:
    out: list[Finding] = []
    for t in tracks.values():
        uniq = Counter(t.classes)
        if len(uniq) < 2:
            continue
        major, n_major = uniq.most_common(1)[0]
        flip_at = next(f for f, c in zip(t.frames, t.classes) if c != major)
        names = "/".join(f"{CLASS_NAMES.get(c, '?')}x{n}"
                         for c, n in uniq.most_common())
        out.append(Finding(
            "CLASSFLIP", flip_at, (t.tid,), CLASS_NAMES.get(major, "?"),
            1.0 - n_major / len(t.classes),
            f"ID {t.tid} doi class giua chung: {names}",
            f"chon 1 class cho ca track (nhieu kha nang: "
            f"{CLASS_NAMES.get(major, '?')})",
        ))
    return out


def find_orphans(tracks: dict[int, Track], merged: list[Finding],
                 n_frames: int, w: int, h: int,
                 min_diag: float = ORPHAN_MIN_DIAG) -> tuple[list[Finding], int]:
    """Interior births/deaths left unpaired by find_merges.

    Per the annotation guide a full occlusion *should* break the track, so
    these are not errors by default — they are places to check whether a
    vehicle actually went missing. In dense traffic almost every track
    breaks this way (296 of them here, median box 21 px diagonal), which
    would bury the queue in stops that are correct as-is. Boxes below
    `min_diag` are therefore counted but not listed: at that size the
    guide says not to label the vehicle at all.

    Returns (findings, n_suppressed).
    """
    paired = {tid for f in merged for tid in f.ids}
    out: list[Finding] = []
    suppressed = 0

    def add(frame: int, box, t: Track, kind: str) -> None:
        nonlocal suppressed
        d = _diag(box)
        if d < min_diag:
            suppressed += 1
            return
        if kind == "end":
            detail = (f"ID {t.tid} bien mat giua khung o frame @{frame} "
                      f"(box {d:.0f}px), khong tim thay track noi tiep")
            action = (f"tua qua frame @{frame}: neu xe bi che han -> de dut la "
                      f"dung; neu xe con thay -> ve tiep box")
        else:
            detail = (f"ID {t.tid} xuat hien giua khung o frame @{frame} "
                      f"(box {d:.0f}px), khong tim thay track truoc do")
            action = (f"tua lui truoc frame @{frame}: neu xe da co tu truoc "
                      f"-> ve box va merge ve cung ID")
        out.append(Finding("ORPHAN", frame, (t.tid,),
                           CLASS_NAMES.get(t.cls, "?"),
                           min(1.0, d / 120.0), detail, action))

    for t in tracks.values():
        if t.tid in paired or t.span < SHORT_TRACK:
            continue
        if t.last < n_frames and not _touches_border(t.boxes[-1], w, h):
            add(t.last, t.boxes[-1], t, "end")
        if t.first > 1 and not _touches_border(t.boxes[0], w, h):
            add(t.first, t.boxes[0], t, "start")
    return out, suppressed


# ── Report ───────────────────────────────────────────────────────────

KIND_TITLE = {
    "MERGE": "MERGE — track vo lam doi (viec chinh, lam truoc)",
    "SPLIT": "SPLIT — mot ID gom nhieu xe khac nhau (nguy hiem nhat)",
    "JUMP": "JUMP — box cua mot ID doi dot ngot (playback kho thay)",
    "DUP": "DUP — hai ID dinh cung mot xe",
    "GHOST": "GHOST — track qua ngan, nhieu kha nang la FP",
    "CLASSFLIP": "CLASSFLIP — mot track doi class giua chung",
    "ORPHAN": "ORPHAN — sinh/chet giua khung khong ghep duoc (kiem tra miss)",
}

KIND_NOTE = {
    "MERGE": "Moi dong = mot cap ID nghi la cung mot xe. Dung phim `M` "
             "hai lan. Day dung la thu C2 claim cai thien, nen sua het "
             "muc nay truoc khi lam gi khac.",
    "SPLIT": "Nguoc voi MERGE: mot ID nhay sang xe khac sau khi bi che. Loai nay nguy hiem nhat cho ablation — GT gop nham hai xe se **thuong** cho tracker nao dam noi qua cho che khuat, tuc la thoi phong dung cai ma C2 claim. Nhin ky hai frame roi Split neu la hai xe.",
    "JUMP": "Track khong dut, nhung box di chuyen/doi kich thuoc bat "
            "thuong so voi chinh doan nay. Hai kha nang, phai nhin moi "
            "biet: (a) ID that su nhay sang xe khac -> Split roi merge "
            "lai; (b) box chi bi che nen thu lai -> keo box cho khit. "
            "Vi khong co birth/death nen tua playback gan nhu khong bao "
            "gio thay loai nay.",
    "DUP": "Hai ID cung trum mot xe. Neu dung la mot xe, xoa bot mot "
           "track — de nguyen se thanh FP trong moi frame.",
    "GHOST": "Track song duoi 10 frame va khong cham mep hinh. Phan lon "
             "la detection rac; xoa nhanh, moi dong ~10 giay.",
    "CLASSFLIP": "GT can mot class cho ca track. Chon class dung roi doi "
                 "o sidebar phai.",
    "ORPHAN": "KHONG phai loi mac dinh: guide noi xe bi che han thi de "
              "track dut moi dung. Muc nay chi de kiem tra co phai xe bi "
              "miss hay khong. Lam sau cung, bo qua duoc neu het gio.",
}


def write_markdown(path: Path, findings: list[Finding], stats: dict,
                   offset: int, per_type: int) -> None:
    by_kind: dict[str, list[Finding]] = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(f)

    total_sec = sum(SECONDS_PER_STOP[f.kind] for f in findings)
    lines: list[str] = []
    lines.append(f"# Review queue — segment `{stats['segment']}`\n")
    lines.append(
        f"Sinh boi `python -m code.scripts.gt_review_queue` tu "
        f"`{stats['prelabels']}`.\n"
        f"{stats['boxes']} box · {stats['tracks']} track ID · "
        f"{stats['frames']} frame.\n")
    lines.append(
        f"**{len(findings)} cho can nhin** (thay vi tua het "
        f"{stats['frames']} frame) — uoc luong "
        f"~{total_sec // 60} phut neu lam tuan tu.\n")
    lines.append(
        "> Danh sach nay **khong phai GT**. No chi noi *cho nao dang ngo*; "
        "dung hay sai van do ban quyet dinh khi nhin anh. Cho nao nhin thay "
        "pre-label da dung thi tick va di tiep — mot dong bi bo qua khong "
        "lam hong GT, mot dong sua bua thi co.\n")

    lines.append("\n## Cach dung\n")
    lines.append(
        "1. Mo task `dense` trong CVAT.\n"
        "2. Go so frame vao o frame (goc tren) de nhay thang toi cho can xem.\n"
        f"3. Cot **CVAT** la so go vao o do (CVAT dem tu 0, file MOT dem tu 1). "
        f"Lan dau nen doi chieu 1 box de chac mapping dung.\n"
        "4. Sua xong tick `[x]`. `Ctrl+S` moi 5-10 phut.\n")
    lines.append(
        f"\nQuy doi frame: `CVAT = MOT - 1` · "
        f"`frame tuyet doi trong video = MOT - 1 + {offset}`\n")

    lines.append(
        f"\n<sub>Nguong JUMP tu calibrate theo phan phoi cua chinh doan nay "
        f"(median + {JUMP_MAD_K:.0f}xMAD): dich chuyen > "
        f"{stats['thr_jump']:.2f} box/frame, "
        f"dien tich doi > {stats['thr_area']:.1f}x. "
        f"ORPHAN da bo qua {stats['orphan_suppressed']} truong hop box < "
        f"{ORPHAN_MIN_DIAG:.0f}px — o kich thuoc do guide noi khong label.</sub>\n")

    for kind in sorted(by_kind, key=lambda k: PRIORITY[k]):
        items = sorted(by_kind[kind], key=lambda f: -f.conf)
        shown = items[:per_type]
        lines.append(f"\n## {KIND_TITLE[kind]}\n")
        lines.append(f"{KIND_NOTE[kind]}\n")
        if len(items) > len(shown):
            lines.append(
                f"\n*{len(items)} muc, hien {len(shown)} muc dang ngo nhat "
                f"(`--per-type` de xem them).*\n")
        lines.append("")
        for f in shown:
            cvat = f.frame - 1
            lines.append(
                f"- [ ] **frame {cvat}** (MOT {f.frame} · abs "
                f"{f.frame - 1 + offset}) · {f.cls} · conf {f.conf:.2f}  \n"
                f"      {to_cvat(f.detail)}  \n"
                f"      -> {to_cvat(f.action)}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, findings: list[Finding], offset: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["kind", "priority", "cvat_frame", "mot_frame",
                     "abs_frame", "ids", "class", "confidence", "detail",
                     "action"])
        for f in sorted(findings, key=lambda f: (PRIORITY[f.kind], -f.conf)):
            wr.writerow([f.kind, PRIORITY[f.kind], f.frame - 1, f.frame,
                         f.frame - 1 + offset,
                         " ".join(str(i) for i in f.ids), f.cls,
                         f"{f.conf:.3f}", to_cvat(f.detail),
                         to_cvat(f.action)])


def frame_size(seg_dir: Path, override) -> tuple[int, int]:
    if override:
        return int(override[0]), int(override[1])
    jpgs = sorted((seg_dir / "img1").glob("*.jpg"))
    if not jpgs:
        raise SystemExit("no frames found; pass --frame-size W H")
    import cv2
    img = cv2.imread(str(jpgs[0]))
    if img is None:
        raise SystemExit(f"cannot read {jpgs[0]}; pass --frame-size W H")
    return img.shape[1], img.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="dense", choices=list(SEGMENTS),
                    help="resolves paths + absolute frame offset")
    ap.add_argument("--root", default="data/annotations/prelabels")
    ap.add_argument("--prelabels", default=None,
                    help="MOT file (default: <root>/<segment>/gt/gt.txt)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the queue (default: segment dir)")
    ap.add_argument("--frame-size", nargs=2, type=int, metavar=("W", "H"),
                    help="skip reading a frame to get image size")
    ap.add_argument("--offset", type=int, default=None,
                    help="absolute frame of segment start (default: from "
                         "--segment)")
    ap.add_argument("--per-type", type=int, default=60,
                    help="max rows shown per check in the markdown")
    ap.add_argument("--min-orphan-px", type=float, default=ORPHAN_MIN_DIAG,
                    help="skip interior breaks on boxes smaller than this "
                         "(box diagonal, px)")
    args = ap.parse_args()

    seg_dir = Path(args.root) / args.segment
    mot_path = Path(args.prelabels) if args.prelabels else seg_dir / "gt" / "gt.txt"
    out_dir = Path(args.out_dir) if args.out_dir else seg_dir
    offset = args.offset if args.offset is not None else SEGMENTS[args.segment][0]

    if not mot_path.exists():
        raise SystemExit(f"pre-labels not found: {mot_path}\n"
                         f"run: python -m code.scripts.export_prelabels")

    w, h = frame_size(seg_dir, args.frame_size)
    frames = load_mot(mot_path)
    n_frames = max(frames) if frames else 0
    tracks = build_tracks(frames)
    n_boxes = sum(len(v) for v in frames.values())
    print(f"{mot_path}: {n_boxes} boxes, {len(tracks)} tracks, "
          f"{n_frames} frames, image {w}x{h}")

    merges = find_merges(tracks, n_frames, w, h)
    splits = find_splits(tracks)
    jumps, calib = find_jumps(tracks, w, h)
    orphans, n_suppressed = find_orphans(tracks, merges, n_frames, w, h,
                                         args.min_orphan_px)
    findings = (merges + splits + jumps
                + find_dups(frames, tracks)
                + find_ghosts(tracks, n_frames, w, h)
                + find_classflips(tracks)
                + orphans)

    print(f"JUMP thresholds (median + {JUMP_MAD_K:.0f}*MAD of this segment): "
          f"move > {calib['jump']:.2f} box/frame, area > {calib['area']:.1f}x")
    print(f"ORPHAN: {n_suppressed} interior breaks skipped "
          f"(box < {args.min_orphan_px:.0f}px)")

    counts = Counter(f.kind for f in findings)
    print("\ncheck        found   est.time")
    for kind in sorted(counts, key=lambda k: PRIORITY[k]):
        secs = counts[kind] * SECONDS_PER_STOP[kind]
        print(f"  {kind:<10}{counts[kind]:>5}{secs // 60:>8} min")
    total = sum(counts[k] * SECONDS_PER_STOP[k] for k in counts)
    print(f"  {'TOTAL':<10}{len(findings):>5}{total // 60:>8} min")

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"segment": args.segment, "prelabels": str(mot_path),
             "boxes": n_boxes, "tracks": len(tracks), "frames": n_frames,
             "thr_jump": calib["jump"], "thr_area": calib["area"],
             "orphan_suppressed": n_suppressed}
    md, csv_path = out_dir / "review_queue.md", out_dir / "review_queue.csv"
    write_markdown(md, findings, stats, offset, args.per_type)
    write_csv(csv_path, findings, offset)
    print(f"\nwrote {md}\n      {csv_path}")


if __name__ == "__main__":
    main()
