"""
code/scripts/render_split_candidates.py
───────────────────────────────────────
Render one visual comparison sheet per suspected over-merged GT identity, so
the re-review can be done by looking rather than by scrubbing CVAT.

`gt_review_queue` flags a SPLIT when an identity disappears for more than a
few frames and comes back displaced or rescaled beyond what one vehicle can
do. That check is deliberately mechanical; deciding whether the two sides are
the *same* vehicle needs eyes. This script puts both sides in front of the
eyes at once.

Each sheet has three rows, in increasing order of how diagnostic they are:

  1. Full frame either side of the gap, the identity boxed in yellow and
     every other box faint. Shows *where on the road* the box sits.
  2. Zoomed crops of just the two boxes. Shows what the vehicle looks like.
  3. The whole track's box-diagonal and centre-y over its lifetime, with the
     gap marked.

Row 3 is the one that settles most cases. Traffic on this clip flows toward
a fixed camera, so a single vehicle's box grows monotonically and its centre
moves steadily down the frame. A diagonal that shrinks and regrows, or a
centre that bounces up the roadway, is more than one vehicle — that is
exactly how GT id 96 was caught.

Usage (repo root, venv active):
  python -m code.scripts.render_split_candidates
  python -m code.scripts.render_split_candidates --top 20
  python -m code.scripts.render_split_candidates --ids 96 107

Writes <out-dir>/NNN_id<ID>_f<A>-<B>.png plus index.md and candidates.csv.

Author  : Trọng Nghĩa
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont

from code.scripts.gt_review_queue import (_area as q_area, _centre as q_centre,
                                          _diag as q_diag, build_tracks)
from code.scripts.import_gt import load_mot

# Same criteria as gt_bias_sensitivity.split_gt, so the sheets correspond
# one-to-one with the rows of the sensitivity analysis.
GAP_MIN = 5
MOVE_MAX = 2.0
AREA_MAX = 2.0

CLASS_NAME = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "accident"}

PANEL_W, PANEL_H = 480, 270
PLOT_H = 132
PAD = 10
HEADER_H = 30
YELLOW = (255, 214, 0)
FAINT = (90, 105, 120)
RED = (235, 70, 70)
BLUE = (90, 170, 255)
BG = (18, 20, 24)
FG = (232, 236, 242)


def _font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def find_candidates(tracks: dict) -> list[dict]:
    """Every join inside a GT identity that looks like two vehicles.

    Uses *raw* displacement across the gap rather than per-frame velocity.
    Dividing by the gap length is what let a 29-frame teleport look
    unremarkable and kept these out of the JUMP check for so long.
    """
    out = []
    for tid, t in tracks.items():
        for i in range(len(t.frames) - 1):
            gap = t.frames[i + 1] - t.frames[i]
            if gap <= GAP_MIN:
                continue
            b0, b1 = t.boxes[i], t.boxes[i + 1]
            (x0, y0), (x1, y1) = q_centre(b0), q_centre(b1)
            move = math.hypot(x1 - x0, y1 - y0) / max(q_diag(b0), 1e-6)
            a0, a1 = q_area(b0), q_area(b1)
            ratio = max(a0, a1) / max(min(a0, a1), 1e-6)
            if move <= MOVE_MAX and ratio <= AREA_MAX:
                continue
            out.append({
                "tid": tid,
                "cls": CLASS_NAME.get(t.classes[i], str(t.classes[i])),
                "f_before": t.frames[i],
                "f_after": t.frames[i + 1],
                "gap": gap,
                "box_before": b0,
                "box_after": b1,
                "move": move,
                "ratio": ratio,
                # Continuous, so the list actually ranks. The queue's own
                # confidence saturates at 1.000 across dozens of rows and
                # therefore orders nothing.
                "severity": max(move / MOVE_MAX, ratio / AREA_MAX),
            })
    out.sort(key=lambda c: -c["severity"])
    return out


def _draw_frame(img_dir: Path, mot_frame: int, gt: dict, tid: int,
                box: tuple) -> Image.Image:
    """Full frame, target identity in yellow, everything else faint.

    A red locator ring is drawn around the box. The identities in `gt.txt`
    are renumbered 1..N by CVAT's MOT export and do not match the IDs the
    CVAT UI shows, so the box has to be found by *position*, not by ID —
    and at this distance a motorcycle box can be 15 px across.
    """
    p = img_dir / f"{mot_frame:06d}.jpg"
    if p.exists():
        base = Image.open(p).convert("RGB")
    else:
        base = Image.new("RGB", (800, 450), BG)
    sx, sy = PANEL_W / base.width, PANEL_H / base.height
    base = base.resize((PANEL_W, PANEL_H), Image.LANCZOS)
    d = ImageDraw.Draw(base)
    for other_id, ob, _ in gt.get(mot_frame, []):
        if other_id == tid:
            continue
        d.rectangle([ob[0] * sx, ob[1] * sy, ob[2] * sx, ob[3] * sy],
                    outline=FAINT)
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = max((x2 - x1), (y2 - y1)) * 0.9 + 16
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=RED, width=2)
    d.rectangle([x1, y1, x2, y2], outline=YELLOW, width=2)
    return base


def _crop(img_dir: Path, mot_frame: int, box: tuple) -> Image.Image:
    """Zoom on the box with context padding, upscaled to panel height."""
    p = img_dir / f"{mot_frame:06d}.jpg"
    if not p.exists():
        return Image.new("RGB", (PANEL_W, PANEL_H), BG)
    base = Image.open(p).convert("RGB")
    x1, y1, x2, y2 = box
    padx, pady = (x2 - x1) * 0.6 + 8, (y2 - y1) * 0.6 + 8
    cx1, cy1 = max(0, x1 - padx), max(0, y1 - pady)
    cx2, cy2 = min(base.width, x2 + padx), min(base.height, y2 + pady)
    crop = base.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
    if crop.width < 2 or crop.height < 2:
        return Image.new("RGB", (PANEL_W, PANEL_H), BG)
    scale = min(PANEL_W / crop.width, PANEL_H / crop.height)
    crop = crop.resize((max(1, int(crop.width * scale)),
                        max(1, int(crop.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (PANEL_W, PANEL_H), BG)
    canvas.paste(crop, ((PANEL_W - crop.width) // 2,
                        (PANEL_H - crop.height) // 2))
    d = ImageDraw.Draw(canvas)
    ox, oy = (PANEL_W - crop.width) // 2, (PANEL_H - crop.height) // 2
    d.rectangle([ox + (x1 - cx1) * scale, oy + (y1 - cy1) * scale,
                 ox + (x2 - cx1) * scale, oy + (y2 - cy1) * scale],
                outline=YELLOW, width=2)
    return canvas


def _profile(track, cut_frame: int, width: int) -> Image.Image:
    """Box diagonal (yellow) and centre-y (blue) over the track's lifetime.

    A vehicle approaching a fixed camera grows monotonically and moves down
    the frame. Non-monotonic diagonal, or centre-y that reverses, means the
    identity spans more than one vehicle.
    """
    img = Image.new("RGB", (width, PLOT_H), BG)
    d = ImageDraw.Draw(img)
    f = _font(11)
    diags = [q_diag(b) for b in track.boxes]
    cys = [q_centre(b)[1] for b in track.boxes]
    frames = track.frames
    if len(frames) < 2:
        return img

    left, right, top, bot = 44, width - 44, 16, PLOT_H - 18
    f0, f1 = frames[0], frames[-1]

    def px(fr):
        return left + (fr - f0) / max(f1 - f0, 1) * (right - left)

    def py(v, lo, hi):
        return bot - (v - lo) / max(hi - lo, 1e-6) * (bot - top)

    d.rectangle([left, top, right, bot], outline=(52, 58, 68))
    for series, lo, hi, colour in (
            (diags, min(diags), max(diags), YELLOW),
            (cys, min(cys), max(cys), BLUE)):
        pts = [(px(fr), py(v, lo, hi)) for fr, v in zip(frames, series)]
        d.line(pts, fill=colour, width=2)

    cx = px(cut_frame)
    d.line([(cx, top), (cx, bot)], fill=RED, width=2)
    d.text((cx + 3, top + 1), "cut", font=f, fill=RED)
    d.text((4, top - 2), f"{max(diags):.0f}", font=f, fill=YELLOW)
    d.text((4, bot - 12), f"{min(diags):.0f}", font=f, fill=YELLOW)
    d.text((left, bot + 3), f"f{f0 - 1}", font=f, fill=FG)
    d.text((right - 26, bot + 3), f"f{f1 - 1}", font=f, fill=FG)
    d.text((right + 4, top + 2), "diag", font=f, fill=YELLOW)
    d.text((right + 4, top + 16), "cy", font=f, fill=BLUE)
    return img


def render(cand: dict, rank: int, gt: dict, tracks: dict,
           img_dir: Path, out_dir: Path) -> Path:
    w = PANEL_W * 2 + PAD * 3
    h = HEADER_H + PANEL_H * 2 + PLOT_H + PAD * 5
    sheet = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(sheet)

    a, b = cand["f_before"], cand["f_after"]
    head = (f"#{rank:02d}  CVAT frame {a - 1} → {b - 1} (gap {cand['gap']}f) "
            f"· {cand['cls']} · move {cand['move']:.1f} diag · "
            f"area {cand['ratio']:.1f}x   [gt.txt id {cand['tid']} — "
            f"khong phai ID tren CVAT]")
    d.text((PAD, 8), head, font=_font(14), fill=FG)

    y = HEADER_H + PAD
    sheet.paste(_draw_frame(img_dir, a, gt, cand["tid"], cand["box_before"]),
                (PAD, y))
    sheet.paste(_draw_frame(img_dir, b, gt, cand["tid"], cand["box_after"]),
                (PAD * 2 + PANEL_W, y))
    y += PANEL_H + PAD
    sheet.paste(_crop(img_dir, a, cand["box_before"]), (PAD, y))
    sheet.paste(_crop(img_dir, b, cand["box_after"]), (PAD * 2 + PANEL_W, y))
    y += PANEL_H + PAD
    sheet.paste(_profile(tracks[cand["tid"]], b, w - PAD * 2), (PAD, y))

    small = _font(12)
    for x0, label, bx in ((PAD, "BEFORE", cand["box_before"]),
                          (PAD * 2 + PANEL_W, "AFTER", cand["box_after"])):
        cx, cy = q_centre(bx)
        d.text((x0 + 4, HEADER_H + PAD + 4),
               f"{label}  frame {a - 1 if label == 'BEFORE' else b - 1}",
               font=small, fill=YELLOW)
        # Pixel position is the only reliable handle on the object, so put
        # it on the sheet rather than only in the CSV.
        d.text((x0 + 4, HEADER_H + PAD + PANEL_H - 15),
               f"tam ({cx:.0f}, {cy:.0f})  box {bx[0]:.0f},{bx[1]:.0f} "
               f"-> {bx[2]:.0f},{bx[3]:.0f}  (anh 800x450)",
               font=small, fill=YELLOW)

    path = out_dir / f"{rank:03d}_id{cand['tid']}_f{a - 1}-{b - 1}.png"
    sheet.save(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/annotations/NKKN_gt/dense/gt/gt.txt")
    ap.add_argument("--images",
                    default="data/annotations/prelabels/dense/img1")
    ap.add_argument("--out-dir", default="experiments/gt_qa/split_candidates")
    ap.add_argument("--offset", type=int, default=10700,
                    help="absolute frame of MOT frame 1")
    ap.add_argument("--top", type=int, default=0,
                    help="render only the N most severe (0 = all)")
    ap.add_argument("--ids", nargs="*", type=int, default=None,
                    help="render only these GT ids")
    args = ap.parse_args()

    gt = load_mot(args.gt)
    tracks = build_tracks(gt)
    cands = find_candidates(tracks)
    print(f"{len(cands)} split candidates over {len(tracks)} GT identities "
          f"({len({c['tid'] for c in cands})} distinct ids)")

    if args.ids:
        cands = [c for c in cands if c["tid"] in args.ids]
    if args.top:
        cands = cands[:args.top]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.images)

    lines = [
        "# SPLIT candidates — `dense` GT\n",
        f"{len(cands)} joins where one GT identity may span two vehicles.",
        "Sorted worst first. Every row is a *question*, not a verdict —",
        "a motorcycle emerging from behind a bus really does reappear",
        "displaced and rescaled.\n",
        "## How to read a sheet\n",
        "- **Top row** — full frame either side of the gap, this identity in",
        "  yellow. Look at where on the road the box sits.",
        "- **Middle row** — zoom on the two boxes. Same vehicle or not?",
        "- **Bottom strip** — box diagonal (yellow) and centre-y (blue) over",
        "  the whole track, red line at the gap. Traffic flows toward the",
        "  camera, so one vehicle grows monotonically and drifts down the",
        "  frame. A diagonal that shrinks then regrows is two vehicles.\n",
        "## The `gt.txt` id is not the CVAT id\n",
        "CVAT renumbers every track `1..N` in export order when it writes",
        "MOT 1.1 — the pre-labels went in with tracker ids (`5849`, `6413`,",
        "…) and came back as `1..335` with no gaps. So the `gt id` column",
        "below exists **only inside `gt.txt`**; typing it into CVAT finds",
        "nothing or, worse, finds the wrong object. It is kept here so the",
        "rows can be traced back to the sensitivity analysis, not for",
        "lookup.\n",
        "**Look things up by frame and position instead:**\n",
        "1. Type the *after* frame number into CVAT's frame box.",
        "2. Find the box inside the red ring on the sheet — the centre",
        "   pixel is printed under each panel (images are 800×450, same",
        "   coordinates CVAT uses).",
        "3. Click it. Whatever id CVAT shows is the real one.",
        "4. Confirm by stepping back to the *before* frame: if it is the",
        "   same CVAT id sitting on a different vehicle, this row is real.",
        "5. Then use *Split* in the object sidebar at the after frame.\n",
        f"Frame numbering: CVAT = MOT − 1 · absolute = MOT − 1 + {args.offset}\n",
        "| # | gt.txt id | class | CVAT before → after | gap | move | area "
        "| centre after | sheet |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for rank, c in enumerate(cands, 1):
        path = render(c, rank, gt, tracks, img_dir, out_dir)
        a, b = c["f_before"] - 1, c["f_after"] - 1
        ca, cb = q_centre(c["box_before"]), q_centre(c["box_after"])
        lines.append(
            f"| {rank} | {c['tid']} | {c['cls']} | f{a} → f{b} | "
            f"{c['gap']}f | {c['move']:.1f} | {c['ratio']:.1f}x | "
            f"({cb[0]:.0f}, {cb[1]:.0f}) | [png]({path.name}) |")
        rows.append({
            "rank": rank, "gt_id_INTERNAL_not_cvat": c["tid"],
            "class": c["cls"],
            "cvat_before": a, "cvat_after": b,
            "mot_before": c["f_before"], "mot_after": c["f_after"],
            "abs_before": a + args.offset, "abs_after": b + args.offset,
            "gap": c["gap"],
            "centre_before": f"{ca[0]:.0f} {ca[1]:.0f}",
            "centre_after": f"{cb[0]:.0f} {cb[1]:.0f}",
            "box_before": " ".join(f"{v:.1f}" for v in c["box_before"]),
            "box_after": " ".join(f"{v:.1f}" for v in c["box_after"]),
            "move_diag": round(c["move"], 3),
            "area_ratio": round(c["ratio"], 3),
            "severity": round(c["severity"], 3),
        })
        if rank % 10 == 0:
            print(f"  rendered {rank}/{len(cands)}")

    (out_dir / "index.md").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
    with open(out_dir / "candidates.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} sheets + index.md + candidates.csv to {out_dir}")


if __name__ == "__main__":
    main()
