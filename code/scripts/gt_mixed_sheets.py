"""Render source evidence, without inferred labels or method scores."""
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw

from code.scripts.render_split_candidates import _crop, _font

WIDTH, HEIGHT = 990, 754
BG, FG = (18, 20, 24), (232, 236, 242)
COLORS = [(255, 214, 0), (65, 220, 255)]


def render_sheet(case: dict, frames: dict, images: Path, out: Path):
    sheet = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"#{case['rank']}  {case['kind']}  ·  ID nguồn {case['ids'] or 'tất cả xe'}",
              font=_font(17), fill=FG)
    for number, (tid, frame) in enumerate((case["before"], case["after"])):
        with Image.open(images / f"{frame:06d}.jpg") as source:
            panel = source.convert("RGB")
        color = COLORS[number]
        target = next((box for identity, box, _ in frames.get(frame, []) if identity == tid), None)
        pd = ImageDraw.Draw(panel)
        for identity, box, _ in frames.get(frame, []):
            pd.rectangle(tuple(box), outline=color if identity == tid else (95, 108, 120),
                         width=3 if identity == tid else 1)
        if target:
            x1, y1, x2, y2 = target
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            radius = max(x2 - x1, y2 - y1) + 16
            pd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=3)
        panel.thumbnail((480, 270), Image.Resampling.LANCZOS)
        x = 10 + number * 490
        draw.text((x, 42), f"{'A' if number == 0 else 'B'} · MOT {frame} · CVAT {frame - 1} · ID {tid or 'tất cả'}",
                  font=_font(14), fill=color)
        sheet.paste(panel, (x, 68))
        if target:
            crop = _crop(images, frame, tuple(target))
            sheet.paste(crop, (x, 350))
        else:
            draw.text((x + 16, 435), "V: xem video và chỉnh box nếu cần", font=_font(17), fill=FG)
    y = 635
    for line in textwrap.wrap(case["question"], width=95):
        draw.text((12, y), line, font=_font(16), fill=FG)
        y += 24
    draw.text((12, 711), "U: chưa rõ   ·   V: video/chỉnh box   ·   E: đã sửa thủ công   ·   Nhãn chưa được xác nhận",
              font=_font(14), fill=(160, 178, 191))
    sheet.save(out)
