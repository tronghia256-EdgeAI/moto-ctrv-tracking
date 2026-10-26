"""Read-only YOLO dataset inventory; candidate flags are not removal decisions.

Detailed reports stay local under data/processed/ (gitignored). Image comparison
uses bytes, decoded RGB pixels, and perceptual hashes across ALL splits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_key(name):
    # A grouping hint ONLY: generic source names can collide across providers.
    return re.sub(r"\.rf\.[^.]+$", "", Path(name).stem)


def parse_label(text, nc, width, height):
    objects, issues, canonical = [], [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = np.array([float(x) for x in line.split()], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite value")
            cid = int(values[0])
            if values[0] != cid or not 0 <= cid < nc:
                raise ValueError("invalid class")
            coords = values[1:]
            if len(coords) == 4:
                kind = "box"
                cx, cy, bw, bh = coords
                bounds = [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2]
                area = bw * bh
                if bw <= 0 or bh <= 0:
                    raise ValueError("nonpositive box")
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                kind = "polygon"
                points = coords.reshape(-1, 2)
                low, high = points.min(axis=0), points.max(axis=0)
                bounds = [*low, *high]
                area = abs(float(cv2.contourArea(points.astype(np.float32))))
                if area <= 1e-12 or len(np.unique(points, axis=0)) < 3:
                    raise ValueError("degenerate polygon")
            else:
                raise ValueError("invalid coordinate count")
            if np.any(coords < -1e-9) or np.any(coords > 1 + 1e-9):
                issues.append((number, "coordinates_outside_0_1"))
            if min(bounds) < -1e-9 or max(bounds) > 1 + 1e-9:
                issues.append((number, "bounds_outside_image"))
            bw_px = (bounds[2] - bounds[0]) * width
            bh_px = (bounds[3] - bounds[1]) * height
            objects.append({"class": cid, "kind": kind, "bounds": bounds,
                            "area_px": area * width * height,
                            "width_px": bw_px, "height_px": bh_px,
                            "tiny": bool(min(bw_px, bh_px) < 4)})
            normalized = (cid, *np.round(coords, 10).tolist())
            if normalized in canonical:
                issues.append((number, "duplicate_annotation_row"))
            canonical.append(normalized)
        except (ValueError, IndexError) as exc:
            issues.append((number, str(exc)))
    # Ignores row order, but does not claim polygon vertex-order equivalence.
    signature = sha(json.dumps(sorted(canonical), separators=(",", ":")).encode())
    return objects, issues, signature


def hash_bits(bits):
    return int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(), "big")


def inspect_image(job):
    root, path, nc = job
    rel = path.relative_to(root).as_posix()
    split = path.relative_to(root).parts[0]
    label = path.parent.parent / "labels" / (path.stem + ".txt")
    raw = path.read_bytes()
    row = {"path": rel, "split": split, "source_hint": source_key(path.name),
           "bytes": len(raw), "sha256": sha(raw), "pixel_sha256": "",
           "width": 0, "height": 0, "phash": "", "dhash": "",
           "blur_laplacian": None, "gray_mean": None, "dark_fraction": None,
           "bright_fraction": None, "label_sha256": "", "label_signature": "",
           "objects": [], "issues": []}
    try:
        with Image.open(path) as opened:
            opened.load()  # Pillow rejects truncated files by default.
            row["exif_orientation"] = opened.getexif().get(274, 1)
            rgb = np.asarray(opened.convert("RGB"))
        h, w = rgb.shape[:2]
        row.update(width=w, height=h, pixel_sha256=sha(f"{w},{h},RGB:".encode() + rgb.tobytes()))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(small.astype(np.float32))[:8, :8]
        threshold = np.median(dct.reshape(-1)[1:])
        bits = dct > threshold
        bits[0, 0] = False
        diff = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        row.update(phash=f"{hash_bits(bits):016x}",
                   dhash=f"{hash_bits(diff[:, 1:] > diff[:, :-1]):016x}",
                   blur_laplacian=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                   gray_mean=float(gray.mean()), dark_fraction=float((gray < 16).mean()),
                   bright_fraction=float((gray > 239).mean()))
    except (OSError, ValueError) as exc:
        row["issues"].append((0, "image_decode: " + str(exc)))
    if not label.exists():
        row["issues"].append((0, "missing_label"))
    else:
        label_raw = label.read_bytes()
        row["label_sha256"] = sha(label_raw)
        try:
            objects, issues, signature = parse_label(label_raw.decode("utf-8-sig"), nc,
                                                   row["width"], row["height"])
            row.update(objects=objects, label_signature=signature)
            row["issues"].extend(issues)
        except UnicodeDecodeError as exc:
            row["issues"].append((0, "label_encoding: " + str(exc)))
    return row


def groups(rows, key):
    buckets = defaultdict(list)
    for row in rows:
        if row[key]:
            buckets[row[key]].append(row)
    return [members for members in buckets.values() if len(members) > 1]


def describe_groups(items):
    return [{"paths": [r["path"] for r in group],
             "splits": sorted({r["split"] for r in group}),
             "label_signatures": len({r["label_signature"] for r in group}),
             "has_label_errors": any(r["issues"] for r in group)} for group in items]


def near_pairs(rows, phash_limit=8, dhash_limit=10):
    """All-pairs search on one representative per exact pixel hash.

    No approximate index: every representative pair is checked. Thresholds are
    triage heuristics, not evidence that objects/labels or events are identical.
    """
    representatives = {}
    for row in rows:
        if row["pixel_sha256"]:
            representatives.setdefault(row["pixel_sha256"], row)
    unique = list(representatives.values())
    hashes = np.array([int(r["phash"], 16) for r in unique], dtype=np.uint64)
    lookup = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    for i, row in enumerate(unique):
        xor = hashes[i+1:] ^ hashes[i]
        distances = lookup[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for offset in np.flatnonzero(distances <= phash_limit):
            other = unique[i + 1 + int(offset)]
            dd = (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count()
            if dd <= dhash_limit:
                yield {"a": row["path"], "b": other["path"],
                       "split_a": row["split"], "split_b": other["split"],
                       "phash_distance": int(distances[offset]), "dhash_distance": dd,
                       "same_source_hint": row["source_hint"] == other["source_hint"]}


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_samples(root, output, rows, pixels, pairs, names):
    rng = np.random.default_rng(42)
    selected = []
    for group in pixels:
        if len({r["split"] for r in group}) > 1 or len({r["label_signature"] for r in group}) > 1:
            selected.extend((r["path"], "exact duplicate") for r in group[:2])
            if len(selected) >= 24:
                break
    cross = [p for p in pairs if p["split_a"] != p["split_b"]]
    for pair in sorted(cross, key=lambda p: (p["phash_distance"], p["dhash_distance"]))[:8]:
        selected.extend([(pair["a"], "near candidate A"), (pair["b"], "near candidate B")])
    readable = [r for r in rows if r["pixel_sha256"]]
    selected.extend((r["path"], "low sharpness: review") for r in
                    sorted(readable, key=lambda r: r["blur_laplacian"])[:8])
    for cid, name in enumerate(names):
        candidates = [r for r in readable if any(o["class"] == cid for o in r["objects"])]
        if candidates:
            selected.extend((candidates[i]["path"], name + ": random") for i in
                            rng.choice(len(candidates), min(8, len(candidates)), replace=False))
    index = {r["path"]: r for r in rows}
    colors = [(255, 50, 50), (255, 190, 0), (60, 230, 100), (50, 190, 255), (230, 90, 255)]
    output.mkdir()
    for page in range(0, len(selected), 12):
        sheet = Image.new("RGB", (1200, 990), (22, 24, 28))
        draw = ImageDraw.Draw(sheet)
        for cell, (path, reason) in enumerate(selected[page:page+12]):
            row = index[path]
            with Image.open(root / path) as opened:
                panel = opened.convert("RGB")
            pd = ImageDraw.Draw(panel)
            for obj in row["objects"]:
                b = obj["bounds"]
                if not 0 <= min(b) <= max(b) <= 1:
                    continue
                xy = (b[0]*panel.width, b[1]*panel.height, b[2]*panel.width, b[3]*panel.height)
                color = colors[obj["class"] % len(colors)]
                pd.rectangle(xy, outline=color, width=2)
                pd.text((xy[0], xy[1]), names[obj["class"]], fill=color)
            panel = ImageOps.contain(panel, (296, 280))
            x, y = cell % 4 * 300, cell // 4 * 330
            sheet.paste(panel, (x + (300-panel.width)//2, y))
            draw.text((x+4, y+282), f"{row['split']} | {reason}", fill="white")
            draw.text((x+4, y+300), Path(path).name[:38], fill="white")
        sheet.save(output / f"sample_{page//12+1:02}.jpg", quality=92)
    write_csv(output / "index.csv", [{"path": p, "reason": reason} for p, reason in selected],
              ["path", "reason"])


def audit(root, output, workers=4):
    # Avoid nested OpenCV thread pools for thousands of small images.
    cv2.setNumThreads(1)
    root, output = root.resolve(strict=True), output.resolve()
    if output == root or root in output.parents:
        raise ValueError("report output must be outside the source dataset")
    if output.exists():
        raise ValueError("report output already exists; choose a fresh directory")
    config_bytes = (root / "data.yaml").read_bytes()
    config = yaml.safe_load(config_bytes)
    raw_names = config["names"]
    names = raw_names if isinstance(raw_names, list) else [raw_names[i] for i in range(len(raw_names))]
    if config.get("nc", len(names)) != len(names):
        raise ValueError("nc does not match names")
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"Scanning {len(images)} images and labels (read-only)", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = []
        for row in pool.map(inspect_image, ((root, p, len(names)) for p in images)):
            rows.append(row)
            if len(rows) % 500 == 0:
                print(f"Decoded and audited {len(rows)}/{len(images)}", flush=True)
    bytes_groups, pixel_groups, hint_groups = (groups(rows, key) for key in
                                              ("sha256", "pixel_sha256", "source_hint"))
    print("Comparing perceptual hashes across all splits", flush=True)
    pairs = list(near_pairs(rows))
    label_paths = {p.parent.parent / "labels" / (p.stem + ".txt") for p in images}
    orphans = [p.relative_to(root).as_posix() for p in root.rglob("labels/*.txt") if p not in label_paths]
    issues = [{"path": row["path"], "line": line, "issue": issue}
              for row in rows for line, issue in row["issues"]]
    counts = {}
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        counts[split] = {"images": len(subset), "objects": sum(len(r["objects"]) for r in subset),
                         "empty_labels": sum(not r["objects"] and not r["issues"] for r in subset),
                         "classes": {name: {"objects": sum(o["class"] == cid for r in subset for o in r["objects"]),
                                             "images": sum(any(o["class"] == cid for o in r["objects"]) for r in subset)}
                                     for cid, name in enumerate(names)}}
    def group_summary(gs):
        cross = [g for g in gs if len({r["split"] for r in g}) > 1]
        return {"groups": len(gs), "images": sum(map(len, gs)),
                "redundant_copies": sum(len(g)-1 for g in gs),
                "cross_split_groups": len(cross), "cross_split_images": sum(map(len, cross)),
                "different_labels_groups": sum(len({r["label_signature"] for r in g}) > 1 for g in gs)}
    readable = [r for r in rows if r["pixel_sha256"]]
    metrics = {}
    for key in ("blur_laplacian", "gray_mean", "dark_fraction", "bright_fraction"):
        metrics[key] = dict(zip(("min", "p01", "p05", "p50", "p95", "p99", "max"),
                               np.percentile([r[key] for r in readable], [0,1,5,50,95,99,100]).tolist())) if readable else {}
    near_cross = [p for p in pairs if p["split_a"] != p["split_b"]]
    summary = {"source": str(root), "images": len(rows), "readable": len(readable),
               "bytes": sum(r["bytes"] for r in rows), "names": names, "splits": counts,
               "dimensions": dict(Counter(f"{r['width']}x{r['height']}" for r in readable)),
               "annotation_formats": dict(Counter(o["kind"] for r in rows for o in r["objects"])),
               "issues": dict(Counter(i["issue"] for i in issues)), "orphan_labels": orphans,
               "exact_bytes": group_summary(bytes_groups), "exact_pixels": group_summary(pixel_groups),
               "source_name_hints": group_summary(hint_groups),
               "near_candidates": {"pairs": len(pairs), "cross_split_pairs": len(near_cross),
                                   "phash_max": 8, "dhash_max": 10,
                                   "cross_split_image_paths": len({p[k] for p in near_cross for k in ("a", "b")})},
               "quality_percentiles": metrics,
               "tiny_annotations_under_4px_side": sum(o["tiny"] for r in rows for o in r["objects"]),
               "yaml_paths": {key: {"value": config.get(key),
                                     "resolves_to": str((root / str(config[key])).resolve()),
                                     "exists": (root / str(config[key])).exists()}
                              for key in ("train", "val", "test") if isinstance(config.get(key), str)},
               "source_data_yaml_sha256": sha(config_bytes),
               "limitations": ["Near matches and filename families require human review.",
                                "Blur/exposure are ranking features, not automatic discard criteria.",
                                "Only label syntax/geometry checked; semantic accuracy not certified.",
                                "No clip/event/camera grouping is known from this export.",
                                "Exact label signatures ignore row order, not polygon vertex order."]}
    # Audit never writes into source; reports are created only after scan completes.
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "groups.json").write_text(json.dumps({"bytes": describe_groups(bytes_groups),
                                                  "pixels": describe_groups(pixel_groups),
                                                  "source_hints": describe_groups(hint_groups)}, indent=2), encoding="utf-8")
    (output / "inventory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    scalar_fields = [key for key in rows[0] if key not in ("objects", "issues")] if rows else ["path"]
    write_csv(output / "images.csv", rows, scalar_fields)
    write_csv(output / "issues.csv", issues, ["path", "line", "issue"])
    write_csv(output / "near_pairs.csv", pairs, ["a", "b", "split_a", "split_b", "phash_distance", "dhash_distance", "same_source_hint"])
    render_samples(root, output / "samples", rows, pixel_groups, pairs, names)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    audit(args.source, args.output, args.workers)
