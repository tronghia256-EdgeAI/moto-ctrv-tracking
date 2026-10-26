"""Standalone integrity checks copied into the Kaggle dataset package."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

MODE = 'DEVELOPMENT_ONLY_NOT_PAPER_EVALUATION'
NAMES = ['Accident', 'bus', 'car', 'motorcycle', 'truck']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        raise ValueError('Manifest path escapes package')
    return target


def verify_package(root):
    from PIL import Image
    root = Path(root).resolve()
    report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
    receipt = json.loads((root / 'package_receipt.json').read_text(encoding='utf-8'))
    if report['mode'] != MODE or report['names'] != NAMES:
        raise ValueError('Expected CAAP 5-class development dataset')
    for name, expected in receipt['file_hashes'].items():
        if digest(inside(root, name)) != expected:
            raise ValueError('Package metadata/checkpoint changed: ' + name)
    with (root / 'manifest.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups, pixels, sources = {}, {}, set()
    counts = {s: {'images': 0, 'objects': 0, 'classes': Counter()} for s in ('train', 'val')}
    for row in rows:
        split = row['split']
        if split not in counts or row['source_path'] in sources:
            raise ValueError('Invalid/duplicate manifest membership')
        sources.add(row['source_path'])
        for table, key in ((groups, 'group_id'), (pixels, 'image_sha256')):
            table.setdefault(row[key], set()).add(split)
        image, label = inside(root, row['image']), inside(root, row['label'])
        if image.parent != root / split / 'images' or label.parent != root / split / 'labels' or image.stem != label.stem:
            raise ValueError('Unexpected image/label location')
        if digest(image) != row['image_sha256'] or digest(label) != row['label_sha256']:
            raise ValueError('Image/label changed: ' + row['source_path'])
        with Image.open(image) as im:
            im.load()
        counts[split]['images'] += 1
        seen = set()
        for line in label.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            values = tuple(float(v) for v in line.split())
            if len(values) != 5 or not all(math.isfinite(v) for v in values):
                raise ValueError('Expected finite YOLO detection label')
            cid, x, y, w, h = values
            if cid != int(cid) or not 0 <= cid < len(NAMES) or min(w, h) <= 0 or min(x-w/2, y-h/2) < -1e-9 or max(x+w/2, y+h/2) > 1+1e-9:
                raise ValueError('Invalid class or box geometry')
            key = tuple(round(v, 10) for v in values)
            if key in seen:
                raise ValueError('Duplicate detection annotation')
            seen.add(key)
            counts[split]['objects'] += 1
            counts[split]['classes'][NAMES[int(cid)]] += 1
    if not rows or any(len(v) != 1 for v in [*groups.values(), *pixels.values()]):
        raise ValueError('Empty dataset or group/duplicate crossing splits')
    for split in counts:
        expected = [r for r in rows if r['split'] == split]
        if not expected or counts[split] != report['counts']['trial'][split]:
            raise ValueError('Counts differ from package report')
        for kind, suffix in (('image', '*'), ('label', '*.txt')):
            actual = set((root / split / (kind + 's')).glob(suffix))
            if actual != {inside(root, r[kind]) for r in expected}:
                raise ValueError('Extra/missing dataset files')
    return report, rows
