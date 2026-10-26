"""Human-defined polygon scope. GT and predictions use identical membership."""
import json
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from code.scripts.gt_review import atomic_write, sha256

RULE = 'bbox_bottom_center_inside_polygon_boundary_included'


def validate_polygon(points, size):
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError('ROI needs at least three vertices')
    w, h = size
    vertices = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p):
            raise ValueError('Invalid ROI vertex')
        x, y = map(float, p)
        if not 0 <= x <= w or not 0 <= y <= h:
            raise ValueError('ROI vertex lies outside image')
        vertices.append((x, y))
    if len(set(vertices)) != len(vertices):
        raise ValueError('ROI has repeated vertices')
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def intersects(a, b, c, d):
        values = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
        if values[0]*values[1] < 0 and values[2]*values[3] < 0:
            return True
        for value, p, q, r in ((values[0],a,b,c),(values[1],a,b,d),(values[2],c,d,a),(values[3],c,d,b)):
            if abs(value) < 1e-8 and min(p[0],q[0])-1e-8 <= r[0] <= max(p[0],q[0])+1e-8 and min(p[1],q[1])-1e-8 <= r[1] <= max(p[1],q[1])+1e-8:
                return True
        return False
    n = len(vertices)
    for i in range(n):
        for j in range(i+1,n):
            if j == i+1 or (i == 0 and j == n-1): continue
            if intersects(vertices[i],vertices[(i+1)%n],vertices[j],vertices[(j+1)%n]):
                raise ValueError('ROI polygon crosses itself')
    area = abs(sum(vertices[i][0]*vertices[(i+1)%n][1]-vertices[(i+1)%n][0]*vertices[i][1] for i in range(n)))/2
    if area < 1:
        raise ValueError('ROI area is too small')
    return [list(p) for p in vertices]


def load_roi(bundle):
    bundle = Path(bundle)
    path = bundle / 'roi.json'
    if not path.exists(): return None
    roi = json.loads(path.read_text(encoding='utf-8'))
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    if roi.get('schema_version') != 1 or roi.get('rule') != RULE or type(roi.get('enabled')) is not bool:
        raise ValueError('Invalid ROI schema/rule')
    if roi.get('manifest_sha256') != sha256(bundle / 'manifest.json') or roi.get('image_size') != manifest['image_size']:
        raise ValueError('ROI belongs to a different camera/review manifest')
    validate_polygon(roi['polygon'], roi['image_size'])
    if not roi.get('reviewer','').strip() or not roi.get('reason','').strip():
        raise ValueError('ROI requires reviewer and scope rationale')
    return roi


def save_roi(bundle, polygon, enabled, reviewer, reason, reference_frame, expected_hash):
    from code.scripts.gt_mixed_review import json_bytes, now
    bundle = Path(bundle)
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    polygon = validate_polygon(polygon, manifest['image_size'])
    if type(enabled) is not bool or not reviewer.strip() or not reason.strip() or not 1 <= reference_frame <= manifest['n_frames']:
        raise ValueError('Provide reviewer, scope rationale and a valid reference frame')
    path = bundle/'roi.json'
    old = load_roi(bundle)
    if manifest.get('review_scope')=='ROI_ONLY' and old and (old['polygon']!=polygon or old['enabled']!=enabled):
        raise ValueError('This ROI review has a frozen scope; edit the full parent ROI and generate a new review version')
    history = old.get('history', []) + [{k: v for k,v in old.items() if k != 'history'}] if old else []
    roi = dict(schema_version=1, enabled=enabled, polygon=polygon, rule=RULE,
               image_size=manifest['image_size'], manifest_sha256=sha256(bundle/'manifest.json'),
               segment=manifest['segment'], reviewer=reviewer.strip(), reason=reason.strip(),
               reference_frame=reference_frame, created_utc=now(), history=history)
    raw = json_bytes(roi)
    if expected_hash is not None:
        atomic_write(path, raw, expected_hash)
    else:
        fd, name = tempfile.mkstemp(prefix='.roi-', dir=bundle)
        temp = Path(name)
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            # Atomic create without replacing another editor's first save.
            os.link(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    return roi


def contains_box(box, roi):
    if not roi or not roi['enabled']: return True
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box) or box[2] < box[0] or box[3] < box[1]:
        raise ValueError('Invalid box for ROI membership')
    point = ((float(box[0])+float(box[2]))/2, float(box[3]))
    return cv2.pointPolygonTest(np.array(roi['polygon'],dtype=np.float32), point, False) >= 0


def filter_frames(frames, roi):
    return {frame: [(tid,box,cls) for tid,box,cls in entries if contains_box(box,roi)]
            for frame,entries in frames.items()}


def pixel_mask(frame,roi):
    if not roi or not roi['enabled']:return None
    h,w=frame.shape[:2]
    if [w,h]!=roi['image_size']:raise ValueError('ROI/frame resolution mismatch; use the correct camera ROI')
    mask=np.zeros((h,w),dtype=np.uint8)
    cv2.fillPoly(mask,[np.rint(roi['polygon']).astype(np.int32)],255)
    return mask


def filter_detections(detections,roi):
    array=np.asarray(detections,dtype=float).reshape(-1,6)
    if not roi or not roi['enabled']:return array
    return array[np.array([contains_box(row[:4],roi) for row in array],dtype=bool)]


def overlay_roi(image, roi):
    if not roi or not roi['enabled']: return image
    mask = Image.new('L',image.size,140)
    points = [tuple(p) for p in roi['polygon']]
    ImageDraw.Draw(mask).polygon(points,fill=0)
    shaded = image.copy().convert('RGBA')
    shaded.alpha_composite(Image.composite(Image.new('RGBA',image.size,(0,0,0,180)),
                                         Image.new('RGBA',image.size,(0,0,0,0)), mask))
    shaded = shaded.convert('RGB')
    ImageDraw.Draw(shaded).line(points+[points[0]],fill='#ffd600',width=2)
    return shaded
