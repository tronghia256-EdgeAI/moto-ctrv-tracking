# C2 on real VN footage — VN-trained detector (2026-07-14)

Video: `NKKN-VoThiSau 2017-07-18_08_00_00_000.asf`, frames 0–7500
(5 min @ 25 fps, 08:00 rush hour, ~3–4 light cycles).
Detector: **co-author-trained yolo11n** (`weights/yolo11_vn.pt`,
classes Accident/bus/car/motorcycle/truck, name-remapped to project ids)
@ imgsz 1280. On a dense frame it clears 32 motorcycles above
track_thresh vs 14 for COCO yolo11s@1280 and 3 for COCO yolo11n@640.

Same protocol as `experiments/c2_real/evidence.md`: one detector pass,
identical detections into both trackers; no-GT proxy metrics.

## Result — 7500 frames

| metric (motorcycle) | baseline | ctrv | Δ |
|---|---|---|---|
| re-id events        | 209  | 177  | **−15.3%** |
| unique IDs          | 2694 | 2641 | −2.0% |
| short tracks (<10f) | 581  | 583  | ±0 |
| mean lifespan (f)   | 56.1 | 56.3 | ±0 |

All classes: re-id 299 vs 272 (−9.0%); unique IDs 3546 vs 3514.
Non-motorcycle re-id: 90 vs 95 — both sides use the linear KF for
cars/buses/trucks; the small difference is a second-order effect of the
shared association pool, worth stating so the paper doesn't overclaim.

26 baseline motorcycle re-id events have no CTRV event within ±30
frames. Densest cluster: 4 baseline fragmentations in frames 6071–6115
(platoon surge), zero for CTRV — rendered in `clip_6040_6140.mp4`.

## Reading across the two runs

COCO s@1280 / 2000 frames: −31% motorcycle re-id events.
VN n@1280 / 7500 frames: −15.3%.
The better detector misses fewer frames, so fewer occlusion coasts are
needed and the motion model's headroom shrinks — the remaining −15% is
the harder core of true occlusion/weaving cases. Report the VN-weights
number as primary (it matches the final system); the COCO run doubles
as evidence that C2's benefit grows as detection degrades (useful for
the discussion of robustness).

## "Ghost track" forensics (2026-07-14, sau khi xem clip)

Ba nguồn "track ma" nhìn thấy trong clip, đã điều tra từng nguồn:

1. **Render bug (nguồn chính, ĐÃ SỬA)** — OpenCV seek trên ASF nhảy vào
   keyframe gần chứ không đúng frame đích, nên overlay lệch vài chục
   frames: box của xe thật vẽ "trước" xe trên đường trống. Track dumps
   không ảnh hưởng (thu tuần tự); chỉ video minh hoạ sai. Fix: render
   đọc tuần tự (`open_video` grab-skip). Mọi clip render TRƯỚC fix này
   đều có artifact — đã render lại.
2. **Xe thật ở vùng xa, conf thấp** — 36/53 track "ít hi-conf support"
   là xe đậu/xếp hàng tít xa (crop xác nhận), detector chấm ~0.3, thỉnh
   thoảng loé ≥0.5 (51/53 track sinh từ một detection ≥0.45). Là object
   thật — không nên diệt ở tracker; C3 det_conf phân loại cho từng task.
3. **FP thật (thiểu số)** — bóng đổ/vân đường sustain qua BYTE second
   association; C3 det_conf (w=0.30) đưa chúng xuống dưới τ_acc.

## Caveats

1. Proxy metrics (no GT) — MOTA/IDF1 await the annotated test split.
2. Single camera, single 5-min daytime segment; night/rain unseen.
3. Severity classifier still disabled (no EfficientNet weights); the
   Accident class now exists at the detector level.

Reproduce:
```
python -m code.scripts.compare_motion_models --video "data/raw/videos/<clip>.asf" \
    --max-frames 7500 --weights weights/yolo11_vn.pt --imgsz 1280 \
    --out-dir experiments/c2_real_vnweights
```
