# C2 on real VN footage — first evidence (2026-07-14)

Video: `NKKN-VoThiSau 2017-07-18_08_00_00_000.asf` (Nam Kỳ Khởi Nghĩa ×
Võ Thị Sáu, HCMC; 800×450 @ 25 fps, 08:00 rush hour, motorcycle-dominant).

## Method

`code/scripts/compare_motion_models.py`: single pass, detector runs once
per frame, the SAME detections feed both trackers (baseline linear KF vs
exp1_C2 CTRV) — any difference is attributable to the motion model.
Detector: COCO yolo11s @ imgsz 1280 (yolo11n@640 was measured inadequate:
3 vs 14 motorcycles above track_thresh on a dense frame; riders detected
as COCO "person" remain invisible until VN weights exist).

No GT ⇒ no-GT proxies: unique IDs, lifespan distribution, and
**re-id events** (ID dies at frame f, same-class ID born ≤40 frames later
with IoU ≥ 0.3 to the death box — the signature of one physical object
fragmenting into two track IDs).

## Result — first 2000 frames (80 s, ~1.5 light cycles)

| metric (motorcycle) | baseline | ctrv | Δ |
|---|---|---|---|
| re-id events        | 52   | 36   | **−31%** |
| unique IDs          | 436  | 415  | −4.8% |
| short tracks (<10f) | 86   | 69   | −20% |
| mean lifespan (f)   | 40.8 | 42.8 | +5% |

All-class totals: 529 vs 508 unique IDs, 56 vs 40 re-id events.

Qualitative: 7 baseline motorcycle re-id events have NO ctrv event within
±30 frames (CTRV kept the ID through the same moment); densest cluster:
4 baseline fragmentations in frames 1279–1359 during a platoon surge.
(The clips once rendered for this run were removed together with its
raw dumps — they predate the ASF seek fix and drew boxes on the wrong
frames; see `c2_real_vnweights/evidence.md`. Current clips live in
`experiments/c2_variants/`.)

## Caveats (state these in the paper)

1. Segment is 2000 frames (runs were killed at the machine; checkpointing
   added). Re-run the full clip for final numbers.
2. COCO detector proxy: motorcycle recall is limited and head-on riders
   are often classed "person" (dropped). VN-trained weights will change
   detection quality; re-run after training.
3. Proxy metrics, not MOTA/IDF1 — those need the annotated test split.
4. Re-id event detection (GAP≤40, IoU≥0.3) is a heuristic; use it to FIND
   qualitative cases and to compare runs on identical detections, not as
   an absolute fragmentation count.

Reproduce:
```
python -m code.scripts.compare_motion_models --video "data/raw/videos/<clip>.asf" \
    --max-frames 7500 --weights yolo11s.pt --imgsz 1280 --out-dir experiments/c2_real
python -m code.scripts.compare_motion_models --video ... --analyze-only --out-dir experiments/c2_real
python -m code.scripts.compare_motion_models --video ... --render 1260 1380 --out-dir experiments/c2_real
```
