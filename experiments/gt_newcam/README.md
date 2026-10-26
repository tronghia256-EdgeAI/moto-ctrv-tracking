# Held-out cameras: parked

**Decision (Nghĩa, 2026-10-13).** The paper reports development results on the
NKKN camera only, and states that as a limitation. No GT for an independent
camera will be reviewed for this submission.

**What exists, if the test is taken up later** (`code/scripts/gt_newcam.py`):

1. **Windows.** One 600-frame window per camera was pre-registered and
   committed in `53d5b1a`, before any tracker ran on these clips.
2. **Detections.** Cached for each window with yolo11_vn@1280 and with best.pt
   at 640 and 448 (gitignored `detections_*.npz`).
3. **Review bundles.** `experiments/gt_qa/<camera>_review` holds pre-labels
   seeded from the union of 12 (detector, variant) outputs. See `seed.json`.
   - This seed roughly triples the review cases relative to a baseline-only
     seed: 825, 974 and 1,647 cases before the ROI.
   - The bundles have not been reviewed.
   - The baseline-only bundles are kept, unreviewed, in
     `data/processed/superseded_bundles/`.

**Open design choice.** The choice between a union seed and a baseline seed
plus a blind crop pass over the disagreements is open
(`experiments/seed_bias/evidence.md`).
