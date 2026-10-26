# Ablation harness — synthetic smoke run, 2026-07-11

Full pipeline implemented and all 4 ablation configs run end-to-end.
Reproduce: `python -m code.scripts.run_experiment --config configs/<name>.yaml --synthetic --save-metrics`

## Synthetic benchmark (9 objects, 300 frames, σ=3 px, 3×14-frame occlusions)

| config | MOTA | IDF1 | frags | IDSW | FPS* |
|---|---|---|---|---|---|
| baseline (Exp0) | 1.000 | 1.000 | 0 | 0 | ~570 |
| exp1_C2 (+CTRV) | 0.998 | 0.922 | 2 | 2 | ~520 |
| exp2_C1C2 | 1.000 | 1.000 | 0 | 0 | ~510 |
| exp3_C1C2C3 (full) | 1.000 | 1.000 | 0 | 0 | ~430 |

*FPS excludes detection (synthetic mode feeds detections directly).

## How to read this table — IMPORTANT

These numbers validate the HARNESS, not the method. Three caveats:

1. **The benchmark is easy for the linear baseline**: sinusoidal weaving
   with 14-frame gaps is the documented adversarial case for CTRV
   (ω-sign-flip, see exp1_C2/synthetic_evidence.md). The 2 lost IDs in
   exp1_C2 are both weaving tracks; the constant-turn gap track survived
   with CTRV (and is exactly the case where CTRV beats the baseline in
   tests/test_adaptive_tracker.py).
2. **exp2/exp3 ≡ baseline here by design**: the C1 controller selects the
   DEFAULT row (motion_model='standard') because the synthetic scene is
   neither dense (9 < 30 tracks) nor dark — so CTRV never activates.
   On real dense VN footage the DENSE_PEAK row ('ctrv') should engage.
   Verified explicitly: `params.motion_model == 'standard'`, no track has
   a CTRV filter in exp2/exp3 on this scene.
3. Real ablation numbers await the VN footage + annotations (co-author).

## Video path smoke test

`--video data/raw/videos/smoke_synthetic.mp4` (60 painted frames) ran the
full detector path: COCO yolov8n fallback engaged (warning logged),
severity classifier disabled (no weights), no crashes, ~12 FPS on CPU
including YOLO inference.
