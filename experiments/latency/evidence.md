# Tracker latency — and why the measurement design mattered more than the result

Reproduce (**one variant per process — see below**):

```bash
for v in baseline ctrv ctrv_nsa ctrv_gate ctrv_gate_nsa c1_ctrv_gate c1c2c3; do
  python -m code.scripts.compare_motion_models \
      --replay experiments/c2_real_vnweights_full/detections_full13726.npz \
      --variants $v --out-dir experiments/lat_iso/$v
done
```

## Results

Tracker-only latency per frame, 13 726 frames, detector excluded (replay feeds
cached detections), so this isolates what this paper's contributions cost.

| variant | mean (ms) | p50 | p95 | p95 dense | vs baseline |
|---|---|---|---|---|---|
| baseline (linear KF) | 3.93 | 3.57 | 6.68 | 9.67 | — |
| + CTRV | 4.44 | 3.96 | 7.51 | 9.87 | +13% |
| + CTRV + NSA | 4.51 | 4.03 | 7.68 | 9.82 | +15% |
| + CTRV + gating | 4.67 | 4.04 | 8.15 | 13.68 | +19% |
| **+ CTRV + gating + NSA (C2)** | **4.78** | 4.16 | 8.28 | **14.09** | **+22%** |
| + C1 (Exp 2) | 4.55 | 4.00 | 7.98 | 10.82 | +16% |
| + C1 + C3 (Exp 3) | 5.42 | 4.67 | 9.71 | 13.80 | +38% |

"dense" = frames 10700–11300, ~47 tracks/frame against a global mean of 25 —
the worst case, and the segment with human GT.

The costs are ordered as the computation is: the EKF is the bulk of it (+13%),
NSA is a single multiply on R and duly costs ~2 points, the gate adds a
Mahalanobis test per lost track. C2 at 4.78 ms is ~209 fps of tracker throughput;
against a YOLO11 inference cost of tens of ms, C2's +0.85 ms is not the term that
decides real-time feasibility. Even the dense-window p95 (14.09 ms) leaves
headroom in a 33 ms budget.

C1 (4.55 ms) sitting below full C2 (4.78 ms) is within measurement noise: after
GT tuning C1's DENSE_PEAK row is conf 0.50 / buffer 30, the same values the fixed
configuration uses, so the two do near-identical work on this segment.

## Methodology — the part worth remembering

The first attempt at this table was **wrong by 3×**, and wrong in a way that
would have flattered the proposed method.

Running all seven variants in one process gave `ctrv_gate_nsa` 5.97 ms against
`baseline` 13.19 ms — the proposed method apparently 2.4× *faster* than the
baseline it strictly adds computation to. Two confounds, stacked:

1. **External contention.** That replay ran in the background while other jobs
   used the CPU. Early variants were starved; later ones ran clean.
2. **Position within the process.** Re-running with the variant order reversed,
   `baseline` went from 13.19 ms (position 1) to 4.07 ms (position 7) — the same
   code, 3.2× apart. Timing tracked *position*, not variant. Even with no
   external load, a residual warm-up gradient remained (5.44 ms at position 1 →
   4.07 ms at position 7), larger than the C2-vs-baseline difference being
   measured.

Only one variant per process removes both: every variant is position 1, with the
same cold-start profile. Baseline then measures 3.93 ms, agreeing with the clean
end of the reversed run (4.07 ms) to within 3.5%.

Two rules follow, and `run_replay()` now warns when they are broken:

- **Never read latency from a multi-variant run.** Those runs remain the right
  way to produce tracking dumps — replay is deterministic, so contention changes
  wall-clock but not a single output box — but their timings are noise.
- **Never benchmark against a busy machine.**

The tracking metrics in `../gt_ablation/` are unaffected by any of this. They
come from deterministic replays where CPU scheduling cannot alter the result.
