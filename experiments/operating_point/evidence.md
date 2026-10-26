# Detector operating points: the accuracy cost of real time

**Question.** How much tracking accuracy does the real-time detector operating
point cost on motorcycle-dominant footage, and where does the cost come from?

**Method.** `python -m code.scripts.operating_points data/processed/paper_runs/best85_roi_20261013_v5`
writes `operating_points.json`. It scores the three reconciled ROI GTs
(`docs/GT_RECONCILED_20261013.md`) with the repository evaluator at IoU 0.5.
Three operating points are compared:

- **yolo11_vn@1280\*** — the old cached tracks. Tracking ran on the full frame,
  and the ROI was applied only at scoring time. The GTs were seeded from these
  trackers' outputs, so this row is favoured by the seed (see
  `experiments/seed_bias/evidence.md`).
- **best.pt@640** and **best.pt@448** — paper run v5: ROI mask before the
  detector, state expiry, one detection cache shared by all trackers.

**Motorcycle IDF1.** Columns are baseline / C2 (ctrv_gate_nsa) / C1+C2
(c1_ctrv_gate) / OC-SORT at matched thresholds (0.5 + BYTE).

| Operating point | dense | freeflow | mixed |
|---|---|---|---|
| yolo11_vn@1280\* | 0.866 / 0.911 / 0.900 / 0.785 | 0.932 / 0.962 / 0.894 / 0.888 | 0.986 / 0.950 / 0.958 / 0.897 |
| best.pt@640 | 0.590 / 0.602 / 0.610 / 0.626 | 0.693 / 0.698 / 0.723 / 0.726 | 0.726 / 0.729 / 0.748 / 0.757 |
| best.pt@448 | 0.477 / 0.499 / 0.517 / 0.551 | 0.516 / 0.536 / 0.558 / 0.588 | 0.565 / 0.574 / 0.587 / 0.646 |

**The drop is driven by misses.** For C2 on dense, FN / FP are:

| Operating point | FN | FP |
|---|---:|---:|
| yolo11_vn@1280\* | 883 | 1,247 |
| best.pt@640 | 9,870 | 1,148 |
| best.pt@448 | 12,263 | 883 |

**Miss rate by GT box height.** C2 on dense, per-frame one-to-one matching
(800×450 frames):

| Operating point | < 20 px | 20–40 px | ≥ 40 px |
|---|---:|---:|---:|
| yolo11_vn@1280\* | 0.10 | 0.03 | 0.03 |
| best.pt@640 | 0.99 | 0.47 | 0.12 |
| best.pt@448 | 1.00 | 0.71 | 0.20 |

Small motorcycles are common in the GT, so these misses weigh heavily:

| Segment | < 20 px | 20–40 px | ≥ 40 px |
|---|---:|---:|---:|
| dense | 28% | 43% | 29% |
| freeflow | 38% | 38% | 24% |
| mixed | 22% | 47% | 31% |

**Findings.**

1. **The real-time operating points lose small motorcycles.**
   - At 640, nearly every motorcycle under 20 px and about half of those at
     20–40 px are missed.
   - At 448, every motorcycle under 20 px and about two thirds of those at
     20–40 px are missed.
   - Motorcycles of 40 px and taller are tracked almost as well as at 1280.

   The input is 800 px wide, so 640 downscales it, whereas 1280 upscaled it.
2. **The tracker ordering is the same at 640 and 448.** C1+C2 beats C2, which
   beats baseline, and OC-SORT at matched thresholds leads.
3. **The detector-level seed caveat does not explain the drop.** Between 59%
   and 92% of best.pt's unmatched predictions lie where the GT has no box of
   any class. Some of these may be real vehicles: the GTs were seeded from
   yolo11_vn trackers, so boxes only best.pt finds were never shown to a
   reviewer. Even so, FP is an order of magnitude smaller than FN.

**Not covered here.**

- Latency at these operating points: v5's benchmark ran on a throttled CPU
  (41–49% processor performance) and is not reportable.
- Whether best.pt at a larger input (960, 1280) recovers the small motorcycles.

## Pre-registered: the input size of Table 2

Fixed on 2026-10-13, before any latency trial of the final runs
(`best85_roi_final`, `yolo11vn_roi_final`). Agreed with the manuscript owner.

**What is measured.** `run_roi_paper --phase benchmark`, with each run's own
launcher arguments. Timing covers all four best.pt sizes (448/640/960/1280) and
yolo11_vn@1280.

- Arms: baseline, C2 (`ctrv_gate_nsa`) and C1+C2+C3 (`c1c2c3`).
- Three trials per arm, each in a fresh process.
- Each trial replays the full prehistory, then runs 20 warm-up frames and
  200 measured frames from the start of the dense GT segment.
- A frame's time covers decode, ROI/context, detection, tracking and C3. It
  excludes the severity branch and network alerts.

**Statistics.**

- Per trial: p95 of the 200 measured frame times.
- Per arm: the median of its three valid trials' p95.

**Valid trial.**

- An external sampler reads `\Processor Information(_Total)\% Processor
  Performance` once per second. It runs during every trial.
- The trial's measured window is reconstructed as [t_end − Σ frame_ms, t_end].
  t_end is the write time of the trial file.
- The trial is valid if the median of the samples inside that window is
  ≥ 90%.
- An invalid trial is moved, with its samples, to `benchmark_trials_invalid/`
  and re-measured. Nothing is deleted.

**Decision.**

1. Table 2 uses the largest best.pt input size at which **every** arm has
   median p95 ≤ 40 ms (the source is 25 fps). This is the maximum over the
   measured arms, not an arm assumed to be slowest.
2. A size with fewer than three valid trials for any arm is re-measured, not
   decided. A throttled night therefore cannot read as "no size meets the
   budget".
3. Only if every size has three valid trials per arm and none meets the
   budget does Table 2 use 448. The text then states that real time was not
   met at p95.
4. The other sizes and yolo11_vn@1280 go to Table 4 and to the
   accuracy–latency curve.

## What the ROI pixel mask costs (2026-10-14)

**Setup.** The paper protocol masks pixels outside the ROI before detection,
then keeps detections whose bottom-centre lies in the ROI. Nghia confirmed this
choice on 14/10 (option B); it was first set on 12/10 in `docs/PAPER_SCOPE.md`.

The GT keeps a box by its bottom-centre. Many small motorcycles straddle the
ROI's far boundary: the top-centre of a GT motorcycle under 20 px lies outside
the polygon in 34% of cases on dense, 58% on freeflow and 44% on mixed. The
masked detector sees only the lower part of these objects.

**Detector-level recall** of GT motorcycles, at conf ≥ 0.5 (the tracker's
threshold) and IoU ≥ 0.5, one-to-one per frame. The "yolo11_vn full frame" row
is the old unmasked cache, which also seeded the GT
(`experiments/c2_real_vnweights_full/detections_full13726.npz`).

| Dense | < 20 px, cut by the mask (n = 1810) | < 20 px, whole (n = 3520) | ≥ 20 px (n = 13883) |
|---|---:|---:|---:|
| yolo11_vn full frame | 0.76 | 0.65 | 0.93 |
| yolo11_vn@1280 masked | 0.20 | 0.37 | 0.90 |
| best.pt@1280 masked | 0.16 | 0.31 | 0.87 |
| best.pt@960 masked | 0.02 | 0.17 | 0.82 |
| best.pt@640 masked | 0.00 | 0.03 | 0.71 |

For objects the mask cuts, freeflow and mixed show the same drop. yolo11_vn
goes from 0.88 to 0.34 (freeflow) and from 0.90 to 0.27 (mixed). best.pt@1280
masked reaches 0.24 and 0.23.

**Findings.**

- **These are real misses, not geometry.** Lowering the IoU threshold from 0.5
  to 0.1 barely changes masked recall at conf ≥ 0.5: on dense, under 20 px, it
  goes from 0.31 to 0.33 (`det_geometry_check.py`).
- **The detectors are about equal under one protocol.** At 1280, yolo11_vn and
  best.pt reach dense C2 IDF1 0.758 and 0.764. Their tracker-level miss rates
  under 20 px are 0.73 and 0.68. The earlier 0.10 for yolo11_vn came from the
  unmasked full-frame row.
- **Within the masked protocol, input size is the large effect** for best.pt.
- **The mask buys no runtime.** The detector still infers on the whole frame.

The tracking-level cost of the mask is measured by the no-mask ablation runs,
`best85_roi_nomask2` and `yolo11vn_roi_nomask2` (worktree branch
`exp/roi-filter-only`, flag `--no-roi-pixel-mask`). Scripts:
`det_cut_split.py` and `det_geometry_check.py`.

## Final runs: masked protocol, primary (2026-10-14)

`operating_points.json` now comes from the final runs. Command:

    python -m code.scripts.operating_points best_448=<best85_roi_final>/448 best_640=…/640 best_960=…/960 best_1280=…/1280 yolo11_vn_1280_roi=<yolo11vn_roi_final2>/1280

The miss rate is that of the baseline tracker, used as a proxy for detector
recall.

| Operating point | dense IDF1 baseline / C2 | baseline miss rate on dense, < 20 / 20–40 / ≥ 40 px |
|---|---|---|
| best.pt@448 | 0.477 / 0.499 | 1.00 / 0.71 / 0.22 |
| best.pt@640 | 0.590 / 0.602 | 0.99 / 0.48 / 0.14 |
| best.pt@960 | 0.685 / 0.698 | 0.94 / 0.24 / 0.12 |
| best.pt@1280 | 0.724 / 0.764 | 0.81 / 0.14 / 0.13 |
| yolo11_vn@1280 (ROI) | 0.746 / 0.758 | 0.74 / 0.11 / 0.10 |

`detector_seed.json` was recomputed with the yolo11_vn ROI row. The reference
is best.pt@1280, whose empty-GT extras add 1,382 boxes on dense. The tables in
the earlier sections use the superseded v5 rows and the full-frame yolo11_vn
row.
