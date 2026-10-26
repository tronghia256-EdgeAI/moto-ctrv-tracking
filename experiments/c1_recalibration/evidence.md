# C1 density threshold, recalibrated outside the GT segments (pre-registered)

**Problem.** Under the paper run v5 (best.pt with ROI), C1's controller never
left its DEFAULT row.

- `DENSE_PEAK` needs the tracker's own density EMA to reach
  `density_threshold`, which is 30. That value was set for the old full-frame
  yolo11_vn@1280 track counts, about 46 per frame on dense.
- Under v5 the density EMA peaked at 29 on dense at 640 and at 24 at 448.
  Brightness never fell below the NIGHT bound of 80 (minimum 89.7).
- In DEFAULT, `motion_model` is "standard". That turns CTRV off, and with it
  C2's gate and NSA (`adaptive_bytetrack.py`, line 284).
- The C1 arms were therefore ByteTrack at a 0.45 confirmation threshold with
  the linear model. Their IDF1 gain over C2 is a threshold effect, and C1's
  premise was not tested (finding by caap-b5, verified on
  `{640,448}/context_c1_ctrv_gate.json`).

**Decision (Nghĩa, 2026-10-13).** Recalibrate on frames outside the GT,
using a rule fixed in advance. No retuning on the evaluation segments.

**Rule.**

- Per input size, the threshold is the 75th percentile of the
  `c1_ctrv_gate` arm's own density EMA (the value the controller compares)
  over every NKKN frame outside the three GT segments. That is 9,500 of the
  11,300 frames run.
- The data source is v5's context log for that size, which recorded the
  DEFAULT regime.
- Computed by `c1_calibration.py`, which sits outside `code/` during the
  code freeze and moves to `code/scripts/` afterwards. Output:
  `calibration.json`.

**Thresholds, fixed before any C1 result under them exists.**

| Input size | p50 | p75 = threshold | p90 | p99 | Share of frames ≥ threshold: dense / freeflow / mixed |
|---|---:|---:|---:|---:|---|
| 640 | 6 | **18** | 21 | 27 | 0.84 / 0.00 / 0.31 |
| 448 | 5 | **14** | 17 | 22 | 0.84 / 0.00 / 0.35 |
| 960 | 7 | **21** | 26 | 32 | 0.87 / 0.00 / 0.27 |
| 1280 | 7 | **23** | 29 | 35 | 0.92 / 0.00 / 0.27 |

The last column is computed from the DEFAULT-regime density only. No GT and
no metric is involved. Once the controller switches, density shifts
(confirmation threshold 0.50 and CTRV in `DENSE_PEAK`), so the re-run must
log the actual regime per frame.

**In-run rule (2026-10-13, before the final run).** The Pareto run's C1 arm
did switch at 960 and 1280 (its density reached 34 and 36 under the old
threshold of 30), so its context log is not DEFAULT-regime density. The rule
is therefore applied inside `run_roi_paper.py` (`c1_calibrate`):

- after caching, `c1_ctrv_gate` is replayed with an infinite density
  threshold, so it never enters `DENSE_PEAK`;
- the threshold is the same percentile over the same frames, written to
  `{size}/c1_threshold.json` before any C1 arm runs;
- the frozen C1 configs carry `density_threshold: null`, and the C1 context
  logs record the regime applied in each frame.

On the v5 caches the probe's density equals v5's logged C1 density frame for
frame at 640 and 448, and gives 18 and 14. The 960 and 1280 rows above come
from the probe on the Pareto caches. The yolo11_vn@1280 threshold comes from
its own run by the same rule.

## Regime actually applied in the final runs (2026-10-14)

The table gives the share of frames in which the C1 arm (`c1_ctrv_gate`)
applied DENSE_PEAK. It comes from the regime log in each final run
(`{size}/context_c1_ctrv_gate.json`, field `regime`, the regime applied in
that frame). The threshold table above gives something else: probe shares,
computed on DEFAULT-regime density. NIGHT never occurred in any run.

| Run / size | dense | mixed | freeflow | NIGHT frames (whole run) |
|---|---:|---:|---:|---:|
| best.pt 448 | 0.80 | 0.35 | 0.00 | 0 |
| best.pt 640 | 0.80 | 0.28 | 0.00 | 0 |
| best.pt 960 | 0.84 | 0.29 | 0.00 | 0 |
| best.pt 1280 | 0.90 | 0.29 | 0.00 | 0 |
| yolo11_vn 1280 | 0.90 | 0.30 | 0.00 | 0 |

## Why C1 adds dense ID switches (masked final runs, 2026-10-14)

The final runs (`best85_roi_final`, `yolo11vn_roi_final2`) use the
pre-registered thresholds. On dense, C1 holds DENSE_PEAK in 80–90% of frames,
yet it switches identities more often than C2. The table shows the cause.

- Under the lifetime rule (`_should_use_ctrv`), a motorcycle track created
  while the controller is in DEFAULT keeps the linear filter for life.
- Dense opens with a DEFAULT stretch: 85, 119, 97, 63 and 59 frames for the
  five rows below. In those frames the density EMA is still below the
  threshold, and the 15-frame hysteresis has to elapse.
- Tracks created there carry the share of dense motorcycle track-frames shown
  in the table.
- Columns: the share of C1's dense switches that involve such a track (on
  either side of the switch), and the switch counts.

| Run / size | Track-frames from DEFAULT-born tracks | Switches involving them | C1 IDSW | C2 IDSW |
|---|---:|---:|---:|---:|
| best.pt 448 | 0.18 | 44 of 100 (0.44) | 100 | 62 |
| best.pt 640 | 0.20 | 55 of 139 (0.40) | 139 | 95 |
| best.pt 960 | 0.16 | 35 of 101 (0.35) | 101 | 81 |
| best.pt 1280 | 0.14 | 13 of 111 (0.12) | 111 | 103 |
| yolo11_vn 1280 | 0.14 | 15 of 131 (0.11) | 131 | 125 |

From 448 to 960 the linear tracks are over-represented in switches by about
2×. Removing their switches leaves counts close to C2's (640: 139 − 55 = 84,
against 95). At 1280 the effect is small.

This is a measured consequence of the lifetime design choice, not a tuning
target. Scripts: `c1_birth_regime.py` and `c1_switch_attrib.py`; the switch
counts reproduce the evaluator's IDSW exactly.
