# C2's motion model inside OC-SORT (pre-registered 2026-10-14, before any metric)

**Question.** Does C2's class-conditioned motion model improve the strongest
baseline? The test is OC-SORT at matched thresholds (`ocsort_matched`: 0.5
plus a BYTE second stage).

**Arms.**

- `ocsort_matched_ctrv`: motorcycle tracks use C2's CTRV EKF with NSA, with
  the same settings as the `ctrv_gate_nsa` arm. Every other class keeps
  OC-SORT's linear Kalman filter.
- `ocsort_matched`: unchanged.

The two arms run on identical cached detections, with the masked protocol
(Nghia's option B). Code: `code/tracker/ocsort_ctrv.py`, branch
`exp/ocsort-ctrv`, commit 2d50fae, fixed before any metric was computed.

**Deviations from a plain filter swap**, stated so they are not tuning knobs:

1. C2's covariance gating is part of ByteTrack's lost-track matching and is
   not carried over. The arm is C2's motion model (CTRV + NSA) only.
2. Observation-centric re-update (ORU) is reproduced as upstream does it.
   - The filter state is saved at the first missed frame.
   - On re-observation it is restored and re-updated along a linear virtual
     trajectory.
   - The re-observed box is then applied again, a double application that
     upstream's linear filter performs too.
3. NSA and the virtual observations of the re-update:
   - The virtual observations have no detector score, so they use the plain
     measurement noise R.
   - Only real detections, including the re-observed one, scale R by their
     score.
4. OC-SORT's association (OCM, OCR, BYTE) and all of its parameters are
   unchanged.

**Sanity checks before the run, without metrics.** On the final 640 masked
cache:

- `ocsort_matched` under the new code reproduces the final dump byte for byte.
- The new arm runs all 11,300 frames with finite boxes and no ROI-rounding
  violation.

**Reading, fixed in advance.**

- **Primary: mixed.** That GT was seeded by the baseline tracker, so it is
  neutral between the two arms. The statistic is the seed-corrected motorcycle
  IDF1 difference `ocsort_matched_ctrv − ocsort_matched` (`seed_bracket`
  mid value, bracket [s_lo, s_hi]) at every input size, with the size of
  Table 2 named first. The verdict is:
  - **helps** if the whole bracket is above 0;
  - **hurts** if it is entirely below 0;
  - **no clear effect** otherwise.
- **Secondary:** dense motorcycle IDSW and the dense and freeflow IDF1
  differences, reported as ranges.
  - Those GTs were seeded by C2's own output, and the new arm shares C2's
    motion model. Its boxes and identities may therefore be partly seed-like,
    which would inflate it against `ocsort_matched` in the hoped-for
    direction. The bracket treats both arms as non-seed and cannot remove
    this.
  - Diagnostic: each arm's exact-box share against the dense GT (from
    `metrics.json`, `exact_box_share`), reported beside the difference.
- **No parameter of either arm is changed after the results exist.** A
  negative or null result is reported as measured.

## Amendment (2026-10-14 14:25, still before any metric)

The bracket spans only the audit share's Wilson range, not frame-sampling
variability. The verdicts above are therefore tightened.

- **helps** requires both of the following:
  - the whole seed-corrected mixed bracket is above 0;
  - the 95% paired moving-block bootstrap interval of the UNcorrected mixed
    motorcycle IDF1 difference `ocsort_matched_ctrv − ocsort_matched` is
    above 0.
- **hurts** requires both intervals entirely below 0.
- **no clear effect** covers everything else.

The bootstrap uses the same deterministic code as the runner:
`eval_hota_ci.block_table` and `run_roi_paper.paired_ci`, with 30-frame
blocks, 10,000 replicates and the seed `[20261013, 30]`. The runner computes
CIs only for its fixed pairs, so this pair is computed post hoc.

## Result: yolo11_vn at 1280 (2026-10-15, read as fixed above)

Run `yolo11vn_roi_ocsortctrv` (2d50fae, masked). Its detections, C1 threshold
and all 12 earlier arms equal `yolo11vn_roi_final2` exactly. Records:
`experiments/paper_final/yolo11vn_roi_ocsortctrv/`,
`experiments/seed_bias/bracket_yolo11vn_roi_ocsortctrv_1280.json` and
`yolo11vn_roi_ocsortctrv_1280.json` (from `read_ocsort_ctrv.py`).

- **Primary, mixed:** `ocsort_matched_ctrv − ocsort_matched`, motorcycle IDF1.
  - Seed-corrected: +0.0003, bracket [−0.0022, +0.0014].
  - Block bootstrap of the uncorrected difference (−0.0062): 95% interval
    [−0.0094, +0.0008].
  - Both intervals include 0. **Verdict: no clear effect.**
- **Secondary:**
  - Dense IDF1: seed-corrected +0.0042 [+0.0037, +0.0052]. Dense IDSW: 155 vs
    138, so the new arm has 17 more switches.
  - Freeflow IDF1: +0.0056 [+0.0034, +0.0076].
  - The exact-box share against the dense GT is the same for both arms (5e-5).

The best.pt sizes follow when `best85_roi_ocsortctrv` completes. Table 2's size
is not fixed yet: the night benchmark of 14/10 stopped on a throttled CPU.

## Result: best.pt at 448/640/960/1280 (2026-10-15, read as fixed above)

Run `best85_roi_ocsortctrv` (2d50fae, masked). At every size its detections,
C1 threshold and all 12 earlier arms equal `best85_roi_final` exactly.

Primary, mixed: `ocsort_matched_ctrv − ocsort_matched`, motorcycle IDF1.
Table 2's size is not fixed yet, so sizes are listed in order.

| Size | Seed-corrected [bracket] | Raw diff | Block bootstrap 95% | Verdict |
|---|---|---|---|---|
| 448 | +0.0065 [+0.0034, +0.0079] | −0.0015 | [−0.0061, +0.0012] | no clear effect |
| 640 | +0.0018 [−0.0017, +0.0034] | −0.0072 | [−0.0135, −0.0014] | no clear effect |
| 960 | −0.0012 [−0.0040, +0.00003] | −0.0084 | [−0.0095, −0.0030] | no clear effect |
| 1280 | +0.0013 [+0.0011, +0.0014] | +0.0008 | [−0.0076, +0.0070] | no clear effect |

- At 960, "hurts" is missed only by the bracket's upper end, which is
  +0.00003. The bootstrap interval is entirely below 0.
- At 640, the two intervals point in opposite directions.

Secondary, seed-corrected (dense and freeflow GT were seeded by C2; see above):

- **Dense IDF1:** −0.0026, +0.0031, −0.0043, −0.0054.
- **Dense IDSW, new arm vs `ocsort_matched`:** 88 vs 39, 104 vs 79, 108 vs 106,
  and 158 vs 121. The new arm switches more at every size, as it does with
  yolo11_vn (155 vs 138).
- **Freeflow IDF1:** +0.0087, +0.0063, +0.0076, −0.0012.
- **Exact-box share against the dense GT:** 0 for both arms at every size.

**Overall:** C2's motion model inside OC-SORT gives no clear effect on mixed
for either detector at any size. On dense it adds identity switches
everywhere. The motorcycle motion model does not carry over to the strongest
baseline.
