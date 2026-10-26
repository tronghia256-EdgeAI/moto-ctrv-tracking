# External trackers, HOTA and block-bootstrap CIs on the reviewed dense GT

**Date:** 2026-10-12. **Scope:** one camera, one 600-frame window (`dense`).
Every number below comes from `metrics.json` in this folder, which
`code/scripts/eval_hota_ci.py` writes.

**Bottom line**

1. The review-run numbers reproduce bit-for-bit.
2. With their published MOT17 defaults, **OC-SORT and BoT-SORT do not beat C2.
   They also score below the plain ByteTrack baseline.** The gap is dominated by
   missed detections (motorcycle FN 7 335 and 4 559, against 1 547 for the
   baseline). Those trace back largely to their default confidence thresholds
   on this detector. This data cannot separate association quality from that
   threshold effect. Read the second caveat below before quoting it. With
   ByteTrack's score thresholds (section f), BoT-SORT ties the baseline and
   OC-SORT closes most of its gap; C2 still leads both, with every CI
   excluding zero.
3. The paired block-bootstrap CIs for C2 − baseline and Exp2/3 − baseline
   exclude zero for IDF1 and MOTA. This holds for both classes and all three
   block lengths.
4. **HOTA is strongly biased towards C2 on this GT.** 90.9 % of the motorcycle
   GT boxes are C2's own output boxes. Read the first caveat below.

## Setup

| Item | Value |
|---|---|
| GT | `experiments/gt_qa/reviewed/20261007T161301Z_126e0b66/gt_reviewed.txt` (gitignored), sha256 `0f1a44aa3c2a44cd1749573c5d53b61b259ae658d60519eb1ec7a24533af898b`. Absolute frames 10 700–11 299, 27 513 boxes, 427 identities (375 motorcycle, 52 car), 0 bus/truck/accident |
| Detections | `experiments/c2_real_vnweights_full/detections_full13726.npz`. All five trackers get the same per-frame `[x1,y1,x2,y2,score,class]` arrays (score ≥ 0.1, no empty frame) |
| Replay | All 13 726 frames from frame 0, so every tracker has a 10 700-frame warm-up. Only the segment is scored, as for `experiments/gt_ablation/` |
| ByteTrack variants | Frozen dumps `experiments/gt_ablation/tracks_{baseline,ctrv_gate_nsa,c1c2c3}.json`. sha256 values equal `review.json` |
| MOTA, IDF1, IDSW, Frag, FN, FP | `code.evaluation.metrics.compute_tracking_metrics`, IoU ≥ 0.5. `metrics.py` sha256 equals `review.json`. Frag uses the repository definition (re-acquisitions after a coverage gap) |
| HOTA, DetA, AssA | Port of TrackEval `HOTA.eval_sequence` (JonathonLuiten/TrackEval @12c8791, `trackeval/metrics/hota.py`, MIT). Averaged over α = 0.05, 0.10, …, 0.95. Similarity is the repository IoU (plain area, no +1) |
| Classes | *Per class:* GT and predictions are both filtered to that class. *ALL:* class-agnostic matching, as the repository evaluator does. This is not TrackEval's class-averaged combination. Each predicted track carries the class of the detection that started it, the rule the repository's ByteTrack uses. This holds for all five trackers |
| Timing | None measured |

## (a) Reproduction check (passed)

The three paper variants were re-scored against the reviewed GT along the code
path `gt_review.score_variants()` used. All 7 stored fields matched
`review.json` exactly for every variant and class: MOTA, IDF1,
fragmentation rate, IDSW, Frag, FN and FP. The script aborts on any mismatch.

| Variant | Class | MOTA | IDF1 | IDSW | identical to review.json |
|---|---|---:|---:|---:|---|
| baseline | motorcycle | 0.8451 | 0.8579 | 137 | yes |
| baseline | ALL | 0.8795 | 0.8891 | 141 | yes |
| ctrv_gate_nsa | motorcycle | 0.8933 | 0.9068 | 27 | yes |
| ctrv_gate_nsa | ALL | 0.9168 | 0.9268 | 29 | yes |
| c1c2c3 | motorcycle | 0.8795 | 0.8956 | 69 | yes |
| c1c2c3 | ALL | 0.9057 | 0.9178 | 73 | yes |

The car rows also match (baseline 0.9968 / 0.9961 / 2, ctrv_gate_nsa
0.9976 / 0.9962 / 2, c1c2c3 0.9955 / 0.9952 / 2). The sha256 of the reviewed GT,
the three dumps and `metrics.py` all equal the values recorded in `review.json`.

## (b) Results

The ByteTrack variants use the frozen dumps. OC-SORT and BoT-SORT use their
published MOT17 defaults on the same detections. Nothing was tuned.

### motorcycle

| Tracker | MOTA | IDF1 | HOTA | DetA | AssA | IDSW | Frag | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack baseline | 0.8451 | 0.8579 | 0.7577 | 0.7536 | 0.7651 | 137 | 156 | 1547 | 1618 |
| C2 (ctrv_gate_nsa) | 0.8933 | 0.9068 | 0.8918 | 0.8810 | 0.9031 | 27 | 60 | 872 | 1376 |
| Exp2/3 (c1c2c3) | 0.8795 | 0.8956 | 0.8677 | 0.8579 | 0.8785 | 69 | 86 | 990 | 1510 |
| OC-SORT | 0.6046 | 0.7210 | 0.6210 | 0.5814 | 0.6641 | 93 | 342 | 7335 | 998 |
| BoT-SORT | 0.7208 | 0.7935 | 0.7189 | 0.6939 | 0.7458 | 147 | 117 | 4559 | 1244 |

### ALL (class-agnostic)

| Tracker | MOTA | IDF1 | HOTA | DetA | AssA | IDSW | Frag | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack baseline | 0.8795 | 0.8891 | 0.8139 | 0.7992 | 0.8356 | 141 | 158 | 1546 | 1629 |
| C2 (ctrv_gate_nsa) | 0.9168 | 0.9268 | 0.9151 | 0.9058 | 0.9249 | 29 | 61 | 873 | 1388 |
| Exp2/3 (c1c2c3) | 0.9057 | 0.9178 | 0.8959 | 0.8866 | 0.9059 | 73 | 87 | 990 | 1532 |
| OC-SORT | 0.6352 | 0.7453 | 0.6466 | 0.6037 | 0.6931 | 101 | 390 | 9049 | 887 |
| BoT-SORT | 0.7447 | 0.8125 | 0.7363 | 0.7114 | 0.7628 | 153 | 123 | 5570 | 1302 |

### car

| Tracker | MOTA | IDF1 | HOTA | DetA | AssA | IDSW | Frag | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack baseline | 0.9968 | 0.9961 | 0.9963 | 0.9969 | 0.9958 | 2 | 1 | 3 | 15 |
| C2 (ctrv_gate_nsa) | 0.9976 | 0.9962 | 0.9967 | 0.9977 | 0.9957 | 2 | 1 | 1 | 12 |
| Exp2/3 (c1c2c3) | 0.9955 | 0.9952 | 0.9944 | 0.9948 | 0.9940 | 2 | 1 | 2 | 24 |
| OC-SORT | 0.6960 | 0.8148 | 0.7192 | 0.6515 | 0.7940 | 2 | 40 | 1854 | 29 |
| BoT-SORT | 0.7921 | 0.8734 | 0.7844 | 0.7460 | 0.8249 | 4 | 6 | 1119 | 166 |

`c1c2c3` is Exp2/3. Its dump is identical to `c1_ctrv_gate` on this segment
because C3 is inert here (`../gt_ablation/evidence.md`, Finding 4).

## Two caveats before quoting the table

### 1. The GT boxes are C2's own boxes, so HOTA is biased towards C2

The dense GT was pre-labelled from `experiments/c2_final/tracks_ctrv_gate_nsa.json`.
On this segment that file is frame-for-frame identical (600/600 frames) to the
`ctrv_gate_nsa` dump scored above as C2. The human review adjudicated identity
splits only. It did not adjudicate box geometry.

| Tracker | Class | LocA | HOTA@α=0.05 | HOTA@α=0.50 | HOTA@α=0.90 | HOTA (avg) | GT boxes reproduced (IoU ≥ 0.999) |
|---|---|---:|---:|---:|---:|---:|---:|
| ByteTrack baseline | motorcycle | 0.8975 | 0.8590 | 0.8446 | 0.4270 | 0.7577 | 0.8% |
| C2 (ctrv_gate_nsa) | motorcycle | 0.9888 | 0.9059 | 0.9004 | 0.8656 | 0.8918 | 90.9% |
| Exp2/3 (c1c2c3) | motorcycle | 0.9782 | 0.8916 | 0.8850 | 0.8094 | 0.8677 | 75.6% |
| OC-SORT | motorcycle | 0.9311 | 0.6611 | 0.6591 | 0.4954 | 0.6210 | 1.3% |
| BoT-SORT | motorcycle | 0.9376 | 0.7638 | 0.7555 | 0.6072 | 0.7189 | 1.5% |
| ByteTrack baseline | ALL | 0.9232 | 0.8904 | 0.8792 | 0.5619 | 0.8139 | 23.1% |
| C2 (ctrv_gate_nsa) | ALL | 0.9914 | 0.9263 | 0.9220 | 0.8943 | 0.9151 | 93.0% |
| Exp2/3 (c1c2c3) | ALL | 0.9831 | 0.9149 | 0.9098 | 0.8497 | 0.8959 | 80.6% |
| OC-SORT | ALL | 0.9322 | 0.6908 | 0.6881 | 0.5113 | 0.6466 | 1.1% |
| BoT-SORT | ALL | 0.9391 | 0.7838 | 0.7761 | 0.6125 | 0.7363 | 1.8% |

On ALL, the baseline's 23.1 % comes from cars. Car tracks use the same linear
Kalman filter in the baseline and in C2.

How to read it:

- **C2 − baseline on motorcycles, HOTA averaged over α: +0.134.**
  - At α = 0.05 the gap is +0.047; at α = 0.50 it is +0.056.
  - So roughly two-thirds of the averaged HOTA margin is high-α box agreement
    with the boxes the GT was copied from. It is not tracking quality.
  - **The averaged HOTA margin must not be presented as C2's improvement.**
- **MOTA and IDF1 at IoU 0.5 are much less sensitive, but not immune.**
  - Between α = 0.05 and 0.5, motorcycle DetA falls by 0.021 for the baseline
    and by 0.010 for C2.
  - That gives a residual localisation advantage of about 0.01 DetA at the
    0.5 threshold.
- **The association advantage is visible at low α:** motorcycle AssA at α = 0.05
  is 0.901 for C2 and 0.838 for the baseline.
- **Cars are nearly perfect for all three ByteTrack variants (HOTA ≈ 0.996).**
  This is partly by construction: the car GT is those same tracks.

### 2. The MOT17 default thresholds do not suit this detector

Each GT box is binned by the score of the best detection that overlaps it at
IoU ≥ 0.5:

| Class | GT boxes | none | (0.1,0.5] | (0.5,0.6] | (0.6,0.7] | (0.7,1] |
|---|---:|---:|---:|---:|---:|---:|
| motorcycle | 21312 | 1.6% | 12.8% | 9.7% | 15.9% | 59.9% |
| car | 6201 | 0.3% | 16.7% | 7.0% | 8.5% | 67.6% |

What each tracker's defaults can reach:

| Tracker | Default thresholds | Effect here |
|---|---|---|
| OC-SORT | Uses only detections > 0.6 (`det_thresh`); `use_byte` is off by default | At IoU 0.5, 24.1 % of motorcycle and 23.9 % of car GT boxes are out of reach whatever it does. A track is only output after 3 consecutive hits, and the count resets after every miss (`min_hits`) |
| BoT-SORT | Starts tracks only from detections ≥ 0.7; associates first at > 0.6, then at 0.1–0.6 for tracks already tracked | 40.1 % of motorcycle GT boxes have no detection above 0.7 |
| ByteTrack baseline | Associates first at > 0.5, then at 0.1–0.5; starts tracks from ≥ 0.6 | Only 14.5 % of motorcycle (16.9 % of car) GT boxes have no detection above 0.5 |

This explains why the FN counts for OC-SORT (7 335) and BoT-SORT (4 559) dwarf
the baseline's (1 547).

The MOT17 defaults were set for a YOLOX pedestrian detector with a different
score calibration. **The gap to C2 is therefore mainly a statement about default
detection thresholds on this detector. It is not a clean comparison of motion
models or association.** Section (f) adds the threshold-matched run that answers
the obvious reviewer question.

## (c) Paired block-bootstrap CIs

### Procedure (exact)

The procedure applies to each block length L ∈ {15, 30, 60}, with n = 600 frames
and K = n/L blocks per replicate (40, 20 and 10).

1. **Score every possible block.**
   - For every start s ∈ {0, …, n−L}, tracker and class, score the block
     [s, s+L) on its own. This uses the repository evaluator's logic at
     IoU 0.5, starting from an empty CLEAR state and with the block's own
     global IDF1 assignment.
   - Store the block's GT dets, FN, FP, IDSW, IDTP and predicted dets.
2. **Draw replicates.**
   - Replicate b = 1…B with B = 10 000.
   - Draw K starts i.i.d. uniform on {0, …, n−L}. This is a moving-block
     bootstrap (Künsch 1989) without wrap-around. The generator is
     `numpy.random.default_rng([20261012, L])`.
   - The resampled sequence is the K blocks concatenated, with every GT and
     predicted identity id offset by the block's position in the replicate
     ((index + 1)·10⁹). No identity therefore spans two blocks.
3. **Compute the replicate's metrics from the stored blocks.**
   - Because ids are offset, the CLEAR state never carries across blocks.
   - The IDF1 bipartite graph is block-diagonal, so the global optimum is the
     sum of the per-block optima.
   - The replicate's metrics are therefore exactly
     MOTA\* = 1 − Σ(FN+FP+IDSW)/ΣGT and IDF1\* = 2·ΣIDTP/(ΣGT+ΣPred), summed
     over its K drawn blocks.
4. **The script checks this rather than assuming it.**
   - It asserts that the fast block counts equal `compute_tracking_metrics` on
     the full sequence of every tracker and class.
   - It asserts the same on two random blocks per tracker, class and L.
   - It asserts that replicate 0, built literally as one concatenated id-offset
     sequence and scored by `compute_tracking_metrics`, equals the summed
     blocks (C2 and baseline, motorcycle, each L).
5. **Pair the trackers.** The same starts are used for both trackers of a pair,
   for every pair and for both classes. The statistic is
   Δ\*_b = metric\*_A − metric\*_B.
6. **Report the interval.** The 95 % CI is the 2.5th and 97.5th percentile of
   {Δ\*_b} (numpy linear interpolation). Also reported:
   - the full-sequence Δ, from the table;
   - the "window Δ", which is the same statistic on the original sequence cut
     into n/L consecutive blocks, i.e. the quantity being resampled;
   - the bootstrap mean and SE;
   - P(Δ\* ≤ 0).

**Note on interpretation.** Cutting into L-frame windows drops identity errors
that straddle a cut. For example, a switch at a block's first frame is not
counted, and a track fragmented across two blocks is matched per block.

- The interval is therefore for the **L-frame-window** version of the metric.
- For IDF1 the window Δ is smaller than the full-sequence Δ (+0.034 against
  +0.049 for C2 − baseline motorcycle at L = 30), and grows with L. That
  difference is the part of C2's gain that lives in identity errors longer
  than L frames.
- In every IDF1 row the full-sequence Δ lies **above** the L = 15 and L = 30
  intervals. For Exp2/3 it also lies above the L = 60 interval.
- In the paper, quote the window Δ together with its interval. Do not put the
  full-sequence Δ next to an interval that does not contain it.
- MOTA is barely affected.

### Results, L = 30 (main)

| Difference | Class | Metric | Full-sequence diff | Window diff (L=30) | Bootstrap mean | SE | 95% CI (L=30) | P(diff*<=0) |
|---|---|---|---:|---:|---:|---:|---|---:|
| C2 - ByteTrack baseline | motorcycle | IDF1 | +0.0489 | +0.0340 | +0.0347 | 0.0033 | [+0.0282, +0.0410] | 0.0000 |
| C2 - ByteTrack baseline | motorcycle | MOTA | +0.0482 | +0.0464 | +0.0469 | 0.0042 | [+0.0389, +0.0551] | 0.0000 |
| C2 - ByteTrack baseline | ALL | IDF1 | +0.0377 | +0.0262 | +0.0267 | 0.0025 | [+0.0219, +0.0315] | 0.0000 |
| C2 - ByteTrack baseline | ALL | MOTA | +0.0373 | +0.0358 | +0.0362 | 0.0030 | [+0.0303, +0.0422] | 0.0000 |
| Exp2/3 - ByteTrack baseline | motorcycle | IDF1 | +0.0377 | +0.0230 | +0.0243 | 0.0029 | [+0.0186, +0.0298] | 0.0000 |
| Exp2/3 - ByteTrack baseline | motorcycle | MOTA | +0.0344 | +0.0332 | +0.0337 | 0.0039 | [+0.0258, +0.0410] | 0.0000 |
| Exp2/3 - ByteTrack baseline | ALL | IDF1 | +0.0288 | +0.0174 | +0.0187 | 0.0023 | [+0.0141, +0.0231] | 0.0000 |
| Exp2/3 - ByteTrack baseline | ALL | MOTA | +0.0262 | +0.0253 | +0.0259 | 0.0031 | [+0.0197, +0.0318] | 0.0000 |

### Sensitivity to block length (window diff, 95% CI)

| Difference | Class | Metric | Full-seq diff | L=15 | L=30 | L=60 |
|---|---|---|---:|---|---|---|
| C2 - ByteTrack baseline | motorcycle | IDF1 | +0.0489 | +0.0290, [+0.0252, +0.0342] | +0.0340, [+0.0282, +0.0410] | +0.0412, [+0.0312, +0.0497] |
| C2 - ByteTrack baseline | motorcycle | MOTA | +0.0482 | +0.0453, [+0.0382, +0.0524] | +0.0464, [+0.0389, +0.0551] | +0.0473, [+0.0414, +0.0562] |
| C2 - ByteTrack baseline | ALL | IDF1 | +0.0377 | +0.0224, [+0.0195, +0.0264] | +0.0262, [+0.0219, +0.0315] | +0.0318, [+0.0242, +0.0378] |
| C2 - ByteTrack baseline | ALL | MOTA | +0.0373 | +0.0350, [+0.0297, +0.0402] | +0.0358, [+0.0303, +0.0422] | +0.0366, [+0.0325, +0.0422] |
| Exp2/3 - ByteTrack baseline | motorcycle | IDF1 | +0.0377 | +0.0204, [+0.0168, +0.0249] | +0.0230, [+0.0186, +0.0298] | +0.0270, [+0.0195, +0.0350] |
| Exp2/3 - ByteTrack baseline | motorcycle | MOTA | +0.0344 | +0.0325, [+0.0258, +0.0392] | +0.0332, [+0.0258, +0.0410] | +0.0338, [+0.0283, +0.0416] |
| Exp2/3 - ByteTrack baseline | ALL | IDF1 | +0.0288 | +0.0155, [+0.0128, +0.0192] | +0.0174, [+0.0141, +0.0231] | +0.0205, [+0.0147, +0.0271] |
| Exp2/3 - ByteTrack baseline | ALL | MOTA | +0.0262 | +0.0248, [+0.0196, +0.0302] | +0.0253, [+0.0197, +0.0318] | +0.0258, [+0.0216, +0.0324] |

Every interval excludes 0 at every L. P(Δ\* ≤ 0) = 0 in all 10 000 replicates
for every row.

### Extra (not requested): C2 against the external trackers, L = 30

| Difference | Class | Metric | Full-seq diff | Window diff (L=30) | 95% CI (L=30) | P(diff*<=0) |
|---|---|---|---:|---:|---|---:|
| C2 - OC-SORT | motorcycle | IDF1 | +0.1858 | +0.1819 | [+0.1618, +0.2020] | 0.0000 |
| C2 - OC-SORT | motorcycle | MOTA | +0.2886 | +0.2864 | [+0.2565, +0.3151] | 0.0000 |
| C2 - OC-SORT | ALL | IDF1 | +0.1814 | +0.1750 | [+0.1575, +0.1912] | 0.0000 |
| C2 - OC-SORT | ALL | MOTA | +0.2816 | +0.2797 | [+0.2533, +0.3029] | 0.0000 |
| C2 - BoT-SORT | motorcycle | IDF1 | +0.1133 | +0.1070 | [+0.0928, +0.1217] | 0.0000 |
| C2 - BoT-SORT | motorcycle | MOTA | +0.1724 | +0.1712 | [+0.1481, +0.1924] | 0.0000 |
| C2 - BoT-SORT | ALL | IDF1 | +0.1143 | +0.1043 | [+0.0945, +0.1145] | 0.0000 |
| C2 - BoT-SORT | ALL | MOTA | +0.1721 | +0.1710 | [+0.1549, +0.1844] | 0.0000 |

These intervals inherit caveat 2: they measure default thresholds as much as
tracking.

**What the CIs do not cover.** They capture frame-sampling variability within
this one window only. They are conditional on the detections and on a GT seeded
from C2. They say nothing about other scenes, cameras or annotators, and
nothing about the seeding bias.

## Verification

- **HOTA port against the real TrackEval.**
  - Setup: TrackEval @12c8791 was installed with `pip --target` outside the
    project environment, with the `np.float`/`np.int` aliases restored at
    runtime.
  - Similarity was recomputed by TrackEval's own `_calculate_box_ious`. It is
    bit-identical to the repository IoU.
  - The HOTA, DetA, AssA and LocA per-α arrays match exactly (max |Δ| = 0) for
    all 5 trackers × 3 classes.
- **TrackEval CLEAR and Identity against the repository evaluator.**
  - IDF1 is identical everywhere.
  - TrackEval counts 2 more IDSW than the repository for baseline, C2, Exp2/3
    and BoT-SORT (motorcycle and ALL). This makes MOTA lower by ≤ 0.0001.
  - The cause is a definitional difference. TrackEval prefers to continue a
    correspondence only if it held in the previous frame. The repository
    (`compute_tracking_metrics`, step 1) keeps the last correspondence across a
    gap in the GT track.
  - The repository numbers are the ones reported.
- **Vendored trackers against unmodified upstream.**
  - `code/scripts/verify_external_upstream.py` runs the unmodified upstream
    code on the same detections. Upstream uses its real dependencies, `lap`
    0.5.13 and `filterpy` 1.4.5, installed outside the project environment.
  - It compares track ids and boxes on every frame. **0 of 13 726 frames
    differ** for:
    - OC-SORT @2d34c67 (230 941 boxes);
    - OC-SORT master @8462e7e, which includes the 2026 Python "efficiency"
      refactor (identical as well, so the version choice does not matter);
    - BoT-SORT @2519854 (285 501 boxes).
  - For BoT-SORT, `cython_bbox` (needs a C compiler) was shimmed in memory by
    the +1-pixel numpy IoU of `code/tracker/vendor/matching.ious`. Its
    `--cmc-method none` ran as upstream.
- **Determinism.** No randomness exists outside the bootstrap, which uses fixed
  seeds. Two full runs of `eval_hota_ci` produced the same `metrics.json`
  sha256.

## (d) Reproduce

Run from the repo root with the venv active (`python` = `.venv\Scripts\python.exe`).
`<SCRATCH>` is any directory outside the repo. Nothing is installed into the
project environment.

```bash
# 1. External trackers on the cached detections. Writes
#    experiments/external_trackers/tracks_{ocsort,botsort}.json
#    (gitignored: box coordinates).
python -m code.scripts.run_external_trackers
# 1b. Threshold-matched controls, section (f).
python -m code.scripts.run_external_trackers --trackers ocsort_matched,ocsort_byte,botsort_matched

# 2. Everything in this file. Writes experiments/external_trackers/metrics.json.
#    --trackeval is only for the cross-check; omit it to skip.
python -m pip install --no-deps --target <SCRATCH>/pydeps \
    https://github.com/JonathonLuiten/TrackEval/archive/12c8791b303e0a0b50f753af204249e622d0281a.zip
python -m code.scripts.eval_hota_ci --trackeval <SCRATCH>/pydeps

# 3. Optional: vendored == unmodified upstream.
python -m pip install --no-deps --only-binary=:all: --target <SCRATCH>/pydeps lap==0.5.13
python -m pip install --no-deps --target <SCRATCH>/pydeps filterpy==1.4.5
git clone https://github.com/noahcao/OC_SORT <SCRATCH>/OC_SORT
git -C <SCRATCH>/OC_SORT checkout 2d34c67d58f89b0762ce912c38f94746175e2903
git clone https://github.com/NirAharon/BoT-SORT <SCRATCH>/BoT-SORT
git -C <SCRATCH>/BoT-SORT checkout 251985436d6712aaf682aaaf5f71edb4987224bd
python -m code.scripts.verify_external_upstream --tracker ocsort  --src <SCRATCH>/OC_SORT  --pydeps <SCRATCH>/pydeps
python -m code.scripts.verify_external_upstream --tracker botsort --src <SCRATCH>/BoT-SORT --pydeps <SCRATCH>/pydeps
# Same check against current OC-SORT master: checkout
# 8462e7e729a93ccd3bd995c0a79a890336cb3a0b, then rerun the ocsort line.
```

Each verify call exits 0 and prints `"frames_differing": 0` when vendored and
upstream agree on every frame.

## (e) Deviations from upstream

### OC-SORT

- **Source:** `code/tracker/external/ocsort/`, noahcao/OC_SORT @2d34c67, MIT.
- **Hyperparameters:** MOT17 defaults, unchanged:
  - `det_thresh` (= `track_thresh`) 0.6
  - `iou_threshold` 0.3
  - `min_hits` 3
  - `max_age` 30 frames
  - `delta_t` 3
  - `inertia` 0.2
  - `asso` = `iou`
  - `use_byte` = False

  These come from `utils/args.py` together with the `OCSort(...)` call in
  `yolox/evaluators/mot_evaluator.py::evaluate_ocsort`.
- **Input interface.** The input is N×6 `[x1,y1,x2,y2,score,class]` in frame
  pixels. The score is used as-is, because the YOLOX path would multiply
  obj × cls confidence. There is no rescaling (`img_info`/`img_size` unused,
  scale ≡ 1).
- **Class metadata.**
  - A track keeps the class of the detection that created it, returned as a
    6th output column.
  - Class never enters association. The upstream MOT17 path is class-agnostic
    as well; the class-aware `update_public` is not used.
- **`filterpy` helpers.** `reshape_z`, `pretty_str` and `logpdf` are copied
  verbatim from filterpy 1.4.5 into `kalmanfilter.py`. Only `reshape_z` is on
  the runtime path.
- **Assignment solver.** `lap` is not installed, so upstream's own fallback
  (`scipy.optimize.linear_sum_assignment`) runs. This needed no code change.
  The output is identical to upstream running with `lap`.
- **Not applied.** These are evaluation-protocol steps of the MOT17 pedestrian
  benchmark, not part of the tracker, and the repository's ByteTrack variants
  do not apply their equivalents either:
  - the output filter `w·h > 100` and `w/h ≤ 1.6`, which would delete most
    cars;
  - offline linear interpolation, which the paper used for its MOT17 test
    numbers.
- **Called on every frame.** Upstream's evaluator skips `update()` on frames
  with no detections. No such frame exists here.

### BoT-SORT

- **Source:** `code/tracker/external/botsort/`, NirAharon/BoT-SORT @2519854, MIT.
- **Hyperparameters:** the MOT17 `--default-parameters` generic values:
  - `track_high_thresh` 0.6
  - `track_low_thresh` 0.1
  - `new_track_thresh` 0.7
  - `track_buffer` 30 at `frame_rate` 30, i.e. 30 frames
  - `match_thresh` 0.8
  - score fusion on (`mot20` = False)
  - `proximity_thresh` 0.5 and `appearance_thresh` 0.25 (inert without ReID)
- **Per-sequence overrides do not apply.** Upstream sets `track_buffer` 14/25
  and `track_high_thresh` 0.65–0.7 for specific MOT17 sequences.
- **No ReID.** This is the paper's "BoT-SORT" row, not "BoT-SORT-ReID".
- **No camera-motion compensation (fixed camera).**
  - Upstream's MOT17 default is `--cmc-method file`, precomputed GMC for the
    MOT17 videos.
  - `none` returns the identity warp, for which `multi_gmc` is an exact no-op.
    So the GMC object and both `multi_gmc` calls are removed.
  - Upstream run with `none` gives identical output.
- **Input interface and class metadata.** The input is N×6. The score is used
  as-is, and the class is set at track birth, as for OC-SORT.
- **Matching module.** It is `code/tracker/vendor/matching.py`. Its
  `linear_assignment`, `ious`, `iou_distance` and `fuse_score` are AST-identical
  to BoT-SORT's, except for `lap` → scipy and `cython_bbox` → numpy with the +1
  pixel convention.
- **`np.float` → `float`.**
- **Not applied:** the output filter `aspect_ratio_thresh` 1.6 and
  `min_box_area` 10, as for OC-SORT.
- **Unchanged upstream behaviours that differ from ByteTrack:**
  - BoT-SORT outputs unconfirmed (newborn) tracks in their birth frame.
  - Its Kalman state is (x, y, w, h).

### Common to both

- Same detections, same 13 726-frame replay, and boxes rounded to 0.1 px in the
  dumps, all exactly as `compare_motion_models.run_replay` does for the
  ByteTrack variants.
- Lost tracks are kept for 30 frames, as in the repository's ByteTrack
  (`frame_rate: 30`). The clip itself is 25 fps.

## (f) Threshold-matched controls, and HOTA for every ablation row

Added after the first version of this file. Same detections, replay, GT and
scorer as above; `metrics.json` now holds all eleven variants.

**Settings.** ByteTrack's detection-score thresholds are: first association
> 0.5, second stage 0.1–0.5, birth ≥ 0.6 (`det_thresh = track_thresh + 0.1`
in `code/tracker/vendor/byte_tracker.py`). Everything not listed keeps the
published value from section (e).

| Variant | Settings |
|---|---|
| `botsort_matched` | `track_high_thresh` 0.5, `new_track_thresh` 0.6. The low threshold is already 0.1 |
| `ocsort_matched` | `det_thresh` 0.5, `use_byte` on. OC-SORT's one threshold sets both association and birth, so births also start at 0.5 |
| `ocsort_byte` | `det_thresh` 0.6 (its published value, = ByteTrack's birth threshold), `use_byte` on |

**Checks.**
- The reproduction gate now also covers the `ctrv`, `ctrv_nsa` and `ctrv_gate`
  dumps. All 18 variant × class rows equal `review.json` in all 7 fields.
- The TrackEval cross-check gives max |port − TrackEval| = 0 over all arrays
  of all eleven variants.
- Every number from the first version of `metrics.json` is unchanged, compared
  key by key (metrics, bootstrap, deltas, detector coverage). The bootstrap
  draws are shared by all pairs, so adding pairs moves nothing.

### Results (HOTA@0.5 = HOTA at α = 0.5, the IoU of the other metrics)

| Tracker | Class | MOTA | IDF1 | HOTA@0.5 | HOTA (avg) | IDSW | Frag | FN | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack baseline | motorcycle | 0.8451 | 0.8579 | 0.8446 | 0.7577 | 137 | 156 | 1547 | 1618 |
| C2 (ctrv_gate_nsa) | motorcycle | 0.8933 | 0.9068 | 0.9004 | 0.8918 | 27 | 60 | 872 | 1376 |
| OC-SORT, 0.5 + BYTE | motorcycle | 0.6700 | 0.7677 | 0.7462 | 0.6963 | 236 | 221 | 1760 | 5037 |
| OC-SORT, 0.6 + BYTE | motorcycle | 0.7336 | 0.8116 | 0.7796 | 0.7267 | 148 | 195 | 1966 | 3563 |
| BoT-SORT, matched | motorcycle | 0.8414 | 0.8581 | 0.8460 | 0.8051 | 221 | 128 | 1136 | 2024 |
| ByteTrack baseline | ALL | 0.8795 | 0.8891 | 0.8792 | 0.8139 | 141 | 158 | 1546 | 1629 |
| C2 (ctrv_gate_nsa) | ALL | 0.9168 | 0.9268 | 0.9220 | 0.9151 | 29 | 61 | 873 | 1388 |
| OC-SORT, 0.5 + BYTE | ALL | 0.6926 | 0.7893 | 0.7654 | 0.7128 | 270 | 232 | 1339 | 6848 |
| OC-SORT, 0.6 + BYTE | ALL | 0.7741 | 0.8358 | 0.8064 | 0.7502 | 172 | 203 | 2011 | 4031 |
| BoT-SORT, matched | ALL | 0.8704 | 0.8754 | 0.8677 | 0.8227 | 230 | 132 | 1165 | 2172 |

HOTA@0.5 of the ablation dumps (motorcycle / ALL), for the paper's Table 2:
`ctrv` 0.8548 / 0.8870, `ctrv_nsa` 0.8770 / 0.9041, `ctrv_gate` 0.8459 / 0.8800,
`c1c2c3` 0.8850 / 0.9098.

Paired block bootstrap, L = 30 (window diff, 95 % CI). Every row also excludes
zero at L = 15 and L = 60, with P(Δ\* ≤ 0) = 0:

| Difference | Class | Metric | Full-seq diff | Window diff | 95% CI |
|---|---|---|---:|---:|---|
| C2 − BoT-SORT, matched | motorcycle | IDF1 | +0.0487 | +0.0369 | [+0.0299, +0.0449] |
| C2 − BoT-SORT, matched | motorcycle | MOTA | +0.0519 | +0.0496 | [+0.0420, +0.0588] |
| C2 − BoT-SORT, matched | ALL | IDF1 | +0.0513 | +0.0322 | [+0.0274, +0.0385] |
| C2 − OC-SORT, 0.6 + BYTE | motorcycle | IDF1 | +0.0952 | +0.0848 | [+0.0767, +0.0942] |
| C2 − OC-SORT, 0.6 + BYTE | motorcycle | MOTA | +0.1596 | +0.1582 | [+0.1464, +0.1716] |
| C2 − OC-SORT, 0.5 + BYTE | motorcycle | IDF1 | +0.1391 | +0.1153 | [+0.1029, +0.1349] |

### Reading

- **BoT-SORT with ByteTrack's thresholds ties the baseline.** Motorcycle IDF1
  is 0.858 for both, and MOTA 0.841 against 0.845. It has fewer misses (1136
  against 1547) but more false positives (2024 against 1618) and more identity
  switches (221 against 137). The published-threshold gap was therefore the
  thresholds, as caveat 2 said.
- **OC-SORT is best with its BYTE stage at its own 0.6.** The BYTE stage alone
  (published 0.6 → 0.6 + BYTE) cuts motorcycle misses from 7335 to 1966 but
  raises false positives from 998 to 3563. Lowering the threshold to 0.5 also
  lets tracks start from 0.5–0.6 detections, which ByteTrack never initializes.
  FP then rises to 5037 and IDF1 falls to 0.768. The paper reports 0.6 + BYTE
  as "OC-SORT, matched", the setting more favourable to OC-SORT.
- **C2 leads every external configuration** on MOTA, IDF1 and HOTA@0.5, for
  both classes. The smallest margin, over matched BoT-SORT, is the same size
  as the margin over the baseline. Caveat 1 still applies: about 0.01 DetA at
  α = 0.5 favours C2 against any tracker whose boxes were not the pre-labels.

## Files

- `code/tracker/external/{__init__.py, ocsort/, botsort/}`: vendored code plus
  upstream `LICENSE` files. Patches are marked `CAAP patch`.
- `code/scripts/run_external_trackers.py`: runs OC-SORT and BoT-SORT on the
  cached detections.
- `code/scripts/eval_hota_ci.py`: reproduction gate, table, HOTA port,
  bootstrap and diagnostics.
- `code/scripts/verify_external_upstream.py`: vendored against unmodified
  upstream.
- `experiments/external_trackers/metrics.json`: all numbers, including per-α
  HOTA/DetA/AssA/LocA arrays, provenance hashes and the cross-check.
- `experiments/external_trackers/tracks_{ocsort,botsort,ocsort_matched,ocsort_byte,botsort_matched}.json`:
  dumps (gitignored).
