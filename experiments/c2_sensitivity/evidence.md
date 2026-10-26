# C2 parameter sensitivity on `dense`: an upper bound, not the verdict

**Date:** 2026-10-12. **Script:** `code/scripts/sweep_c2_sensitivity.py`.
**Numbers:** `sensitivity_10700_11300.json` in this folder. The dumps are kept
under `dumps_10700_11300/`; they are gitignored because they hold box
coordinates.

**Bottom line**

1. On `dense`, C2's gain is **sensitive to Q and R**. Scaling either by 2x
   in either direction drops motorcycle IDF1 from 0.907 to 0.85–0.88 and
   raises ID switches from 27 to 97–185. The baseline is 0.858 with 137.
2. **This GT cannot say how much of that is real.** It was corrected from the
   default C2 run's own output, so it keeps that run's boxes (90.9% of
   motorcycle GT boxes, IoU ≥ 0.999) and every association the reviewers did
   not change. The Q/R variants agree with the default run on only 5–14% of
   their boxes. Any such divergence is penalised by construction.
3. **The gate constants show a real mechanism.** χ² and c0 variants keep
   78–91% agreement, yet a looser gate still costs heavily: χ² at 99% gives
   173 switches, c0 = 0.3 gives 265. That is the wrong-rescue mechanism of the
   paper's super-additivity finding. A tighter gate costs less (69 and 77
   switches).
4. **The NSA floor is irrelevant within 0.01–0.1** (identical metrics). A
   floor of 0.2 costs 0.005 IDF1.
5. **Seeding explains part, not all.** Across the 17 non-default variants,
   Pearson r = 0.59 between the IDF1 drop and divergence from the default.
6. **The decisive run is the baseline-seeded `mixed` segment** once its GT is
   final. Until then, quote the dense result only as an upper bound on
   sensitivity (a lower bound on robustness), and never as "C2 is robust".

## Results (motorcycle, IoU 0.5, `dense` 10700–11300, full frame)

| Variant | Change | MOTA | IDF1 | IDSW | HOTA α=0.5 | Exact-box share vs GT | Agreement with default |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | linear KF | 0.845 | 0.858 | 137 | 0.845 | 0.008 | 0.008 |
| **c2_default** | shipped constants | **0.893** | **0.907** | **27** | **0.900** | 0.909 | 1.000 |
| q_x0.25 | Q × 0.25 | 0.859 | 0.865 | 110 | 0.851 | 0.075 | 0.080 |
| q_x0.5 | Q × 0.5 | 0.867 | 0.879 | 97 | 0.866 | 0.125 | 0.134 |
| q_x2 | Q × 2 | 0.853 | 0.862 | 152 | 0.843 | 0.134 | 0.139 |
| q_x4 | Q × 4 | 0.817 | 0.811 | 304 | 0.792 | 0.065 | 0.067 |
| q_ang_x0.1 | θ, ω noise × 0.1 | 0.874 | 0.885 | 71 | 0.875 | 0.363 | 0.394 |
| q_ang_x10 | θ, ω noise × 10 | 0.810 | 0.798 | 612 | 0.785 | 0.228 | 0.242 |
| r_x0.25 | R × 0.25 | 0.840 | 0.854 | 113 | 0.839 | 0.052 | 0.054 |
| r_x0.5 | R × 0.5 | 0.859 | 0.873 | 110 | 0.861 | 0.110 | 0.114 |
| r_x2 | R × 2 | 0.859 | 0.867 | 143 | 0.852 | 0.077 | 0.081 |
| r_x4 | R × 4 | 0.843 | 0.849 | 185 | 0.830 | 0.032 | 0.032 |
| chi2_p90 | gate at χ²₂ 90% (4.605) | 0.880 | 0.895 | 69 | 0.883 | 0.826 | 0.909 |
| chi2_p99 | gate at χ²₂ 99% (9.210) | 0.855 | 0.862 | 173 | 0.846 | 0.761 | 0.828 |
| c0_0.3 | rescue base cost 0.3 | 0.847 | 0.851 | 265 | 0.834 | 0.723 | 0.781 |
| c0_0.7 | rescue base cost 0.7 | 0.871 | 0.888 | 77 | 0.877 | 0.790 | 0.871 |
| nsa_floor_0.01 | NSA floor 0.01 | 0.893 | 0.907 | 27 | 0.900 | 0.909 | 1.000 |
| nsa_floor_0.1 | NSA floor 0.1 | 0.893 | 0.907 | 27 | 0.900 | 0.901 | 0.992 |
| nsa_floor_0.2 | NSA floor 0.2 | 0.887 | 0.902 | 56 | 0.893 | 0.661 | 0.726 |

The defaults were not tuned on this segment. Q was tuned on synthetic
trajectories and frozen before any real footage was processed; the gate uses
the textbook 95% value; NSA was adopted on the development frames 0–8,999. A
sharp optimum at exactly those values is therefore what the seeding would
produce: the GT inherits the default run's choices.

## Protocol

- **Same detections everywhere:** every variant replays the cached detections
  (`experiments/c2_real_vnweights_full/detections_full13726.npz`).
- **One knob per variant:** each variant changes one knob; every other CTRV
  setting is set explicitly on every variant, so nothing leaks between them.
- **Shortened warm-up, checked:** each variant replays the 1,000 frames before
  the segment, not all 10,700. The default replayed this way reproduces the
  frozen full-replay dump exactly (MOTA 0.8933, IDF1 0.9068, 27 switches), and
  the script stops if it does not.
- **Scoring:** `gt_review.score_variants`, i.e. the paper's evaluator at
  IoU 0.5, against the reviewed dense GT. HOTA is reported at α = 0.5 only,
  because the α-averaged HOTA credits the seeding run.
- **Diagnostics:** `eval_hota_ci.exact_box_share` against the GT, and the same
  function against the default run's boxes (agreement).

## Reproduce

```powershell
python -m code.scripts.sweep_c2_sensitivity                     # dense, ~25 min
python -m code.scripts.sweep_c2_sensitivity --gt <mixed reviewed gt> --segment 2500 3100
python -m code.scripts.sweep_c2_sensitivity --score-only --roi-bundle <bundle>   # re-score kept dumps
```
