# Seed bias in the reviewed NKKN ground truth

**Question.** Each reviewed GT started from one tracker's output:

- `dense` and `freeflow` from `ctrv_gate_nsa` (C2); `freeflow` via
  `export_prelabels.py`, which slices that dump;
- `mixed` from `baseline`.

Does a GT favour the tracker that seeded it, and through what?

**Method.** `python -m code.scripts.check_seed_bias` scores every frozen dump in
`experiments/gt_ablation` on the three ROI exports listed in
`docs/GT_COMPLETION_20261012.md`. Those exports come *before* the audit
reconciliation (`docs/GT_RECONCILED_20261013.md`). On the reconciled GT, the
old-protocol margins are in `experiments/operating_point/operating_points.json`
(row `yolo11_vn_1280`): +0.045 dense, +0.030 freeflow, −0.037 mixed.
Motorcycles only, within each review's ROI.
Scored two ways:

- **as published**;
- **detection-anchored**: every GT and predicted box is replaced by the cached
  detection it overlaps one-to-one at IoU ≥ 0.5, so both sides share the
  detector's geometry.

Output is in `seed_bias.json`.

| GT (seeded from) | seed's exact-box share | IDF1 baseline | IDF1 C2 | C2 − baseline | IDSW baseline / C2 |
|---|---:|---:|---:|---:|---:|
| dense (C2) | 0.902 | 0.864 | 0.909 | +0.045 | 124 / 48 |
| freeflow (C2) | 0.983 | 0.929 | 0.965 | +0.036 | 11 / 7 |
| mixed (baseline) | 0.997 | 0.983 | 0.945 | −0.038 | 9 / 10 |

**Findings.**

1. **Each GT favours its seed by about 0.04 IDF1.** In every segment, C2's
   margin has the sign of the seed.
2. **Box geometry does not explain the IDF1 gap.** Anchoring both sides to the
   same detections moves IDF1 and HOTA at α = 0.5 by at most 0.002, for all 7
   variants on all three GTs.

   Alpha-averaged HOTA is the exception: it *is* sensitive to geometry. After
   anchoring, the C2 − baseline gap shrinks:

   | GT | published | anchored |
   |---|---:|---:|
   | dense | +0.124 | +0.055 |
   | freeflow | +0.156 | +0.049 |
   | mixed | −0.127 | −0.049 |

   On every GT the seed's geometry inflates the averaged score in the seed's
   favour. That is why the manuscript reports α = 0.5 only. Anchored scores make
   the averaged value usable.
3. **On `mixed` the gap comes from which boxes are in the GT, not from
   identity.** ID switches are 9 vs 10. C2 outputs 220 more boxes than
   `baseline`, which gives 281 FPs against 23 (repository metric, IoU 0.5). Of
   C2's anchored boxes that the GT lacks:
   - 73% are detections with score ≥ 0.5;
   - 61% lie where the GT has no box of any class;
   - 23% overlap a GT box at IoU ≥ 0.5 (duplicates or class confusion).
4. **`dense` is the mirror image.** This was measured on the earlier
   full-frame dense GT: 73% of `baseline`'s extra boxes lay where the C2-seeded
   GT had no box.

**The completed blind audit (`paper_gt_audit`, 148 verdicts by Nghĩa)**

The per-stratum shares are in `audit_shares.json`. A run is "GT wrong" under
the rule `gt_issue` in `paper_gt_audit.py`:

- a tracker's extra run (a box the GT lacks) shows a real motorcycle;
- or a missing run (a GT box the tracker lacks) shows no motorcycle, or a
  duplicate;
- or the two crops of an ID switch show two different vehicles.

Intervals are Wilson 95%. "Left" is the share applied to the runs the audit
did not sample. Those runs are still in the reconciled GT, because the
32-group reconciliation fixed sampled cases only.

| Stratum (tracker that did not seed the GT) | GT wrong | Wilson 95% | runs left (low / mid / high) |
|---|---:|---|---|
| dense, baseline extras (GT seeded from C2) | 8/10 | 0.49–0.94 | 106 / 174 / 205 of 217 |
| mixed, C2 extras (GT seeded from baseline) | 8/10 | 0.49–0.94 | 72 / 118 / 139 of 147 |
| dense, baseline misses | 2/10 | 0.06–0.51 | 28 / 99 / 253 of 497 |
| mixed, C2 misses | 1/10 | 0.02–0.40 | 0 / 2 / 6 of 16 |
| dense, baseline ID switches | 1/10 | 0.02–0.40 | 2 / 11 / 46 of 114 |
| mixed, C2 ID switches | 1/10 | 0.02–0.40 | 0 / 0 / 0 (all sampled) |

The seed trackers' own extras are mostly duplicates the reviewer removed
correctly (dense C2: 7 of 10 are `D`). In freeflow every stratum was sampled
completely, so nothing is left.

**What the audit settles.**

1. **The membership bias is real and symmetric.** About 80% of the boxes that
   the non-seed tracker outputs and the GT lacks are real motorcycles. The
   reviewer was never shown them. In the reconciled GT used by the paper run,
   an estimated 106–205 such runs remain in dense and 72–139 in mixed. Each GT
   therefore still charges the non-seed tracker false positives for real
   vehicles.
2. **ID switches are mostly genuine on both sides,** 9/10 in each direction.
   Counts of identity switches are therefore not a seeding artefact. C2's
   reduction in switches on dense (124 → 48 on the ROI GT) survives the
   seed-bias critique. Its IDF1 margin does not.

The audit sampled the old cached tracks (`experiments/gt_ablation`, from
yolo11_vn@1280 detections), not best.pt. The shares describe how incomplete
the GT is. They are not a property of the paper run's dumps. The script lives
outside `code/` during the paper run's code freeze and will be moved to
`code/scripts/` afterwards.

**Brackets on the paper run v5** (best.pt; `code/scripts/seed_bracket.py`;
`bracket_best85_roi_20261013_v5_{640,448}.json`).

The bracket gives a lower and an upper bound for each motorcycle IDF1 gap:

- **lower** — the reconciled GT as it is;
- **upper** — the GT plus the unmatched boxes of the pair's non-seed-like
  trackers, linked by `union_seed`;
- **corrected** — lower + s × (upper − lower), where s is the audit's real
  share. C1, C1+C2+C3 and the external trackers have no audit stratum of their
  own, so they borrow the segment's non-seed share; each record notes this.

Corrected gaps, 640 / 448:

| Comparison | dense | freeflow | mixed |
|---|---|---|---|
| C2 − baseline | +0.002 / +0.012 | +0.004 / +0.018 | +0.011 / +0.020 |
| C1 (c1_ctrv_gate − ctrv_gate_nsa) | +0.025 / +0.035 | +0.027 / +0.023 | +0.029 / +0.022 |
| C3 (c1c2c3 − its pre-C3 output) | 0 / 0 | 0 / 0 | 0 / 0 |
| OC-SORT at matched thresholds − C2 | +0.062 / +0.071 | +0.032 / +0.055 | +0.033 / +0.071 |

ID switches, baseline / C2 / C1+C2 / OC-SORT at matched thresholds:

| Segment | 640 | 448 |
|---|---|---|
| dense | 154 / 95 / 159 / 79 | 141 / 62 / 138 / 39 |
| mixed | 9 / 8 / 13 / 11 | 19 / 9 / 27 / 6 |

On the C2-seeded dense GT, the seed favours C2's identities. OC-SORT's lead
in ID switches is therefore, if anything, understated. The identity audit
never sampled OC-SORT's switches, so no correction is applied to them.

The pattern in finding 1 fits two explanations equally well:

- a seed advantage of about 0.04 with no IDF1 effect of C2 at all;
- a C2 effect that differs by segment.

Only ground truth whose disagreements are adjudicated on both sides can tell
them apart. Two sources can provide it: the blind audits, and the held-out
cameras' union pre-labels (`union_seed` in `code/scripts/gt_newcam.py`), which
put every tracker's boxes before the reviewer.

**Why the seed wins.** The seeding tracker's boxes were all shown to the
reviewer, so its wrong ones could be deleted. Boxes that only the other tracker
outputs were never shown. Their absence from the GT is therefore a default, not
a judgement.

**Consequence.** A GT pre-labelled from one tracker cannot rank that tracker
against another by IDF1 or MOTA unless the other tracker's disagreements are
adjudicated as well.

**Two blind audits.**

| | `paper_gt_audit` | `audit_disagreements` (this sample) |
|---|---|---|
| Added | 2026-10-12, 18:59 | earlier |
| Defined in | `docs/PAPER_STEP1_PROTOCOL.md`, `code/scripts/paper_gt_audit.py` | below |
| GTs | final ROI GTs of all three segments | full-frame dense and ROI mixed |
| Sample | at most 10 runs per stratum | 40 runs per stratum |
| Purpose | find GT errors to fix | estimate shares with intervals |

**This sample was retired on 2026-10-12, before any verdict was entered.**

- Its dense strata were drawn from the full-frame dense GT. That GT has since
  been superseded by the ROI export.
- `paper_gt_audit` samples all three final ROI GTs.
- Shares with Wilson intervals will be computed from `paper_gt_audit`'s
  verdicts instead.
- `audit_disagreements` stays pinned to the GTs it sampled, so the record
  below remains reproducible.

**Record of the retired sample.** `python -m code.scripts.audit_disagreements sample`
(seed 20261012) drew 146 runs, where a run is consecutive frames of one ID:

| Stratum | Runs available | Runs sampled |
|---|---:|---:|
| C2 extras on `mixed` | 157 | 40 |
| C2 misses on `mixed` | 26 | 26 (all) |
| `baseline` extras on `dense` | 233 | 40 |
| `baseline` misses on `dense` | 453 | 40 |

The judge sees numbered crops only, in `data/annotations/seed_bias_audit/`
(gitignored). `score` reports, per stratum, the share of runs where the GT is
wrong, with Wilson 95% CIs.

**Identity strata.** Membership does not settle whether the dense ID-switch gap
(137 vs 27) is real. Wherever the reviewer missed a swap in the C2-seeded
identities, a baseline that tracked correctly is still charged a switch.
`switches` (seed 20261013) replicates the CLEAR matching of
`compute_tracking_metrics` and aborts if its count differs. It draws:

| GT | Tracker | Switches | Sampled |
|---|---|---:|---:|
| dense | baseline | 137 | 30 |
| dense | ctrv_gate_nsa | 27 | 27 (all) |
| mixed | baseline | 9 | 9 (all) |
| mixed | ctrv_gate_nsa | 10 | 10 (all) |

Each switch is shown blind as a crop pair of the GT identity: one crop at its
last match before the switch, one at the switch.

- Same vehicle in both crops: the switch is real.
- Two different vehicles: the switch is an artefact of the GT identities.

A seed's own switches are reviewer corrections, so they also act as a control
on the judge.

**Stopping rule.** Fixed on 2026-10-12, before any verdict was entered. At
commit 1ddaa05 both verdict files were empty: 0 of 146 in `verdicts.csv` and 0
of 76 in `switch_verdicts.csv`.

- **Order:** the judge works in sheet order. That order is a seeded shuffle,
  so every prefix is a random subsample of each stratum.
- **Early stop:** a stratum stops once 20 items are judged (excluding `?`)
  and its Wilson 99% interval excludes 0.5. Otherwise it runs to its end.
- **Why 99%:** `score` is checked after every sheet, and repeated looks at 95%
  would inflate the error rate. The intervals reported are 95%.
- **Blind interim output:** until every stratum has stopped, `score` prints
  only "continue" or "you may stop" for each file, so the judge sees no
  interim shares.
- **Gaps:** a blank verdict before a filled one is reported, because a
  skipped item breaks the random prefix.
