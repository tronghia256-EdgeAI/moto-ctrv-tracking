# C2 ablation on human GT — first real MOTA/IDF1 for the paper

> ⚠️ **The identity numbers below are not publishable as they stand.**
> The GT contains ~74 over-merged identities (22% of tracks), two verified
> by eye, and they inflate the proposed method specifically: correcting
> them moves the C2-over-baseline margin from **+0.089 down to as little
> as +0.014** motorcycle IDF1. See `../gt_qa/WRONG_MERGES.md`. C2 still
> leads on every variant of the ground truth, but the magnitude is not
> settled. Detection metrics, latency, and the C1/C3 findings are
> unaffected.

**Segment:** `dense` = absolute frames 10700–11300 (600 frames, ~47 tracks/frame),
the densest window of the NKKN clip and inside the frames-9000+ held-out region.
**GT:** `data/annotations/NKKN_gt/dense/gt/gt.txt` — 27 513 boxes, 335 track IDs,
human-corrected in CVAT (gitignored, IP).
**Detections:** identical for every row (`detections_full13726.npz` replay), so all
differences are attributable to the tracker alone.

Supersedes every no-GT proxy number (re-id events, unique-id counts). Those were
development signals; the gate sweep already showed they can be gamed.

## Acceptance gates (both pass)

| gate | result |
|---|---|
| GT scored against itself | MOTA = IDF1 = 1.000, 0 IDsw / miss / FP |
| GT↔frame alignment (mean best IoU by shift) | −2: 0.559 · −1: 0.705 · **0: 0.960** · +1: 0.715 · +2: 0.568 → no off-by-one |
| GT edit vs pre-labels ↔ pre-label source metrics | 873 boxes added / 1388 deleted **exactly equals** `ctrv_gate_nsa` miss 873 / FP 1388 |

That third row is the strong one: it closes the loop end-to-end
(export → CVAT → export → import → metrics) with no free parameters.

## Results (dense segment, all classes)

| variant | MOTA | IDF1 | IDsw | frag | miss | FP |
|---|---|---|---|---|---|---|
| baseline (linear KF) | 0.878 | 0.878 | 194 | 202 | 1546 | 1629 |
| ctrv | 0.889 | 0.883 | 162 | 175 | 1375 | 1521 |
| ctrv + nsa | 0.898 | 0.898 | 125 | 132 | 1252 | 1432 |
| ctrv + gate | 0.886 | 0.896 | 188 | 157 | 1219 | 1721 |
| **ctrv + gate + nsa (C2)** | **0.917** | **0.947** | **31** | **65** | **873** | **1388** |
| + C1 (Exp2) | 0.905 | 0.929 | 103 | 98 | 990 | 1532 |
| + C1 + C3 (Exp3) | 0.905 | 0.929 | 103 | 98 | 990 | 1532 |

Exp2/Exp3 use the GT-tuned `DENSE_PEAK` row (conf 0.50, buffer 30). With the
originally shipped guess (0.35 / 45) they scored 0.731 / 0.795 with 6255 FPs —
see Finding 3.

Motorcycle only — the class C2 is about:

| variant | MOTA | IDF1 | IDsw |
|---|---|---|---|
| baseline | 0.843 | 0.844 | 189 |
| ctrv | 0.857 | 0.850 | 158 |
| ctrv + nsa | 0.869 | 0.869 | 123 |
| ctrv + gate | 0.854 | 0.868 | 182 |
| **ctrv + gate + nsa (C2)** | **0.893** | **0.933** | **29** |
| + C1 (Exp2, tuned) | 0.878 | 0.909 | 99 |

**C2 vs baseline on motorcycles: IDF1 0.844 → 0.933, ID switches 189 → 29 (−85%).**

## Detection metrics — tracking beats every per-frame operating point

```bash
python -m code.scripts.import_gt --gt data/annotations/NKKN_gt/dense/gt/gt.txt \
    --detection --det-npz experiments/c2_real_vnweights_full/detections_full13726.npz \
    --pred-tracks experiments/gt_ablation/tracks_ctrv_gate_nsa.json \
    --segment 10700 11300
```

| source | P | R | F1 | mAP@0.5 |
|---|---|---|---|---|
| detector, all detections | 0.497 | 0.984 | 0.661 | **0.925** |
| detector @conf ≥ 0.30 | 0.746 | 0.943 | 0.833 | — |
| detector @conf ≥ 0.50 | 0.890 | 0.847 | 0.868 | — |
| detector @conf ≥ 0.70 | 0.946 | 0.615 | 0.745 | — |
| baseline tracker output | 0.941 | 0.944 | 0.942 | — |
| **C2 tracker output** | **0.950** | **0.968** | **0.959** | — |

mAP@0.5 = 0.925 is the detector ceiling — the bound every tracker row works
under, and the context for the miss counts in the ablation table.

The interesting line is the last one. The tracker runs at track_thresh 0.50,
where the detector alone scores P 0.890 / R 0.847. After tracking: P 0.950 /
R 0.968. **Both rise at once**, which no confidence threshold can do — moving
the cut-off only slides along the PR curve, trading one for the other (0.30 →
0.70 buys +0.20 precision for −0.33 recall). Temporal association recovers
objects the detector drops in individual frames while discarding detections that
fail to persist, so it leaves the per-frame curve entirely rather than moving
along it.

C2 over baseline at detection level is smaller than at identity level (F1 0.942
→ 0.959): the motion model's value is mostly in *who* a box belongs to, not
whether it exists. That is the expected shape for a motion-model contribution and
worth saying in the paper before a reviewer asks why detection gains look modest.

## Finding 1 — gating and NSA are super-additive; neither works alone

Motorcycle ID switches over the 2×2:

| | NSA off | NSA on |
|---|---|---|
| **gate off** | 158 | 123 (−22%) |
| **gate on** | 182 (**+15%, worse**) | 29 (−82%) |

Uncertainty gating *on its own makes identity worse*. The mechanism is the one the
earlier gate sweep was circling ("wider gates game the re-id proxy via wrong
merges"): the gate rescues a lost track when a detection falls inside the EKF
innovation ellipse. Without NSA the measurement noise R stays large, the EKF
over-smooths, the predicted box lags a weaving motorcycle — so the ellipse is
both wide and mis-centred, and rescues land on the *wrong* detection. NSA shrinks
R on confident detections (R ← R·clip(1−score, 0.05, 1)), the prediction tracks
the bike, and the same gate then rescues correctly.

So C2 must be reported as one mechanism, not two independent tricks. An ablation
row for "gating alone" is a row where the method is worse than not having it.

## Finding 2 — the NSA arm was never actually ablated (fixed)

`configs/exp1_C2.yaml` gained `ctrv.nsa_noise: true` in commit 7cc9928 ("enable by
default"). `VARIANTS` in `compare_motion_models.py` only overrode keys it listed,
so `nsa` fell through to the config for every CTRV arm: `ctrv`, `ctrv_gate` and
`ctrv_gate_nsa` **all ran with NSA on**. `ctrv_gate` and `ctrv_gate_nsa` were
therefore the same configuration — their dumps are identical on all 13 726
frames, which is what surfaced the bug.

Fixed by making every toggled feature explicit on every variant. The pre-GT
numbers in `experiments/c2_final/` are affected: what they label "ctrv" and
"ctrv_gate" are both really CTRV+NSA. The −60% re-id-event headline from commit
9640201 compared baseline against a mislabelled arm.

## Finding 3 — C1's shipped tuning is harmful; C1 itself is still untested

MOTA 0.917 → 0.731, false positives 1388 → 6255 (4.5×). C1's `DENSE_PEAK` row
drops the confidence threshold to 0.35 and this segment sits in DENSE_PEAK for
579 of 600 frames, so the threshold is pinned at 0.35 throughout. Recall does
improve as intended (miss 873 → 808), nowhere near enough to pay for the FPs.

No proxy could have caught this: false positives are only measurable against GT.

A full sweep of the DENSE_PEAK row (`../c1_sweep/evidence.md`) settles the
tuning and reframes the finding:

- 0.35 is simply the wrong value; 0.50 / buffer 30 is the optimum here.
- That optimum *is* the tracker's fixed default — the best the controller can do
  on this segment is imitate the threshold it was meant to improve on, and it
  still loses (0.905 vs 0.917 MOTA).
- **But `dense` cannot test C1's premise at all.** Density never drops below the
  switching threshold (0/600 frames), so C1 holds one condition throughout. What
  was measured is "is 0.35 a good *fixed* threshold" (no), not "does adaptation
  help" (unmeasured).

So: fix the constants, and do not write C1 up as a negative result yet — the
experiment justifying that claim has not been run. `mixed` (which switches
264/336 frames) is the segment that would settle it.

## Finding 4 — C3's MAPPO branch is inert on this data

`tracks_c1c2c3.json` and `tracks_c1_ctrv_gate.json` are identical on all 13 726
frames. Cause: τ_mappo = 0.40, but the *lowest* reliability score observed on the
segment is 0.435, so the recall branch filters nothing.

| branch | threshold | kept |
|---|---|---|
| MAPPO (recall) | 0.40 | 32 960 / 32 960 = **100.0%** |
| accident (precision) | 0.70 | 29 158 / 32 960 = 88.5% |

Score distribution: min 0.435 · p1 0.453 · p5 0.621 · median 0.854 · max 0.982.

C3 is working — it is just doing all its work on the accident branch, which the
replay harness never dumps.

A full τ_mappo sweep (`../c3_sweep/evidence.md`) then showed that raising the
threshold is not the fix, and that the MAPPO branch cannot be made to help at
all: MOTA falls monotonically with τ, because the reliability score separates
true from false tracks only weakly (**ROC AUC 0.609**) while false positives are
just **5.5%** of the population. Even at its best cut-off the filter discards
four true tracks per false one. Leave τ_mappo at 0.40 — inert is the correct
behaviour for a recall branch — and treat AUC 0.609, not the threshold, as the
number to improve if C3 is to be salvaged.

## Caveat that must go in the paper — now measured, not just disclosed

**The pre-labels were `ctrv_gate_nsa`'s own output**, so the GT was seeded from
the proposed method. Errors the annotator did not catch stay in the GT as
agreement, biasing that row upward relative to every other. Scale of the human
correction, which bounds how much can remain: 873 boxes added, 1388 deleted,
1019 moved, 15 merges, 9 splits — ~8% of boxes touched.

Rather than leave it at a disclosure, the bias is bracketed by applying the
still-flagged merge candidates to the GT and re-scoring everything
(`python -m code.scripts.gt_bias_sensitivity`):

| merges applied | GT IDs | baseline moto IDF1 | C2 moto IDF1 | margin |
|---|---|---|---|---|
| 0 (as labelled) | 335 | 0.844 | 0.933 | **+0.089** |
| 3 | 332 | 0.844 | 0.931 | +0.087 |
| 6 | 329 | 0.843 | 0.930 | +0.087 |
| 10 | 325 | 0.842 | 0.928 | +0.086 |
| all 29 | 306 | 0.831 | 0.916 | **+0.085** |

Correcting *every* remaining candidate — a deliberate worst case, since visual
sampling of the top six found only those above confidence ≈0.43 to be genuine —
moves C2's margin over baseline from +0.089 to +0.085. Both rows fall together,
because a merged GT identity penalises whichever variant split it, and baseline
splits in more places.

**The bias is real, one-directional, and bounded at roughly 0.004 IDF1.** That is
a far better answer to a reviewer than a caveat sentence.

Still worth doing: re-seed a second segment (`mixed`, already exported) from the
**baseline** tracker, which removes the caveat entirely rather than bounding it —
and is the only segment that can test C1.

## Reproduce

```bash
python -m code.scripts.compare_motion_models \
    --replay experiments/c2_real_vnweights_full/detections_full13726.npz \
    --variants baseline,ctrv,ctrv_nsa,ctrv_gate,ctrv_gate_nsa,c1_ctrv_gate,c1c2c3 \
    --out-dir experiments/gt_ablation

python -m code.scripts.import_gt \
    --gt data/annotations/NKKN_gt/dense/gt/gt.txt \
    --pred-tracks experiments/gt_ablation/tracks_ctrv_gate_nsa.json \
    --segment 10700 11300
```
