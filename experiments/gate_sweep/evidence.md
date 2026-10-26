# Gate parameter sweep — and a Goodhart lesson (2026-07-14)

Question: is the default gate (χ² = 5.991 @95%, base cost 0.5) leaving
gains on the table? Replay sweep on the 9000-frame NKKN detections cache.

## Sweep (motorcycle re-id events; baseline ByteTrack = 259)

| variant | χ² | base | moto re-id | moto uIDs | mean life | short |
|---|---|---|---|---|---|---|
| ctrv_gate (default) | 5.99 | 0.5 | **125** | 2628 | 71.8 | 353 |
| c9_b5  | 9.21  | 0.5 | 120 | 2525 | 76.7 | 325 |
| c9_b4  | 9.21  | 0.4 | 104 | 2473 | 82.1 | 321 |
| c14_b5 | 13.82 | 0.5 | 93  | 2443 | 82.0 | 324 |
| c14_b4 | 13.82 | 0.4 | 93  | 2335 | 91.0 | 307 |
| c21_b4 | 21.0  | 0.4 | 68  | 2105 | 104.7 | 272 |

Every proxy improves monotonically as the gate widens. Too good — and
the jump proxy (intra-track >40 px steps) stays flat (2→4 per ~215k),
because a WRONG re-association after an occlusion is smooth: track A
grabs motorcycle B and simply continues. It produces no jump and it
REMOVES a re-id event. The optimization target is gameable.

## The check that caught it: heading-flip rate

For every bridged gap ≥5 frames, compare mean heading over the 5 frames
before the gap vs after it. A >90° flip across a short gap is the
signature of a wrong merge (bikes weave, but rarely reverse mid-block):

| variant | bridges | heading flips >90° |
|---|---|---|
| default (χ²=5.99) | 1200 | 284 (**23.7%**) |
| c14_b5 | 2555 | 932 (**36.5%**) |
| c14_b4 | 3788 | 1518 (**40.1%**) |
| c21_b4 | 5287 | 2346 (**44.4%**) |

The flip RATE nearly doubles at the widest setting: the extra "rescues"
are substantially wrong merges. The 23.7% base rate itself includes
stop-and-go jitter, so absolute values are noisy — the relative trend
across variants is the evidence.

## Decision

**Keep χ² = 5.991, base cost 0.5.** Wider gates buy proxy points by
manufacturing smooth wrong merges. Final tuning waits for ground truth
(MOTA/IDF1), where a wrong merge costs an ID switch instead of looking
like a win.

Standing rule going forward — every tracker change must pass all three
checks before its flag is enabled by default:
1. re-id event count (target metric),
2. intra-track jump proxy (teleport check),
3. heading-flip rate on bridged gaps (wrong-merge check).

Paper material: this is a Discussion paragraph on why proxy metrics
cannot drive association tuning in dense mixed traffic, and why the
evaluation protocol needs annotated GT.

Reproduce: sweep script inline in session notes; variants replayed via
`compare_motion_models --replay experiments/c2_real_vnweights_full/detections.npz`
with `AdaptiveBYTETracker.GATE_CHI2 / GATE_BASE_COST` overridden.
