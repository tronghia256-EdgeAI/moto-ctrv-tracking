# C1 sweep — what it settles, and what it cannot

Raw grid: `sweep.md` / `sweep.json`. Reproduce:
`python -m code.scripts.sweep_c1_params`

## 1. The shipped value is bad, and the sweep says why

| C1 dense conf | buffer | MOTA | moto IDF1 | miss | FP |
|---|---|---|---|---|---|
| 0.35 (**shipped**) | 45 | 0.731 | 0.761 | 807 | 6254 |
| 0.50 | 45 | 0.884 | 0.855 | 1087 | 1876 |
| 0.50 | 30 | **0.905** | **0.909** | 990 | 1532 |
| 0.60 | 30 | 0.757 | 0.808 | 5299 | 1284 |
| — (**C1 off**, fixed 0.50) | 30 | **0.917** | **0.933** | 875 | 1390 |

The FP/miss trade is monotone and steep: 0.35 → 0.60 moves FP 6254 → 1284 while
miss goes 807 → 5299. The optimum sits at 0.50, and falls off hard either side.

## 2. C1 never beats having C1 off

Best C1 point is conf **0.50**, buffer **30** — which is *exactly* the tracker's
fixed default (`track_thresh` 0.5, `track_buffer` 30). So the best thing the
adaptive controller can do on this segment is imitate the fixed threshold it was
meant to improve on, and it still loses: 0.905 vs 0.917 MOTA, 0.909 vs 0.933
motorcycle IDF1.

The residual gap is explained: C1 spends 21 of 600 frames in DEFAULT (conf 0.45)
during warm-up and hysteresis transitions. Those frames are strictly worse than
staying at 0.50, and nothing in the segment ever compensates.

## 3. …but `dense` structurally cannot test C1's actual claim

C1's premise is that *switching* between operating points beats any fixed one.
Measured condition occupancy over the segment (`DENSITY_THRESHOLD` = 30,
hysteresis 15 frames):

| segment | density mean | min | max | frames below threshold | C1 condition |
|---|---|---|---|---|---|
| `dense` | 54.9 | 30 | 69 | **0 / 600** | DENSE_PEAK 579, DEFAULT 21 |
| `mixed` | 30.0 | 12 | 54 | 369 / 600 | DENSE_PEAK 264, **DEFAULT 336** |

`dense` never drops below the density threshold, so C1 holds one condition for
96.5% of the segment. What this sweep actually measured is therefore **"is 0.35 a
good fixed threshold for dense traffic"** (no) — not **"does context adaptation
help"** (untested). A segment that never changes condition cannot demonstrate
adaptation, by construction.

`mixed` switches 264/336 and straddles the threshold almost exactly (mean density
30.0 vs threshold 30). That is the segment where C1 either earns its place or
does not, and it is already exported and waiting in
`data/annotations/prelabels/mixed_*.zip`.

## 4. The tuning inverts C1's design premise

C1 was built on the intuition that dense traffic wants a **lower** confidence
threshold — more occlusion, so accept weaker detections to avoid losing objects.
DENSE_PEAK shipped at 0.35 against a DEFAULT of 0.45 for exactly that reason.

Measured, the opposite holds. The optimum for dense traffic is 0.50, *stricter*
than DEFAULT: in a 47-vehicle scene the extra weak detections generate far more
false positives than they recover real objects (FP +4722 against miss −183 going
from 0.50 to 0.35). After tuning, the "dense" row is the most conservative row in
the table, which is the reverse of what its name describes.

Worth stating plainly in the paper if C1 survives: the adaptation direction that
seemed obvious is contradicted by ground truth on this footage. Pinned as a test
(`tests/test_context.py::test_dense_peak_is_stricter_than_default`) so that if
`mixed` GT overturns it, the reversal is deliberate rather than silent.

## Conclusions

**Settled:** `PARAM_TABLE[DENSE_PEAK].yolo_conf = 0.35` is wrong and should be
0.50 regardless of what happens next; `buffer_len = 45` should be 30. Both are
strict improvements at every conf value tested.

**Not settled:** whether C1 helps at all. Needs GT on `mixed`.

**Do not** write C1 up as a negative result on this evidence — the experiment
that would justify that claim has not been run yet. What can be said today is
that C1's shipped tuning was harmful and its concept is untested.

## Recommended next experiment (one job, two problems)

Label `mixed`, **seeding pre-labels from the baseline tracker rather than from
`ctrv_gate_nsa`**:

```bash
python -m code.scripts.export_prelabels \
    --tracks experiments/gt_ablation/tracks_baseline.json --segments mixed
python -m code.scripts.gt_review_queue --segment mixed
```

That single segment simultaneously (a) tests C1 where it actually switches
conditions, and (b) gives a segment whose GT was *not* seeded from the proposed
method, removing the circularity caveat on the headline C2 numbers
(`../gt_ablation/evidence.md`). Cost is the same ~1 hour as `dense`.
