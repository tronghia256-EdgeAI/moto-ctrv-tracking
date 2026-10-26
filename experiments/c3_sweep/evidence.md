# C3 on human GT — the MAPPO branch cannot be made to help

Raw grid: `sweep.md` / `sweep.json`. Reproduce:
`python -m code.scripts.sweep_c3_threshold --diagnose`

**Scope.** Everything here is the **MAPPO (recall) branch**. C3's precision
claim lives on the accident branch, and this clip contains no accidents, so
that half of the contribution is neither supported nor refuted below.

## 1. No threshold beats not filtering

| τ_mappo | kept | MOTA | IDF1 | IDsw | miss | FP |
|---|---|---|---|---|---|---|
| **0.40 (shipped)** | **100.0%** | **0.905** | **0.929** | 103 | 990 | 1532 |
| 0.50 | 98.7% | 0.895 | 0.924 | 93 | 1301 | 1492 |
| 0.60 | 97.9% | 0.892 | 0.923 | 72 | 1484 | 1428 |
| 0.65 | 96.6% | 0.885 | 0.921 | 59 | 1760 | 1348 |
| 0.70 | 93.6% | 0.861 | 0.910 | 53 | 2506 | 1261 |
| 0.75 | 88.0% | 0.814 | 0.886 | 35 | 3950 | 1128 |
| 0.80 | 75.9% | 0.705 | 0.820 | 24 | 7153 | 940 |

MOTA falls monotonically. 0.40 → 0.70 buys 271 fewer false positives and costs
1516 extra misses — roughly six good detections destroyed per bad one removed.

Do not read the falling IDsw column as a gain. Switches drop from 103 to 24
because tracks are being deleted wholesale, not because identity improved; IDF1
falls at the same time, which is what gives it away.

## 2. Why — the score is only weakly discriminative

Per (frame, track) on the segment, labelled TP/FP by IoU ≥ 0.5 against GT:

| | value |
|---|---|
| tracks scored | 28 055 |
| false-positive base rate | **5.5%** (1531 of 28 055) |
| reliability score, TP | mean 0.850, median 0.870 |
| reliability score, FP | mean 0.805, median 0.836 |
| **ROC AUC** | **0.609** (0.5 = no signal) |

The two distributions overlap almost entirely. There *is* signal — filtering is
enriched for false positives at every cut-off — but nowhere near enough:

| τ | tracks dropped | share of them actually FP |
|---|---|---|
| 0.50 | 351 | 11.4% |
| 0.60 | 598 | 17.4% |
| **0.65** | 954 | **19.3%** (best, 3.5× base rate) |
| 0.70 | 1787 | 15.2% |
| 0.80 | 6755 | 8.7% |

Even at its best operating point the filter discards **four true tracks for
every false one**. With false positives only 5.5% of the population, there is no
threshold at which that arithmetic comes out ahead on MOTA. The shape of the
sweep table follows directly from AUC 0.609 plus a 5.5% base rate; it is not a
tuning failure.

## 3. C2 made C3's job harder

The 5.5% base rate is what C2 leaves behind. The pipeline C3 was designed
against was noisier — C3 assumes a meaningful population of junk tracks to
remove, and C2's gating+NSA already removed most of it (FP 1629 → 1388, short
tracks 1061 → 529 on the full video). A filter downstream of a strong tracker
has little left to filter.

Worth stating in the paper: the contributions are not independent, and the
ordering matters. This is the same lesson as Finding 1 in `../gt_ablation/`,
where gating and NSA turned out to be one mechanism rather than two.

## Conclusions

**τ_mappo = 0.40 should stay as it is.** It is inert on this data, but inert is
the correct behaviour for a recall branch: keeping everything maximises the
observation set MAPPO consumes. Raising it is measurably worse. Leave the
constant and change the *description* — it is not "tuned for recall", it is
"below the score floor, therefore pass-through".

**C3 currently has no measurable evidence.** Its MAPPO half cannot help for
structural reasons; its accident half is untestable without accident footage.
Do not claim C3 as a validated contribution on what exists today.

**If C3 is to be salvaged, the target is the score, not the threshold.**
AUC 0.609 is the number to move. The components (age, consistency, smoothness,
class stability, det-conf) are equally weighted by assumption — measuring each
component's individual AUC would show which ones carry signal and which are
dead weight. That is a cheap next experiment and it needs no new data.

**The accident branch may still be the real story.** Dropping four true tracks
to remove one false one is a bad trade for MAPPO observations and a possibly
excellent one for a Telegram alert that must not cry wolf. That asymmetry is
precisely C3's dual-task argument — it just cannot be demonstrated until the
accident dataset arrives.
