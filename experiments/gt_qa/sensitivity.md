# Annotation-bias sensitivity on `dense` GT

The ground truth can be wrong in two opposite directions, and they
do **not** matter equally. Both are reported here; quoting only the
first would understate the risk to the headline result.


## Under-merging: fragmentations the annotator did not join

Pre-labels were seeded from `ctrv_gate_nsa`, so a fragmentation it
produced and the annotator missed survives as agreement. Applying
the top-K flagged merge candidates:


| correction | GT IDs | baseline moto IDF1 | ctrv moto IDF1 | ctrv_nsa moto IDF1 | ctrv_gate moto IDF1 | ctrv_gate_nsa moto IDF1 | C2 − baseline |
|---|---|---|---|---|---|---|---|
| 0 | 335 | 0.844 | 0.850 | 0.869 | 0.868 | 0.933 | +0.089 |
| 29 | 306 | 0.831 | 0.837 | 0.856 | 0.855 | 0.916 | +0.085 |

The margin barely moves. This direction is not the problem.


## Over-merging: distinct vehicles chained into one identity

The damaging direction. An over-merged identity rewards a tracker
that bridges the gap between two vehicles and penalises one that
correctly refuses — which is exactly what C2's uncertainty gate
does. Splitting joins whose box moves more than the given number of
box-diagonals (raw, across the gap) or rescales by more than it:


| correction | GT IDs | baseline moto IDF1 | ctrv moto IDF1 | ctrv_nsa moto IDF1 | ctrv_gate moto IDF1 | ctrv_gate_nsa moto IDF1 | C2 − baseline |
|---|---|---|---|---|---|---|---|
| split @3.0 (59 cuts) | 394 | 0.855 | 0.863 | 0.884 | 0.864 | 0.916 | +0.061 |
| split @2.0 (105 cuts) | 440 | 0.861 | 0.870 | 0.889 | 0.854 | 0.898 | +0.037 |
| split @1.5 (147 cuts) | 482 | 0.866 | 0.877 | 0.896 | 0.843 | 0.880 | +0.014 |

The margin collapses, the baseline *improves* and the proposed
method *degrades* — the signature of a ground truth that rewards
over-merging. See `WRONG_MERGES.md` for the verified cases and what
to do about them.

