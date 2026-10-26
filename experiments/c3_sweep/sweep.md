# C3 tau_mappo sweep on human GT

Segment `dense` (frames 10700-11300) vs `data/annotations/NKKN_gt/dense/gt/gt.txt`.

**MAPPO branch only.** The accident branch — where C3's precision claim lives — cannot be tested on this clip, which contains no accidents.


| tau_mappo | tracks kept | MOTA | IDF1 | IDsw | miss | FP | moto IDF1 |
|---|---|---|---|---|---|---|---|
| 0.40 (shipped) | 100.0% | 0.905 | 0.929 | 103 | 990 | 1532 | 0.909 |
| 0.65 | 96.6% | 0.885 | 0.921 | 59 | 1760 | 1348 | 0.901 |

Best MOTA: tau 0.40 (0.905, keeping 100.0%).
