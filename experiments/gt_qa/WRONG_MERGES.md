# The `dense` GT is over-merged, and it inflates the C2 result

**This supersedes the headline numbers in `../gt_ablation/evidence.md`.**
Read this before quoting motorcycle IDF1 $0.933$ or the $+0.089$ margin
anywhere.

> **Identity numbers below are internal to `gt.txt`.** CVAT renumbers every
> track `1..N` in export order when writing MOT 1.1 — the pre-labels went in
> with tracker ids (`5849`, `6413`, …) and came back `1..335` with no gaps.
> These numbers do not exist in the CVAT UI. To find a case in CVAT, use the
> frame number and pixel position in
> `split_candidates/candidates.csv`, or the rendered sheets in
> `split_candidates/`, which ring the box in red.

## What happened

While building the qualitative figure I picked the most extreme example of
"baseline splits an identity, C2 keeps it whole" — GT id 96, where the
baseline spends 11 track IDs and C2 spends 1. Rendering it showed what
looked like several different motorcycles. Checking the trajectory
confirmed it:

| frame | box diagonal | centre y |
|---|---|---|
| 277 | 104 | 386 |
| 301 | 47 | 307 |
| 315 | 74 | 371 |
| 386 | 50 | 302 |
| 446 | 39 | 291 |
| 468 | 62 | 335 |

A vehicle approaching a fixed camera grows monotonically. This one shrinks
and grows repeatedly while its centre bounces up and down the roadway.
**GT id 96 is several motorcycles chained into one identity.**

A second case, GT id 107, was confirmed visually at full frame: at f210 the
box is on a motorcycle at the bottom edge; at f216 the same identity is a
different motorcycle further *up* the road. Traffic flows toward the
camera, so a vehicle at the bottom edge leaves — it does not jump back.

## Scale

Flagging identities whose box moves more than 2 box-diagonals of **raw**
displacement, or changes area more than 2$\times$, across a gap of $>5$
frames:

| | count |
|---|---|
| GT identities | 335 |
| flagged as possibly over-merged | **74 (22.1%)** |
| boxes inside flagged identities | 5605 / 27513 (20.4%) |

## Why it inflates C2 specifically

A ground-truth identity that wrongly spans two vehicles **rewards any
tracker willing to bridge the gap between them, and penalises one that
correctly refuses.** C2's uncertainty gate exists precisely to bridge
occlusion gaps. So the error is not neutral noise — it is aligned with the
mechanism under test.

Splitting the flagged joins and re-scoring (motorcycle IDF1):

| GT variant | baseline | ctrv | ctrv+gate | **C2** | C2 − baseline |
|---|---|---|---|---|---|
| as annotated | 0.844 | 0.850 | 0.868 | **0.933** | **+0.089** |
| split, conservative (59 cuts) | 0.855 | 0.863 | 0.864 | 0.916 | +0.061 |
| split, moderate (105 cuts) | 0.861 | 0.870 | 0.854 | 0.898 | +0.037 |
| split, aggressive (147 cuts) | 0.866 | 0.877 | 0.843 | 0.880 | **+0.014** |

Three things to note:

1. **The margin collapses by up to 84%** ($+0.089 \rightarrow +0.014$).
2. **Baseline improves while C2 degrades** as the GT is split — the exact
   signature of a GT that rewards over-merging.
3. **It is the gate that is being rewarded**: plain `ctrv` *improves*
   ($0.850 \rightarrow 0.877$) while `ctrv+gate` degrades
   ($0.868 \rightarrow 0.843$). This is the same failure the earlier gate
   sweep warned about — "wider gates game the re-id proxy via wrong
   merges" — reappearing in the ground truth itself.

## What is and is not established

**Established.** The GT contains a substantial number of over-merged
identities; two are verified by eye. The headline margin is highly
sensitive to them, in the direction that flatters the proposed method.
Motorcycle IDF1 $0.933$ and the $+0.089$ margin **should not be published
as they stand**.

**Not established.** That C2 provides no benefit. The splitting criterion
is a heuristic, and the aggressive setting will certainly cut some genuine
tracks — a motorcycle emerging from behind a bus does legitimately
reappear displaced and rescaled. The true margin lies somewhere between
$+0.014$ and $+0.089$ and cannot be pinned down without re-annotation.
Note also that C2 still leads on every row.

## Why the earlier sensitivity analysis missed this

`gt_bias_sensitivity` originally tested only the *under*-merging direction
(fragmentations the annotator failed to join), and found the margin barely
moved. That was correct and irrelevant: the damaging error is the opposite
one. `gt_review_queue` was blind to it for a mechanical reason — its JUMP
check divides displacement by the gap length, so a box that teleports
across a 29-frame occlusion looks unremarkable per frame.

Both tools are now fixed: `gt_review_queue` has a `SPLIT` check using raw
cross-gap displacement (74 hits on the current GT, ranked first after
MERGE), and `gt_bias_sensitivity` runs the analysis in both directions.

## What to do

1. **Re-review the GT** against the 74 `SPLIT` candidates in
   `review_queue.md`. Roughly 45 minutes, and it is the highest-value
   annotation work available — it decides whether the paper's main result
   holds.
2. **Do not report the tracking numbers until that pass is done.**
3. When labelling `mixed`, note that its pre-labels are seeded from
   *baseline*, which fragments more and merges less — so it is far less
   exposed to this failure, and is the cleaner segment for the headline
   claim.

Reproduce:
```bash
python -m code.scripts.gt_review_queue \
    --prelabels data/annotations/NKKN_gt/dense/gt/gt.txt \
    --out-dir experiments/gt_qa --frame-size 800 450 --offset 10700
python -m code.scripts.gt_bias_sensitivity --k 0 29 --split 3.0 2.0 1.5
```
