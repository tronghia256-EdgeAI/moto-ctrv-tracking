# End-to-end real-time: the detector is 91% of the frame, and the mask head is half of that

**Supersedes the framing in `../latency/evidence.md` for anything called
"real-time".** That file measured tracker-side cost with the detector excluded,
which is the right number for "what does C2 cost" and the wrong number for
"does this run in real time". These are the first end-to-end numbers the
project has.

Measured 27/8/2026 on `NKKN-VoThiSau 2017-07-18_08_00_00_000.asf`
(800x450, 25 fps), Intel Core i5-12500H, CPU only (no CUDA device present),
torch 2.11.0+cpu, ultralytics 8.4.39. One variant per process, warm-up
discarded — see `code/scripts/benchmark_realtime.py` for why both are
enforced rather than left to the operator.

**The budget is 40 ms/frame, not 33.3.** The source is 25 fps. Grading against
a round 30 fps would be grading against the wrong target.

## 1. Where the frame actually goes

Shipped configuration (`weights/yolo11_vn.pt` @ 640, baseline tracker):

| stage | mean (ms) | p50 | p95 | share |
|---|---|---|---|---|
| decode | 1.44 | 1.43 | 1.79 | 1.0% |
| context (C1) | 1.02 | 1.01 | 1.21 | 0.7% |
| **detect** | **130.72** | 130.63 | 141.42 | **91.3%** |
| track (C2) | 10.05 | 10.22 | 11.65 | 7.0% |
| c3 | 0.00 | 0.00 | 0.00 | 0.0% |
| **TOTAL** | **143.24** | 142.51 | 153.91 | |

**7.0 fps.** The 40 ms budget is missed by 3.6x.

This contradicts a sentence currently in the paper
(`paper/sections/05_results.tex`, Runtime): *"Against a YOLO11 inference cost
of tens of milliseconds, C2 is not the term that decides real-time
feasibility."* The inference cost is 130 ms, not tens of ms, and nothing else
on the frame matters next to it. The conclusion about C2 survives — C2 really
is not the deciding term — but the premise it rests on was wrong, and the
sentence has to be rewritten rather than kept because its verdict happened to
hold.

## 1b. What the components cost as a share of a real frame

All four ablation configs, end to end, 100 frames each, one process each:

| variant | detect | track | total | p95 | fps |
|---|---|---|---|---|---|
| Exp 0 baseline | 109.82 | 5.83 | 117.08 | 131.47 | 8.5 |
| Exp 1 (+C2) | 109.50 | 9.53 | 120.39 | 134.97 | 8.3 |
| Exp 2 (+C1, C2) | 110.87 | 7.80 | 120.09 | 140.79 | 8.3 |
| Exp 3 (+C1, C2, C3) | 110.42 | 7.40 | 120.31 | 133.28 | 8.3 |

**This is the number the paper should quote.** Tracker-side, C2 costs +22%
(`../latency/evidence.md`). End to end it costs **+2.8%** — 3.3 ms on a
117 ms frame. Both are true; they answer different questions, and only the
second one is about real-time feasibility. The tracker-side figure stays as
the component analysis and stops being the headline.

C1 and C3 are free at this resolution: Exp 2 and Exp 3 land inside the noise
of Exp 1, and the context stage measures ~1 ms.

Two honesty notes on this table:

- Detector cost reads 109.8 ms here against 130.7 ms in §1. Same video, same
  machine, different frame window (20–120 vs 10–70) and a different run. A
  ~16% cross-run spread on the dominant stage means **absolute end-to-end
  numbers are not comparable across sessions**, only within a sweep. Every
  row above was measured in one sweep, so the comparison between rows holds.
  Anything quoted against a different session's numbers does not.
- `track` at 5.83 ms for baseline against 9.53 ms for Exp 1 reproduces the
  tracker-side ordering, so the sweep is measuring what it claims to.

## 2. The mask head costs 2.25x for an output nothing reads

`weights/yolo11_vn.pt` is a **segment** checkpoint (`task='segment'`,
2.84M params, 9.6 GFLOPs). `Detector.detect()` reads `.boxes` and never
touches a mask.

| model | head | params | imgsz | detect (ms) |
|---|---|---|---|---|
| yolo11_vn | segment | 2.84M | 640 | **130.72** |
| yolo11n | detect | 2.62M | 640 | **58.20** |

Same scale, same image, same machine, same process discipline: **2.25x**. The
extra is the 160x160x32 prototype tensor and the coefficient branch, computed
every frame and discarded every frame.

Caveat, stated because it limits the claim: these are different checkpoints
(yolo11n is COCO-trained, yolo11_vn is the 5-class VN model), so the
comparison isolates the head only under the assumption that class count and
training set do not move inference cost much — reasonable at 5 vs 80 classes,
but not free. The clean version needs a detect-head model trained on the same
data, which is why `docs/ACCIDENT_DATA_SPEC.md` §4b asks the co-author for the
detection training set.

## 3. Export backends are all slower than PyTorch eager here

The expected move for CPU real-time — export to ONNX Runtime or OpenVINO — is
a **regression** on this machine, in every combination tried:

| backend | head | imgsz | detect (ms) | vs PyTorch |
|---|---|---|---|---|
| PyTorch | segment | 640 | 130.72 | — |
| OpenVINO fp32 | segment | 640 | 163.92 | +25% |
| ONNX Runtime | segment | 640 | 228.67 | +75% |
| PyTorch | detect | 640 | 58.20 | — |
| OpenVINO fp32 | detect | 640 | 73.03 | +25% |
| OpenVINO fp16 | detect | 640 | 79.34 | +36% |

Consistent +25% for OpenVINO across both heads, so this is the backend, not a
fluke of one model. fp16 is slower than fp32, which is what one expects on a
CPU with no native half-precision path: the conversion is real work and the
compute is not cheaper.

The likely mechanism is that ultralytics runs only the backbone through the
exported graph and keeps postprocessing (NMS, mask assembly) in torch/numpy
outside it, so the export gives up torch's fused CPU kernels without replacing
the part that dominates. **Not established** — no profile inside the backend
was taken, and the exports use ultralytics' default settings with no
thread/stream tuning. Reported as measured, with the settings named, rather
than as a general claim that OpenVINO is slow.

INT8 was not reached: it needs a calibration set, which is the same missing
detection dataset as above.

## 4. Real-time is reachable, and the operating point is the whole question

PyTorch detect head, sweeping input size:

| imgsz | detect (ms) | total (ms) | p95 | fps | 40 ms budget |
|---|---|---|---|---|---|
| 640 | 58.20 | 62.89 | 69.61 | 15.9 | MISSED |
| 448 | 24.33 | 26.57 | 32.90 | **37.6** | **MET** |
| 320 | 17.33 | 19.47 | 24.65 | 51.4 | MET |
| 256 | 14.24 | 16.29 | 20.53 | 61.4 | MET |

So the pipeline does run in real time on a laptop CPU with no GPU — at 448 and
below, with a detect head.

**What this does not yet establish.** Accuracy at each operating point is
unmeasured, so the table above is one axis of a Pareto curve with the other
axis missing. Two reasons it cannot be filled yet:

1. The accuracy axis has to come from the VN model, and the VN model is the
   segment one. yolo11n is COCO — note its track stage falls to 0.81 ms at
   448 against 10.05 ms for the VN model at 640, because it simply finds far
   fewer vehicles in this domain. Some of the speed above is a detector doing
   less work, not a detector doing the same work faster.
2. Scoring against ground truth is blocked on the `dense` GT re-review
   (`../gt_qa/WRONG_MERGES.md`), which has to land before any accuracy number
   is quoted.

Until both are resolved, the honest statement is: **the input-size knob buys
real-time, at an accuracy cost that has not been measured.**

## 5. One bug found and fixed on the way

Exported weights (`.onnx`, openvino dirs) do not always carry their task, and
ultralytics guesses. Guessing `detect` for a segment model is silent and
total: the 32 mask coefficients are parsed as class scores, producing
confidences above 1.0 (measured max 9.16) and 300 garbage boxes per frame,
all of which the project's name-based class remap then discards. The result is
**zero detections with no error anywhere** — the first ONNX benchmark run
measured 227 ms of a detector finding nothing, and looked plausible.

Caught because the `track` stage read 0.26 ms instead of ~10 ms. `Detector`
now takes an explicit `task=` and warns when loading non-`.pt` weights
without one (`code/detection/detector.py`).

Verification after the fix, same frame, three backends:

| weights | detections |
|---|---|
| yolo11_vn.pt | 103 |
| yolo11_vn.onnx | 111 |
| yolo11_vn_openvino_model | 111 |

The 103 vs 111 gap is export numerics near the 0.1 confidence floor, not a
correctness problem.

## Reproduce

```powershell
# single variant, full breakdown
python -m code.scripts.benchmark_realtime --config configs/baseline.yaml --video "data/raw/videos/NKKN-VoThiSau 2017-07-18_08_00_00_000.asf" --frames 100 --warmup 20 --save

# operating point
python -m code.scripts.benchmark_realtime --config configs/baseline.yaml --video <path> --weights weights/yolo11n.pt --imgsz 448 --task detect --label pt-det-n@448 --save

# every ablation config, one subprocess each
python -m code.scripts.benchmark_realtime --sweep --video <path> --frames 100 --warmup 20 --save
```

Raw JSON per run in `experiments/realtime/*.json`, including the full
environment block.
