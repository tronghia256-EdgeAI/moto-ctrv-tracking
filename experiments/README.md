# Experiments

Committed results and evidence notes. Raw footage, annotations, detection caches and weights are not in the
repository, so the scripts that produced these files need local data to re-run.

## Results reported in the paper

| Folder | Contents |
|---|---|
| `paper_final/` | The final runs: the real-time detector at 448/640/960/1,280 px and the reference detector at 1,280 px, with and without the region-of-interest mask, plus OC-SORT with the CTRV motion model. Each run has `protocol.json`, `provenance.json` and per-size `metrics.json`. `figures.py` draws the paper's figures from these files. |
| `seed_bias/` | Blind-audit shares and the seed-bias brackets per run and input size |
| `operating_point/` | Accuracy by input size and object height, mask analysis, CPU-validity tooling |
| `c1_recalibration/` | The pre-registered C1 density thresholds and the switch attribution |
| `c3_accident/` | The pre-registered C3 accident test: protocol, amendments, results at 448 px |
| `ocsort_ctrv/` | The pre-registered test of C2's motion model inside OC-SORT |
| `realtime/`, `latency/` | CPU timing sweeps and the measurement-protocol notes |
| `external_trackers/` | OC-SORT and BoT-SORT on identical detections |
| `detector_leakage/` | Perceptual-hash check of the training images against the evaluation camera |
| `gt_qa/` | Ground-truth review queues and decisions (identifiers only, no images) |
| `c2_final/`, `ablation_synthetic_summary.md` | C2 ablations and the synthetic sanity check |

The other folders hold earlier exploratory runs (parameter sweeps, motion-model variants, preliminary ground-truth
ablations). They are kept for transparency; the paper does not report them.

## Redacted paths

Machine-specific absolute paths in these records were replaced by `<workspace>/` and `<scratch>` before
publication. `provenance.json` files keep the `protocol_sha256` of the original, unredacted `protocol.json`, so
that hash will not match the redacted copy.
