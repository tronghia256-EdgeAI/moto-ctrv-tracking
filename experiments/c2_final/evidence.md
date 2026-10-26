# C2 final numbers — full video, all upgrades (2026-07-14)

Replay on the COMPLETE NKKN clip. Discovery: the ASF metadata claims
22 876 frames but the stream decodes exactly **13 726 (9.2 min @ 25fps)**
— surveillance-DVR exports routinely lie in the header; frame counts
must come from decoding, not metadata. The cache
(`c2_real_vnweights_full/detections_full13726.npz`) therefore covers
100% of the footage.

## Headline (motorcycle re-id events, identical detections)

| variant | moto re-id | moto uIDs | mean life | short tracks |
|---|---|---|---|---|
| ByteTrack baseline    | 356 | 4755 | 57.3 | 1061 |
| + CTRV + NSA          | 264 | 4599 | 58.0 | 1021 |
| + uncertainty gate    | **141** | **3806** | **72.9** | **529** |

**Full system vs baseline: −60.4% motorcycle re-id events,
−20% unique IDs, +27% mean lifespan, −50% short tracks.**
(Note: since NSA became a config default, the "ctrv" row includes NSA;
the gate contributes the largest single step.)

## Held-out consistency

Frames 9000–13726 were detected AFTER all tuning decisions were frozen
(gate params, NSA) — a natural held-out segment. Its live dual-run gave
96 → 49 (−49%), consistent with the tuned-segment numbers: the gains
are not overfit to the first 9000 frames
(`c2_real_vnweights_full_p2/summary.json`).

## Quality checks (full video)

- Jumps >40px: baseline 1/331k, ours 9/336k — noise level.
- Heading-flip on bridged gaps: ours 334/1413 (23.6%), stable across
  the whole video vs the tuned segment (23.7%). Baseline shows 4.6% on
  560 bridges — NOT better: the baseline rarely bridges at all (it
  fragments instead, which is the failure being fixed). The honest
  reading: of our ~1400 bridges, an unknown minority are wrong merges;
  splitting them from correct rescues requires GT (Phase 2/3).

Reproduce:
```
python -m code.scripts.compare_motion_models \
    --replay experiments/c2_real_vnweights_full/detections_full13726.npz \
    --variants baseline,ctrv,ctrv_gate,ctrv_gate_nsa \
    --out-dir experiments/c2_final
```
