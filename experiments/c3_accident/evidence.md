# C3's accident branch: false alerts and one real accident (pre-registered 2026-10-14)

Fixed before the detector or any pipeline has run on the clips below. Nghia
chose this test on 14/10.

**Question.** Does C3's precision branch keep the real accident while it
suppresses false accident alerts? The branch is the reliability score ≥ 0.7
that feeds incident alerting.

**Footage.** All files are in `data/raw/videos/`; none has GT.

- **Normal traffic, 9 clips:**
  - NKKN-VoThiSau, CongHoa-TruongChinh, CongHoa-UtTich1 (14:15 and 17:15),
    DuongBaTrac-TaQuangBuu1 and HAMQ1. These are 6 cameras, about 75 min in
    all. No accident is known in them, so every alert counts as false.
  - The two `7_20_2017` clips are a camera with 75 frames in best.pt's
    training export (`experiments/detector_leakage/`). They are reported
    separately as a seen camera.
- **`accident_tonghop.mp4`:** 1920×1080, 30 fps, 481 frames. It is one CCTV
  scene in rain in which a car hits a motorcycle.
  - Onset is frame 55, the first contact, fixed by eye on frames 30–95 before
    any detector output existed. The rider is down by frames 70–75.
  - The accident interval is frames 55–480, through the end of the clip,
    including the aftermath.
  - Leakage check: pHash against best.pt's 4,980 training images gives a
    nearest distance of 12 bits. The 8 nearest images are other scenes on
    visual check, so the clip is not in best.pt's training data.

**Pipeline.**

- The paper's full pipeline (`c1c2c3`) with best.pt, at the input size that
  the ce8f3d5 rule selects for Table 2.
- Detection runs on the full frame (these cameras have no ROI) with the
  paper's confidence floor of 0.1. The tracker frame rate is each clip's own,
  and the hour comes from each file name; all are inside the peak hours.
- C1's density threshold is the NKKN value for that size, applied unchanged.
  Nothing is recalibrated on these clips.

**Alerts.**

- Raw (without C3): a frame alerts if the pipeline's confirmed tracks before
  C3 include a track of class Accident.
- With C3: the same, using C3's accident output (reliability ≥ 0.7).
- An alert episode is a maximal run of alerting frames; gaps shorter than 1 s
  are merged.

**Metrics.**

- Normal clips: false alert episodes per hour, raw vs with C3, per clip and
  pooled. The seen camera is reported apart.
  - Every alert episode is checked visually. One showing a real accident would
    be reclassified and reported as such.
- Accident clip: whether an episode overlaps frames 55–480, the first alerting
  frame and the latency from frame 55, raw vs with C3.

**Reading.**

- C3's precision branch helps if it lowers false alert episodes per hour on
  the normal clips and still alerts on the accident.
- With one positive event, the accident side is a case study, not a recall
  estimate.
- No threshold or weight of C3 or C1 is changed after the results exist.
  Whatever happens is reported.

## Amendment (2026-10-14 15:21, before any clip has run)

Small false-alert counts need explicit handling.

1. **Counts and rates.** False-alert episodes are reported as counts per
   camera and in total, raw and with C3. Each rate per hour carries an exact
   Poisson 95% interval (chi-square bounds).
2. **Verdict.**
   - "C3 helps" requires both of the following:
     - C3 removes at least one visually confirmed false episode;
     - C3 still alerts on the accident.
   - If the raw pipeline has 0–2 false episodes in total, the counts are
     reported and the false-alert comparison is called inconclusive.
3. **Accident clip, raw and C3 side by side:**
   - whether each alerts;
   - the first alerting frame;
   - the latency from frame 55;
   - the duration of the episode that covers the accident.

   C3's cost in delay is therefore visible even when both alert.
4. **Blind visual check.**
   - Only raw episodes are reviewed. C3's alerting frames are a subset of the
     raw ones, so C3's episodes inherit those verdicts.
   - Review sheets carry neutral random ids and no variant label.
   - The verdicts (accident / false / unclear) are written before C3's
     episodes are computed: `--review`, then the verdict file, then
     `--analyze`, which refuses to run until every raw episode has a verdict.

## Amendment 2 (2026-10-15, before any C3-test pipeline output)

Nghia decided on 15/10 that the paper will not measure latency, so the ce8f3d5
rule cannot select Table 2's size. **S = 448**, fixed now, before any pipeline
has run on these clips, for these reasons:

- The rule picks the largest best.pt size whose arms all have a median p95 of
  at most 40 ms. If no size meets that, it falls back to 448.
- The only clean sweep is `experiments/realtime/evidence.md`, Sect. 4, from
  27/8 on an unthrottled CPU. It measured the detect head at a p95 of
  32.9 ms at 448 and 69.6 ms at 640.
- So 448 is the outcome unless 640 met the budget, which that sweep rules
  out. One caveat is inferred, not verified: the sweep timed yolo11n, not
  best.pt.

Everything else in the test is unchanged.

## Result at 448 px (2026-10-15, read as pre-registered)

Records are in `results_448/`: `summary.json` from `--analyze`, the blind
verdicts and the episode index. The review sheets are derived from footage, so
they stay in the gitignored `data/processed/c3_accident/448/review/`.

- **Clips:** 8 normal clips (the list above names 8 files, although its heading
  says 9) and the accident clip.
- **Normal clips:** 0 raw false-alert episodes and 0 with C3.
  - Unseen cameras pooled: 1.01 h, 0/h, exact Poisson 95% interval [0, 3.65]/h.
  - Seen camera `7_20_2017`: 0.5 h, 0/h, [0, 7.37]/h.
- **Verdict on false alerts:** inconclusive, by the small-count rule. The raw
  pipeline gives C3 no false alert to suppress.
- **Accident clip:**
  - Raw: two episodes, frames 60–86 and 360–480. The blind review judged both
    to show the accident. The first alert is at frame 60, 0.17 s after onset,
    and the episode lasts 0.9 s.
  - With C3: episodes at 62–77 and 362–480. The first alert is at frame 62,
    0.23 s after onset, so C3 costs 2 frames (0.07 s) of delay and shortens
    the first episode to 0.53 s.
- **Reading:** C3 keeps the real accident alert, at a two-frame delay.
  Whether it lowers false alerts cannot be measured here, because the raw
  pipeline raises none in about 1.5 h of normal traffic.
