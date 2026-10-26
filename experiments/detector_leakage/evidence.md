# Camera overlap between the detector's training export and the raw clips

**Date:** 2026-10-12. **Question:** do the detector's training images include
frames from any of the 8 raw CCTV clips, in particular the evaluation camera
(NKKN–Võ Thị Sáu)? Before this audit the manuscript's sentence "contains no
frames from the evaluation camera" rested on a filename search alone.

**Bottom line**

1. **Evaluation camera (NKKN): no overlap found.** Of the 68 training images
   nearest to the NKKN clip (≤ 14 of 64 bits), none shows the NKKN view on
   visual inspection.
2. **One other camera is in the training set.** All 75 images named
   `7_20_2017-*` are frames of the two `7_20_2017 … .mkv` clips (same camera,
   11:59 and 16:59). That camera cannot serve as a held-out test camera.
3. **Four further cameras are clean** and can be used as held-out test cameras
   for new GT: Cộng Hòa–Út Tịch (14:15 and 17:15), Cộng Hòa–Trường Chinh,
   Dương Bá Trạc–Tạ Quang Bửu and HAMQ1.

## Method

- **Training set:** the Roboflow export `data/annotations/accident_vehicle/`
  (v12, 4,980 images in train/valid/test; the training data of `best.pt`).
  Roboflow letterboxed every image into 640×640 with white edges, so each image
  is cropped to its non-white content before hashing. The horizontal flip is
  hashed too.
- **Clips:** every clip is decoded sequentially, and one frame per second is
  hashed (544–900 frames per clip).
- **Hash:** a 64-bit DCT perceptual hash (32×32 grey, top-left 8×8 block,
  median threshold). A fixed camera's background dominates the hash. Distances
  are always even, because every hash has exactly 32 set bits.
- **Distance:** for each training image and clip, the minimum Hamming distance
  over the clip's hashed frames, image and flip combined.
- **Verdict:** pHash alone has a grey zone (next section), so every candidate
  within 12 bits of a clip is inspected visually on a contact sheet. For NKKN
  the cut-off is 14 bits. Each sheet puts one clip frame next to all of the
  clip's candidates.

## Positive control and the grey zone

The 75 `7_20_2017-*` images are known frames of the `.mkv` clips:

- **Known frames:** 55 of 55 `11_59_59` images and 19 of 20 `4_59_59` images
  lie within 10 bits of their own clip; the last one is at 12 bits. All 75 lie
  within 12 bits, including Roboflow's augmented copies. The sheets confirm
  that they show that camera.
- **Unrelated scenes:** these start at the same distances. Five images lie at
  10 bits from a Cộng Hòa–Út Tịch or HAMQ1 clip, and all five are other scenes
  (Russian dash and CCTV footage, a rainy HCMC road from `cam_06`). Many
  unrelated images lie at 12 bits.

So no single threshold separates matches from non-matches. "Within 12 bits" is
the candidate list whose recall the control measured, and visual inspection
decides each case.

## Results

| Clip | Frames decoded | Candidates ≤10 / ≤12 bits | Verdict |
|---|---:|---:|---|
| NKKN–VoThiSau 08:00 (evaluation) | 13,726 | 0 / 12 (68 at ≤14) | no image shows this view |
| 7_20_2017 11:59:59 AM (.mkv) | 13,500 | 55 / 57 | **in training**: 55 named frames, distances 0–10 |
| 7_20_2017 4:59:59 PM (.mkv) | 13,500 | 19 / 24 | **in training**: 20 named frames, 19 at ≤10 and 1 at 12 |
| CongHoa–TruongChinh 17:00 | 13,707 | 0 / 1 | no match |
| CongHoa–UtTich1 14:15 | 13,662 | 2 / 9 | no match |
| CongHoa–UtTich1 17:15 | 13,586 | 1 / 8 | no match |
| DuongBaTrac–TaQuangBuu1 08:00 | 13,721 | 0 / 3 | no match |
| HAMQ1 15:00 (59.94 fps) | 53,936 | 2 / 23 | no match |

The NKKN candidates are other HCMC traffic cameras (`cam_06`, `07`, `09`,
`11`, `14`), Brazilian street cameras, crash photographs, and toy-car and
dashcam scenes.

In the original export the `7_20_2017` frames split 61 / 6 / 8 across
train / valid / test. The cleaned Kaggle package moves the original test
images into train.

Decoded lengths: the `.asf` headers report about 22,800 frames, but every
`.asf` clip decodes to 13,586–13,726 frames, about 9.1 min at 25 fps. The
`.mkv` clips are 15 min at 15 fps; HAMQ1 is 15 min at 59.94 fps.

## What this does and does not certify

- **Certifies:** the Roboflow v12 export (the training set of `best.pt`)
  contains no frame recognisable as the NKKN view, and it does contain the
  `7_20_2017` camera.
- **`weights/yolo11_vn.pt`, the segmentation checkpoint, only partly.**
  Nghĩa stated on 2026-10-13 that it was trained from the same Roboflow
  project, `Vehicle-Accident_Final`, the source of this export. The finding
  therefore carries over to every image the version behind yolo11_vn shares
  with v12.
  - Which version that was has not been recorded.
  - Images in that version and absent from v12 have not been audited.
  - Closing the gap needs that version's export run through this audit
    (`--images <export dir>`).
- **Not covered:** crops or rotations stronger than Roboflow's augmentations.
  The positive control shows that Roboflow's own copies stay within 12 bits.
- **Inspection:** the visual verdicts were made on 2026-10-12. An author should
  look at the regenerated sheets once (about 2 minutes) before the manuscript
  sentence is final.

## Reproduce

```powershell
# hashes and distances -> overlap.csv, summary.json (about 10 min on the laptop CPU)
python -m code.scripts.audit_camera_overlap
# contact sheets for the visual verdict. They show dataset images, so keep
# them under gitignored data/:
python -m code.scripts.audit_camera_overlap --sheets data/processed/leakage_sheets --sheet-bits 12
```

`overlap.csv` holds one row per training image: its best clip, its best
distance, and its distance to every clip. `summary.json` holds the histogram of
best distances, per-clip counts at 10 bits (before visual verification) and
the positive control.
