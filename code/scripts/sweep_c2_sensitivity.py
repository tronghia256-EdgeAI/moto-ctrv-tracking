"""
code/scripts/sweep_c2_sensitivity.py
────────────────────────────────────
One-at-a-time sensitivity of C2 (CTRV + gate + NSA) to the constants a reviewer
will ask about: process noise Q (all entries, or only the angular θ/ω ones),
measurement noise R, the gate's χ² level, its base cost c0 and the NSA floor.
Everything else stays at the shipped exp1_C2 values, and every variant replays
the SAME cached detections, so differences are the constant's alone.

Seeding bias: a GT corrected from one run's output keeps that run's boxes and
association decisions (on `dense`, 90.9% of motorcycle GT boxes are the default
C2 run's boxes). Scored there, ANY deviation from the defaults looks worse. So
each variant also reports its exact-box share against the GT and its agreement
with the default run; the correlation of its IDF1 change with that divergence
is printed. On a C2-seeded GT the sweep is an upper bound on sensitivity; the
baseline-seeded `mixed` segment is the decisive run.

Warm-up: the paper's dumps replay from frame 0 (10,700 frames before `dense`).
Each variant replays only --warmup frames before the segment; the shortened
default is scored against the frozen full-replay dump first, and the sweep
stops if the warm-up changes the result.

Usage (repo root):
    python -m code.scripts.sweep_c2_sensitivity
    python -m code.scripts.sweep_c2_sensitivity --gt <mixed gt> --segment 2500 3100
    python -m code.scripts.sweep_c2_sensitivity --roi-bundle <dir with roi.json>
    python -m code.scripts.sweep_c2_sensitivity --score-only   # re-score kept dumps
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import code.models.motorcycle_motion_model as mmm
from code.pipeline import Pipeline
from code.scripts.compare_motion_models import CONFIG_BASELINE, CONFIG_CTRV
from code.scripts.eval_hota_ci import ALPHAS, exact_box_share, frame_data, hota, hota_data
from code.scripts.gt_review import score_variants
from code.scripts.gt_roi import filter_frames
from code.scripts.import_gt import load_mot, to_framelist
from code.tracker.adaptive_bytetrack import AdaptiveBYTETracker, ClassAwareSTrack

DEFAULT_GT = "experiments/gt_qa/reviewed/20261007T161301Z_126e0b66/gt_reviewed.txt"
DEFAULT_NPZ = "experiments/c2_real_vnweights_full/detections_full13726.npz"
FROZEN_C2 = "experiments/gt_ablation/tracks_ctrv_gate_nsa.json"
MOTO = 1
ALPHA_05 = int(np.argmin(np.abs(ALPHAS - 0.5)))

# name -> {knob: value}; knobs absent from a variant keep the shipped value.
VARIANTS = {
    "c2_default": {},
    **{f"q_x{s:g}": {"q_scale": s} for s in (0.25, 0.5, 2, 4)},
    **{f"q_ang_x{s:g}": {"q_ang_scale": s} for s in (0.1, 10)},
    **{f"r_x{s:g}": {"r_scale": s} for s in (0.25, 0.5, 2, 4)},
    "chi2_p90": {"chi2": 4.605}, "chi2_p99": {"chi2": 9.210},
    **{f"c0_{c:g}": {"c0": c} for c in (0.3, 0.7)},
    **{f"nsa_floor_{f:g}": {"nsa_floor": f} for f in (0.01, 0.1, 0.2)},
}


def build(config: str, knobs: dict) -> Pipeline:
    """Pipeline for one variant; class-level CTRV settings are set explicitly
    every time so no knob leaks from the previous variant."""
    pipe = Pipeline.from_config(config, dry_run_alerts=True, build_detector=False)
    pipe.reset()
    pipe.context_estimator.hour_override = 8          # 08:00 footage, as run_replay
    if config == CONFIG_BASELINE:
        return pipe
    q = np.array(ClassAwareSTrack.ctrv_q_diag or mmm.Q_DIAG, dtype=float)
    q *= knobs.get("q_scale", 1.0)
    q[3:] *= knobs.get("q_ang_scale", 1.0)
    r = np.array(ClassAwareSTrack.ctrv_r_diag or mmm.R_DIAG, dtype=float)
    ClassAwareSTrack.ctrv_q_diag = q.tolist()
    ClassAwareSTrack.ctrv_r_diag = (r * knobs.get("r_scale", 1.0)).tolist()
    ClassAwareSTrack.ctrv_nsa = True
    ClassAwareSTrack.ctrv_scale_adaptive = False
    pipe.tracker.uncertainty_gating = True
    pipe.tracker.GATE_CHI2 = knobs.get("chi2", AdaptiveBYTETracker.GATE_CHI2)
    pipe.tracker.GATE_BASE_COST = knobs.get("c0", AdaptiveBYTETracker.GATE_BASE_COST)
    mmm.NSA_FLOOR = knobs.get("nsa_floor", 0.05)
    return pipe


def replay(pipe: Pipeline, dets: list, first: int, end: int) -> list:
    """Track frames [first, end); frames before `first` are left empty so the
    dump keeps absolute indexing (start_frame 0), as score_variants expects."""
    dummy = np.full((450, 800, 3), 128, dtype=np.uint8)
    frames = [[] for _ in range(first)]
    for d in dets[first:end]:
        result = pipe.process(dummy, detections=d)
        frames.append([(int(t.track_id), int(t.class_id),
                        [round(float(v), 1) for v in t.tlbr])
                       for t in result["tracks_mappo"]])
    return frames


def moto_framelist(path: Path, start: int, end: int, roi) -> list:
    """Motorcycle FrameList of one dump over [start, end), ROI-filtered as in
    gt_review.score_variants."""
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"][start:end]
    pred = {k: [(tid, box, cls) for tid, cls, box in f] for k, f in enumerate(frames, 1)}
    return to_framelist(filter_frames(pred, roi), end - start, MOTO)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--segment", nargs=2, type=int, default=(10700, 11300))
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--roi-bundle", default=None)
    ap.add_argument("--out", default="experiments/c2_sensitivity")
    ap.add_argument("--score-only", action="store_true",
                    help="skip the replays and re-score the kept dumps")
    args = ap.parse_args()

    start, end = args.segment
    first = max(0, start - args.warmup)
    roi = None
    if args.roi_bundle:
        from code.scripts.gt_roi import load_roi
        roi = load_roi(Path(args.roi_bundle))
    out = Path(args.out)
    dumps = out / f"dumps_{start}_{end}"          # tracks_*.json are gitignored
    runs = [("baseline", CONFIG_BASELINE, {})] + \
           [(n, CONFIG_CTRV, k) for n, k in VARIANTS.items()]

    if not args.score_only:
        npz = np.load(args.npz, allow_pickle=True)
        meta = json.loads(str(npz["meta"]))
        dets = [npz[f"f{i}"] for i in range(meta["num_frames"])]
        dumps.mkdir(parents=True, exist_ok=True)
        try:
            for name, config, knobs in runs:
                frames = replay(build(config, knobs), dets, first, end)
                (dumps / f"tracks_{name}.json").write_text(json.dumps(
                    {**meta, "start_frame": 0, "variant": name, "knobs": knobs,
                     "warmup_from": first, "frames": frames}), encoding="utf-8")
                print(f"  replayed {name}", flush=True)
        finally:
            mmm.NSA_FLOOR = 0.05

    # Warm-up check: shortened default vs the frozen full-replay C2 dump.
    check = Path(tempfile.mkdtemp(prefix="c2_frozen_"))
    shutil.copy(dumps / "tracks_baseline.json", check)
    shutil.copy(FROZEN_C2, check / "tracks_c2_full.json")
    frozen = score_variants(Path(args.gt), check, args.segment, roi=roi)
    shutil.rmtree(check)
    scores = score_variants(Path(args.gt), dumps, args.segment, roi=roi)
    pick = lambda s, n, c: {k: s[n]["metrics"][c][k] for k in ("mota", "idf1", "num_switches")}
    short, full = pick(scores, "c2_default", "motorcycle"), pick(frozen, "c2_full", "motorcycle")
    print(f"warm-up check (motorcycle): short {short} vs full {full}")
    if abs(short["idf1"] - full["idf1"]) > 0.002 or \
            abs(short["num_switches"] - full["num_switches"]) > 2:
        raise SystemExit("Shortened warm-up changes the C2 result; rerun "
                         "with a larger --warmup (or 99999 for frame 0).")

    # Seeding diagnostics on motorcycles.
    truth = filter_frames(load_mot(Path(args.gt)), roi)
    gt_fl = to_framelist(truth, end - start, MOTO)
    default_fl = moto_framelist(dumps / "tracks_c2_default.json", start, end, roi)
    table = {}
    for name, _, knobs in runs:
        fl = moto_framelist(dumps / f"tracks_{name}.json", start, end, roi)
        d = frame_data(gt_fl, fl)
        table[name] = {
            "knobs": knobs,
            "motorcycle": pick(scores, name, "motorcycle"),
            "ALL": pick(scores, name, "ALL"),
            "hota_alpha05_motorcycle": float(hota(hota_data(d))["HOTA"][ALPHA_05]),
            "exact_box_share_vs_gt": exact_box_share(d),
            # share of this run's boxes that the default run also produced
            "agreement_with_default": exact_box_share(frame_data(fl, default_fl)),
        }
    ctrv = [n for n in VARIANTS if n != "c2_default"]
    gap = np.array([table["c2_default"]["motorcycle"]["idf1"] - table[n]["motorcycle"]["idf1"]
                    for n in ctrv])
    diverge = np.array([1 - table[n]["agreement_with_default"] for n in ctrv])
    r = float(np.corrcoef(diverge, gap)[0, 1]) if gap.std() and diverge.std() else None

    out.mkdir(parents=True, exist_ok=True)
    result = {"gt": args.gt, "npz": args.npz, "segment": list(args.segment),
              "warmup_frames": start - first, "roi": bool(roi and roi["enabled"]),
              "warmup_check": {"short": short, "full_replay": full},
              "pearson_r_idf1_drop_vs_divergence": r, "variants": table}
    (out / f"sensitivity_{start}_{end}.json").write_text(json.dumps(result, indent=2),
                                                         encoding="utf-8")
    print(f"{'variant':<16}{'MOTA':>8}{'IDF1':>8}{'IDSW':>6}{'HOTA.5':>8}{'exact':>7}{'agree':>7}")
    for name, row in table.items():
        m = row["motorcycle"]
        print(f"{name:<16}{m['mota']:>8.4f}{m['idf1']:>8.4f}{m['num_switches']:>6}"
              f"{row['hota_alpha05_motorcycle']:>8.4f}{row['exact_box_share_vs_gt']:>7.3f}"
              f"{row['agreement_with_default']:>7.3f}")
    print(f"Pearson r (IDF1 drop vs divergence from default, {len(ctrv)} variants): {r}")


if __name__ == "__main__":
    main()
