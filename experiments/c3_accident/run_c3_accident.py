"""C3 accident-branch test, pre-registered in experiments/c3_accident/evidence.md
(e333a4d, amended 58269b1).

Runs the paper's full pipeline (c1c2c3, best.pt at the Table 2 size, C1 threshold of
that size from the final run) on every clip of data/raw/videos, full frame, and records
per frame the Accident-class tracks before C3 (raw, with boxes) and in C3's accident
output. One JSON per clip under --out (gitignored data/), resumable clip by clip.
Order of use (the review stays blind to C3):
    PYTHONPATH=. python experiments/c3_accident/run_c3_accident.py --size S
    PYTHONPATH=. python experiments/c3_accident/run_c3_accident.py --size S --review
        -> <out>/review/<id>.jpg + verdicts.csv to fill (accident / false / unclear)
    PYTHONPATH=. python experiments/c3_accident/run_c3_accident.py --size S --analyze
"""
import argparse
import csv
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.stats import chi2

ACCIDENT = 4                       # PROJECT_CLASSES['accident']
FINAL = Path("data/processed/paper_runs/best85_roi_final")
SEEN = "7_20_2017"                 # camera with frames in best.pt's training export
ACCIDENT_CLIP, ONSET, END = "accident_tonghop.mp4", 55, 480
VIDEO_EXT = {".asf", ".mkv", ".mov", ".mp4"}


def clip_hour(name):
    """Hour from the file name; the accident clip has none, its overlay reads 13:30."""
    if name == ACCIDENT_CLIP:
        return 13
    m = re.search(r"(\d{1,2})_\d\d_\d\d (AM|PM)", name)
    if m:
        return int(m.group(1)) % 12 + (12 if m.group(2) == "PM" else 0)
    return int(re.search(r"\d{4}-\d\d-\d\d[ _](\d\d)", name).group(1))


def camera(name):
    return name.split(" ")[0]


def pipeline(size, fps, hour):
    from code.pipeline import Pipeline
    from code.detection.detector import Detector       # attached as run_roi_paper.benchmark_trial does
    cfg = yaml.safe_load((FINAL / str(size) / "configs/c1c2c3.yaml").read_text(encoding="utf-8"))
    cfg.pop("roi", None)                                # these cameras have no ROI
    cfg["tracker"]["frame_rate"] = float(fps)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    pipe = Pipeline.from_config(f.name, build_detector=False, dry_run_alerts=True)
    Path(f.name).unlink()
    pipe.detector = Detector(cfg["weights"]["yolo11"], img_size=size)
    pipe.alert, pipe.accident_classifier = None, None
    pipe.reset()
    pipe.adaptive_controller.density_threshold = json.loads(
        (FINAL / str(size) / "c1_threshold.json").read_text(encoding="utf-8"))["threshold"]
    pipe.context_estimator.hour_override = hour
    return pipe


def run_clip(path, size, out, max_frames=None):
    import torch
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    pipe = pipeline(size, fps, clip_hour(path.name))
    ok, frame = cap.read()
    pipe.detector.detect(frame)                         # warm-up before fixing threads, as run_roi_paper
    torch.set_num_threads(8)
    raw, c3, n_tracks, k = {}, {}, [], 0
    while ok and (max_frames is None or k < max_frames):
        r = pipe.process(frame)
        a = [[int(t.track_id), *[round(float(v), 1) for v in t.tlbr]]
             for t in pipe._last_tracks if int(t.class_id) == ACCIDENT]
        b = [int(t.track_id) for t in r["tracks_accident"] if int(t.class_id) == ACCIDENT]
        if a:
            raw[k] = a
        if b:
            c3[k] = b
        n_tracks.append(len(pipe._last_tracks))
        k += 1
        ok, frame = cap.read()
    cap.release()
    out.write_text(json.dumps(dict(
        clip=path.name, fps=fps, frames=k, size=size, hour=clip_hour(path.name),
        c1_threshold=pipe.adaptive_controller.density_threshold, seen_camera=SEEN in path.name,
        commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        mean_confirmed_tracks=float(np.mean(n_tracks)), accident_raw=raw, accident_c3=c3), indent=1), encoding="utf-8")


def episodes(frames, fps):
    """Maximal runs of alerting frames, merging gaps shorter than 1 s: [(start, end)]."""
    runs = []
    for k in sorted(int(k) for k in frames):
        if runs and k - runs[-1][1] <= max(1, round(fps)):
            runs[-1][1] = k
        else:
            runs.append([k, k])
    return [tuple(r) for r in runs]


def poisson_rate(count, hours):
    """Rate per hour with the exact (chi-square) Poisson 95% interval."""
    lo = 0.0 if count == 0 else chi2.ppf(0.025, 2 * count) / 2
    hi = chi2.ppf(0.975, 2 * count + 2) / 2
    return dict(count=count, hours=round(hours, 3), per_hour=round(count / hours, 2),
                ci95=[round(lo / hours, 2), round(hi / hours, 2)])


def clip_files(out_dir):
    return sorted(p for p in out_dir.glob("*.json") if p.name != "summary.json")


def review(out_dir, videos):
    """One sheet per RAW episode under neutral random ids; C3 is never computed here."""
    rdir = out_dir / "review"
    rdir.mkdir(exist_ok=True)
    rng, index = random.Random(), {}
    for f in clip_files(out_dir):
        d = json.loads(f.read_text(encoding="utf-8"))
        eps = episodes(d["accident_raw"], d["fps"])
        if not eps:
            continue
        cap, frames, k = cv2.VideoCapture(str(Path(videos) / d["clip"])), {}, 0
        wanted = {int(x) for s, e in eps for x in np.linspace(s, e, 6).round()}
        while k <= max(wanted):
            ok, img = cap.read()
            if not ok:
                break
            if k in wanted:
                frames[k] = img
            k += 1
        cap.release()
        for s, e in eps:
            eid = f"{rng.getrandbits(24):06x}"
            tiles = []
            for x in sorted({int(x) for x in np.linspace(s, e, 6).round()}):
                img = frames[x].copy()
                for _, x1, y1, x2, y2 in d["accident_raw"].get(str(x), d["accident_raw"].get(x, [])):
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                tiles.append(cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1]))))
            h = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
            rows = [np.hstack(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[i:i + 3])))
                    for i in range(0, len(tiles), 3)]
            cv2.imwrite(str(rdir / f"{eid}.jpg"), np.vstack(rows))
            index[eid] = dict(clip=d["clip"], start=s, end=e)
    (rdir / "episodes.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    ids = list(index)
    rng.shuffle(ids)
    with open(rdir / "verdicts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "verdict"])
        w.writerows([[i, ""] for i in ids])
    print(f"{len(ids)} raw episodes -> {rdir}; fill verdicts.csv with accident / false / unclear")


def analyze(out_dir):
    rdir = out_dir / "review"
    index = json.loads((rdir / "episodes.json").read_text(encoding="utf-8")) if (rdir / "episodes.json").exists() else {}
    verdict = {}
    if index:
        with open(rdir / "verdicts.csv", encoding="utf-8") as fh:
            verdict = {r["episode"]: r["verdict"].strip() for r in csv.DictReader(fh)}
        missing = [e for e in index if verdict.get(e) not in ("accident", "false", "unclear")]
        if missing:
            raise SystemExit(f"{len(missing)} raw episodes have no verdict yet; review first")
    by_clip = {}
    for eid, ep in index.items():
        by_clip.setdefault(ep["clip"], []).append((ep["start"], ep["end"], verdict[eid]))
    cams, rows = {}, []
    for f in clip_files(out_dir):
        d = json.loads(f.read_text(encoding="utf-8"))
        hours = d["frames"] / d["fps"] / 3600
        judged = by_clip.get(d["clip"], [])

        def label(s, e):  # a C3 episode inherits the verdict of the raw episode containing it
            return next((v for a, b, v in judged if a <= s and e <= b), "unclear")
        ep = {k: [(s, e, label(s, e)) for s, e in episodes(d[f"accident_{k}"], d["fps"])] for k in ("raw", "c3")}
        row = dict(clip=d["clip"], camera=camera(d["clip"]), minutes=round(hours * 60, 1),
                   seen_camera=d["seen_camera"], episodes=ep)
        if d["clip"] == ACCIDENT_CLIP:
            for k, v in ep.items():
                hit = [(s, e) for s, e, _ in v if e >= ONSET and s <= END]
                first = max(hit[0][0], ONSET) if hit else None
                row[k] = dict(alerts=bool(hit), first_alert_frame=first,
                              latency_s=round((first - ONSET) / d["fps"], 2) if hit else None,
                              duration_s=round((hit[0][1] - first + 1) / d["fps"], 2) if hit else None,
                              false_before_onset=sum(1 for s, e, _ in v if e < ONSET))
        else:
            c = cams.setdefault((row["camera"], d["seen_camera"]), {"raw": 0, "c3": 0, "hours": 0.0, "real": 0})
            for k in ("raw", "c3"):
                c[k] += sum(1 for *_, lab in ep[k] if lab == "false")
            c["real"] += sum(1 for *_, lab in ep["raw"] if lab == "accident")
            c["hours"] += hours
        rows.append(row)
    per_camera = {f"{cam}{' (seen)' if seen else ''}": dict(
        raw=poisson_rate(c["raw"], c["hours"]), c3=poisson_rate(c["c3"], c["hours"]), real_accident_episodes=c["real"])
        for (cam, seen), c in cams.items()}
    unseen = [c for (cam, seen), c in cams.items() if not seen]
    tot = {k: sum(c[k] for c in unseen) for k in ("raw", "c3", "hours")}
    pooled = dict(raw=poisson_rate(tot["raw"], tot["hours"]), c3=poisson_rate(tot["c3"], tot["hours"]))
    acc = next((r for r in rows if r["clip"] == ACCIDENT_CLIP), None)
    if tot["raw"] <= 2:
        verdict_text = "inconclusive: the raw pipeline has 0-2 false episodes on unseen normal clips"
    elif tot["raw"] - tot["c3"] >= 1 and acc and acc["c3"]["alerts"]:
        verdict_text = "C3 helps: it removes confirmed false episodes and still alerts on the accident"
    else:
        verdict_text = "C3 does not help on this test"
    summary = dict(per_camera=per_camera, pooled_unseen_normal=pooled, accident_clip=acc and {k: acc[k] for k in ("raw", "c3")},
                   verdict=verdict_text, clips=rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "clips"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--videos", default="data/raw/videos")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, help="smoke test only")
    ap.add_argument("--clips", nargs="*", help="subset of clip file names")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out or f"data/processed/c3_accident/{args.size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.review:
        return review(out_dir, args.videos)
    if args.analyze:
        return analyze(out_dir)
    clips = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in VIDEO_EXT)
    clips = [p for p in clips if not args.clips or p.name in args.clips]
    for p in clips:
        out = out_dir / (p.stem + ".json")
        if out.exists():
            continue
        print("running", p.name, flush=True)
        run_clip(p, args.size, out, args.max_frames)


if __name__ == "__main__":
    main()
