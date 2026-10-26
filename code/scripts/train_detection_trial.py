"""Explicit development-only detector trial; never completes human review."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from code.scripts.clean_detection_dataset import (
    BUNDLE, REPO, CleanStore, child, csv_bytes, final_groups, label_text, log, payload, stamp,
)
from code.scripts.audit_detection_dataset import parse_label
from code.scripts.gt_review import sha256

TRIALS = REPO / "data/processed/detector_trials"
MODE = "DEVELOPMENT_ONLY_NOT_PAPER_EVALUATION"


def plan(store, seed=42):
    """80/20 development split, preserving every conservative source constraint."""
    usable, skipped = {}, []
    for path in sorted(store.rows):
        entry = store.image(path)
        if entry["status"] != "KEEP":
            skipped.append({"source_path": path, "reason": entry["status"] + ": " + entry["reason"]})
            continue
        boxes = store.boxes(path)
        # Distinct polygons can collapse to duplicate boxes. Do not choose a label
        # on the user's behalf; omit that image from this trial only.
        canonical = [tuple(round(float(v), 10) for v in box) for box in boxes]
        if len(set(canonical)) != len(canonical):
            skipped.append({"source_path": path, "reason": "Unresolved duplicate detection boxes"})
            continue
        usable[path] = label_text(boxes, len(store.names))
    groups = [sorted(set(group) & usable.keys()) for group in final_groups(store)]
    groups = [group for group in groups if group]
    if len(groups) < 2:
        raise ValueError("At least two conservative groups are required for a trial")
    groups.sort(key=lambda g: hashlib.sha256((str(seed) + "|" + "|".join(g)).encode()).hexdigest())
    groups.sort(key=len, reverse=True)
    features = {}
    for path in usable:
        values = np.zeros(len(store.names) + 1); values[0] = 1
        for box in store.boxes(path):
            values[int(box[0]) + 1] += 1
        features[path] = values
    total = sum(features.values())
    used = np.zeros((2, len(total)))
    assignment = {}
    for index, group in enumerate(groups):
        vector = sum(features[p] for p in group)
        empty = [i for i in range(2) if not used[i, 0]]
        options = empty if len(groups) - index == len(empty) else [0, 1]
        scores = []
        for candidate in options:
            updated = used.copy(); updated[candidate] += vector
            scores.append(np.square((updated - np.array([.8, .2])[:, None] * total) / np.maximum(total, 1)).sum())
        chosen = options[int(np.argmin(scores))]
        used[chosen] += vector
        group_id = hashlib.sha256("|".join(group).encode()).hexdigest()[:16]
        for path in group:
            assignment[path] = ("train" if chosen == 0 else "val", group_id)
    return usable, assignment, skipped


def select_smoke(store, paths, limit, seed):
    """Prioritize class coverage, then deterministic image ordering."""
    ordered = sorted(paths, key=lambda p: hashlib.sha256((str(seed) + p).encode()).hexdigest())
    selected, covered = [], set()
    while ordered and len(selected) < limit:
        best = max(ordered, key=lambda p: len({int(b[0]) for b in store.boxes(p)} - covered))
        selected.append(best)
        covered.update(int(b[0]) for b in store.boxes(best))
        ordered.remove(best)
    return set(selected)


def prepare_trial(store, output, seed=42, smoke_train=32, smoke_val=8):
    output = Path(output).resolve()
    source = Path(store.manifest["source"]).resolve()
    if any(output == root or root in output.parents or output in root.parents
           for root in (store.bundle, source)):
        raise ValueError("Trial output must be separate from source and review bundle")
    if output.exists():
        raise FileExistsError(output)
    if min(smoke_train, smoke_val) < 1:
        raise ValueError("Smoke subsets must be nonempty")
    log("Checking original images, labels and immutable prepared pool...")
    store.verify_files(original=True)
    frozen = store.fingerprint
    labels, assignment, skipped = plan(store, seed)
    smoke = set()
    for split, limit in (("train", smoke_train), ("val", smoke_val)):
        smoke.update(select_smoke(store, [p for p, value in assignment.items() if value[0] == split], limit, seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".trial-", dir=output.parent))
    try:
        for prefix in (Path(), Path("smoke")):
            for split in ("train", "val"):
                (stage / prefix / split / "images").mkdir(parents=True)
                (stage / prefix / split / "labels").mkdir()
        manifest = []
        counts = {profile: {split: {"images": 0, "objects": 0, "classes": Counter()} for split in ("train", "val")}
                  for profile in ("trial", "smoke")}
        for index, (path, (split, gid)) in enumerate(sorted(assignment.items()), 1):
            row = store.rows[path]
            # Source names can exceed Windows MAX_PATH after staging is renamed.
            # Keep the original name in the manifest, use a short stable copy ID.
            name = 'img_' + hashlib.sha256(path.encode()).hexdigest()[:24] + Path(row['name']).suffix.lower()
            target = stage / split / "images" / name
            shutil.copy2(store.bundle / "pool/images" / row["name"], target)
            if sha256(target) != row["image_sha256"]:
                raise ValueError(f"Image changed during copying: {path}")
            label = stage / split / "labels" / (Path(name).stem + ".txt")
            label.write_text(labels[path], encoding="utf-8")
            if path in smoke:
                shutil.copy2(target, stage / "smoke" / split / "images" / target.name)
                shutil.copy2(label, stage / "smoke" / split / "labels" / label.name)
            for profile in ("trial", "smoke") if path in smoke else ("trial",):
                count = counts[profile][split]
                count["images"] += 1
                for box in store.boxes(path):
                    count["objects"] += 1
                    count["classes"][store.names[int(box[0])]] += 1
            manifest.append({"source_path": path, "split": split, "group_id": gid,
                             "image": target.relative_to(stage).as_posix(), "image_sha256": row["image_sha256"],
                             "label": label.relative_to(stage).as_posix(), "label_sha256": sha256(label),
                             "smoke": path in smoke})
            if index % 1000 == 0:
                log(f"Copied {index}/{len(assignment)} trial images")
        for profile in ("trial", "smoke"):
            root = output if profile == "trial" else output / "smoke"
            config = {"path": str(root), "train": "train/images", "val": "val/images", "names": store.names, "nc": len(store.names)}
            (stage / ("data.yaml" if profile == "trial" else "smoke.yaml")).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        report = {"schema": 1, "mode": MODE, "created": stamp(), "seed": seed,
                  "review_sha256": frozen, "review_status": store.tally, "names": store.names,
                  "counts": counts, "skipped_images": len(skipped), "source_preserved": True,
                  "limitations": ["Incomplete review and unknown source/augmentation metadata are explicitly accepted for this development trial only.",
                                  "Validation is internal development data, not an independent test set or paper result.",
                                  "Groups use heuristic image/name/camera constraints; unrecognized shared sources may remain.",
                                  "No severity labels or event timeline; no historical training/benchmark overlap certification."]}
        (stage / "report.json").write_bytes(payload(report))
        (stage / "manifest.csv").write_bytes(csv_bytes(manifest, list(manifest[0])))
        (stage / "skipped.csv").write_bytes(csv_bytes(skipped, ["source_path", "reason"]))
        (stage / "review_snapshot.json").write_bytes(payload(store.state))
        if sha256(store.path) != frozen:
            raise ValueError("Review changed during preparation; retry with a fresh snapshot")
        verify_trial(stage)
        os.replace(stage, output)
    except Exception:
        if stage.exists() and stage.resolve().parent == output.parent and stage.name.startswith(".trial-"):
            shutil.rmtree(stage)
        raise
    log(f"Development-only dataset: {output}")
    log(json.dumps(report["counts"], ensure_ascii=False))
    return output


def verify_trial(dataset):
    """Check copied labels/images and group boundaries before fitting a model."""
    dataset = Path(dataset).resolve()
    report = json.loads((dataset / "report.json").read_text(encoding="utf-8"))
    if report["mode"] != MODE:
        raise ValueError("Expected an explicitly marked development trial dataset")
    with (dataset / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("Empty trial manifest")
    groups, pixels, seen = {}, {}, set()
    for row in records:
        if row["source_path"] in seen or row["split"] not in ("train", "val"):
            raise ValueError("Invalid trial manifest membership")
        seen.add(row["source_path"])
        for key in ("group_id", "image_sha256"):
            table = groups if key == "group_id" else pixels
            table.setdefault(row[key], set()).add(row["split"])
        for prefix in ("", "smoke/") if row["smoke"] == "True" else ("",):
            for kind in ("image", "label"):
                target = child(dataset, prefix + row[kind])
                if not target.is_file() or sha256(target) != row[kind + "_sha256"]:
                    raise ValueError(f"Trial file changed: {target}")
            label = child(dataset, prefix + row["label"])
            objects, issues, _ = parse_label(label.read_text(encoding="utf-8"), len(report["names"]), 1, 1)
            if issues or any(obj["kind"] != "box" for obj in objects):
                raise ValueError(f"Invalid detection trial label: {label}")
    if any(len(splits) != 1 for splits in [*groups.values(), *pixels.values()]):
        raise ValueError("Trial group or duplicate crosses train/val")
    # Extra images could otherwise be silently picked up by Ultralytics.
    for profile, prefix in (("trial", ""), ("smoke", "smoke/")):
        for split in ("train", "val"):
            members = [r for r in records if r["split"] == split and (not prefix or r["smoke"] == "True")]
            expected_images = {child(dataset, prefix + r["image"]) for r in members}
            expected_labels = {child(dataset, prefix + r["label"]) for r in members}
            images = set((dataset / prefix / split / "images").glob("*"))
            labels = set((dataset / prefix / split / "labels").glob("*.txt"))
            if images != expected_images or labels != expected_labels or len(members) != report["counts"][profile][split]["images"]:
                raise ValueError("Trial inventory differs from frozen manifest")
    return report


def train_trial(dataset, weights, profile="smoke", epochs=None, device="cpu", batch=None, imgsz=None):
    """Train and validate only; no test-set selection or production replacement."""
    import torch
    import ultralytics
    from ultralytics import YOLO

    dataset, weights = Path(dataset).resolve(), Path(weights).resolve()
    report = verify_trial(dataset)
    if not weights.is_file():
        raise FileNotFoundError(f"Local detection checkpoint required: {weights}")
    if profile not in ("smoke", "trial"):
        raise ValueError("Unknown trial profile")
    root = dataset / "smoke" if profile == "smoke" else dataset
    run_id = profile + "_" + uuid.uuid4().hex[:8]
    runs = dataset / "runs"; runs.mkdir(exist_ok=True)
    # Regenerate absolute roots after moving a dataset to a GPU machine.
    config_path = runs / (run_id + "_data.yaml")
    config_path.write_text(yaml.safe_dump({"path": str(root), "train": "train/images", "val": "val/images",
                                         "names": report["names"], "nc": len(report["names"])}, sort_keys=False), encoding="utf-8")
    args = dict(data=str(config_path), epochs=epochs if epochs is not None else (1 if profile == "smoke" else 30),
                imgsz=imgsz if imgsz is not None else (320 if profile == "smoke" else 640),
                batch=batch if batch is not None else (2 if profile == "smoke" else 8),
                device=device, workers=0, seed=report["seed"], deterministic=True,
                project=str(runs), name=run_id, exist_ok=False, cache=False, plots=False,
                amp=device != "cpu", patience=10, val=True)
    if min(args["epochs"], args["imgsz"], args["batch"]) < 1:
        raise ValueError("Epochs, image size and batch must be positive")
    model = YOLO(str(weights))
    if model.task != "detect":
        raise ValueError("Use a detection checkpoint, not the old segmentation model")
    receipt = {"mode": MODE, "started": stamp(), "status": "RUNNING", "profile": profile,
               "dataset_report_sha256": sha256(dataset / "report.json"), "weights_sha256": sha256(weights),
               "torch": torch.__version__, "ultralytics": ultralytics.__version__, "arguments": args}
    receipt_path = runs / (run_id + "_receipt.json")
    receipt_path.write_bytes(payload(receipt))
    try:
        metrics = model.train(**args)
        receipt.update(status="COMPLETED", finished=stamp(),
                       metrics={k: float(v) for k, v in metrics.results_dict.items()},
                       output=str(model.trainer.save_dir))
    except Exception as error:
        receipt.update(status="FAILED", finished=stamp(), error=str(error))
        raise
    finally:
        receipt_path.write_bytes(payload(receipt))
    log(f"Trial completed; development metrics only: {receipt_path}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--weights", type=Path, default=REPO / "weights/yolo11n.pt")
    parser.add_argument("--profile", choices=("smoke", "trial"), default="smoke")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.prepare and not args.train:
        parser.error("Choose --prepare and/or --train")
    dataset = args.dataset
    if args.prepare:
        dataset = dataset or TRIALS / ("trial_" + uuid.uuid4().hex[:8])
        prepare_trial(CleanStore(args.bundle), dataset, args.seed)
    if args.train:
        if dataset is None:
            parser.error("Provide --dataset or use --prepare --train")
        train_trial(dataset, args.weights, args.profile, args.epochs, args.device, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
