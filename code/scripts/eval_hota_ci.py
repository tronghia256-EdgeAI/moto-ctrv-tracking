"""
code/scripts/eval_hota_ci.py
────────────────────────────
Scores frozen track dumps on the human-REVIEWED dense GT
(experiments/gt_qa/reviewed/<run>/gt_reviewed.txt: absolute frames
10700-11299, 427 identities) and adds what the review run did not report.

  1. Reproduction gate. The three paper variants are re-scored with the
     repository evaluator exactly the way gt_review.score_variants() produced
     review.json; every field must match bit-for-bit (and the sha256 of GT,
     dumps and metrics.py are compared). Any mismatch aborts.
  2. Table. MOTA / IDF1 / IDSW / Frag / FN / FP from
     code.evaluation.metrics.compute_tracking_metrics (IoU 0.5) plus HOTA /
     DetA / AssA (+ DetRe/DetPr/AssRe/AssPr/LocA in the JSON) from a
     single-sequence port of TrackEval's HOTA (alpha = 0.05 ... 0.95), for
     every ByteTrack ablation dump, OC-SORT and BoT-SORT (published and
     threshold-matched). Per class = GT and predictions both filtered to that class;
     ALL = class-agnostic, as in the repository evaluator.
  3. Paired moving-block bootstrap CIs for IDF1 and MOTA differences
     (see block_bootstrap()).
  4. Optional --trackeval DIR: cross-check the HOTA port (and report
     TrackEval CLEAR/Identity next to the repository MOTA/IDF1) against the
     real TrackEval package installed in DIR (pip install --target DIR,
     outside the project environment).
  5. Diagnostics for reading the table: share of GT boxes each tracker
     reproduces exactly (the GT was pre-labelled from ctrv_gate_nsa), and
     the detector-score band of the best detection behind each GT box
     (which GT each tracker's thresholds can reach at all).

Usage (repo root):
  python -m code.scripts.eval_hota_ci [--trackeval DIR]
Writes experiments/external_trackers/metrics.json and prints markdown tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.evaluation.metrics import IOU_MATCH_THRESHOLD, compute_tracking_metrics
from code.scripts.import_gt import load_mot, to_framelist, tracks_json_slice

REVIEW_DIR = "experiments/gt_qa/reviewed/20261007T161301Z_126e0b66"
SEGMENT = (10700, 11300)                  # absolute, end-exclusive
VARIANTS = {                              # label -> dump (repo-relative)
    "baseline": "experiments/gt_ablation/tracks_baseline.json",
    "ctrv_gate_nsa": "experiments/gt_ablation/tracks_ctrv_gate_nsa.json",
    "c1c2c3": "experiments/gt_ablation/tracks_c1c2c3.json",
    "ctrv": "experiments/gt_ablation/tracks_ctrv.json",
    "ctrv_nsa": "experiments/gt_ablation/tracks_ctrv_nsa.json",
    "ctrv_gate": "experiments/gt_ablation/tracks_ctrv_gate.json",
    "ocsort": "experiments/external_trackers/tracks_ocsort.json",
    "botsort": "experiments/external_trackers/tracks_botsort.json",
    "ocsort_matched": "experiments/external_trackers/tracks_ocsort_matched.json",
    "ocsort_byte": "experiments/external_trackers/tracks_ocsort_byte.json",
    "botsort_matched": "experiments/external_trackers/tracks_botsort_matched.json",
}
NAMES = {"baseline": "ByteTrack baseline", "ctrv_gate_nsa": "C2 (ctrv_gate_nsa)",
         "c1c2c3": "Exp2/3 (c1c2c3)", "ctrv": "+ CTRV", "ctrv_nsa": "+ CTRV, NSA",
         "ctrv_gate": "+ CTRV, gate", "ocsort": "OC-SORT", "botsort": "BoT-SORT",
         "ocsort_matched": "OC-SORT (0.5 + BYTE)", "ocsort_byte": "OC-SORT (0.6 + BYTE)",
         "botsort_matched": "BoT-SORT (matched thr.)"}
REPRO = ("baseline", "ctrv_gate_nsa", "c1c2c3", "ctrv", "ctrv_nsa", "ctrv_gate")
CLASSES = {"motorcycle": 1, "car": 0, "ALL": None}
PAIRS = [("ctrv_gate_nsa", "baseline"), ("c1c2c3", "baseline"),
         ("ctrv_gate_nsa", "ocsort"), ("ctrv_gate_nsa", "botsort"),
         ("ctrv_gate_nsa", "ocsort_matched"), ("ctrv_gate_nsa", "ocsort_byte"),
         ("ctrv_gate_nsa", "botsort_matched")]
BLOCKS = (15, 30, 60)
B = 10_000
SEED = 20261012
ALPHAS = np.arange(0.05, 0.99, 0.05)      # TrackEval HOTA.array_labels


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Per-frame data ───────────────────────────────────────────────────

def iou_matrix(a, b) -> np.ndarray:
    """Vectorised code.evaluation.metrics.compute_iou: same operations in the
    same order (plain geometric area, no +1), hence bit-identical values."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    iw = np.maximum(0.0, np.minimum(a[:, None, 2], b[None, :, 2])
                    - np.maximum(a[:, None, 0], b[None, :, 0]))
    ih = np.maximum(0.0, np.minimum(a[:, None, 3], b[None, :, 3])
                    - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = inter / (area_a[:, None] + area_b[None, :] - inter)
    return np.where(inter == 0.0, 0.0, out)


def frame_data(gt_fl, pr_fl) -> dict:
    """FrameLists (as fed to compute_tracking_metrics) -> ids, boxes, IoU."""
    return {
        "gt_ids": [[g for g, _ in f] for f in gt_fl],
        "pr_ids": [[p for p, _ in f] for f in pr_fl],
        "gt_boxes": [np.asarray([b for _, b in f], float).reshape(-1, 4) for f in gt_fl],
        "pr_boxes": [np.asarray([b for _, b in f], float).reshape(-1, 4) for f in pr_fl],
        "iou": [iou_matrix([b for _, b in g], [b for _, b in p])
                for g, p in zip(gt_fl, pr_fl)],
    }


# ── Replica of the repository evaluator on precomputed IoU ───────────

def clear_counts(d: dict, lo: int, hi: int, thr: float = IOU_MATCH_THRESHOLD):
    """compute_tracking_metrics' CLEAR loop, line for line, over frames
    [lo, hi) starting from an empty state. Returns
    (num_gt, misses, false_positives, switches, fragmentations, gt_tracks)."""
    misses = fps = switches = frags = total_gt = 0
    last_match: dict = {}
    frag_state: dict = {}
    for k in range(lo, hi):
        gids, pids, iou = d["gt_ids"][k], d["pr_ids"][k], d["iou"][k]
        total_gt += len(gids)
        matched: dict = {}
        used: set = set()
        pid_to_idx = {pid: j for j, pid in enumerate(pids)}
        for i, gid in enumerate(gids):
            prev = last_match.get(gid)
            if prev is not None and prev in pid_to_idx:
                j = pid_to_idx[prev]
                if j not in used and iou[i, j] >= thr:
                    matched[i] = j
                    used.add(j)
        free_g = [i for i in range(len(gids)) if i not in matched]
        free_p = [j for j in range(len(pids)) if j not in used]
        if free_g and free_p:
            rows, cols = linear_sum_assignment(1.0 - iou[np.ix_(free_g, free_p)])
            for r, c in zip(rows, cols):
                i, j = free_g[r], free_p[c]
                if iou[i, j] >= thr:
                    matched[i] = j
                    used.add(j)
        for i, gid in enumerate(gids):
            st = frag_state.setdefault(gid, [False, False])   # ever, prev
            if i in matched:
                pid = pids[matched[i]]
                if gid in last_match and last_match[gid] != pid:
                    switches += 1
                if st[0] and not st[1]:
                    frags += 1
                last_match[gid] = pid
                st[0] = st[1] = True
            else:
                misses += 1
                st[1] = False
        fps += len(pids) - len(used)
    return total_gt, misses, fps, switches, frags, len(frag_state)


def idtp_counts(d: dict, lo: int, hi: int, thr: float = IOU_MATCH_THRESHOLD):
    """_compute_idf1's global ID matching over frames [lo, hi).
    Returns (IDTP, num_gt_dets, num_pred_dets)."""
    pairs: Counter = Counter()
    n_gt = n_pr = 0
    for k in range(lo, hi):
        gids, pids = d["gt_ids"][k], d["pr_ids"][k]
        n_gt += len(gids)
        n_pr += len(pids)
        for i, j in zip(*np.nonzero(d["iou"][k] >= thr)):
            pairs[(gids[i], pids[j])] += 1
    if not pairs:
        return 0.0, n_gt, n_pr
    g_ix = {g: n for n, g in enumerate(sorted({g for g, _ in pairs}))}
    p_ix = {p: n for n, p in enumerate(sorted({p for _, p in pairs}))}
    counts = np.zeros((len(g_ix), len(p_ix)))
    for (g, p), c in pairs.items():
        counts[g_ix[g], p_ix[p]] = c
    rows, cols = linear_sum_assignment(-counts)
    return float(counts[rows, cols].sum()), n_gt, n_pr


def mota_idf1(n_gt, misses, fps, switches, idtp, n_pr):
    """The repository's formulas (metrics.py) on summed counts."""
    mota = 1.0 - (misses + fps + switches) / np.maximum(n_gt, 1)
    idf1 = 2.0 * idtp / (n_pr + n_gt)
    return mota, idf1


# ── HOTA: port of TrackEval for one sequence ─────────────────────────

def hota_data(d: dict, similarity=None) -> dict:
    """TrackEval eval_sequence input; ids remapped to 0..n-1 in sorted order
    like TrackEval's loaders (HOTA is invariant to the numbering)."""
    sims = similarity or d["iou"]
    g_all = np.unique([g for f in d["gt_ids"] for g in f])
    p_all = np.unique([p for f in d["pr_ids"] for p in f])
    g_map = {g: n for n, g in enumerate(g_all)}
    p_map = {p: n for n, p in enumerate(p_all)}
    return {
        "num_timesteps": len(d["gt_ids"]),
        "num_gt_ids": len(g_all), "num_tracker_ids": len(p_all),
        "num_gt_dets": sum(len(f) for f in d["gt_ids"]),
        "num_tracker_dets": sum(len(f) for f in d["pr_ids"]),
        "gt_ids": [np.array([g_map[g] for g in f], dtype=int) for f in d["gt_ids"]],
        "tracker_ids": [np.array([p_map[p] for p in f], dtype=int) for f in d["pr_ids"]],
        "similarity_scores": sims,
    }


def hota(data: dict) -> dict:
    """Port of HOTA.eval_sequence + HOTA._compute_final_fields from
    JonathonLuiten/TrackEval @12c8791b303e0a0b50f753af204249e622d0281a,
    trackeval/metrics/hota.py. Copyright (c) 2020 Jonathon Luiten, MIT
    License (https://github.com/JonathonLuiten/TrackEval/blob/master/LICENSE).
    Same statements in the same order; np.float -> float is the only change.
    Cross-checked bit-for-bit against TrackEval itself via --trackeval."""
    eps = np.finfo("float").eps
    res = {f: np.zeros(len(ALPHAS), dtype=float) for f in
           ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA",
            "OWTA", "HOTA_TP", "HOTA_FN", "HOTA_FP")}
    if data["num_tracker_dets"] == 0:
        res["HOTA_FN"] = data["num_gt_dets"] * np.ones(len(ALPHAS), dtype=float)
        res["LocA"] = np.ones(len(ALPHAS), dtype=float)
        return res
    if data["num_gt_dets"] == 0:
        res["HOTA_FP"] = data["num_tracker_dets"] * np.ones(len(ALPHAS), dtype=float)
        res["LocA"] = np.ones(len(ALPHAS), dtype=float)
        return res

    potential_matches_count = np.zeros((data["num_gt_ids"], data["num_tracker_ids"]))
    gt_id_count = np.zeros((data["num_gt_ids"], 1))
    tracker_id_count = np.zeros((1, data["num_tracker_ids"]))
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        similarity = data["similarity_scores"][t]
        sim_iou_denom = (similarity.sum(0)[np.newaxis, :]
                         + similarity.sum(1)[:, np.newaxis] - similarity)
        sim_iou = np.zeros_like(similarity)
        sim_iou_mask = sim_iou_denom > 0 + eps
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
        potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tracker_ids_t] += 1

    global_alignment_score = potential_matches_count / (
        gt_id_count + tracker_id_count - potential_matches_count)
    matches_counts = [np.zeros_like(potential_matches_count) for _ in ALPHAS]

    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        if len(gt_ids_t) == 0:
            for a, _ in enumerate(ALPHAS):
                res["HOTA_FP"][a] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            for a, _ in enumerate(ALPHAS):
                res["HOTA_FN"][a] += len(gt_ids_t)
            continue
        similarity = data["similarity_scores"][t]
        score_mat = global_alignment_score[gt_ids_t[:, np.newaxis],
                                           tracker_ids_t[np.newaxis, :]] * similarity
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        for a, alpha in enumerate(ALPHAS):
            actually_matched_mask = similarity[match_rows, match_cols] >= alpha - eps
            alpha_match_rows = match_rows[actually_matched_mask]
            alpha_match_cols = match_cols[actually_matched_mask]
            num_matches = len(alpha_match_rows)
            res["HOTA_TP"][a] += num_matches
            res["HOTA_FN"][a] += len(gt_ids_t) - num_matches
            res["HOTA_FP"][a] += len(tracker_ids_t) - num_matches
            if num_matches > 0:
                res["LocA"][a] += sum(similarity[alpha_match_rows, alpha_match_cols])
                matches_counts[a][gt_ids_t[alpha_match_rows],
                                  tracker_ids_t[alpha_match_cols]] += 1

    for a, _ in enumerate(ALPHAS):
        matches_count = matches_counts[a]
        ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
        res["AssA"][a] = np.sum(matches_count * ass_a) / np.maximum(1, res["HOTA_TP"][a])
        ass_re = matches_count / np.maximum(1, gt_id_count)
        res["AssRe"][a] = np.sum(matches_count * ass_re) / np.maximum(1, res["HOTA_TP"][a])
        ass_pr = matches_count / np.maximum(1, tracker_id_count)
        res["AssPr"][a] = np.sum(matches_count * ass_pr) / np.maximum(1, res["HOTA_TP"][a])

    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
    res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
    res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
    res["DetA"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"])
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])
    return res


HOTA_SUMMARY = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA")


def hota_summary(res: dict) -> dict:
    """TrackEval's summary convention: array fields averaged over alpha."""
    out = {k: float(np.mean(res[k])) for k in HOTA_SUMMARY}
    out["HOTA(0)"] = float(res["HOTA"][0])
    out["per_alpha"] = {k: [float(x) for x in res[k]] for k in ("HOTA", "DetA", "AssA", "LocA")}
    return out


DETS = "experiments/c2_real_vnweights_full/detections_full13726.npz"
SCORE_BANDS = {"(0.1,0.5]": (0.1, 0.5), "(0.5,0.6]": (0.5, 0.6),
               "(0.6,0.7]": (0.6, 0.7), "(0.7,1]": (0.7, 1.0)}


def detector_coverage(truth: dict, n: int) -> dict:
    """Per class, GT boxes binned by the best score among same-frame detections
    with IoU >= 0.5. Shows what each tracker's thresholds can reach: ByteTrack
    >0.5 (0.1-0.5 only continues tracks), BoT-SORT >0.6 (0.1-0.6 continues),
    OC-SORT MOT17 default >0.6 only (no BYTE stage)."""
    z = np.load(REPO_ROOT / DETS, allow_pickle=True)
    out = {}
    for c, cid in (("motorcycle", 1), ("car", 0)):
        best = []
        for k in range(1, n + 1):
            g = [b for _, b, cls in truth.get(k, []) if cls == cid]
            if g:
                d = np.asarray(z[f"f{SEGMENT[0] + k - 1}"], dtype=float).reshape(-1, 6)
                iou = iou_matrix(g, d[:, :4])
                best.extend(np.where(iou >= IOU_MATCH_THRESHOLD, d[None, :, 4], 0.0)
                            .max(axis=1, initial=0.0))
        best = np.asarray(best)
        out[c] = {"gt_boxes": int(len(best)), "no_detection": float((best == 0).mean()),
                  **{k: float(((best > lo) & (best <= hi)).mean())
                     for k, (lo, hi) in SCORE_BANDS.items()}}
    return out


def exact_box_share(d: dict, thr: float = 0.999) -> float:
    """Share of GT boxes reproduced (IoU >= thr) by some box of the tracker in
    the same frame. The dense GT was pre-labelled from ctrv_gate_nsa's own
    dump, so this measures how much of the GT is that tracker's output."""
    hit = sum(int((m.max(axis=1) >= thr).sum()) for m in d["iou"] if m.size)
    return hit / max(1, sum(len(f) for f in d["gt_ids"]))


# ── Block bootstrap ──────────────────────────────────────────────────

def block_table(d: dict, L: int) -> np.ndarray:
    """Row s = counts of the block [s, s+L) scored on its own (fresh CLEAR
    state, own ID matching): [num_gt, misses, FP, IDSW, IDTP, num_pred].
    Offsetting IDs per resampled block makes every block its own set of
    identities, so a replicate's counts are exactly the sums of its rows:
    CLEAR has no state across blocks and the global IDF1 assignment is
    block-diagonal."""
    rows = []
    for s in range(len(d["gt_ids"]) - L + 1):
        n_gt, mi, fp, sw, _, _ = clear_counts(d, s, s + L)
        idtp, n_gt2, n_pr = idtp_counts(d, s, s + L)
        assert n_gt == n_gt2
        rows.append((n_gt, mi, fp, sw, idtp, n_pr))
    return np.asarray(rows, dtype=float)


def metrics_from_rows(rows: np.ndarray):
    n_gt, mi, fp, sw, idtp, n_pr = np.moveaxis(rows, -1, 0)
    return mota_idf1(n_gt, mi, fp, sw, idtp, n_pr)


def replicate_framelists(d: dict, starts, L: int):
    """The literal resampled sequence: drawn blocks concatenated, every id
    offset by (block index + 1) * 1e9 so no identity spans two blocks."""
    pr, gt = [], []
    for b, s in enumerate(starts):
        off = (b + 1) * 10**9
        for k in range(s, s + L):
            gt.append([(off + g, x) for g, x in zip(d["gt_ids"][k], d["gt_boxes"][k].tolist())])
            pr.append([(off + p, x) for p, x in zip(d["pr_ids"][k], d["pr_boxes"][k].tolist())])
    return pr, gt


def block_bootstrap(tables: dict, L: int, n: int) -> dict:
    """Moving-block bootstrap (Kuensch 1989), paired.

    Each replicate draws K = n/L block starts uniformly with replacement from
    {0, ..., n-L}; the SAME starts are used for every variant (paired) and
    every class. Identity ids are offset per drawn block, so no identity spans
    two blocks. The statistic is the difference of the pooled MOTA / IDF1 of
    the concatenated replicate. 95% CI = 2.5 / 97.5 percentiles of B
    replicates (numpy linear interpolation)."""
    k = n // L
    starts = np.random.default_rng([SEED, L]).integers(0, n - L + 1, size=(B, k))
    cut = np.arange(0, n - L + 1, L)          # the original sequence, cut
    out = {}
    for (a, b) in PAIRS:
        for cls in ("motorcycle", "ALL"):
            ta, tb = tables[(a, cls)], tables[(b, cls)]
            ma, ia = metrics_from_rows(ta[starts].sum(axis=1))
            mb, ib = metrics_from_rows(tb[starts].sum(axis=1))
            ca, cb = metrics_from_rows(ta[cut].sum(0)), metrics_from_rows(tb[cut].sum(0))
            for metric, da, dcut in (("IDF1", ia - ib, ca[1] - cb[1]),
                                     ("MOTA", ma - mb, ca[0] - cb[0])):
                lo, hi = np.percentile(da, [2.5, 97.5])
                out[f"{a}-{b}|{cls}|{metric}"] = {
                    "pair": [a, b], "class": cls, "metric": metric,
                    "block_len": L, "blocks_per_replicate": k, "B": B,
                    "delta_cut": float(dcut), "boot_mean": float(da.mean()),
                    "boot_se": float(da.std(ddof=1)), "ci95": [float(lo), float(hi)],
                    "p_le_0": float((da <= 0).mean()),
                }
    return out


# ── TrackEval cross-check ────────────────────────────────────────────

def trackeval_crosscheck(path: str, data_by_key: dict) -> dict:
    """Same per-frame data through the real TrackEval (HOTA, CLEAR, Identity);
    similarity recomputed with TrackEval's own _calculate_box_ious."""
    sys.path.insert(0, path)
    for alias, typ in {"float": float, "int": int, "bool": bool}.items():
        if not hasattr(np, alias):         # TrackEval @12c8791 predates NumPy 1.24
            setattr(np, alias, typ)
    import trackeval
    from trackeval.datasets._base_dataset import _BaseDataset
    out = {"trackeval_file": trackeval.__file__, "results": {}}
    for key, d in data_by_key.items():
        sims = [_BaseDataset._calculate_box_ious(g, p, box_format="x0y0x1y1")
                for g, p in zip(d["gt_boxes"], d["pr_boxes"])]
        data = hota_data(d, sims)
        te_h = trackeval.metrics.HOTA().eval_sequence(data)
        te_c = trackeval.metrics.CLEAR().eval_sequence(data)
        te_i = trackeval.metrics.Identity().eval_sequence(data)
        mine = hota(hota_data(d))
        out["results"][key] = {
            "max_abs_diff_HOTA_array": float(np.max(np.abs(te_h["HOTA"] - mine["HOTA"]))),
            "max_abs_diff_DetA_array": float(np.max(np.abs(te_h["DetA"] - mine["DetA"]))),
            "max_abs_diff_AssA_array": float(np.max(np.abs(te_h["AssA"] - mine["AssA"]))),
            "max_abs_diff_LocA_array": float(np.max(np.abs(te_h["LocA"] - mine["LocA"]))),
            "max_abs_diff_similarity": float(max(
                (np.max(np.abs(a - b)) if a.size else 0.0)
                for a, b in zip(sims, d["iou"]))),
            "trackeval_HOTA": float(np.mean(te_h["HOTA"])),
            "trackeval_MOTA": float(te_c["MOTA"]), "trackeval_IDSW": int(te_c["IDSW"]),
            "trackeval_Frag": int(te_c["Frag"]), "trackeval_IDF1": float(te_i["IDF1"]),
        }
    return out


# ── Main ─────────────────────────────────────────────────────────────

def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="HOTA, external trackers and "
                                             "block-bootstrap CIs on reviewed GT.")
    ap.add_argument("--review-dir", default=REVIEW_DIR)
    ap.add_argument("--out", default="experiments/external_trackers/metrics.json")
    ap.add_argument("--trackeval", default=None,
                    help="dir with the TrackEval package (pip --target), optional")
    args = ap.parse_args()

    review_dir = REPO_ROOT / args.review_dir
    review = json.loads((review_dir / "review.json").read_text(encoding="utf-8"))
    gt_path = review_dir / "gt_reviewed.txt"
    start, end = SEGMENT
    n = end - start
    assert tuple(review["segment_zero_based_end_exclusive"]) == SEGMENT

    prov = {
        "gt": {"path": str(gt_path.relative_to(REPO_ROOT).as_posix()), "sha256": sha256(gt_path),
               "matches_review_json": sha256(gt_path) == review["reviewed_gt_sha256"]},
        "metrics_py_sha256": sha256(REPO_ROOT / "code/evaluation/metrics.py"),
        "metrics_py_matches_review_json":
            sha256(REPO_ROOT / "code/evaluation/metrics.py") == review["metric_engine_sha256"],
        "dumps": {},
    }
    for v, rel in VARIANTS.items():
        h = sha256(REPO_ROOT / rel)
        prov["dumps"][v] = {"path": rel, "sha256": h,
                            "matches_review_json": (h == review["variants"][v]["sha256"])
                            if v in review["variants"] else None}
    assert prov["gt"]["matches_review_json"], "reviewed GT differs from review.json"

    truth = load_mot(gt_path)
    gt_fl = {c: to_framelist(truth, n, cid) for c, cid in CLASSES.items()}
    coverage = detector_coverage(truth, n)

    table, data, repro = {}, {}, []
    for v, rel in VARIANTS.items():
        pred = tracks_json_slice(REPO_ROOT / rel, start, end)
        for c, cid in CLASSES.items():
            pr_fl = to_framelist(pred, n, cid)
            m = asdict(compute_tracking_metrics(pr_fl, gt_fl[c]))
            d = frame_data(gt_fl[c], pr_fl)
            # The fast replica used by the bootstrap must equal the evaluator.
            n_gt, mi, fp, sw, fr, n_tracks = clear_counts(d, 0, n)
            idtp, _, n_pr = idtp_counts(d, 0, n)
            mota, idf1 = mota_idf1(n_gt, mi, fp, sw, idtp, n_pr)
            assert (mi, fp, sw, fr) == (m["num_misses"], m["num_false_positives"],
                                        m["num_switches"], m["num_fragmentations"])
            assert float(mota) == m["mota"] and float(idf1) == m["idf1"], (v, c)
            if v in REPRO:
                ref = review["variants"][v]["metrics"][c]
                ok = ref == m
                repro.append({"variant": v, "class": c, "ok": ok,
                              "review_json": ref, "recomputed": m})
                assert ok, f"reproduction failed for {v}/{c}: {ref} != {m}"
            table[(v, c)] = {**m, **hota_summary(hota(hota_data(d))),
                             "exact_box_share": exact_box_share(d),
                             "num_gt_dets": n_gt, "num_pred_dets": n_pr}
            data[(v, c)] = d
        print(f"scored {v}", flush=True)

    # Bootstrap: block tables only for what PAIRS needs.
    boot = {}
    needed = sorted({x for p in PAIRS for x in p})
    for L in BLOCKS:
        tables = {(v, c): block_table(data[(v, c)], L)
                  for v in needed for c in ("motorcycle", "ALL")}
        # Spot-check block rows against the repository evaluator itself.
        rng = np.random.default_rng([SEED, L, 1])
        for (v, c), t in tables.items():
            for s in rng.integers(0, n - L + 1, size=2):
                d = data[(v, c)]
                ref = compute_tracking_metrics(
                    [list(zip(d["pr_ids"][k], d["pr_boxes"][k].tolist())) for k in range(s, s + L)],
                    [list(zip(d["gt_ids"][k], d["gt_boxes"][k].tolist())) for k in range(s, s + L)])
                mota, idf1 = metrics_from_rows(t[s])
                assert (t[s, 1], t[s, 2], t[s, 3]) == (ref.num_misses, ref.num_false_positives,
                                                       ref.num_switches), (v, c, L, s)
                assert float(mota) == ref.mota and float(idf1) == ref.idf1, (v, c, L, s)
        # And the shortcut itself: replicate 0, scored literally as one
        # concatenated id-offset sequence by the repository evaluator.
        first = np.random.default_rng([SEED, L]).integers(0, n - L + 1, size=(B, n // L))[0]
        for v in ("ctrv_gate_nsa", "baseline"):
            ref = compute_tracking_metrics(*replicate_framelists(data[(v, "motorcycle")], first, L))
            mota, idf1 = metrics_from_rows(tables[(v, "motorcycle")][first].sum(0))
            assert float(mota) == ref.mota and float(idf1) == ref.idf1, (v, L)
        boot.update({f"L{L}|{k}": r for k, r in block_bootstrap(tables, L, n).items()})
        print(f"bootstrap L={L} done", flush=True)

    cross = trackeval_crosscheck(args.trackeval, {f"{v}|{c}": data[(v, c)] for v, c in data}) \
        if args.trackeval else None

    result = {
        "segment_abs_end_exclusive": list(SEGMENT), "iou_threshold": IOU_MATCH_THRESHOLD,
        "hota_alphas": [round(float(a), 2) for a in ALPHAS],
        "bootstrap": {"B": B, "seed": SEED, "block_lengths": list(BLOCKS),
                      "method": "paired moving-block bootstrap, ids offset per block, "
                                "percentile 95% CI"},
        "provenance": prov, "reproduction": repro,
        "metrics": {f"{v}|{c}": r for (v, c), r in table.items()},
        "full_sequence_deltas": {
            f"{a}-{b}|{c}": {"IDF1": table[(a, c)]["idf1"] - table[(b, c)]["idf1"],
                             "MOTA": table[(a, c)]["mota"] - table[(b, c)]["mota"]}
            for a, b in PAIRS for c in ("motorcycle", "ALL")},
        "block_bootstrap": boot, "trackeval_crosscheck": cross,
        "detector_coverage": coverage,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ── Markdown ──
    print("\n## Reproduction (review.json vs recomputed, all 7 fields)\n")
    for r in repro:
        m = r["recomputed"]
        print(f"- {r['variant']:<14} {r['class']:<10} MOTA {fmt(m['mota'])} IDF1 "
              f"{fmt(m['idf1'])} IDSW {m['num_switches']:>3}  identical: {r['ok']}")
    for c in CLASSES:
        print(f"\n### {c}\n\n| Tracker | MOTA | IDF1 | HOTA | DetA | AssA | IDSW | Frag | FN | FP |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for v in VARIANTS:
            r = table[(v, c)]
            print(f"| {NAMES[v]} | {fmt(r['mota'])} | {fmt(r['idf1'])} | {fmt(r['HOTA'])} | "
                  f"{fmt(r['DetA'])} | {fmt(r['AssA'])} | {r['num_switches']} | "
                  f"{r['num_fragmentations']} | {r['num_misses']} | {r['num_false_positives']} |")
    print("\n## Localisation diagnostic (GT pre-labelled from ctrv_gate_nsa)\n")
    i05, i50, i90 = 0, 9, 17
    assert np.isclose(ALPHAS[[i05, i50, i90]], [0.05, 0.5, 0.9]).all()
    print("| Tracker | Class | LocA | HOTA@0.05 | HOTA@0.50 | HOTA@0.90 | HOTA (avg) "
          "| GT boxes reproduced (IoU>=0.999) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for c in ("motorcycle", "ALL"):
        for v in VARIANTS:
            r = table[(v, c)]
            h = r["per_alpha"]["HOTA"]
            print(f"| {NAMES[v]} | {c} | {fmt(r['LocA'])} | {fmt(h[i05])} | {fmt(h[i50])} | "
                  f"{fmt(h[i90])} | {fmt(r['HOTA'])} | {r['exact_box_share']:.1%} |")
    print("\n## Detector score of the best IoU>=0.5 detection per GT box\n")
    print("| Class | GT boxes | none | " + " | ".join(SCORE_BANDS) + " |")
    print("|---|---:|---:|" + "---:|" * len(SCORE_BANDS))
    for c, r in coverage.items():
        print(f"| {c} | {r['gt_boxes']} | {r['no_detection']:.1%} | "
              + " | ".join(f"{r[k]:.1%}" for k in SCORE_BANDS) + " |")
    print("\n## Paired block bootstrap (difference = first - second)\n")
    print("| Pair | Class | Metric | L | Full-seq diff | Cut diff | Boot mean | 95% CI | P(d<=0) |")
    print("|---|---|---|---:|---:|---:|---:|---|---:|")
    for k, r in boot.items():
        a, b = r["pair"]
        full = result["full_sequence_deltas"][f"{a}-{b}|{r['class']}"][r["metric"]]
        print(f"| {NAMES[a]} - {NAMES[b]} | {r['class']} | {r['metric']} | {r['block_len']} | "
              f"{full:+.4f} | {r['delta_cut']:+.4f} | {r['boot_mean']:+.4f} | "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['p_le_0']:.4f} |")
    if cross:
        worst = max(max(x[k] for k in x if k.startswith("max_abs_diff"))
                    for x in cross["results"].values())
        print(f"\nTrackEval cross-check: max |port - TrackEval| over all arrays = {worst:.3e}")
        for key, x in cross["results"].items():
            v, c = key.split("|")
            r = table[(v, c)]
            print(f"- {key:<26} HOTA {x['trackeval_HOTA']:.6f} (port {r['HOTA']:.6f}) | "
                  f"TrackEval MOTA {x['trackeval_MOTA']:.4f} IDSW {x['trackeval_IDSW']} "
                  f"IDF1 {x['trackeval_IDF1']:.4f} | repo MOTA {r['mota']:.4f} "
                  f"IDSW {r['num_switches']} IDF1 {r['idf1']:.4f}")
    print(f"\nWrote {out.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
