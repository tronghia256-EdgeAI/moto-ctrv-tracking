"""
code/scripts/seed_bracket.py
────────────────────────────
Seed-bias bracket for a run_roi_paper output (motorcycle class).

Each segment's GT pre-labels came from one old tracker (C2 for dense and
freeflow, baseline for mixed). A box a tracker outputs and the GT lacks was
never shown to the reviewer unless that tracker was the seed, and the blind
audit found most such boxes real (experiments/seed_bias/audit_shares.json).
Per comparison X - Y:
  sources  the trackers of the pair that are not seed-like (ctrv_gate_nsa on
           dense/freeflow, baseline on mixed); c1c2c3 enters with its pre-C3
           output, so a box C3 drops still counts against the filtered output
  lower    the GT as it is (favours seed-like trackers)
  upper    the GT plus every source's unmatched boxes (any class, IoU < 0.5 to
           all GT boxes), linked to GT identities by gt_newcam.union_seed
  share    each source's real share: its own audit stratum where one exists,
           else borrowed from the segment's non-seed stratum (recorded); the
           pair's share is their mean, its interval the widest of theirs
  corrected = lower + share * (upper - lower)

    python -m code.scripts.seed_bracket <run dir> <size>
        -> experiments/seed_bias/bracket_<run>_<size>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from code.evaluation.metrics import compute_tracking_metrics
from code.scripts.eval_hota_ci import ALPHAS, frame_data, hota, hota_data
from code.scripts.gt_newcam import union_seed
from code.scripts.import_gt import load_mot, to_framelist, tracks_json_slice

MOTO = 1
A05 = int(np.argmin(np.abs(ALPHAS - 0.5)))
SEED_LIKE = {"dense": "ctrv_gate_nsa", "freeflow": "ctrv_gate_nsa", "mixed": "baseline"}
NON_SEED_STRATUM = {"dense": "baseline", "freeflow": "baseline", "mixed": "ctrv_gate_nsa"}
# c1_ctrv_gate has nsa_noise on in the run's resolved configs, so C1 alone is
# c1_ctrv_gate - ctrv_gate_nsa; c1c2c3 adds C3 on top of it.
EXTERNAL = ("ocsort", "ocsort_matched", "ocsort_byte", "botsort", "botsort_matched")
COMPARISONS = ([("ctrv_gate_nsa", "baseline"), ("c1_ctrv_gate", "ctrv_gate_nsa"), ("c1c2c3", "c1c2c3_raw"),
                ("c1c2c3", "baseline")] + [(e, "ctrv_gate_nsa") for e in EXTERNAL] + [("ocsort_matched", "baseline")])


def sources_and_shares(seg: str, x: str, y: str, audit: dict) -> dict:
    """{source: share record} for comparison x - y in segment seg (rule in the module doc)."""
    sources = sorted({"c1c2c3_raw" if t == "c1c2c3" else t for t in (x, y) if t != SEED_LIKE[seg]})
    out = {}
    for t in sources:
        own = audit.get((seg, t))
        st = own or audit[(seg, NON_SEED_STRATUM[seg])]
        out[t] = dict(share=st["share"], wilson95=st["wilson95"],
                      borrowed_from=None if own else f"{seg}/{NON_SEED_STRATUM[seg]}/extra")
    return out


def dump(run: Path, size: str, name: str) -> Path:
    return run / size / ("raw_tracks_c1c2c3.json" if name == "c1c2c3_raw" else f"tracks_{name}.json")


def as_list(frames: dict, n: int) -> list:
    return [[(tid, cls, box) for tid, box, cls in frames.get(k, [])] for k in range(1, n + 1)]


def metrics(truth: dict, preds: dict, n: int) -> dict:
    gt, pr = to_framelist(truth, n, MOTO), to_framelist(preds, n, MOTO)
    m = compute_tracking_metrics(pr, gt)
    return dict(idf1=m.idf1, mota=m.mota, idsw=m.num_switches, fn=m.num_misses, fp=m.num_false_positives,
                hota05=float(hota(hota_data(frame_data(gt, pr)))["HOTA"][A05]))


def main() -> None:
    run, size = Path(sys.argv[1]), sys.argv[2]
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    audit = {(s["segment"], s["tracker"]): s for s in json.loads(
        Path("experiments/seed_bias/audit_shares.json").read_text(encoding="utf-8"))["strata"] if s["kind"] == "extra"}
    out = dict(run=str(run), size=int(size), method=__doc__.split("\n\n")[2].strip(), segments={})
    for seg, item in protocol["segments"].items():
        (start, end), n = item["segment"], item["segment"][1] - item["segment"][0]
        truth = load_mot(item["gt"])
        preds = {t: tracks_json_slice(dump(run, size, t), start, end) for pair in COMPARISONS for t in pair}
        rows = {}
        for x, y in COMPARISONS:
            shares = sources_and_shares(seg, x, y, audit)
            aug, merged = union_seed([as_list(truth, n)] + [as_list(preds[t], n) for t in shares], 0, n)
            aug = {k + 1: [(tid, box, cls) for tid, cls, box in f] for k, f in enumerate(aug)}
            s_mid = float(np.mean([v["share"] for v in shares.values()]))
            s_lo = min(v["wilson95"][0] for v in shares.values())
            s_hi = max(v["wilson95"][1] for v in shares.values())
            lower = {t: metrics(truth, preds[t], n) for t in (x, y)}
            upper = {t: metrics(aug, preds[t], n) for t in (x, y)}
            row = dict(sources=list(shares), shares=shares, added_boxes=merged["linked"] + merged["fresh_boxes"],
                       lower=lower, upper=upper, gap={})
            for key in ("idf1", "hota05", "mota", "idsw"):
                g_lo, g_up = lower[x][key] - lower[y][key], upper[x][key] - upper[y][key]
                row["gap"][key] = dict(lower=g_lo, upper=g_up,
                                       corrected=[g_lo + s * (g_up - g_lo) for s in (s_mid, s_lo, s_hi)])
            rows[f"{x} - {y}"] = row
            c = row["gap"]["idf1"]["corrected"]
            print(f"{seg:<9}{x + ' - ' + y:<32} IDF1 lower {row['gap']['idf1']['lower']:+.4f} "
                  f"upper {row['gap']['idf1']['upper']:+.4f} corrected {c[0]:+.4f} "
                  f"[{min(c[1:]):+.4f}, {max(c[1:]):+.4f}]  IDSW {lower[x]['idsw']}/{lower[y]['idsw']}")
        out["segments"][seg] = dict(seed_like=SEED_LIKE[seg], segment=[start, end], comparisons=rows)
    dest = Path(f"experiments/seed_bias/bracket_{run.name}_{size}.json")
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
