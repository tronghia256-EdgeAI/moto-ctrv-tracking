"""
code/scripts/audit_shares.py
────────────────────────────
Per-stratum GT error shares from the completed blind audit (paper_gt_audit),
with Wilson 95% intervals, and the errors expected among the runs the audit
did not sample. Those are still in the reconciled GT, which fixed sampled
cases only.

    python -m code.scripts.audit_shares   -> experiments/seed_bias/audit_shares.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from code.scripts.audit_disagreements import wilson
from code.scripts.paper_gt_audit import gt_issue

AUDIT = Path("data/annotations/paper_gt_audit/roi_241dc106947b1723")
SEED = {"dense": "ctrv_gate_nsa", "freeflow": "ctrv_gate_nsa", "mixed": "baseline"}


def main() -> None:
    keys = json.loads((AUDIT / "key.json").read_text(encoding="utf-8"))
    verdict = {r["n"]: r["verdict"] for r in csv.DictReader((AUDIT / "verdicts.csv").open(encoding="utf-8"))}
    strata = json.loads((AUDIT / "protocol.json").read_text(encoding="utf-8"))["strata"]
    if not all(verdict.get(k["n"]) for k in keys):
        raise SystemExit("the audit is not complete")
    rows = []
    for s in strata:
        v = [verdict[k["n"]] for k in keys if (k["segment"], k["tracker"], k["kind"]) == (s["segment"], s["tracker"], s["kind"])]
        judged = [x for x in v if x != "?"]
        wrong = sum(gt_issue(s["kind"], x) for x in judged)
        share, (lo, hi) = wrong / max(1, len(judged)), wilson(wrong, len(judged))
        left = s["available"] - s["sampled"]
        rows.append(dict(**s, seed=SEED[s["segment"]] == s["tracker"], counts={x: v.count(x) for x in sorted(set(v))},
                         judged=len(judged), gt_wrong=wrong, share=round(share, 3), wilson95=[lo, hi],
                         unsampled=left, expected_left=[round(lo * left), round(share * left), round(hi * left)]))
    out = Path("experiments/seed_bias/audit_shares.json")
    out.write_text(json.dumps(dict(source=str(AUDIT), rule="gt_issue as in paper_gt_audit.py", seed_tracker=SEED,
                                   strata=rows), indent=1, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        print(f"{r['segment']:<9}{r['tracker']:<15}{r['kind']:<9}{'seed' if r['seed'] else '':<5}"
              f"{r['gt_wrong']}/{r['judged']} share {r['share']:.2f} {r['wilson95']} left {r['expected_left']}")


if __name__ == "__main__":
    main()
