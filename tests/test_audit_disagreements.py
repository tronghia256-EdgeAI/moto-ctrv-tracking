"""Run grouping, IoU matching and switch replication behind the disagreement audit."""

from code.evaluation.metrics import compute_tracking_metrics
from code.scripts.audit_disagreements import runs, stratum, switch_events, unmatched


def test_unmatched_keeps_only_boxes_without_an_iou_partner():
    a = [(1, [0, 0, 10, 10]), (2, [50, 50, 60, 60])]
    assert unmatched(a, [(9, [1, 1, 11, 11])]) == [1]
    assert unmatched(a, []) == [0, 1]


def test_runs_break_on_frame_gaps_and_id_changes():
    b = [0, 0, 1, 1]
    fl = [[(1, b), (2, b)], [(1, b)], [], [(1, b)]]
    found = runs(fl, [[0, 1], [0], [], [0]])
    assert sorted((r[0][1], len(r)) for r in found) == [(1, 1), (1, 2), (2, 1)]


def test_switch_events_match_the_metric_including_a_gap():
    a, far = [0, 0, 10, 10], [100, 100, 110, 110]
    gt = [[(1, a)], [(1, a)], [(1, a)], [(1, a)]]
    pr = [[(10, a)], [(10, far)], [(11, a)], [(11, a)]]     # frame 1 unmatched, new ID after the gap
    assert switch_events(gt, pr) == [(0, 2, 1, a, a)]
    assert compute_tracking_metrics(pr, gt).num_switches == 1


def test_stratum_stops_early_only_on_a_decided_gap_free_prefix():
    rows = [dict(n=f"S{i}", verdict="Y" if i <= 20 else "") for i in range(1, 26)]
    s = stratum(rows, lambda x: x == "Y")
    assert (s["status"], s["judged"], s["prefix"]) == ("stopped", 20, True)
    rows[0]["verdict"] = ""                                  # a blank before filled verdicts
    s = stratum(rows, lambda x: x == "Y")
    assert (s["status"], s["prefix"]) == ("open", False)
    split = [dict(n=str(i), verdict="YN"[i % 2] if i <= 24 else "") for i in range(1, 31)]
    assert stratum(split, lambda x: x == "Y")["status"] == "open"   # 12/24: undecided, keep going
