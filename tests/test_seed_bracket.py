"""Which trackers' boxes enter the seed-bias bracket, and whose audit share applies."""

from code.scripts.seed_bracket import sources_and_shares

AUDIT = {("dense", "baseline"): dict(share=0.8, wilson95=[0.49, 0.94]),
         ("mixed", "ctrv_gate_nsa"): dict(share=0.8, wilson95=[0.49, 0.94])}


def test_seed_like_trackers_add_nothing_and_missing_strata_are_borrowed():
    assert list(sources_and_shares("dense", "ctrv_gate_nsa", "baseline", AUDIT)) == ["baseline"]
    s = sources_and_shares("dense", "c1c2c3", "baseline", AUDIT)
    assert list(s) == ["baseline", "c1c2c3_raw"]                     # C3's pre-filter output, not the filtered one
    assert s["baseline"]["borrowed_from"] is None
    assert s["c1c2c3_raw"]["borrowed_from"] == "dense/baseline/extra"
    s = sources_and_shares("mixed", "ocsort_matched", "ctrv_gate_nsa", AUDIT)
    assert list(s) == ["ctrv_gate_nsa", "ocsort_matched"]            # baseline is the seed-like one on mixed
    assert s["ocsort_matched"]["borrowed_from"] == "mixed/ctrv_gate_nsa/extra"
