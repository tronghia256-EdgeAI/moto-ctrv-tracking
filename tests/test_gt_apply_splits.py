"""
tests/test_gt_apply_splits.py
──────────────────────────────
Tests for the SPLIT-verdict applier.

This module rewrites ground truth, so the properties worth pinning are the
conservative ones: it must never lose a box, never touch an identity nobody
voted on, and never silently invent a verdict for a blank row.
"""

from __future__ import annotations

import csv

import pytest

from code.scripts.gt_apply_splits import (CUT, KEEP, apply_cuts,
                                          load_decisions, unreviewed_cuts)


def write_gt(path, rows):
    """rows: (frame, track_id, class1based). Boxes are arbitrary but distinct."""
    with open(path, "w", encoding="utf-8") as f:
        for frame, tid, cls in rows:
            f.write(f"{frame},{tid},{10*tid},{20*frame},30,40,1,{cls},1\n")
    return path


def ids_by_frame(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            out[(int(p[0]), int(p[2]) // 10)] = int(p[1])   # keyed by orig tid
    return out


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [int(l.split(",")[1]) for l in f if l.strip()]


@pytest.fixture
def gt(tmp_path):
    # Identity 1 spans frames 1-6, identity 2 spans 1-3, identity 3 spans 4-6.
    rows = ([(f, 1, 2) for f in range(1, 7)]
            + [(f, 2, 2) for f in range(1, 4)]
            + [(f, 3, 1) for f in range(4, 7)])
    return write_gt(tmp_path / "gt.txt", rows)


class TestApplyCuts:
    def test_no_cuts_is_a_faithful_copy(self, gt, tmp_path):
        out = tmp_path / "out.txt"
        stats = apply_cuts(gt, [], out)
        assert stats["ids_before"] == stats["ids_after"] == 3
        assert stats["identities_touched"] == 0
        assert out.read_text(encoding="utf-8") == gt.read_text(encoding="utf-8")

    def test_cut_splits_one_identity_at_the_frame(self, gt, tmp_path):
        out = tmp_path / "out.txt"
        stats = apply_cuts(gt, [(1, 4)], out)          # cut identity 1 at f4
        assert stats["ids_before"] == 3
        assert stats["ids_after"] == 4                  # exactly one new id
        assert stats["boxes"] == 12                     # nothing lost

        seen = ids_by_frame(out)
        head = {seen[(f, 1)] for f in (1, 2, 3)}
        tail = {seen[(f, 1)] for f in (4, 5, 6)}
        assert head == {1}, "frames before the cut keep the original id"
        assert len(tail) == 1 and tail != head, "frames from the cut on move"

    def test_untouched_identities_keep_their_ids(self, gt, tmp_path):
        out = tmp_path / "out.txt"
        apply_cuts(gt, [(1, 4)], out)
        seen = ids_by_frame(out)
        assert {seen[(f, 2)] for f in (1, 2, 3)} == {2}
        assert {seen[(f, 3)] for f in (4, 5, 6)} == {3}

    def test_two_cuts_give_three_segments(self, gt, tmp_path):
        out = tmp_path / "out.txt"
        stats = apply_cuts(gt, [(1, 3), (1, 5)], out)
        seen = ids_by_frame(out)
        segs = [{seen[(f, 1)] for f in fs} for fs in ((1, 2), (3, 4), (5, 6))]
        assert all(len(s) == 1 for s in segs)
        assert len({next(iter(s)) for s in segs}) == 3
        assert stats["ids_after"] == 5                  # 3 + 2 new

    def test_new_ids_never_collide_with_existing_ones(self, gt, tmp_path):
        out = tmp_path / "out.txt"
        apply_cuts(gt, [(1, 4), (2, 2)], out)
        ids = read_ids(out)
        assert len(ids) == 12
        # Every id still maps to a contiguous, single-purpose identity: no id
        # may carry boxes from two different original tracks.
        assert len(set(ids)) == 5


class TestDecisionSheet:
    def _sheet(self, tmp_path, decisions):
        p = tmp_path / "decisions.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "sheet", "gt_id", "cut_at_frame", "gap",
                        "move_diag", "area_ratio", "decision", "note"])
            for i, d in enumerate(decisions, start=1):
                w.writerow([i, f"{i:03d}.png", i, i + 10, 9, 1.0, 1.0, d, ""])
        return p

    def test_only_cut_rows_become_cuts(self, tmp_path):
        p = self._sheet(tmp_path, [CUT, KEEP, CUT])
        cuts, tally = load_decisions(p)
        assert tally == {CUT: 2, KEEP: 1, "BLANK": 0}
        assert cuts == [(1, 11), (3, 13)]

    def test_blank_row_is_refused_rather_than_read_as_keep(self, tmp_path):
        p = self._sheet(tmp_path, [CUT, "", KEEP])
        with pytest.raises(SystemExit, match="blank"):
            load_decisions(p)

    def test_unrecognised_verdict_is_refused(self, tmp_path):
        p = self._sheet(tmp_path, [CUT, "maybe", KEEP])
        with pytest.raises(SystemExit, match="not CUT or KEEP"):
            load_decisions(p)

    def test_verdicts_are_case_insensitive(self, tmp_path):
        p = self._sheet(tmp_path, ["cut", " keep ", "Cut"])
        cuts, tally = load_decisions(p)
        assert tally == {CUT: 2, KEEP: 1, "BLANK": 0}

    def test_partial_mode_reads_a_half_filled_sheet(self, tmp_path):
        p = self._sheet(tmp_path, [CUT, "", KEEP, ""])
        cuts, tally = load_decisions(p, allow_partial=True)
        assert tally == {CUT: 1, KEEP: 1, "BLANK": 2}
        assert cuts == [(1, 11)]

    def test_unreviewed_cuts_are_exactly_the_blanks(self, tmp_path):
        p = self._sheet(tmp_path, [CUT, "", KEEP, ""])
        # The two ends of the bound differ by precisely these, so a blank must
        # never leak into the decided set nor a decided row into the pending
        # one -- otherwise the bracket stops being a bracket.
        assert unreviewed_cuts(p) == [(2, 12), (4, 14)]

    def test_the_two_ends_partition_every_candidate(self, tmp_path):
        p = self._sheet(tmp_path, [CUT, "", KEEP, ""])
        decided, tally = load_decisions(p, allow_partial=True)
        pending = unreviewed_cuts(p)
        assert tally[CUT] + tally[KEEP] + tally["BLANK"] == 4
        assert len(pending) == tally["BLANK"]
        assert not set(decided) & set(pending)
