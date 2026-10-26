"""
code/scripts/make_paper_figures.py
──────────────────────────────────
Data figures for the JRTIP manuscript (paper/jrtip/figures/*.pdf).

Every number is copied from an evidence file, named next to it; nothing is
recomputed here. Re-run after any of those files changes:

  python -m code.scripts.make_paper_figures

Fonts are embedded as TrueType (pdf.fonttype 42): journal production
rejects Type 3 fonts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "paper" / "jrtip" / "figures"

# Validated categorical slots (dataviz reference palette, light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WIDTH = 5.1  # in; sn-jnl text width, used by the figures of the long version
COL = 3.3    # in; JRTIP double-column (svjour3) column width is ~3.3 in

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_c1_sweep() -> None:
    """MOTA vs the DENSE-PEAK confirmation threshold, both buffer lengths."""
    rows = json.loads((REPO / "experiments/c1_sweep/sweep.json")
                      .read_text(encoding="utf-8"))["rows"]
    off = next(r for r in rows if not r["c1"])["all"]["mota"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    # the two end points are 0.013 apart: nudge the labels apart vertically
    for buf, color, dy in ((30, BLUE, 6), (45, AQUA, -6)):
        pts = sorted((r["conf"], r["all"]["mota"]) for r in rows
                     if r["c1"] and r["buffer"] == buf)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=1.6, marker="o", ms=4.5,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(f"C1 on, $L_{{\\mathrm{{buf}}}}$ = {buf}", (xs[-1], ys[-1]),
                    xytext=(6, dy), textcoords="offset points", va="center",
                    color=INK)
    ax.axhline(off, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.annotate(f"C1 off (fixed 0.50 / 30): {off:.3f}", (0.35, off),
                xytext=(0, 4), textcoords="offset points", color=INK2)
    shipped = next(r for r in rows if r["c1"] and r["conf"] == 0.35
                   and r["buffer"] == 45)["all"]["mota"]
    ax.annotate("shipped row\n(0.35 / 45)", (0.35, shipped), xytext=(14, -2),
                textcoords="offset points", va="center", color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlabel(r"DENSE-PEAK confirmation threshold $c_{\mathrm{trk}}$")
    ax.set_ylabel("MOTA (all classes)")
    ax.set_xlim(0.33, 0.68)
    ax.set_ylim(0.70, 0.94)
    save(fig, "fig_c1_sweep.pdf")


def fig_c3_diagnostics() -> None:
    """(a) MOTA vs tau_obs; (b) share of dropped track-frames that are FP."""
    # experiments/c3_sweep/evidence.md, section 1 (full tau sweep).
    tau = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
    mota = [0.905, 0.895, 0.892, 0.885, 0.861, 0.814, 0.705]
    diag = json.loads((REPO / "experiments/c3_sweep/sweep.json")
                      .read_text(encoding="utf-8"))["diagnostics"]
    enr = sorted((float(k), v["fp_share"]) for k, v in diag["enrichment"].items())
    base = diag["fp_base_rate"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.2))
    a.plot(tau, mota, color=BLUE, lw=1.6, marker="o", ms=4.5,
           markeredgecolor="white", markeredgewidth=0.8)
    a.set_xlabel(r"observation threshold $\tau_{\mathrm{obs}}$")
    a.set_ylabel("MOTA (all classes)")
    a.set_title("(a) filtering the observation branch", loc="left",
                fontsize=8, color=INK)

    xs, ys = zip(*enr)
    b.bar(xs, [100 * y for y in ys], width=0.035, color=BLUE,
          edgecolor="white", linewidth=1.0)
    for x, y in enr:
        b.annotate(f"{100 * y:.1f}", (x, 100 * y), xytext=(0, 2),
                   textcoords="offset points", ha="center", color=INK,
                   fontsize=7)
    b.axhline(100 * base, color=INK2, lw=1.0, ls=(0, (4, 3)))
    b.text(0.02, 0.97, f"dashed: FP base rate ({100 * base:.1f}%)",
           transform=b.transAxes, va="top", color=INK2, fontsize=7)
    b.set_xlabel(r"threshold $\tau$")
    b.set_ylabel("FP share of dropped (%)")
    b.set_ylim(0, 24)
    b.set_title("(b) what a threshold removes", loc="left", fontsize=8,
                color=INK)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_c3_diagnostics.pdf")


def fig_realtime() -> None:
    """Per-frame latency: detector vs rest of pipeline, 40 ms budget."""
    # experiments/realtime/evidence.md: section 1 (deployed, segmentation
    # head) and section 4 (detection head, PyTorch). Same machine, one
    # process per row, warm-up discarded.
    rows = [  # label, detect ms, total ms, fps
        ("Seg. head, 640 px", 130.72, 143.24, 7.0),
        ("Det. head, 640 px", 58.20, 62.89, 15.9),
        ("Det. head, 448 px", 24.33, 26.57, 37.6),
        ("Det. head, 320 px", 17.33, 19.47, 51.4),
        ("Det. head, 256 px", 14.24, 16.29, 61.4),
    ]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    ys = list(range(len(rows)))[::-1]
    for y, (label, det, tot, fps) in zip(ys, rows):
        ax.barh(y, det, color=BLUE, height=0.6, edgecolor="white", lw=1.0)
        ax.barh(y, tot - det, left=det, color=ORANGE, height=0.6,
                edgecolor="white", lw=1.0)
        # start labels right of the budget line so it never crosses text
        ax.annotate(f"{tot:.1f} ms, {fps:.1f} fps", (max(tot, 40), y),
                    xytext=(3, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=6.5)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.axvline(40, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.annotate("40 ms budget", (40, ys[0] + 0.45), xytext=(3, 0),
                textcoords="offset points", va="bottom", color=INK2,
                fontsize=6.5)
    ax.set_ylim(-0.5, ys[0] + 1.0)
    ax.set_xlabel("per-frame latency (ms), i5-12500H CPU")
    ax.set_xlim(0, 215)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["detector", "other stages"], loc="lower right",
              fontsize=6.5)
    save(fig, "fig_realtime.pdf")


def fig_gt_versions() -> None:
    """Motorcycle IDF1 of baseline and C2 under each version of the GT."""
    # experiments/gt_qa/sensitivity.md (heuristic splits) and the reviewed
    # run experiments/gt_qa/reviewed/20261007T161301Z_126e0b66/summary.md.
    rows = [  # label, baseline, C2
        ("As annotated (335 IDs)", 0.844, 0.933),
        ("Heuristic, 59 cuts", 0.855, 0.916),
        ("Human review, 92 cuts", 0.858, 0.907),
        ("Heuristic, 105 cuts", 0.861, 0.898),
        ("Heuristic, 147 cuts", 0.866, 0.880),
    ]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ys = list(range(len(rows)))[::-1]
    for y, (label, base, c2) in zip(ys, rows):
        ax.plot([base, c2], [y, y], color=GRID, lw=2.0, zorder=1)
        ax.scatter([base], [y], s=30, color=ORANGE, zorder=2,
                   edgecolors="white", linewidths=0.8)
        ax.scatter([c2], [y], s=30, color=BLUE, zorder=2,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(f"+{c2 - base:.3f}", (max(base, c2), y), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK,
                    fontsize=6.5)
    ax.set_yticks(ys, [r[0] for r in rows])
    for t in ax.get_yticklabels():
        if t.get_text().startswith("Human"):
            t.set_fontweight("bold")
    ax.set_xlabel("motorcycle IDF1")
    ax.set_xlim(0.835, 0.965)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", color=ORANGE),
                       plt.Line2D([], [], ls="", marker="o", color=BLUE)],
              labels=["baseline (ByteTrack)", "C2"], loc="lower right",
              fontsize=6.5)
    save(fig, "fig_gt_versions.pdf")


if __name__ == "__main__":
    fig_c1_sweep()
    fig_c3_diagnostics()
    fig_realtime()
    fig_gt_versions()
