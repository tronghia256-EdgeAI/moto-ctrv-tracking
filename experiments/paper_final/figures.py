"""Figures for the JRTIP manuscript, drawn only from committed result files.

    python experiments/paper_final/figures.py  ->  paper/jrtip/figures/*.pdf

Springer svjour3 sizes: one column 84 mm, two columns 174 mm. Times-like serif at
8 pt to match the body, vector PDF with TrueType fonts embedded, Okabe-Ito colours
plus distinct markers so that the figures also read in greyscale.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/jrtip/figures'
SIZES = (448, 640, 960, 1280)
SEGMENTS = ('dense', 'mixed', 'freeflow')
SEG_LABEL = {'dense': 'Dense', 'mixed': 'Mixed', 'freeflow': 'Free-flow'}
SEG_STYLE = {'dense': ('#0072B2', 'o'), 'mixed': ('#D55E00', 's'), 'freeflow': ('#009E73', '^')}
ARM_STYLE = {'baseline': ('ByteTrack', '#000000', 'o', '-'),
             'ctrv_gate_nsa': ('C2', '#0072B2', 's', '-'),
             'ocsort_matched': ('OC-SORT (0.5)', '#D55E00', '^', '--')}
MM = 1 / 25.4
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 4,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'axes.spines.top': False, 'axes.spines.right': False})


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def bracket(size):
    return load(f'experiments/seed_bias/bracket_best85_roi_final_{size}.json')['segments']


def corrected(b, seg, pair):
    """Seed-corrected IDF1 gap: (mid, low, high) over the audit-share interval."""
    mid, a, c = b[seg]['comparisons'][pair]['gap']['idf1']['corrected']
    return mid, min(a, c), max(a, c)


def size_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.minorticks_off()
    ax.set_xlabel('Input size (px)')


def fig_sizes():
    """Dense ID switches, and the seed-corrected IDF1 gaps C2-ByteTrack and OC-SORT-C2."""
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 42 * MM))
    ax = axes[0]
    for arm, (label, colour, marker, style) in ARM_STYLE.items():
        idsw = [load(f'experiments/paper_final/best85_roi_final/{s}/metrics.json')['segments']['dense']['metrics'][arm]
                ['motorcycle']['num_switches'] for s in SIZES]
        ax.plot(SIZES, idsw, color=colour, marker=marker, linestyle=style, label=label)
    size_axis(ax)
    ax.set_ylabel('Identity switches')
    ax.set_title('(a) Dense, motorcycle identity switches', loc='left')
    ax.set_ylim(0, 165)
    ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.02, 0.0), fontsize=6.5,
              handlelength=1.6, labelspacing=0.25, borderaxespad=0.2)
    for ax, pair, title in ((axes[1], 'ctrv_gate_nsa - baseline', '(b) C2 $-$ ByteTrack, IDF1'),
                            (axes[2], 'ocsort_matched - ctrv_gate_nsa', '(c) OC-SORT (0.5) $-$ C2, IDF1')):
        for k, seg in enumerate(SEGMENTS):
            colour, marker = SEG_STYLE[seg]
            g = [corrected(bracket(s), seg, pair) for s in SIZES]
            x = [s * (1 + 0.035 * (k - 1)) for s in SIZES]
            ax.errorbar(x, [m for m, _, _ in g], yerr=[[m - lo for m, lo, _ in g], [hi - m for m, _, hi in g]],
                        color=colour, marker=marker, linestyle='-', capsize=1.5, elinewidth=0.6, label=SEG_LABEL[seg])
        ax.axhline(0, color='0.5', linewidth=0.6, zorder=0)
        size_axis(ax)
        ax.set_ylabel('Seed-corrected $\\Delta$IDF1')
        ax.set_title(title, loc='left')
    axes[1].legend(frameon=False, loc='upper left')
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / 'fig_input_size.pdf')


def fig_seed_bias(size=448):
    """Lower (GT as is), upper (GT plus the non-seed trackers' unmatched boxes) and
    corrected IDF1 gaps at the Table 2 size: how far the seed moves each comparison."""
    b = bracket(size)
    groups = (('ocsort_matched - ctrv_gate_nsa', 'OC-SORT (0.5) $-$ C2'), ('ctrv_gate_nsa - baseline', 'C2 $-$ ByteTrack'))
    fig, ax = plt.subplots(figsize=(84 * MM, 47 * MM))
    ticks, labels, y = [], [], 0
    for pair, name in groups:
        for seg in reversed(SEGMENTS):
            g = b[seg]['comparisons'][pair]['gap']['idf1']
            lo, up = g['lower'], g['upper']
            mid, a, c = corrected(b, seg, pair)
            first = not ticks
            ax.plot([lo, up], [y, y], color='0.6', linewidth=0.8, zorder=1)
            ax.plot(lo, y, marker='o', markerfacecolor='white', markeredgecolor='black', linestyle='none', zorder=2,
                    label='GT as annotated' if first else None)
            ax.plot(up, y, marker='s', markerfacecolor='white', markeredgecolor='#D55E00', linestyle='none', zorder=2,
                    label='GT + non-seed boxes' if first else None)
            ax.errorbar(mid, y, xerr=[[mid - a], [c - mid]], marker='D', color='#0072B2', markersize=3.5, capsize=1.5,
                        elinewidth=0.8, linestyle='none', zorder=3, label='Seed-corrected' if first else None)
            ticks.append(y); labels.append(SEG_LABEL[seg]); y += 1
        ax.text(-0.004, y - 0.35, name, fontsize=7, fontstyle='italic', ha='left', va='bottom')
        y += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, y - 0.3)
    ax.set_xlim(-0.005, None)
    ax.axvline(0, color='0.5', linewidth=0.6, zorder=0)
    ax.set_xlabel(f'IDF1 difference at {size} px')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=3, handletextpad=0.2,
              columnspacing=0.6, fontsize=6.5)
    fig.savefig(OUT / 'fig_seed_bias.pdf')


def fig_operating():
    """Accuracy cost of the operating point (replaces the old Table 4): (a) C2's motorcycle
    IDF1 per segment and (b) ByteTrack's dense miss rate by ground-truth box height, over
    the real-time detector's input sizes; hollow markers at 1,280 px: reference detector."""
    ops = load('experiments/operating_point/operating_points.json')['operating_points']
    ref = ops['yolo11_vn_1280_roi']
    fig, axes = plt.subplots(1, 2, figsize=(84 * MM, 44 * MM))
    ax = axes[0]
    for seg in SEGMENTS:
        colour, marker = SEG_STYLE[seg]
        ax.plot(SIZES, [ops[f'best_{s}'][seg]['ctrv_gate_nsa']['motorcycle']['idf1'] for s in SIZES],
                color=colour, marker=marker, label=SEG_LABEL[seg])
        ax.plot(1280 * 1.06, ref[seg]['ctrv_gate_nsa']['motorcycle']['idf1'], marker=marker, linestyle='none',
                markerfacecolor='white', markeredgecolor=colour)
    size_axis(ax)
    ax.set_ylabel('C2 motorcycle IDF1')
    ax.set_ylim(0.45, 1.0)
    ax.set_title('(a)', loc='left')
    ax.legend(frameon=False, loc='upper left', handlelength=1.2)
    ax = axes[1]
    bins = (('0-20', 'height $<$20 px', 'o', '#000000', (800, 0.83)),
            ('20-40', '20$-$40 px', 's', '#0072B2', (760, 0.31)),
            ('40-10000', '$\geq$40 px', '^', '#D55E00', (760, 0.03)))
    for key, label, marker, colour, (tx, ty) in bins:
        ax.plot(SIZES, [ops[f'best_{s}']['dense']['baseline']['motorcycle']['miss_by_height'][key]['rate']
                        for s in SIZES], marker=marker, color=colour)
        ax.plot(1280 * 1.06, ref['dense']['baseline']['motorcycle']['miss_by_height'][key]['rate'], marker=marker,
                linestyle='none', markerfacecolor='white', markeredgecolor=colour)
        ax.text(tx, ty, label, color=colour, fontsize=6.5, ha='center')
    size_axis(ax)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel('Dense miss rate')
    ax.set_title('(b)', loc='left')
    for a in axes:
        a.set_xlabel('Input size (px)', fontsize=7)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / 'fig_operating.pdf')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_sizes()
    fig_seed_bias()
    fig_operating()
    print('written:', sorted(p.name for p in OUT.glob('fig_*.pdf')))
