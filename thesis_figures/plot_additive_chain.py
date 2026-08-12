#!/usr/bin/env python
"""Render the additive-chain figure for Results 4.4 (fig:additive-chain).

A descending step plot of EMDB PA-MPJPE across the four cumulative changes that
turn a non-working cross-entropy configuration into the best model in the thesis.
Every arm is a from-scratch 200k-step run on one identical schedule, so the
deltas annotated between the steps are attributable to the change named.

The pose-supervised baseline (Table 4.7, 57.06 mm) is drawn as a horizontal
reference line: the chain crosses it at the second step, which is the single
most important thing the figure has to show.

Numbers are hard-coded from Tables 4.7/4.11/4.10 in experimentsandresults.tex --
if a cell there changes, update the STEPS list below.

Run with the project env (from the tokenhmr folder):
    ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/plot_additive_chain.py

Output: <repo>/thesis/images/downstream/additive_chain.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/tokenhmr/thesis_figures
REPO = os.path.dirname(os.path.dirname(HERE))              # <repo>
OUTDIR = os.path.join(REPO, "thesis", "images", "downstream")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 10.5,
    "axes.linewidth": 0.8, "pdf.fonttype": 42,
})

# (label over two lines, EMDB PA-MPJPE [mm], run tag, measured?)
# All four entries are now MEASURED (run_additive_queue.sh phases A-E, finished 2026-08-11).
# NOTE: this figure was removed from the thesis on 2026-08-11 (the table says it more
# cleanly); the script is kept in sync with tab:additive-chain but is no longer referenced.
STEPS = [
    ("Token CE only\n(no gate)",        75.55, "chain_nogate",  True),
    ("$+$ label-purity\ngate",          54.37, "chain_gate",    True),
    ("$+$ straight-through\ndecode",    52.28, "chain_st",      True),
    ("$+$ Gumbel\nsampling",            52.08, "chain_gumbel",  True),
]

POSE_BASELINE = 57.06        # Table 4.7, pose losses only, EMDB PA-MPJPE
LINE = "#0072B2"             # Okabe-Ito blue
DROP = "#009E73"             # Okabe-Ito green (improvements)
REF = "#D55E00"              # Okabe-Ito vermillion (baseline line)

fig, ax = plt.subplots(figsize=(6.6, 3.9))

xs = list(range(len(STEPS)))
ys = [s[1] for s in STEPS]

# horizontal pose-supervised reference
ax.axhline(POSE_BASELINE, color=REF, lw=1.1, ls=(0, (5, 3)), zorder=1)
ax.text(len(STEPS) - 0.52, POSE_BASELINE + 0.9,
        "pose-supervised baseline (57.06)", color=REF, fontsize=8.6,
        ha="right", va="bottom")

# the descending staircase: a flat tread at each level, a riser between them
for i, (x, y) in enumerate(zip(xs, ys)):
    ax.plot([x - 0.30, x + 0.30], [y, y], color=LINE, lw=2.6,
            solid_capstyle="round", zorder=3)
    if i > 0:
        ax.plot([x - 0.30, x - 0.30], [ys[i - 1], y], color=LINE, lw=1.0,
                ls=":", zorder=2)

# value labels on each tread; projected values are red, as in the thesis tables
for x, y, meas in zip(xs, ys, [s[3] for s in STEPS]):
    ax.text(x, y + 1.1, f"{y:.2f}", ha="center", va="bottom",
            fontsize=10, color=(LINE if meas else "#D62728"))

# delta annotations on the risers
for i in range(1, len(STEPS)):
    dy = ys[i] - ys[i - 1]
    ymid = 0.5 * (ys[i] + ys[i - 1])
    ax.annotate("", xy=(i - 0.30, ys[i]), xytext=(i - 0.30, ys[i - 1]),
                arrowprops=dict(arrowstyle="-|>", color=DROP, lw=1.4,
                                shrinkA=0, shrinkB=0))
    # left of the riser: the value labels are centred on the tread, so anything
    # placed to the right of a short riser collides with them.
    ax.text(i - 0.37, ymid, f"$-{abs(dy):.2f}$ mm", color=DROP,
            fontsize=9.4, ha="right", va="center")

ax.set_xticks(xs)
ax.set_xticklabels([s[0] for s in STEPS], fontsize=9.3)
ax.set_xlim(-0.55, len(STEPS) - 0.45)
ax.set_ylim(45.5, 81.0)
ax.set_ylabel("EMDB PA-MPJPE [mm]  (lower is better)")
ax.tick_params(axis="x", length=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="0.88", lw=0.6, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
out = os.path.join(OUTDIR, "additive_chain.pdf")
fig.savefig(out)
fig.savefig(os.path.join(HERE, "additive_chain_preview.png"), dpi=150)
print("wrote", out)
