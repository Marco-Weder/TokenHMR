#!/usr/bin/env python
"""Render the wrong-token pose-damage figure for Results 4.4 (fig:token-ambiguity).

For every token the gated cross-entropy model predicts incorrectly, the otherwise
ground-truth token sequence is decoded with only that one position replaced by the
prediction, and the resulting MPJPE against the unmodified decode is the "damage"
of that single mistake. If token accuracy tracked pose quality the damages would be
large; the point of the figure is that almost all of them are not.

Left panel: the damage distribution, binned finely enough to show its shape (the
supervisor's coarse <5/5-10/10-20/20-50/>50 bins put 95% of the mass in one bar).
Right panel: the cumulative fraction, which is what the text quotes.

Input:  <repo>/tokenhmr/results/ambiguity_gate/damages.npz  (raw per-event damages,
        written by tokenhmr/analyze_token_ambiguity.py)
Output: <repo>/thesis/images/downstream/token_ambiguity.pdf

Run with the project env (from the tokenhmr folder):
    ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/plot_token_ambiguity.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro import paths

HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/tokenhmr/thesis_figures
TOKENHMR = os.path.dirname(HERE)                           # <repo>/tokenhmr
REPO = os.path.dirname(TOKENHMR)                           # <repo>
INDIR = os.path.join(str(paths.RESULTS_DIR), "ambiguity_gate")
OUTDIR = str(paths.figure_out_dir("downstream"))
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 10.5,
    "axes.linewidth": 0.8, "pdf.fonttype": 42,
})

BAR = "#0072B2"      # Okabe-Ito blue
CUM = "#D55E00"      # Okabe-Ito vermillion
MARK = "#009E73"     # Okabe-Ito green

d = np.load(os.path.join(INDIR, "damages.npz"))["damages"]
with open(os.path.join(INDIR, "ambiguity_stats.json")) as f:
    stats = json.load(f)

median = float(np.median(d))
frac_under_10 = float((d < 10.0).mean())
frac_under_5 = float((d < 5.0).mean())

fig, ax = plt.subplots(figsize=(6.2, 3.3))

XMAX = 12.0
bins = np.linspace(0.0, XMAX, 49)
w = np.ones_like(d) / d.size
ax.hist(np.clip(d, 0, XMAX), bins=bins, weights=w, color=BAR, edgecolor="none",
        label="distribution (rightmost bar: all $\\geq 12$ mm)")
ax.axvline(median, color=MARK, lw=1.2, ls=(0, (4, 2)))
ax.text(median + 0.25, ax.get_ylim()[1] * 0.97, f"median {median:.1f} mm",
        color=MARK, fontsize=9, va="top")
ax.set_xlim(0, XMAX)
ax.set_xlabel("Pose damage of a single wrong token [mm]")
ax.set_ylabel("Fraction of wrong tokens")

# cumulative fraction on the right axis: this is what the text quotes
axc = ax.twinx()
xs = np.linspace(0.0, XMAX, 400)
cum = np.searchsorted(np.sort(d), xs, side="right") / d.size
axc.plot(xs, cum, color=CUM, lw=1.8, label="cumulative")
for x, va, dy in ((5.0, "top", -0.04), (10.0, "bottom", 0.03)):
    y = float((d < x).mean())
    axc.plot([0, x], [y, y], color="0.65", lw=0.7, ls=":", zorder=0)
    axc.plot([x, x], [0, y], color="0.65", lw=0.7, ls=":", zorder=0)
    axc.plot([x], [y], "o", color=CUM, ms=3.8)
    axc.text(x - 0.25, y + dy, f"{y * 100:.1f}% below {x:.0f} mm",
             fontsize=8.8, color=CUM, ha="right", va=va)
axc.set_ylim(0, 1.04)
axc.set_ylabel("Cumulative fraction", color=CUM)
axc.tick_params(axis="y", colors=CUM)
axc.spines["top"].set_visible(False)
axc.spines["right"].set_color(CUM)

for side in ("top",):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="0.9", lw=0.6, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
out = os.path.join(OUTDIR, "token_ambiguity.pdf")
fig.savefig(out)
fig.savefig(os.path.join(HERE, "token_ambiguity_preview.png"), dpi=150)

print(f"n wrong tokens : {d.size}")
print(f"top-1 accuracy : {stats['top1_accuracy'] * 100:.1f}%")
print(f"median damage  : {median:.2f} mm")
print(f"< 5 mm         : {frac_under_5 * 100:.1f}%")
print(f"< 10 mm        : {frac_under_10 * 100:.1f}%")
print(f"max damage     : {d.max():.1f} mm")
print("wrote", out)
