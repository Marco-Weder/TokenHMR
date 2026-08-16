#!/usr/bin/env python
"""Render Figure 4.3: the reconstruction-versus-consistency frontier.

A scatter of label stability S(1 deg) (y, higher better) against reconstruction
error (x, lower better), one point per tokenizer, coloured AND shaped by quantizer
family (so it survives greyscale printing). The learned codebooks trace a
trade-off frontier; the convolutional baseline sits below it and FSQ steps above
it. The four tokenizers carried into the token-supervision comparison of
Table 4.10 (tab:ce-tokenizer) are ringed: FSQ d=4, transformer-l2 d=256,
convolutional-l2 d=256 and cosine d=4. Their rings descend monotonically in
stability (63, 53, 24, 22) while their reconstruction error improves
(2.58, 2.45, 2.27, 1.51), which is the inversion Section 4.4.5 rests on.

The numbers are hard-coded from Table 4.3 (tab:token-target-quality) in
experimentsandresults.tex -- if a cell there changes, update the P list below.

Run with the project env (from the tokenhmr folder):
    ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/plot_recon_frontier.py

Output: <repo>/thesis/images/tokenizer/recon_stability_frontier.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from repro import manifest, paths

HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/tokenhmr/thesis_figures
REPO = os.path.dirname(os.path.dirname(HERE))              # <repo>
OUTDIR = str(paths.figure_out_dir("tokenizer"))
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 10.5,
    "axes.linewidth": 0.8, "pdf.fonttype": 42,
})

# family -> (colour, marker)  Okabe-Ito + shape (CVD- and greyscale-safe)
FAM = {
    "conv": ("#E69F00", "s"),   # convolutional (EMA l2)
    "tfl2": ("#0072B2", "^"),   # transformer   (EMA l2)
    "cos":  ("#009E73", "o"),   # cosine        (EMA cosine)
    "fsq":  ("#D55E00", "D"),   # FSQ
}
# Label, family, whether the tokenizer is carried into tab:ce-tokenizer, and the
# label offset. Those are typography. The coordinates are not: recon MPJPE and
# S(1 deg) are read from the tokenizer registry, so this figure and
# tab:token-target-quality cannot disagree.
STYLE = {
    "CNN":                ("Conv ($\\ell_2$)", "conv", True,  ( 6,  9), "left",  "bottom"),
    "Transformer tier1":  ("Tf ($\\ell_2$)",   "tfl2", True,  ( 9,  0), "left",  "center"),
    "Transformer cosine": ("cos $d$256",      "cos",  False, ( 7,  4), "left",  "bottom"),
    "Skeleton-masked":    ("skel-mask",       "cos",  False, ( 8, -3), "left",  "top"),
    "VQ d4":              ("cos $d$4",        "cos",  True,  ( 8, -3), "left",  "top"),
    "VQ d2":              ("cos $d$2",        "cos",  False, ( 7,  2), "left",  "bottom"),
    "FSQ d4":             ("FSQ $d$4",        "fsq",  True,  (-8,  6), "right", "bottom"),
    "FSQ d5":             ("FSQ $d$5",        "fsq",  False, (-8,  6), "right", "bottom"),
}
P = []
for _t in manifest.tokenizers():
    _label, _fam, _carried, _off, _ha, _va = STYLE[_t["label"]]
    # Rounded to the precision tab:token-target-quality prints, so the figure
    # plots the same values the table does rather than fuller ones.
    P.append((_label, round(_t["recon_mpjpe_mm"], 2), round(_t["stability_pct"]["1"]),
              _fam, _carried, _off, _ha, _va))

fig, ax = plt.subplots(figsize=(6.6, 4.5))

# No fitted trend line: it reads as a claimed correlation. The points speak for
# themselves -- the learned codebooks trail down-left, FSQ sits up and to the right.

for (lab, x, y, fam, cf, off, ha, va) in P:
    c, mk = FAM[fam]
    if cf:  # carried-forward: black ring halo
        ax.scatter([x], [y], s=185, facecolors="none", edgecolors="black",
                   linewidths=1.4, marker=mk, zorder=4)
    ax.scatter([x], [y], s=95, color=c, marker=mk, edgecolors="white",
               linewidths=0.8, zorder=5)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, ha=ha, va=va,
                fontsize=9, fontweight=("bold" if cf else "normal"), color="0.15")

ax.set_xlabel("Reconstruction MPJPE  [mm]   (lower better)")
ax.set_ylabel("Label stability  $S(1^\\circ)$  [%]   (higher better)")
ax.set_xlim(0.35, 2.85)
ax.set_ylim(0, 70)
ax.grid(True, color="0.9", lw=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fam_names = {"conv": "Convolutional", "tfl2": "Transformer ($\\ell_2$)",
             "cos": "Cosine", "fsq": "FSQ"}
handles = [Line2D([0], [0], marker=FAM[k][1], color="none", markerfacecolor=FAM[k][0],
                  markeredgecolor="white", markersize=9, label=fam_names[k])
           for k in ["conv", "tfl2", "cos", "fsq"]]
handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                       markeredgecolor="black", markersize=11, markeredgewidth=1.3,
                       label="carried to token supervision"))
ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95,
          edgecolor="0.8", fontsize=8.8, handletextpad=0.5, borderpad=0.7)

fig.tight_layout()
out = os.path.join(OUTDIR, "recon_stability_frontier.pdf")
fig.savefig(out)                                           # the file wired into the thesis
fig.savefig(os.path.join(HERE, "recon_stability_frontier_preview.png"), dpi=170)  # local preview
print("wrote", out)
