#!/usr/bin/env python
"""Render fig:cb-separation (Results Sec. 4.3.1): code spacing versus quantization noise.

Two panels (FSQ d4 vs Transformer cosine d256, the same contrast as Fig 4.7). In each
panel the latent-to-code quantization error e(z) is drawn as a histogram, and the
inter-code nearest-neighbour spacing rho_k as a median line with a 10-90% band (a sharp
line for the perfectly regular FSQ grid, a band for the irregular cosine codes). Both
axes are normalised by that tokenizer's median code spacing, so the code spacing sits at
1 and the error mass sits at 1 / R, where R is the separation ratio: a wide gap means the
discrete token is unambiguous. FSQ's error mass sits far to the left (R = 12.1); the
cosine codes crowd closer to their own quantization noise (R = 4.1).

Reads the arrays dumped by tokenization/dump_code_separation.py (same metric and pose
set as Table 4.4, so the annotated R matches).

Run with the project env (from the tokenhmr folder):
    ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/plot_code_separation.py

Output: <repo>/thesis/images/latent/code_separation.pdf
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENHMR = os.path.dirname(HERE)
REPO = os.path.dirname(TOKENHMR)
NPZ = os.path.join(TOKENHMR, "tokenization", "output", "codebook_geometry", "separation_arrays.npz")
OUTDIR = os.path.join(REPO, "thesis", "images", "latent")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 10.5,
    "axes.linewidth": 0.8, "pdf.fonttype": 42,
})

ERR_COLOR = "#D55E00"       # quantization error e(z)   (vermillion)
RHO_COLOR = "#0072B2"       # code spacing rho_k         (blue)
TITLE = {"FSQ d4": "FSQ ($d{=}4$)", "Transformer cosine": "Cosine ($d{=}256$)"}
METRIC = {"l2": "Euclidean", "cosine": "chordal"}


def main():
    z = np.load(NPZ, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    labels = meta["labels"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True)
    xmax = 1.85
    bins = np.linspace(0, xmax, 70)

    for ax, i, label in zip(axes, range(len(labels)), labels):
        s = meta["scalars"][label]
        rho_med = s["rho_med"]
        err = z[f"err_{i}"] / rho_med                       # normalise by this tokenizer's median spacing
        rho = z[f"rhoused_{i}"] / rho_med                   # used codes only
        R = s["sep"]

        # quantization error distribution
        ax.hist(err, bins=bins, density=True, color=ERR_COLOR, alpha=0.55,
                edgecolor=ERR_COLOR, linewidth=0.5, zorder=3)
        err_med = np.median(err)
        ax.axvline(err_med, color=ERR_COLOR, ls="-", lw=1.6, zorder=5)

        # code spacing: 10-90% band + median line (band collapses to a line for the FSQ grid)
        lo, hi = np.percentile(rho, 10), np.percentile(rho, 90)
        if hi - lo > 1e-3:
            ax.axvspan(lo, hi, color=RHO_COLOR, alpha=0.15, zorder=1)
        ax.axvline(1.0, color=RHO_COLOR, ls="-", lw=1.6, zorder=5)

        # headroom for the gap annotation, then draw the two-headed arrow between medians
        top = ax.get_ylim()[1] * 1.18
        ax.set_ylim(0, top)
        ax.annotate("", xy=(1.0, 0.90 * top), xytext=(err_med, 0.90 * top),
                    arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.1), zorder=6)
        ax.text((err_med + 1.0) / 2, 0.91 * top, f"$R = {R:.1f}$", ha="center", va="bottom",
                fontsize=10, color="0.15")

        ax.set_title(TITLE[label], fontsize=11)
        ax.set_xlabel("distance / median code spacing")
        ax.text(0.97, 0.97, f"{METRIC[s['metric']]} metric", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.3, color="0.4")
        ax.set_xlim(0, xmax)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("density")

    handles = [
        Patch(facecolor=ERR_COLOR, alpha=0.55, edgecolor=ERR_COLOR,
              label="quantization error $e(z)$"),
        Line2D([0], [0], color=ERR_COLOR, lw=1.6, label="median $e(z)$"),
        Line2D([0], [0], color=RHO_COLOR, lw=1.6, label="code spacing $\\rho_k$ (median)"),
        Patch(facecolor=RHO_COLOR, alpha=0.15, label="$\\rho_k$ 10–90%"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.6,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = os.path.join(OUTDIR, "code_separation.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(HERE, "code_separation_preview.png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
