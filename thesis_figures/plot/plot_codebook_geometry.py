#!/usr/bin/env python
"""Render Figure 4.4: codebook distinctness across tokenizers.

Two horizontal-bar panels, one row per tokenizer (order of Table 4.4), coloured by
quantizer family (same palette as Figure 4.3):

  (a) code separation ratio = median nearest-neighbour code distance / median
      latent-to-code quantization error. A dashed line at 1 marks where codes are
      no more separated than the encoder's own quantization noise.
  (b) the distance-based entropy decomposition H(k) = I(z;k) + H(k|z) in bits at
      tau = half the median NN code distance: the solid part is the information a
      token index carries, the faded part the confusion; a tick marks log2(K).

Reads the consistent 8-tokenizer / 2048-pose run produced by
tokenization/analyze_codebook_geometry.py.

Run with the project env (from the tokenhmr folder):
    python -m thesis_figures.plot.plot_codebook_geometry

Output: <repo>/thesis/images/latent/codebook_distinctness.pdf
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from repro import paths

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENHMR = os.path.dirname(HERE)
REPO = os.path.dirname(TOKENHMR)
SUMMARY = os.path.join(str(paths.TOKENIZER_OUT), "codebook_geometry", "summary.json")
OUTDIR = str(paths.figure_out_dir("latent"))
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 10.5,
    "axes.linewidth": 0.8, "pdf.fonttype": 42,
})

FAM_COLOR = {"conv": "#E69F00", "tfl2": "#0072B2", "cos": "#009E73", "fsq": "#D55E00"}
FAM_NAME = {"conv": "Convolutional", "tfl2": "Transformer ($\\ell_2$)",
            "cos": "Cosine", "fsq": "FSQ"}
SHORT = {"CNN": "Conv", "Transformer tier1": "Tf ($\\ell_2$)",
         "Transformer cosine": "cos $d$256", "Skeleton-masked": "skel-mask",
         "VQ d4": "cos $d$4", "VQ d2": "cos $d$2", "FSQ d4": "FSQ $d$4", "FSQ d5": "FSQ $d$5"}

runs = json.load(open(SUMMARY))["runs"]
labels = [SHORT.get(r["label"], r["label"]) for r in runs]
colors = [FAM_COLOR[r["family"]] for r in runs]
sep = [r["separation_ratio"] for r in runs]
I = [r["I_zk_bits"] for r in runs]
Hkz = [r["H_confusion_bits"] for r in runs]
log2K = [r["log2K"] for r in runs]
y = np.arange(len(runs))[::-1]          # first tokenizer at the top

fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})

# ---- (a) separation ratio ----
ax[0].barh(y, sep, color=colors, height=0.66, edgecolor="white", linewidth=0.6, zorder=3)
ax[0].axvline(1.0, color="0.45", ls="--", lw=1.1, zorder=2)
for yi, v in zip(y, sep):
    ax[0].text(v + 0.35, yi, f"{v:.1f}", va="center", ha="left", fontsize=8.3, color="0.15")
ax[0].set_yticks(y); ax[0].set_yticklabels(labels)
ax[0].set_xlabel("Separation ratio   (NN dist / quant. error)")
ax[0].set_xlim(0, max(sep) * 1.17)
ax[0].set_title("(a)  Code distinctness", fontsize=10.5, loc="left")

# ---- (b) entropy decomposition H(k) = I(z;k) + H(k|z) ----
ax[1].barh(y, I, color=colors, height=0.66, edgecolor="white", linewidth=0.6, zorder=3)
ax[1].barh(y, Hkz, left=I, color=colors, height=0.66, alpha=0.30, hatch="////",
           edgecolor="white", linewidth=0.6, zorder=3)
for yi, lk in zip(y, log2K):
    ax[1].plot([lk, lk], [yi - 0.36, yi + 0.36], color="0.25", lw=1.4, zorder=5)
for yi, v in zip(y, I):
    ax[1].text(v / 2, yi, f"{v:.1f}", va="center", ha="center", fontsize=8.0,
               color="white", zorder=6)
ax[1].set_yticks(y); ax[1].set_yticklabels([])
ax[1].set_xlabel("$H(k)=I(z;k)+H(k|z)$   [bits]")
ax[1].set_xlim(0, max(log2K) * 1.04)
ax[1].set_title("(b)  Token information", fontsize=10.5, loc="left")

# legends
fam_present = list(dict.fromkeys(r["family"] for r in runs))
fam_handles = [Patch(facecolor=FAM_COLOR[f], label=FAM_NAME[f]) for f in fam_present]
ax[0].legend(handles=fam_handles, loc="lower right", fontsize=8.2, frameon=True,
             edgecolor="0.8", framealpha=0.95, borderpad=0.6, handlelength=1.2,
             title="quantizer family", title_fontsize=8.2)
dec_handles = [Patch(facecolor="0.35", label="$I(z;k)$ information"),
               Patch(facecolor="0.35", alpha=0.30, hatch="////", label="$H(k|z)$ confusion"),
               plt.Line2D([0], [0], color="0.25", lw=1.4, label="$\\log_2 K$  (max)")]
ax[1].legend(handles=dec_handles, loc="upper right", fontsize=8.2, frameon=True,
             edgecolor="0.8", framealpha=0.95, borderpad=0.6, handlelength=1.2)

for a in ax:
    a.grid(True, axis="x", color="0.9", lw=0.7, zorder=0)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

fig.tight_layout(w_pad=2.2)
out = os.path.join(OUTDIR, "codebook_distinctness.pdf")
fig.savefig(out)
fig.savefig(os.path.join(HERE, "codebook_distinctness_preview.png"), dpi=170)
print("wrote", out)
