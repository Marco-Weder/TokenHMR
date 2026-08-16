#!/usr/bin/env python
"""Figure 4.5 (main) and Figure B.1 (appendix): token->joint influence maps.

Reads the per-tokenizer influence matrices from
tokenization/output/token_maps/<label>.npz (produced by analyze_token_maps.py).

  - Fig 4.5: the carried-forward FSQ (d=4) map, tokens x joints, with per-token
    specialisation (right strip) and per-joint spread (bottom bar) as marginals.
  - Fig B.1: an 8-panel grid, one influence map per tokenizer (each normalised to
    its own scale so the token->joint structure is visible across scales).

Run: python -m thesis_figures.plot.plot_influence_maps
Outputs: thesis/images/latent/token_influence_fsq.pdf
         thesis/images/appendix/influence_all_tokenizers.pdf
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from repro import paths

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENHMR = os.path.dirname(HERE)
REPO = os.path.dirname(TOKENHMR)
MAPS = os.path.join(str(paths.TOKENIZER_OUT), "token_maps")
LAT_OUT = str(paths.figure_out_dir("latent"))
APP_OUT = str(paths.figure_out_dir("appendix"))
os.makedirs(LAT_OUT, exist_ok=True); os.makedirs(APP_OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "cm", "font.size": 9, "pdf.fonttype": 42})

index = json.load(open(os.path.join(MAPS, "index.json")))["runs"]
SHORT = {"CNN": "Conv ($\\ell_2$)", "Transformer tier1": "Transformer ($\\ell_2$)",
         "Transformer cosine": "Cosine $d$256", "Skeleton-masked": "Skeleton-masked",
         "VQ d4": "Cosine $d$4", "VQ d2": "Cosine $d$2",
         "FSQ d4": "FSQ $d$4", "FSQ d5": "FSQ $d$5"}


def load(label):
    d = np.load(os.path.join(MAPS, [r["file"] for r in index if r["label"] == label][0]),
                allow_pickle=True)
    return d["influence"], [str(x) for x in d["joint_names"]]


# ----------------------------- Fig 4.5: FSQ d4 with marginals -----------------------------
infl, joints = load("FSQ d4")                              # (T, J)
T, J = infl.shape
row_sum = infl.sum(1, keepdims=True); row_sum[row_sum == 0] = 1
spec = infl.max(1) / row_sum[:, 0]                         # per-token specialisation (0..1)
col = infl / np.clip(infl.sum(0, keepdims=True), 1e-8, None)
srt = np.sort(col, axis=0)[::-1]
spread = (np.cumsum(srt, axis=0) < 0.8).sum(0) + 1         # per-joint spread (# tokens for 80%)

fig = plt.figure(figsize=(5.2, 7.2))
gs = GridSpec(2, 3, width_ratios=[1, 0.10, 0.05], height_ratios=[1, 0.14],
              wspace=0.06, hspace=0.06)
axH = fig.add_subplot(gs[0, 0])
im = axH.imshow(infl, aspect="auto", cmap="viridis", interpolation="nearest")
axH.set_ylabel("latent token  $i$  (0..%d)" % (T - 1))
axH.tick_params(axis="x", labelbottom=False, bottom=False)
axS = fig.add_subplot(gs[0, 1], sharey=axH)
axS.imshow(spec[:, None], aspect="auto", cmap="magma", vmin=0, vmax=1)
axS.set_xticks([]); axS.set_yticks([]); axS.set_title("spec.", fontsize=7)
axCb = fig.add_subplot(gs[0, 2]); fig.colorbar(im, cax=axCb, label="influence $\\mathrm{Infl}_{ij}$ [deg]")
axB = fig.add_subplot(gs[1, 0], sharex=axH)
axB.bar(range(J), spread, color="#E45756"); axB.set_xticks(range(J))
axB.set_xticklabels(joints, rotation=90, fontsize=6.5)
axB.set_ylabel("spread", fontsize=7.5); axB.margins(x=0.005)
for s in ("top", "right"): axB.spines[s].set_visible(False)
fig.suptitle("Token$\\to$joint influence  |  FSQ ($d=4$)", fontsize=10, y=0.995)
fig.savefig(os.path.join(LAT_OUT, "token_influence_fsq.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "token_influence_fsq_preview.png"), dpi=150, bbox_inches="tight")
print("wrote token_influence_fsq.pdf")

# ----------------------------- Fig B.1: 8-panel grid -----------------------------
order = ["CNN", "Transformer tier1", "Transformer cosine", "Skeleton-masked",
         "VQ d4", "VQ d2", "FSQ d4", "FSQ d5"]
fig, axes = plt.subplots(2, 4, figsize=(9.2, 7.6))
for ax, label in zip(axes.ravel(), order):
    m, jn = load(label)
    ax.imshow(m, aspect="auto", cmap="viridis", interpolation="nearest")   # own scale per panel
    ax.set_title(SHORT[label], fontsize=8.5)
    ax.set_xticks(range(len(jn))); ax.set_xticklabels(jn, rotation=90, fontsize=4.2)
    ax.set_yticks([0, m.shape[0] - 1]); ax.set_yticklabels([0, m.shape[0] - 1], fontsize=6)
    ax.set_ylabel("token", fontsize=6.5)
fig.suptitle("Token$\\to$joint influence maps (each panel normalised to its own range)",
             fontsize=10.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(APP_OUT, "influence_all_tokenizers.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "influence_all_preview.png"), dpi=130, bbox_inches="tight")
print("wrote influence_all_tokenizers.pdf")
