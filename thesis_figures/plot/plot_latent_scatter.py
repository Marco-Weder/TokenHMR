#!/usr/bin/env python
"""Figure B.3: per-tokenizer latent-space geometry.

For each tokenizer, scatters a subsample of the encoder's pre-quant token latents
(grey) with the codebook (coloured) overlaid, reduced to 2D: d=2 codebooks are shown
in the plane, wider ones via PCA fit on the latents (codebook projected into the same
basis; cosine latents/codes unit-normalised first). Shows how each codebook occupies
its space -- the cosine codebooks crowd onto a low-dimensional shell (low effective
rank) while the FSQ grids fill the space.

Reads tokenization/output/token_maps/<label>.npz (from analyze_token_maps.py).
Run: ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/plot_latent_scatter.py
Output: thesis/images/appendix/latent_space_all.pdf
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENHMR = os.path.dirname(HERE)
REPO = os.path.dirname(TOKENHMR)
MAPS = os.path.join(str(paths.TOKENIZER_OUT), "token_maps")
APP_OUT = str(paths.figure_out_dir("appendix"))
os.makedirs(APP_OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "cm", "font.size": 9, "pdf.fonttype": 42})

index = json.load(open(os.path.join(MAPS, "index.json")))["runs"]
SHORT = {"CNN": "Conv ($\\ell_2$), $d$256", "Transformer tier1": "Transformer ($\\ell_2$), $d$256",
         "Transformer cosine": "Cosine $d$256", "Skeleton-masked": "Skeleton-masked $d$256",
         "VQ d4": "Cosine $d$4", "VQ d2": "Cosine $d$2",
         "FSQ d4": "FSQ $d$4", "FSQ d5": "FSQ $d$5"}
order = ["CNN", "Transformer tier1", "Transformer cosine", "Skeleton-masked",
         "VQ d4", "VQ d2", "FSQ d4", "FSQ d5"]


def reduce2d(lat, cb, cosine):
    lat = lat.astype(np.float64); cb = cb.astype(np.float64)
    if cosine:
        lat = lat / np.clip(np.linalg.norm(lat, axis=1, keepdims=True), 1e-9, None)
        cb = cb / np.clip(np.linalg.norm(cb, axis=1, keepdims=True), 1e-9, None)
    if lat.shape[1] == 2:
        return lat, cb, ("$z_1$", "$z_2$")
    mu = lat.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(lat - mu, full_matrices=False)
    P = Vt[:2].T
    return (lat - mu) @ P, (cb - mu) @ P, ("PC 1", "PC 2")


fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.4))
rng = np.random.default_rng(0)
for ax, label in zip(axes.ravel(), order):
    d = np.load(os.path.join(MAPS, [r["file"] for r in index if r["label"] == label][0]),
                allow_pickle=True)
    lat, cb, cosine = d["latents"], d["codebook"], bool(d["cosine"])
    if cb.shape[0] > 2500:                                  # subsample big codebooks (FSQ d5)
        cb = cb[rng.choice(cb.shape[0], 2500, replace=False)]
    L2, C2, (xl, yl) = reduce2d(lat, cb, cosine)
    ax.scatter(L2[:, 0], L2[:, 1], s=2, c="0.72", linewidths=0, rasterized=True, zorder=1)
    ax.scatter(C2[:, 0], C2[:, 1], s=3, c="#D55E00", linewidths=0, rasterized=True, zorder=2)
    ax.set_title(SHORT[label], fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7)
    ax.set_aspect("equal", "datalim")

from matplotlib.lines import Line2D

from repro import paths
handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="0.72", markersize=5, label="encoder latents"),
           Line2D([0], [0], marker="o", color="none", markerfacecolor="#D55E00", markersize=5, label="codebook")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Latent space of each tokenizer  (codebook over encoder latents; $d>2$ via PCA)",
             fontsize=10.5, y=1.0)
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(os.path.join(APP_OUT, "latent_space_all.pdf"), bbox_inches="tight", dpi=200)
fig.savefig(os.path.join(HERE, "latent_space_all_preview.png"), dpi=140, bbox_inches="tight")
print("wrote latent_space_all.pdf")
