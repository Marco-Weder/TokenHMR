#!/usr/bin/env python
"""Figure 4.7: the two decoding rules, rendered in the Figure 4.6 / Figure 3.1 style.

plot_decoding_figure.py --keep N caches the N highest-scoring 3DPW frames. This script
renders them so the frame can be chosen by eye, then draws the chosen one as the thesis
figure. The meshes use the same Blender setup and the same plain-turbo vertex colouring
as Figure 4.6, so the two error maps in the thesis read on one visual convention.

  contact sheet:  ... render_decode_candidates.py --sheet
  final figure:   ... render_decode_candidates.py --pick 3

Output: <repo>/thesis/images/downstream/decoding.pdf
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_token_viz_blender as B          # render(), blend(), BASE_GREY, TURBO, YFOV

TOKENHMR = os.path.dirname(HERE)
REPO = os.path.dirname(TOKENHMR)
NPZ = os.path.join(HERE, "cache", "decfig_cache_candidates.npz")
OUT = os.path.join(REPO, "thesis", "images", "downstream")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "cm", "font.size": 10.5, "pdf.fonttype": 42})


def cam_dist(v):
    c = 0.5 * (v.max(0) + v.min(0))
    return (np.abs(v - c).max(0)[:2].max() * 1.1) / np.tan(B.YFOV / 2)


def upright(v):
    """SMPL is +y down for pyrender's camera; flip so the body stands up."""
    w = v.copy()
    w[:, 1] *= -1.0
    w[:, 2] *= -1.0
    return w


def panels(z, j, vmax, res):
    B.RES = res
    v_gt, v_s, v_h = (upright(z[f"v_{k}_{j}"]) for k in ("gt", "s", "h"))
    e_s, e_h = z[f"e_s_{j}"], z[f"e_h_{j}"]
    d = max(cam_dist(v_gt), cam_dist(v_s), cam_dist(v_h))
    grey = np.concatenate([np.tile(B.BASE_GREY, (len(v_gt), 1)), np.ones((len(v_gt), 1))], 1)
    return [B.render(v_gt, z["faces"], grey, d),
            B.render(v_s, z["faces"], B.blended_colors(e_s, vmax), d),
            B.render(v_h, z["faces"], B.blended_colors(e_h, vmax), d)]


def sheet(z, n):
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.5 * n)); fig.patch.set_facecolor("white")
    for j in range(n):
        m = z[f"meta_{j}"]
        vmax = float(np.percentile(z[f"e_h_{j}"], 95))
        for ax, im, t in zip(axes[j], panels(z, j, vmax, 300),
                             ["ground truth", f"soft {m[1]:.0f} mm", f"hard {m[2]:.0f} mm"]):
            ax.imshow(im); ax.axis("off"); ax.set_title(t, fontsize=8)
        axes[j][0].set_ylabel(f"cand {j}")
        axes[j][0].text(-0.05, 0.5, f"cand {j}\nframe {int(m[0])}\nartic {m[3]:.0f}$^\\circ$",
                        transform=axes[j][0].transAxes, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "decode_candidates_sheet.png")
    fig.savefig(out, dpi=70, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


def final(z, j, topk=12):
    m = z[f"meta_{j}"]
    vmax = float(np.percentile(z[f"e_h_{j}"], 95))
    imgs = panels(z, j, vmax, 900)
    logits = z[f"logits_{j}"]
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    # A TYPICAL position, by median entropy. Taking the most confident position would
    # flatter the model and contradict the near-uniform claim the figure makes; taking
    # the least confident would overstate it. Matches plot_decoding_figure.py.
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)
    pos = int(np.argsort(ent)[len(ent) // 2])
    top = np.sort(p[pos])[::-1][:topk]
    K = p.shape[-1]

    fig = plt.figure(figsize=(9.4, 2.9)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.05, left=0.06,
                          right=0.99, top=0.84, bottom=0.24)
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(topk), top * 1e3, color=["#D55E00"] + ["#9ecae1"] * (topk - 1))
    ax.axhline(1e3 / K, ls=":", lw=1, color="0.35")
    ax.text(topk - 1, 1e3 / K, "uniform", ha="right", va="bottom", fontsize=7.5, color="0.35")
    ax.set_ylabel(r"$p_k$  [$\times 10^{-3}$]", fontsize=9)
    ax.set_xlabel(f"top {topk} of $K = {K}$ codes", fontsize=8.5)
    ax.set_xticks([]); ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("predicted distribution\nat one token position", fontsize=9)

    for c, (im, t) in enumerate(zip(imgs, ["ground truth\n(reference)",
                                           f"soft decode\n{m[1]:.0f} mm",
                                           f"hard decode\n{m[2]:.0f} mm"])):
        a = fig.add_subplot(gs[0, c + 1]); a.imshow(im); a.axis("off")
        a.set_title(t, fontsize=9)

    cax = fig.add_axes([0.55, 0.10, 0.30, 0.035])
    sm = plt.cm.ScalarMappable(cmap=B.BLEND_CMAP, norm=Normalize(0, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("per-vertex error [mm]", fontsize=8.5, labelpad=2)
    cb.ax.tick_params(labelsize=8)

    out = os.path.join(OUT, "decoding.pdf")
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "decoding_preview.png"), dpi=150, facecolor="white",
                bbox_inches="tight")
    print(f"wrote {out}  (candidate {j}, frame {int(m[0])}, soft {m[1]:.1f} / hard {m[2]:.1f} mm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--pick", type=int, default=None)
    a = ap.parse_args()
    z = np.load(NPZ, allow_pickle=True)
    n = int(z["n"])
    if a.sheet:
        sheet(z, n)
    if a.pick is not None:
        final(z, a.pick)
