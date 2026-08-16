#!/usr/bin/env python
"""Figure B.1 ("what tokenization buys"): candidate sheets and the final figure.

Scores every frame cached by scan_token_benefit.py by how far the CONTINUOUS
prediction sits off the pose manifold, measured as the residual when its predicted body
pose is passed through the frozen FSQ tokenizer (encode then decode). The token
classifier's own output is on that manifold by construction, so the residual is the
quantity the figure is about. Frames are preferred where the two models' vertex errors
against the ground truth are similar, so the panel shows a plausibility difference and
not simply a more accurate model.

    --sheet          contact sheet of the top candidates
    --pick J         write the final figure for candidate J

Output: <repo>/thesis/images/appendix/token_benefit.pdf
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from repro import paths

HERE = Path(__file__).resolve().parent
TOKENHMR = HERE.parent
REPO = TOKENHMR.parent
CACHE = HERE / 'cache' / 'token_benefit.npz'
OUT = paths.figure_out_dir("appendix")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                     'mathtext.fontset': 'cm', 'font.size': 10.5, 'pdf.fonttype': 42})


def manifold_residual(poses):
    """Mean per-joint geodesic residual (deg) of a batch of body poses under the
    frozen FSQ tokenizer's encode/decode round trip."""
    from tokenization.analysis import analyze_latent_pose_info as ali
    ali.DEVICE = torch.device('cpu')
    import models.transformer_pose_vqvae as _t; _t.body_model.to('cpu')
    from tokenization.analysis.analyze_latent_pose_info import (load_net, encode_full, decode_from_codes,
                                          pose6d_to_rotmat)
    runs = {r['label']: os.path.join(str(TOKENHMR / 'tokenization'), r['ckpt'])
            for r in json.load(open(TOKENHMR / 'tokenization/output/token_stability/summary.json'))['runs']}
    net, _ = load_net(runs['FSQ d4']); net = net.cpu()
    out = []
    with torch.no_grad():
        for P in poses:
            p = torch.tensor(P, dtype=torch.float32)[None]
            rt = pose6d_to_rotmat(net, decode_from_codes(net, encode_full(net, p)['idx']))
            rt = rt.reshape(-1, 3, 3)[:21]                 # decoder returns 6D rotations
            R = torch.matmul(p[0], rt.transpose(-1, -2))
            tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
            out.append(float(torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1))).mean()))
    return np.array(out)


def rank(z, by='manifold'):
    n = int(z['n'])
    meta = np.stack([z[f'meta_{j}'] for j in range(n)])
    res = manifold_residual([z[f'pose_c_{j}'] for j in range(n)])
    e_c, e_t = meta[:, 1], meta[:, 2]
    artic = meta[:, 3] if meta.shape[1] > 3 else np.zeros(n)
    if by == 'extreme':
        # most articulated ground-truth poses, de-duplicated so one clip cannot fill
        # the sheet: greedily take the next candidate only if its GT pose differs from
        # every one already chosen.
        cand = np.argsort(-artic)
        chosen, kept = [], []
        for j in cand:
            g = z[f'v_gt_{j}']
            if all(np.linalg.norm(g - z[f'v_gt_{k}'], axis=-1).mean() * 1000 > 120 for k in chosen):
                chosen.append(j); kept.append(j)
            if len(chosen) >= 40:
                break
        order = np.array(kept)
    elif by == 'tokwin':
        # where the TOKEN classifier is most ahead of the continuous regressor, which is
        # the case the figure is meant to illustrate; de-duplicated as above
        adv = e_c - e_t
        cand = np.argsort(-adv)
        chosen = []
        for j in cand:
            g = z[f'v_gt_{j}']
            if all(np.linalg.norm(g - z[f'v_gt_{k}'], axis=-1).mean() * 1000 > 120 for k in chosen):
                chosen.append(j)
            if len(chosen) >= 40:
                break
        order = np.array(chosen)
    elif by == 'disagree':
        # where the two models produce the most different bodies, which is where a
        # visual difference exists at all
        d = np.array([np.linalg.norm(z[f'v_c_{j}'] - z[f'v_t_{j}'], axis=-1).mean() * 1000
                      for j in range(n)])
        order = np.argsort(-d)
    else:
        similar = np.exp(-np.abs(e_c - e_t) / 25.0)
        order = np.argsort(-(res * similar))
    return order, res, e_c, e_t


def panels(z, j, res_state=None):
    import render_token_viz_blender as B
    B.RES = res_state or 700
    faces = np.load(TOKENHMR / 'tokenization/output/token_viz/swap_geometry.npz')['faces']
    def up(v):
        w = v.copy(); w[:, 1] *= -1; w[:, 2] *= -1; return w
    vs = [up(z[f'v_gt_{j}']), up(z[f'v_c_{j}']), up(z[f'v_t_{j}'])]
    d = max((np.abs(v - 0.5 * (v.max(0) + v.min(0))).max(0)[:2].max() * 1.1) / np.tan(B.YFOV / 2) for v in vs)
    grey = np.concatenate([np.tile(B.BASE_GREY, (len(vs[0]), 1)), np.ones((len(vs[0]), 1))], 1)
    return [B.render(v, faces, grey, d) for v in vs]


def crop(img):
    x = img.transpose(1, 2, 0)
    x = x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(x, 0, 1)


def sheet(z, order, res, e_c, e_t, k=8):
    fig, ax = plt.subplots(k, 4, figsize=(9, 2.3 * k)); fig.patch.set_facecolor('white')
    for r, j in enumerate(order[:k]):
        ims = panels(z, j, 300)
        ax[r, 0].imshow(crop(z[f'img_{j}'])); ax[r, 0].set_title('input', fontsize=8)
        for c, (im, t) in enumerate(zip(ims, ['ground truth', f'continuous {e_c[j]:.0f} mm',
                                              f'tokenized {e_t[j]:.0f} mm'])):
            ax[r, c + 1].imshow(im); ax[r, c + 1].set_title(t, fontsize=8)
        for c in range(4):
            ax[r, c].axis('off')
        mm = z[f'meta_{j}']
        lab = (f'cand {r}\nframe {int(mm[0])}\noff-man {res[j]:.1f}$^\\circ$'
               + (f'\nartic {mm[3]:.0f}$^\\circ$' if len(mm) > 3 else ''))
        ax[r, 0].text(-0.08, 0.5, lab,
                      transform=ax[r, 0].transAxes, ha='right', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / 'token_benefit_sheet.png', dpi=70, facecolor='white', bbox_inches='tight')
    print(f'wrote {HERE / "token_benefit_sheet.png"}')


def final_multi(z, js, res, e_c, e_t):
    """Figure B.1: several frames, one row each (input / GT / continuous / tokenized)."""
    rows = len(js)
    fig, ax = plt.subplots(rows, 4, figsize=(7.6, 2.05 * rows))
    fig.patch.set_facecolor('white')
    ax = np.atleast_2d(ax)
    for r, j in enumerate(js):
        ims = panels(z, j, 640)
        ax[r, 0].imshow(crop(z[f'img_{j}']))
        for c, im in enumerate(ims):
            ax[r, c + 1].imshow(im)
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for sp in ax[r, c].spines.values():
                sp.set_visible(False)
        ax[r, 2].set_xlabel(f'{e_c[j]:.0f} mm', fontsize=8.5)
        ax[r, 3].set_xlabel(f'{e_t[j]:.0f} mm', fontsize=8.5)
    for c, t in enumerate(['input', 'ground truth', 'continuous regressor', 'token classifier']):
        ax[0, c].set_title(t, fontsize=9.5, pad=6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02, wspace=0.02, hspace=0.10)
    out = OUT / 'token_benefit.pdf'
    fig.savefig(out, dpi=200, facecolor='white', bbox_inches='tight')
    fig.savefig(HERE / 'token_benefit_preview.png', dpi=130, facecolor='white', bbox_inches='tight')
    for j in js:
        print(f'  frame {int(z[f"meta_{j}"][0])}: continuous {e_c[j]:.1f} mm  tokenized {e_t[j]:.1f} mm  '
              f'off-manifold {res[j]:.2f} deg')
    print(f'wrote {out}')


def final(z, j, res, e_c, e_t):
    ims = panels(z, j)
    fig, ax = plt.subplots(1, 4, figsize=(8.6, 2.9)); fig.patch.set_facecolor('white')
    ax[0].imshow(crop(z[f'img_{j}'])); ax[0].set_title('input', fontsize=9.5)
    for c, (im, t) in enumerate(zip(ims, ['ground truth', 'continuous regressor', 'token classifier'])):
        ax[c + 1].imshow(im); ax[c + 1].set_title(t, fontsize=9.5)
    for a in ax:
        a.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.02)
    out = OUT / 'token_benefit.pdf'
    fig.savefig(out, dpi=200, facecolor='white', bbox_inches='tight')
    print(f'wrote {out}  (frame {int(z[f"meta_{j}"][0])}, off-manifold {res[j]:.1f} deg, '
          f'continuous {e_c[j]:.1f} mm vs tokenized {e_t[j]:.1f} mm)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', action='store_true'); ap.add_argument('--pick', type=int)
    ap.add_argument('--picks', type=str, help='comma-separated candidate ranks for the final figure')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--by', default='manifold', choices=['manifold','disagree','extreme','tokwin'])
    a = ap.parse_args()
    z = np.load(CACHE, allow_pickle=True)
    order, res, e_c, e_t = rank(z, a.by)
    print('rank  frame  off-manifold  continuous  tokenized')
    for r, j in enumerate(order[:12]):
        print(f'{r:4d}  {int(z[f"meta_{j}"][0]):5d}   {res[j]:8.2f}   {e_c[j]:8.1f}   {e_t[j]:8.1f}')
    if a.sheet: sheet(z, order, res, e_c, e_t, a.k)
    if a.pick is not None: final(z, order[a.pick], res, e_c, e_t)
    if a.picks: final_multi(z, [order[int(x)] for x in a.picks.split(',')], res, e_c, e_t)
