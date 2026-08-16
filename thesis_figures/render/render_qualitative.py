#!/usr/bin/env python
"""Figure B.6: cosine-d4 and FSQ-d4, pose-supervised against CE-supervised.

  --stats           test the distal-joint claim over the whole sample
  --sheet --by K    contact sheet, K in {distal, cosfail, pve}
  --picks a,b,c     final figure from the chosen candidate ranks

Meshes reuse the Figure 4.6 Blender path, so every mesh figure in the thesis shares one
renderer and one material.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from repro import paths

HERE = Path(__file__).resolve().parent
TOKENHMR = HERE.parent
REPO = TOKENHMR.parent
CACHE = HERE / 'cache' / 'qualitative.npz'
OUT = paths.figure_out_dir("appendix")
NAMES = ['cos4_pose', 'cos4_ce', 'fsq4_pose', 'fsq4_ce']
NICE = {'cos4_pose': 'Cosine $d$4\npose loss', 'cos4_ce': 'Cosine $d$4\ncross-entropy',
        'fsq4_pose': 'FSQ $d$4\npose loss', 'fsq4_ce': 'FSQ $d$4\ncross-entropy'}
DISTAL = ['Head', 'L_Foot', 'R_Foot', 'L_Ankle', 'R_Ankle']

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                     'mathtext.fontset': 'cm', 'font.size': 10.5, 'pdf.fonttype': 42})


def arrays(z):
    n = int(z['n']); J = [str(x) for x in z['joints']]
    pve = {k: np.array([z[f'pve_{k}_{j}'] for j in range(n)]) for k in NAMES}
    jt = {k: np.stack([z[f'joint_{k}_{j}'] for j in range(n)]) for k in NAMES}
    return n, J, pve, jt


def stats(z):
    n, J, pve, jt = arrays(z)
    di = [J.index(d) for d in DISTAL]
    print(f'{n} frames of 3DPW-TEST\n')
    print(f"{'model':16}{'PVE':>8}{'all joints':>12}{'distal':>10}{'Head':>8}{'feet':>8}")
    for k in NAMES:
        feet = jt[k][:, [J.index('L_Foot'), J.index('R_Foot')]].mean()
        print(f'{k:16}{pve[k].mean():8.1f}{jt[k].mean():12.2f}{jt[k][:, di].mean():10.2f}'
              f'{jt[k][:, J.index("Head")].mean():8.2f}{feet:8.2f}')
    print('\nper-joint rotation error, pose loss -> cross-entropy (deg):')
    print(f"{'joint':12}{'cos4 pose':>10}{'cos4 CE':>9}{'delta':>8}   {'fsq4 pose':>10}{'fsq4 CE':>9}{'delta':>8}")
    for i, name in enumerate(J):
        cp, cc = jt['cos4_pose'][:, i].mean(), jt['cos4_ce'][:, i].mean()
        fp, fc = jt['fsq4_pose'][:, i].mean(), jt['fsq4_ce'][:, i].mean()
        mark = '  <-- distal' if name in DISTAL else ''
        print(f'{name:12}{cp:10.2f}{cc:9.2f}{cc-cp:+8.2f}   {fp:10.2f}{fc:9.2f}{fc-fp:+8.2f}{mark}')


def order_by(z, by):
    n, J, pve, jt = arrays(z)
    di = [J.index(d) for d in DISTAL]
    hi = J.index('Head')
    gm = np.stack([z[f'gtmag_{j}'] for j in range(n)]) if 'gtmag_0' in z else None
    if by == 'headturn':
        s = gm[:, hi]
    elif by == 'headwin':
        adv = jt['fsq4_pose'][:, hi] - jt['fsq4_ce'][:, hi]
        s = adv * (gm[:, hi] > np.percentile(gm[:, hi], 70))
    elif by == 'distal':          # CE improves distal joints most, averaged over both tokenizers
        s = ((jt['cos4_pose'][:, di].mean(1) - jt['cos4_ce'][:, di].mean(1))
             + (jt['fsq4_pose'][:, di].mean(1) - jt['fsq4_ce'][:, di].mean(1))) / 2
    elif by == 'cosfail':       # cosine CE fails where FSQ CE succeeds
        s = pve['cos4_ce'] - pve['fsq4_ce']
    else:
        s = pve['cos4_ce'] - pve['fsq4_ce'] + pve['cos4_pose'] - pve['fsq4_pose']
    cand = np.argsort(-s)
    chosen = []
    for j in cand:                                   # de-duplicate near-identical frames
        g = z[f'v_gt_{j}'].astype(np.float32)
        if all(np.linalg.norm(g - z[f'v_gt_{k}'].astype(np.float32), axis=-1).mean() * 1000 > 120
               for k in chosen):
            chosen.append(j)
        if len(chosen) >= 40:
            break
    return np.array(chosen), s, pve


def crop(img):
    x = img.astype(np.float32).transpose(1, 2, 0)
    x = x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(x, 0, 1)


def meshes(z, j, res):
    import render_token_viz_blender as B
    B.RES = res
    faces = np.load(TOKENHMR / 'tokenization/output/token_viz/swap_geometry.npz')['faces']
    def up(v):
        w = v.astype(np.float32).copy(); w[:, 1] *= -1; w[:, 2] *= -1; return w
    vs = [up(z[f'v_gt_{j}'])] + [up(z[f'v_{k}_{j}']) for k in NAMES]
    d = max((np.abs(v - 0.5 * (v.max(0) + v.min(0))).max(0)[:2].max() * 1.1) / np.tan(B.YFOV / 2)
            for v in vs)
    grey = np.concatenate([np.tile(B.BASE_GREY, (len(vs[0]), 1)), np.ones((len(vs[0]), 1))], 1)
    return [B.render(v, faces, grey, d) for v in vs]


def sheet(z, order, s, pve, k, tag):
    cols = ['input', 'ground truth'] + [NICE[m].replace('\n', ' ') for m in NAMES]
    fig, ax = plt.subplots(k, 6, figsize=(13, 2.2 * k)); fig.patch.set_facecolor('white')
    for r, j in enumerate(order[:k]):
        ims = meshes(z, j, 280)
        ax[r, 0].imshow(crop(z[f'img_{j}']))
        for c, im in enumerate(ims):
            ax[r, c + 1].imshow(im)
        for c in range(6):
            ax[r, c].axis('off')
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=8.5)
        for c, m in enumerate(NAMES):
            ax[r, c + 2].set_title(f'{pve[m][j]:.0f} mm', fontsize=7.5,
                                   color='0.35') if r else None
        ax[r, 0].text(-0.06, 0.5, f'cand {r}\nframe {int(z[f"idx_{j}"])}\nscore {s[j]:.1f}',
                      transform=ax[r, 0].transAxes, ha='right', va='center', fontsize=8)
    fig.tight_layout()
    p = HERE / f'qualitative_sheet_{tag}.png'
    fig.savefig(p, dpi=68, facecolor='white', bbox_inches='tight')
    print(f'wrote {p}')


def final(z, js, pve):
    """Figure B.6, laid out as two tokenizer groups of two supervisions each."""
    rows = len(js)
    fig, ax = plt.subplots(rows, 6, figsize=(10.2, 2.15 * rows))
    fig.patch.set_facecolor('white')
    ax = np.atleast_2d(ax)
    for r, j in enumerate(js):
        ims = meshes(z, j, 900)
        ax[r, 0].imshow(crop(z[f'img_{j}']))
        for c, im in enumerate(ims):
            ax[r, c + 1].imshow(im)
        best = min(pve[m][j] for m in NAMES)
        J = [str(x) for x in z['joints']]; hidx = J.index('Head')
        hbest = min(float(z[f'joint_{m}_{j}'][hidx]) for m in NAMES)
        for c, m in enumerate(NAMES):
            v = pve[m][j]; hv = float(z[f'joint_{m}_{j}'][hidx])
            ax[r, c + 2].set_xlabel(
                f'{v:.0f} mm\nhead {hv:.0f}$^\\circ$', fontsize=8.4, linespacing=1.5,
                color='0.15' if (v == best or hv == hbest) else '0.45',
                fontweight='bold' if hv == hbest else 'normal')
        for c in range(6):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for sp in ax[r, c].spines.values():
                sp.set_visible(False)
    fig.subplots_adjust(left=0.008, right=0.992, top=0.855, bottom=0.05,
                        wspace=0.02, hspace=0.20)

    def cx(c):
        p = ax[0, c].get_position(); return 0.5 * (p.x0 + p.x1)
    for c, t in enumerate(['input', 'ground truth']):
        fig.text(cx(c), 0.895, t, ha='center', va='bottom', fontsize=10)
    for c0, t in [(2, 'Cosine $d = 4$'), (4, 'FSQ $d = 4$')]:
        a0, a1 = ax[0, c0].get_position(), ax[0, c0 + 1].get_position()
        fig.text(0.5 * (a0.x0 + a1.x1), 0.945, t, ha='center', va='bottom', fontsize=10.5)
        fig.add_artist(plt.Line2D([a0.x0 + 0.004, a1.x1 - 0.004], [0.938, 0.938],
                                  color='0.55', lw=0.8, transform=fig.transFigure))
    for c, t in zip(range(2, 6), ['pose loss', 'cross-entropy', 'pose loss', 'cross-entropy']):
        fig.text(cx(c), 0.895, t, ha='center', va='bottom', fontsize=9.2, color='0.3')

    for ca, cb in [(1, 2), (3, 4)]:          # after ground truth, and between tokenizers
        xg = 0.5 * (ax[0, ca].get_position().x1 + ax[0, cb].get_position().x0)
        fig.add_artist(plt.Line2D([xg, xg], [0.03, 0.93], color='0.85', lw=0.8,
                                  transform=fig.transFigure))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'qualitative.pdf', dpi=200, facecolor='white', bbox_inches='tight')
    fig.savefig(HERE / 'qualitative_preview.png', dpi=130, facecolor='white', bbox_inches='tight')
    for j in js:
        print(f'  frame {int(z[f"idx_{j}"])}: ' +
              '  '.join(f'{m} {pve[m][j]:.0f}' for m in NAMES))
    print(f'wrote {OUT / "qualitative.pdf"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--sheet', action='store_true')
    ap.add_argument('--by', default='distal', choices=['distal', 'cosfail', 'pve', 'headturn', 'headwin'])
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--picks', type=str)
    a = ap.parse_args()
    z = np.load(CACHE, allow_pickle=True)
    if a.stats: stats(z)
    order, s, pve = order_by(z, a.by)
    if a.sheet: sheet(z, order[a.offset:], s, pve, a.k, a.by + (f'_{a.offset}' if a.offset else ''))
    if a.picks: final(z, [order[int(x)] for x in a.picks.split(',')], pve)
