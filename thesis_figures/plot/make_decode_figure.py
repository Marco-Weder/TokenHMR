#!/usr/bin/env python
"""Assemble the chosen frames into the thesis figure for Results 4.4.

`render_decode_comparison.py` renders the whole browsing set. This script takes the
handful of frame ids picked out of it and lays them out as one publication figure,
reusing the stills that pass already wrote so the panels are guaranteed identical
to the ones that were reviewed.

Pick the frame ids from <out>/<dataset>/ranking.csv (the `dataset_index` column,
which is also the filename in compare/ and individual/).

    ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/make_decode_figure.py \
        --frames 23000,29500,4500

Output: <repo>/thesis/images/downstream/decode_qualitative.pdf
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from repro import paths

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENHMR = os.path.dirname(HERE)
REPO = os.path.dirname(TOKENHMR)
OUTDIR = str(paths.figure_out_dir("downstream"))

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'font.size': 8.5, 'pdf.fonttype': 42,
})

# (model key, column header). Keep to four columns: the point is baseline-soft vs
# baseline-hard vs a hard-decoding model, and more columns shrink every panel.
DEFAULT_COLUMNS = [
    ('fsq_soft', 'Baseline, soft decode'),
    ('fsq_hard', 'Baseline, hard decode'),
    ('chain_st', 'Gated CE $+$ ST (hard)'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True,
                    help='comma-separated dataset_index values from ranking.csv')
    ap.add_argument('--dataset', default='3DPW-TEST')
    ap.add_argument('--src', default=os.path.join(TOKENHMR, 'results/qualitative_decode'))
    ap.add_argument('--columns', default=','.join(k for k, _ in DEFAULT_COLUMNS),
                    help='comma-separated model keys, in order')
    ap.add_argument('--out', default=os.path.join(OUTDIR, 'decode_qualitative.pdf'))
    args = ap.parse_args()

    frames = [int(x) for x in args.frames.split(',')]
    headers = dict(DEFAULT_COLUMNS)
    cols = [(k, headers.get(k, k)) for k in args.columns.split(',')]

    root = os.path.join(args.src, args.dataset)
    cache = np.load(os.path.join(args.src, 'cache', f'images_{args.dataset}.npz'))
    index_of = {int(v): i for i, v in enumerate(cache['indices'])}

    n_rows, n_cols = len(frames), len(cols) + 1
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.55 * n_cols, 1.55 * n_rows + 0.25),
                             squeeze=False)

    for r, fid in enumerate(frames):
        if fid not in index_of:
            raise SystemExit(f'frame {fid} is not in the rendered set for {args.dataset}')
        axes[r][0].imshow(cache['images'][index_of[fid]])
        for c, (key, _label) in enumerate(cols, start=1):
            p = os.path.join(root, 'individual', key, f'{fid:06d}.png')
            if not os.path.exists(p):
                raise SystemExit(f'missing panel {p} (was --individual on?)')
            axes[r][c].imshow(plt.imread(p))

    for r in range(n_rows):
        for c in range(n_cols):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            for s in axes[r][c].spines.values():
                s.set_linewidth(0.4)
                s.set_color('#999999')

    for c, label in enumerate(['Input'] + [l for _, l in cols]):
        axes[0][c].set_title(label, fontsize=8.5, pad=4)

    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.005, right=0.995,
                        top=0.93, bottom=0.005)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight', pad_inches=0.01)
    print(f'wrote {args.out}  ({n_rows} frames x {n_cols} columns)')


if __name__ == '__main__':
    main()
