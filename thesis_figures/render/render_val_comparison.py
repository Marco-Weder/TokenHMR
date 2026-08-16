#!/usr/bin/env python
"""Qualitative no-CE vs CE comparison on the COCO-VAL validation set.

These are the frames that appear as `val/predictions` in the wandb runs, so what
this renders is directly comparable to what was watched during training.

Columns:
  fsq_soft      FSQ downstream model WITHOUT token cross-entropy. Decodes a
                softmax-weighted mixture of codes, so its pose need not be any
                single entry of the codebook.
  chain_gate    WITH gated cross-entropy, argmax decode
  chain_st      + straight-through
  chain_gumbel  + Gumbel sampling

Validation is a shuffled webdataset (an IterableDataset, so it cannot be indexed),
so "the same frames for every model" has to be arranged explicitly: one pool of
frames is pulled once, cached, and replayed through every model.

Random validation frames are mostly easy standing and walking poses, where all the
models agree. To find the frames that separate them, the pool is scored before
rendering and only the top --top are drawn. See --sort.

    # 512-frame pool, render the 48 most off-manifold
    ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/render_val_comparison.py \
        --n 512 --top 48 --sort offmanifold

Output: tokenhmr/results/qualitative_val/
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import sys
import argparse

import numpy as np
import torch
import cv2
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENHMR = os.path.dirname(HERE)
from tokenhmr.lib.configs import dataset_config                     # noqa: E402
from tokenhmr.lib.datasets import MixedWebDataset                   # noqa: E402
from tokenhmr.lib.models import load_tokenhmr                       # noqa: E402
from tokenhmr.lib.models.smpl_wrapper import SMPL                   # noqa: E402
from tokenhmr.lib.utils import recursive_to                         # noqa: E402

from render_decode_comparison import (                     # noqa: E402
    MODELS, LOGS, IMAGENET_MEAN, IMAGENET_STD, Renderer, set_decode_mode, _label)


def slice_batch(batch, s, e):
    """Slice a nested batch dict along the batch dimension."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[s:e]
        elif isinstance(v, dict):
            out[k] = {kk: (vv[s:e] if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        elif isinstance(v, list):
            out[k] = v[s:e]
        else:
            out[k] = v
    return out


def grab_pool(args, model_cfg):
    """Pull --n validation samples once and cache them, so all models see the same frames."""
    dst = os.path.join(args.out, f'val_pool_{args.n}.pt')
    if os.path.exists(dst) and not args.overwrite:
        print(f'[pool] cached: {dst}')
        return torch.load(dst)
    ds = MixedWebDataset(model_cfg, dataset_config(), train=False)
    loader = torch.utils.data.DataLoader(ds, args.n, drop_last=True, num_workers=1)
    batch = next(iter(loader))
    torch.save(batch, dst)
    print(f'[pool] wrote {dst} ({args.n} COCO-VAL frames)')
    return batch


def mesh_dist(a, b):
    """Mean per-vertex distance in mm after removing translation.

    Centring means this measures a change in the body itself, not in where the
    camera put it, which is what matters when comparing pose predictions.
    """
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return np.linalg.norm(a - b, axis=2).mean(1) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=512, help='validation frames in the scored pool')
    ap.add_argument('--top', type=int, default=48, help='how many to render (0 = all)')
    ap.add_argument('--sort', default='offmanifold',
                    choices=['offmanifold', 'ce_change', 'none'],
                    help='offmanifold: no-CE mixture vs its own nearest code. '
                         'ce_change: how much adding CE moves the body.')
    ap.add_argument('--dataset_dir', default=os.path.join(TOKENHMR, 'dataset_dir'))
    ap.add_argument('--out', default=os.path.join(TOKENHMR, 'results/qualitative_val'))
    ap.add_argument('--render_models', default='fsq_soft,chain_gate,chain_st,chain_gumbel',
                    help='columns to draw, in order')
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--per_page', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    render_keys = args.render_models.split(',')
    # fsq_hard is always run: it is what makes the offmanifold score computable. It is
    # only drawn if the user asks for it.
    run_keys = list(dict.fromkeys(render_keys + ['fsq_soft', 'fsq_hard']))
    models = [m for m in MODELS if m[0] in run_keys]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _m, model_cfg = load_tokenhmr(
        checkpoint_path=os.path.join(LOGS, MODELS[0][1], 'checkpoints', MODELS[0][2]),
        model_cfg=os.path.join(LOGS, MODELS[0][1], 'model_config.yaml'),
        dataset_dir=args.dataset_dir)
    del _m
    batch = grab_pool(args, model_cfg)

    imgs = batch['img'].cpu().numpy().transpose(0, 2, 3, 1)
    imgs = (np.clip(imgs * IMAGENET_STD + IMAGENET_MEAN, 0, 1) * 255).astype(np.uint8)
    n = imgs.shape[0]

    preds, caps = {}, {}
    for label, run, ckpt, mode, cap in models:
        ckpt_path = os.path.join(LOGS, run, 'checkpoints', ckpt)
        if not os.path.exists(ckpt_path):
            print(f'[run] {label}: MISSING {ckpt_path}, skipping')
            continue
        model, _cfg = load_tokenhmr(checkpoint_path=ckpt_path,
                                    model_cfg=os.path.join(LOGS, run, 'model_config.yaml'),
                                    dataset_dir=args.dataset_dir)
        model = model.to(device).eval()
        ok, why = set_decode_mode(model, mode)
        print(f'[run] {label}: decode={mode} ({why})' + ('' if ok else '  <-- INERT'))
        v, c, fl = [], [], []
        for s in range(0, n, args.chunk):
            with torch.no_grad():
                out = model(recursive_to(slice_batch(batch, s, s + args.chunk), device))
            b = out['pred_vertices'].shape[0]
            v.append(out['pred_vertices'].detach().reshape(b, -1, 3).cpu().numpy())
            c.append(out['pred_cam_t'].detach().reshape(b, 3).cpu().numpy())
            fl.append(out['focal_length'].detach().reshape(b, 2).cpu().numpy())
        preds[label] = {'vertices': np.concatenate(v), 'cam_t': np.concatenate(c),
                        'focal': np.concatenate(fl)}
        caps[label] = cap
        del model
        torch.cuda.empty_cache()

    scores = {}
    if 'fsq_soft' in preds and 'fsq_hard' in preds:
        scores['offmanifold'] = mesh_dist(preds['fsq_soft']['vertices'],
                                          preds['fsq_hard']['vertices'])
    if 'fsq_soft' in preds and 'chain_st' in preds:
        scores['ce_change'] = mesh_dist(preds['fsq_soft']['vertices'],
                                        preds['chain_st']['vertices'])

    if args.sort in scores:
        ranking = np.argsort(-scores[args.sort])
        s = scores[args.sort]
        print(f'[rank] {args.sort} over {n} frames: max {s[ranking[0]]:.1f} mm, '
              f'p90 {np.percentile(s, 90):.1f} mm, median {np.median(s):.1f} mm')
    else:
        ranking = np.arange(n)

    keys = sorted(scores)
    with open(os.path.join(args.out, 'scores.csv'), 'w') as f:
        f.write('rank,frame,' + ','.join(keys) + '\n')
        for rank, i in enumerate(ranking):
            f.write(f'{rank},{i},' + ','.join(f'{scores[k][i]:.2f}' for k in keys) + '\n')

    order = ranking[:args.top] if args.top else ranking
    labels = [k for k in render_keys if k in preds]

    faces = SMPL(**{k.lower(): v for k, v in dict(model_cfg.SMPL).items()}).faces
    r = Renderer(faces, imgs.shape[1])
    for sub in ['compare', 'sheets'] + [f'individual/{l}' for l in labels]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    strips = []
    for rank, i in enumerate(tqdm(order, desc='render/COCO-VAL')):
        tag = f'{rank:03d}_f{i:04d}'
        tiles = [_label(imgs[i], f'input  #{i}')]
        for label in labels:
            p = preds[label]
            front = r(p['vertices'][i], p['cam_t'][i], imgs[i], float(p['focal'][i][0]))
            side = r(p['vertices'][i], p['cam_t'][i], imgs[i], float(p['focal'][i][0]), side=True)
            note = f'  off {scores["offmanifold"][i]:.0f}mm' if label == 'fsq_soft' and \
                'offmanifold' in scores else ''
            tiles.append(_label(np.concatenate([front, side], 1), caps[label] + note))
            cv2.imwrite(os.path.join(args.out, 'individual', label, f'{tag}.png'),
                        cv2.cvtColor(front, cv2.COLOR_RGB2BGR))
        h = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0)), constant_values=255)
                 for t in tiles]
        strip = np.concatenate(tiles, 1)
        cv2.imwrite(os.path.join(args.out, 'compare', f'{tag}.png'),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        strips.append(strip)

    for p in range((len(strips) + args.per_page - 1) // args.per_page):
        chunk = strips[p * args.per_page:(p + 1) * args.per_page]
        width = max(c.shape[1] for c in chunk)
        chunk = [np.pad(c, ((0, 0), (0, width - c.shape[1]), (0, 0)), constant_values=255)
                 for c in chunk]
        sheet = np.concatenate([x for c in chunk for x in
                                (c, np.full((6, width, 3), 200, np.uint8))], 0)
        cv2.imwrite(os.path.join(args.out, 'sheets', f'page{p:03d}.png'),
                    cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    print(f'\nrendered {len(order)} of {n} COCO-VAL frames to {args.out}')
    print('  columns: ' + ' | '.join(labels))
    print(f'  ranked by {args.sort}; filenames are <rank>_f<frame>, scores.csv has the pool')


if __name__ == '__main__':
    main()
