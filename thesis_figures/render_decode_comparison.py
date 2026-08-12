#!/usr/bin/env python
"""Qualitative soft-vs-hard decode comparison for Results 4.4.

Renders the SAME frames through several token-decoding configurations so the
visual claim behind Table 4.6 (`tab:decoding-gap`) can be checked by eye: a model
trained with a softmax-weighted decode is free to mix codebook entries, which the
metrics only partly capture, while every hard-decoding model emits exactly one
code per token and therefore stays inside the codebook.

Five columns, chosen so the comparison is controlled in both directions:

  fsq_soft      the pose-supervised FSQ baseline, decoded the way it was trained
  fsq_hard      the SAME checkpoint decoded argmax -- isolates the decode alone
  chain_gate    gated token CE, hard decode, severed pose gradient
  chain_st      + straight-through decode
  chain_gumbel  + Gumbel sampling

fsq_soft vs fsq_hard is one checkpoint under two decodes (the decoding gap).
fsq_soft vs chain_* is soft-trained vs hard-trained (the recipe).

Two passes, cached separately so re-rendering never re-runs the network:

  collect  one Subset pass per model; stores predicted vertices, camera and
           per-sample PA-MPJPE/MPJPE/PVE into <out>/cache/<model>_<ds>.npz
  render   one pyrender pass; writes per-model stills, per-frame comparison
           strips, ranked contact sheets and ranking.csv

Run with the project env, from the tokenhmr folder:
    ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/render_decode_comparison.py \
        --dataset 3DPW-TEST --stride 20

Output goes to tokenhmr/results/qualitative_decode/ -- deliberately NOT into the
thesis repo, which syncs to Overleaf and should only ever receive the handful of
frames that are actually chosen.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import sys
import csv
import argparse

import numpy as np
import torch
import cv2
import trimesh
import pyrender
import smplx
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENHMR = os.path.dirname(HERE)                     # <repo>/tokenhmr
sys.path.insert(0, os.path.join(TOKENHMR, 'tokenhmr'))

from lib.configs import dataset_eval_config          # noqa: E402
from lib.datasets import create_dataset              # noqa: E402
from lib.models import load_tokenhmr                 # noqa: E402
from lib.models.smpl_wrapper import SMPL             # noqa: E402
from lib.utils import Evaluator, recursive_to        # noqa: E402

LOGS = os.path.join(TOKENHMR, 'logs')

# (label, run directory under logs/, checkpoint file, decode mode, caption)
MODELS = [
    ('fsq_soft', 'tokenhmr_fsq/runs/tokenhmr_fsq_0',
     'epoch=9-step=600000.ckpt', 'soft', 'FSQ baseline (soft decode)'),
    ('fsq_hard', 'tokenhmr_fsq/runs/tokenhmr_fsq_0',
     'epoch=9-step=600000.ckpt', 'hard', 'FSQ baseline (hard decode)'),
    ('chain_gate', 'tokenhmr_chain2_gate/runs/chain_gate',
     'last.ckpt', 'hard', 'gated CE (hard)'),
    ('chain_st', 'tokenhmr_chain2_st/runs/chain_st',
     'last.ckpt', 'hard', '+ straight-through'),
    ('chain_gumbel', 'tokenhmr_chain2_gumbel/runs/chain_gumbel',
     'last.ckpt', 'hard', '+ Gumbel'),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _dataset_cfg(args):
    """Eval dataset config with dataset_dir prefixed, exactly as eval.py does it."""
    cfg = dataset_eval_config()[args.dataset]
    if 'DATASET_FILE' in cfg:
        cfg['DATASET_FILE'] = os.path.join(args.dataset_dir, cfg['DATASET_FILE'])
    if 'IMG_DIR' in cfg:
        cfg['IMG_DIR'] = os.path.join(args.dataset_dir, cfg['IMG_DIR'])
    return cfg


def selected_indices(n, stride, limit):
    idx = list(range(0, n, max(1, stride)))
    return idx[:limit] if limit else idx


def set_decode_mode(model, mode):
    """Point the token head at `mode`, and report whether it can actually honour it.

    Runs with MODEL.VQHPS_HARD_DECODE=true decode argmax in the forward pass no
    matter what `decode_mode` says (token_classifier.py), so for those the flag is
    inert and 'soft' would silently measure the hard decode. The caller needs to
    know that, otherwise a column of the figure would be a duplicate.
    """
    head = getattr(getattr(model, 'smpl_head', None), 'decpose', None)
    if head is None:
        return False, 'no token head'
    forced_hard = bool(getattr(head, 'vqhps_hard_decode', False))
    if hasattr(head, 'decode_mode'):
        head.decode_mode = mode
    if forced_hard:
        return (mode == 'hard'), 'vqhps_hard_decode=True, always argmax'
    return True, 'decode_mode honoured'


def collect(args):
    """One Subset forward pass per model; cache vertices, camera and per-sample error."""
    cache_dir = os.path.join(args.out, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_images = False

    for label, run, ckpt, mode, _cap in MODELS:
        dst = os.path.join(cache_dir, f'{label}_{args.dataset}.npz')
        if os.path.exists(dst) and not args.overwrite:
            print(f'[collect] {label}: cached, skipping')
            continue

        ckpt_path = os.path.join(LOGS, run, 'checkpoints', ckpt)
        cfg_path = os.path.join(LOGS, run, 'model_config.yaml')
        if not os.path.exists(ckpt_path):
            print(f'[collect] {label}: MISSING {ckpt_path}, skipping')
            continue

        model, model_cfg = load_tokenhmr(checkpoint_path=ckpt_path,
                                         model_cfg=cfg_path,
                                         dataset_dir=args.dataset_dir)
        model = model.to(device).eval()
        ok, why = set_decode_mode(model, mode)
        print(f'[collect] {label}: decode={mode} ({why})' + ('' if ok else '  <-- INERT'))

        ds_cfg = _dataset_cfg(args)
        dataset = create_dataset(model_cfg, ds_cfg, train=False)
        picks = selected_indices(len(dataset), args.stride, args.max)
        subset = torch.utils.data.Subset(dataset, picks)
        loader = torch.utils.data.DataLoader(subset, args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)

        j_reg = smplx.SMPL(model_path=model_cfg.SMPL.MODEL_PATH).J_regressor.to(device).float()
        evaluator = Evaluator(dataset_length=len(picks), keypoint_list=ds_cfg.KEYPOINT_LIST,
                              pelvis_ind=model_cfg.EXTRA.PELVIS_IND,
                              metrics=['mode_re', 'mode_mpjpe', 'mode_pve'],
                              J_regressor_24_SMPL=j_reg, dataset=args.dataset)

        verts, cams, focals, imgs = [], [], [], []
        for batch in tqdm(loader, desc=f'{label}/{args.dataset}'):
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            evaluator(out, batch)
            b = batch['keypoints_2d'].shape[0]
            verts.append(out['pred_vertices'].detach().reshape(b, -1, 3).cpu().numpy().astype(np.float16))
            cams.append(out['pred_cam_t'].detach().reshape(b, 3).cpu().numpy().astype(np.float32))
            focals.append(out['focal_length'].detach().reshape(b, 2).cpu().numpy().astype(np.float32))
            if not saved_images:
                im = batch['img'].cpu().numpy().transpose(0, 2, 3, 1)
                im = np.clip(im * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
                imgs.append((im * 255).astype(np.uint8))

        n = len(picks)
        np.savez_compressed(
            dst,
            indices=np.array(picks),
            vertices=np.concatenate(verts)[:n],
            cam_t=np.concatenate(cams)[:n],
            focal=np.concatenate(focals)[:n],
            pa_mpjpe=evaluator.mode_re[:n],
            mpjpe=evaluator.mode_mpjpe[:n],
            pve=evaluator.mode_pve[:n],
            decode_effective=np.array([ok]),
        )
        # Evaluator already reports millimetres.
        print(f'[collect] {label}: wrote {dst}  '
              f'(PA-MPJPE {np.mean(evaluator.mode_re[:n]):.2f} mm over {n} frames)')

        if not saved_images:
            np.savez_compressed(os.path.join(cache_dir, f'images_{args.dataset}.npz'),
                                images=np.concatenate(imgs)[:n], indices=np.array(picks))
            saved_images = True

        del model
        torch.cuda.empty_cache()


class Renderer:
    """Single reusable offscreen renderer.

    MeshRenderer.__call__ builds and deletes an OffscreenRenderer per image, which
    dominates the runtime when rendering thousands of frames. One context is
    created here and reused for every mesh.
    """

    def __init__(self, faces, res):
        self.faces = faces
        self.res = res
        self.r = pyrender.OffscreenRenderer(viewport_width=res, viewport_height=res,
                                            point_size=1.0)
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))

    def __call__(self, vertices, cam_t, image, focal, side=False):
        cam_t = cam_t.copy()
        cam_t[0] *= -1.0
        mesh = trimesh.Trimesh(vertices.astype(np.float64), self.faces.copy())
        if side:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=self.material), 'mesh')
        pose = np.eye(4)
        pose[:3, 3] = cam_t
        scene.add(pyrender.IntrinsicsCamera(fx=focal, fy=focal,
                                            cx=self.res / 2., cy=self.res / 2.), pose=pose)
        for node in _raymond_lights():
            scene.add_node(node)
        color, _ = self.r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        mask = (color[:, :, -1] > 0.8)[:, :, None]
        bg = np.ones_like(color[:, :, :3]) if side else image.astype(np.float32) / 255.0
        out = color[:, :, :3] * mask + (1 - mask) * bg
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _raymond_lights():
    thetas = np.pi * np.array([1. / 6., 1. / 6., 1. / 6.])
    phis = np.pi * np.array([0., 2. / 3., 4. / 3.])
    nodes = []
    for phi, theta in zip(phis, thetas):
        z = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        x = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(x) == 0 else x / np.linalg.norm(x)
        m = np.eye(4)
        m[:3, :3] = np.c_[x, np.cross(z, x), z]
        nodes.append(pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                                   matrix=m))
    return nodes


def _label(img, text, height=22):
    bar = np.full((height, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (4, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def render(args):
    cache_dir = os.path.join(args.out, 'cache')
    data, caps = {}, {}
    for label, _run, _ckpt, _mode, cap in MODELS:
        p = os.path.join(cache_dir, f'{label}_{args.dataset}.npz')
        if os.path.exists(p):
            # Materialise every array up front. NpzFile is lazy and these caches are
            # compressed, so `f['vertices'][i]` inside the render loop would inflate
            # the whole ~300 MB array once per frame (measured: 6 s/frame vs 6/s).
            with np.load(p) as f:
                data[label] = {k: f[k] for k in f.files}
            caps[label] = cap
    if not data:
        sys.exit('no cached predictions; run --stage collect first')
    images = np.load(os.path.join(cache_dir, f'images_{args.dataset}.npz'))['images']
    labels = [m[0] for m in MODELS if m[0] in data]
    picks = data[labels[0]]['indices']
    res = images.shape[1]

    faces = SMPL(**{k.lower(): v for k, v in
                    dict(load_cfg(args).SMPL).items()}).faces
    r = Renderer(faces, res)

    for sub in ['compare', 'sheets'] + [f'individual/{l}' for l in labels]:
        os.makedirs(os.path.join(args.out, args.dataset, sub), exist_ok=True)

    # Ranking: how far the soft baseline's mesh sits from each hard-decoding model,
    # in mean per-vertex distance after removing translation. Large values are the
    # frames where the decode actually changes the body, which is what to look at.
    rank_rows, strips = [], []
    for i in tqdm(range(len(picks)), desc=f'render/{args.dataset}'):
        img = images[i]
        tiles = [_label(img, 'input')]
        row = {'pos': i, 'dataset_index': int(picks[i])}
        base_v = data['fsq_soft']['vertices'][i].astype(np.float32) if 'fsq_soft' in data else None
        for label in labels:
            d = data[label]
            v = d['vertices'][i].astype(np.float32)
            tile = r(v, d['cam_t'][i], img, float(d['focal'][i][0]))
            side = r(v, d['cam_t'][i], img, float(d['focal'][i][0]), side=True)
            # Evaluator already stores millimetres (pose_utils.py multiplies by 1000).
            pa = float(d['pa_mpjpe'][i])
            row[f'{label}_pa_mpjpe'] = round(pa, 2)
            row[f'{label}_mpjpe'] = round(float(d['mpjpe'][i]), 2)
            if base_v is not None and label != 'fsq_soft':
                a = base_v - base_v.mean(0)
                b = v - v.mean(0)
                row[f'{label}_vs_soft_mm'] = round(float(np.linalg.norm(a - b, axis=1).mean()) * 1000, 2)
            tiles.append(_label(np.concatenate([tile, side], 1), f'{caps[label]}  PA {pa:.0f}mm'))
            if args.individual:
                cv2.imwrite(os.path.join(args.out, args.dataset, 'individual', label,
                                         f'{picks[i]:06d}.png'),
                            cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

        h = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
        strip = np.concatenate(tiles, 1)
        cv2.imwrite(os.path.join(args.out, args.dataset, 'compare', f'{picks[i]:06d}.png'),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        strips.append(strip)
        rank_rows.append(row)

    # Rank by the biggest soft-vs-hard mesh disagreement, so the frames where the
    # two decodes genuinely disagree come first instead of being buried.
    key = 'chain_st_vs_soft_mm' if 'chain_st_vs_soft_mm' in rank_rows[0] else None
    order = sorted(range(len(rank_rows)),
                   key=lambda k: -rank_rows[k].get(key, 0)) if key else list(range(len(rank_rows)))

    with open(os.path.join(args.out, args.dataset, 'ranking.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
        w.writeheader()
        for k in order:
            w.writerow(rank_rows[k])

    # Sheets cover only the top of the ranking. Every frame still has its own strip
    # in compare/, but a sheet per 8 frames over the whole set would be thousands of
    # pages, which defeats the point of having a ranked shortlist.
    per_page = args.per_page
    top = order[:args.sheet_top] if args.sheet_top else order
    for p in range((len(top) + per_page - 1) // per_page):
        chunk = [strips[k] for k in top[p * per_page:(p + 1) * per_page]]
        width = max(c.shape[1] for c in chunk)
        chunk = [np.pad(c, ((0, 0), (0, width - c.shape[1]), (0, 0)), constant_values=255)
                 for c in chunk]
        sheet = np.concatenate([x for c in chunk for x in
                                (c, np.full((6, width, 3), 200, np.uint8))], 0)
        cv2.imwrite(os.path.join(args.out, args.dataset, 'sheets', f'page{p:03d}.png'),
                    cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    print(f'wrote {len(picks)} frames to {os.path.join(args.out, args.dataset)}')
    print(f'  compare/    per-frame strips (input + {len(labels)} models, front and side)')
    print(f'  sheets/     {(len(top) + per_page - 1) // per_page} contact sheets over the top '
          f'{len(top)} frames, ranked by soft-vs-hard mesh disagreement')
    print(f'  ranking.csv per-frame PA-MPJPE per model + disagreement, same order as the sheets')


def _sort_value(row, key):
    """Ranking keys, including two derived ones.

    ce_gain      how much cross-entropy improves this frame, in PA-MPJPE. The
                 headline comparison: no-CE baseline minus the CE+ST model.
    offmanifold  how far the no-CE baseline's softmax-mixture pose sits from its
                 own nearest code, i.e. how deep into the convex hull of the
                 codebook that frame's prediction is. High values are the frames
                 where the baseline is relying on a pose no single code encodes.
    """
    if key == 'ce_gain':
        return float(row['fsq_soft_pa_mpjpe']) - float(row['chain_st_pa_mpjpe'])
    if key == 'offmanifold':
        return float(row['fsq_hard_vs_soft_mm'])
    return float(row[key])


def sheets(args):
    """Rebuild contact sheets from strips that are already rendered.

    Re-ranking should not cost another render pass, so this reads ranking.csv and
    the existing compare/ strips and writes a sheet directory per sort key.
    """
    root = os.path.join(args.out, args.dataset)
    with open(os.path.join(root, 'ranking.csv')) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit('ranking.csv is empty; run --stage render first')
    rows.sort(key=lambda r: -_sort_value(r, args.sort))
    top = rows[:args.sheet_top] if args.sheet_top else rows

    dst = os.path.join(root, f'sheets_{args.sort}')
    os.makedirs(dst, exist_ok=True)
    per_page = args.per_page
    for p in range((len(top) + per_page - 1) // per_page):
        chunk = []
        for r in top[p * per_page:(p + 1) * per_page]:
            img = cv2.imread(os.path.join(root, 'compare', f"{int(r['dataset_index']):06d}.png"))
            if img is not None:
                chunk.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not chunk:
            continue
        width = max(c.shape[1] for c in chunk)
        chunk = [np.pad(c, ((0, 0), (0, width - c.shape[1]), (0, 0)), constant_values=255)
                 for c in chunk]
        sheet = np.concatenate([x for c in chunk for x in
                                (c, np.full((6, width, 3), 200, np.uint8))], 0)
        cv2.imwrite(os.path.join(dst, f'page{p:03d}.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    with open(os.path.join(dst, 'order.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in top:
            w.writerow(r)
    print(f'wrote {(len(top) + per_page - 1) // per_page} sheets to {dst} '
          f'(top {len(top)} by {args.sort})')


def load_cfg(args):
    from lib.configs import get_config
    return get_config(os.path.join(LOGS, MODELS[0][1], 'model_config.yaml'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='3DPW-TEST', choices=['3DPW-TEST', 'EMDB'])
    ap.add_argument('--dataset_dir', default=os.path.join(TOKENHMR, 'dataset_dir/evaluation_data'))
    ap.add_argument('--out', default=os.path.join(TOKENHMR, 'results/qualitative_decode'))
    ap.add_argument('--stride', type=int, default=20, help='take every Nth frame (1 = all)')
    ap.add_argument('--max', type=int, default=0, help='cap on frames (0 = no cap)')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--per_page', type=int, default=8)
    ap.add_argument('--sheet_top', type=int, default=240,
                    help='build contact sheets for the top N ranked frames (0 = all)')
    ap.add_argument('--individual', action='store_true', default=True)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--stage', default='all', choices=['all', 'collect', 'render', 'sheets'])
    ap.add_argument('--sort', default='ce_gain',
                    help="sheets stage ranking: 'ce_gain' (no-CE minus CE+ST PA-MPJPE), "
                         "'offmanifold', or any ranking.csv column")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.stage in ('all', 'collect'):
        collect(args)
    if args.stage in ('all', 'render'):
        render(args)
    if args.stage == 'sheets':
        sheets(args)


if __name__ == '__main__':
    main()
