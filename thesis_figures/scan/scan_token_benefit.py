#!/usr/bin/env python
"""Find frames for Figure B.1 ("what tokenization buys").

The claim is that the token classifier can only emit poses assembled from codes fitted
to motion capture, while the continuous regressor is free to leave that space, and that
aggregate joint error does not penalise it for doing so. So we look for frames where

  * the CONTINUOUS prediction sits far off the pose manifold, measured as the residual
    when its predicted body pose is passed through the frozen tokenizer (encode then
    decode). A pose the codebook cannot represent comes back changed; one on the
    manifold comes back almost unaltered.
  * the two models' per-vertex errors against the ground truth are COMPARABLE, so the
    figure shows a plausibility difference rather than an accuracy difference.

Stage 1 (GPU) runs both models; stage 2 (CPU) scores the manifold residual.

    ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/scan_token_benefit.py --scan 300
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TOKENHMR = HERE.parent
CACHE = HERE / 'cache' / 'token_benefit.npz'
CACHE.parent.mkdir(parents=True, exist_ok=True)

CONT = TOKENHMR / 'logs/tokenhmr_continuous/runs/continuous_400k'
TOKD = TOKENHMR / 'logs/tokenhmr_chain2_st/runs/chain_st'


def load(run, device):
    from tokenhmr.lib.models import load_tokenhmr
    ck = sorted(Path(run, 'checkpoints').glob('epoch=*.ckpt'))[-1]
    m, cfg = load_tokenhmr(checkpoint_path=str(ck), model_cfg=str(Path(run, 'model_config.yaml')),
                           dataset_dir=str(TOKENHMR / 'dataset_dir/evaluation_data'))
    print(f'[load] {ck.name}', flush=True)
    return m.to(device).eval(), cfg


def main(a):
    from tokenhmr.lib.configs import dataset_eval_config
    from tokenhmr.lib.datasets import create_dataset
    from tokenhmr.lib.utils import recursive_to
    from tokenhmr.lib.models.smpl_wrapper import SMPL
    from tokenhmr.lib.utils.geometry import aa_to_rotmat

    dev = torch.device(a.device)
    m_c, cfg = load(CONT, dev)
    m_t, _ = load(TOKD, dev)
    smpl = SMPL(**{k.lower(): v for k, v in dict(cfg.SMPL).items()}).to(dev)

    ds_cfg = dataset_eval_config()[a.dataset]
    for k in ('DATASET_FILE', 'IMG_DIR'):
        if k in ds_cfg:
            ds_cfg[k] = os.path.join(str(TOKENHMR / 'dataset_dir/evaluation_data'), ds_cfg[k])
    full = create_dataset(cfg, ds_cfg, train=False)
    # Spread the sample over the WHOLE dataset. Taking the first N frames gives
    # consecutive frames of one clip, i.e. the same pose many times over.
    idx = np.linspace(0, len(full) - 1, min(a.scan, len(full))).astype(int)
    print(f'[data] {a.dataset}: {len(full)} frames, sampling {len(idx)} spread over all of it', flush=True)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full, idx.tolist()),
                                         batch_size=1, shuffle=False, num_workers=4)

    rec = []
    for i, batch in enumerate(loader):
        batch = recursive_to(batch, dev)
        if not bool(batch['has_smpl_params']['body_pose'].reshape(-1)[0] > 0.5):
            continue
        with torch.no_grad():
            oc, ot = m_c(batch), m_t(batch)
            gt_p = {k: v.float() for k, v in batch['smpl_params'].items()}
            is_aa = batch['smpl_params_is_axis_angle']
            for k in ('global_orient', 'body_pose'):
                gt_p[k] = (aa_to_rotmat(gt_p[k].reshape(-1, 3)).reshape(1, -1, 3, 3)
                           if bool(np.asarray(is_aa[k].cpu()).reshape(-1)[0])
                           else gt_p[k].reshape(1, -1, 3, 3))
            gt_p['betas'] = gt_p['betas'].reshape(1, -1)
            v_gt = smpl(**gt_p, pose2rot=False).vertices[0]
            v_c, v_t = oc['pred_vertices'][0], ot['pred_vertices'][0]
            c = lambda v: v - v.mean(0, keepdim=True)
            v_gt, v_c, v_t = c(v_gt), c(v_c), c(v_t)
            e_c = float((v_c - v_gt).norm(dim=-1).mean() * 1000)
            e_t = float((v_t - v_gt).norm(dim=-1).mean() * 1000)
            Rg = gt_p['body_pose'][0][:21]
            trg = Rg[:, 0, 0] + Rg[:, 1, 1] + Rg[:, 2, 2]
            artic = float(torch.rad2deg(torch.acos(((trg - 1) / 2).clamp(-1, 1))).mean())
            rec.append(dict(idx=int(idx[i]), e_c=e_c, e_t=e_t, artic=artic,
                            pose_c=oc['pred_smpl_params']['body_pose'][0, :21].cpu().numpy(),
                            pose_t=ot['pred_smpl_params']['body_pose'][0, :21].cpu().numpy(),
                            v_gt=v_gt.cpu().numpy(), v_c=v_c.cpu().numpy(), v_t=v_t.cpu().numpy(),
                            img=batch['img'][0].cpu().numpy()))
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(idx)}  kept {len(rec)}', flush=True)

    out = {'n': len(rec)}
    for j, r in enumerate(rec):
        for k in ('pose_c', 'pose_t', 'v_gt', 'v_c', 'v_t', 'img'):
            out[f'{k}_{j}'] = r[k]
        out[f'meta_{j}'] = np.array([r['idx'], r['e_c'], r['e_t'], r['artic']])
    np.savez_compressed(CACHE, **out)
    print(f'[cache] {CACHE}  ({len(rec)} frames)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', type=int, default=300)
    ap.add_argument('--dataset', default='3DPW-TEST')
    ap.add_argument('--device', default='cuda')
    main(ap.parse_args())
