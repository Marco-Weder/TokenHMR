#!/usr/bin/env python
"""Frames for Figure B.6: cosine-d4 and FSQ-d4, each pose-supervised and CE-supervised.

Records, per frame: each model's mesh, its per-vertex error, and its per-joint geodesic
rotation error against the ground-truth fit. The per-joint errors let the distal-joint
claim (head and feet recovered better under cross-entropy) be checked rather than
assumed, and let frames be ranked by it.

    ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/scan_qualitative.py --scan 400
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TOKENHMR = HERE.parent
sys.path.insert(0, str(TOKENHMR)); sys.path.insert(0, str(TOKENHMR / 'tokenhmr'))
CACHE = HERE / 'cache' / 'qualitative.npz'
CACHE.parent.mkdir(parents=True, exist_ok=True)

RUNS = [('cos4_pose', 'logs/tokenhmr_transformer_dim4/runs/tokenhmr_transformer_dim4_0'),
        ('cos4_ce',   'logs/tokenhmr_cetok_cosd4/runs/cetok_cosd4'),
        ('fsq4_pose', 'logs/tokenhmr_fsq/runs/tokenhmr_fsq_0'),
        ('fsq4_ce',   'logs/tokenhmr_chain2_gate/runs/chain_gate')]
JOINTS = ['L_Hip','R_Hip','Spine1','L_Knee','R_Knee','Spine2','L_Ankle','R_Ankle','Spine3',
          'L_Foot','R_Foot','Neck','L_Collar','R_Collar','Head','L_Shoulder','R_Shoulder',
          'L_Elbow','R_Elbow','L_Wrist','R_Wrist']


def load(run, device):
    from lib.models import load_tokenhmr
    d = TOKENHMR / run
    ck = sorted(Path(d, 'checkpoints').glob('epoch=*.ckpt'))[-1]
    m, cfg = load_tokenhmr(checkpoint_path=str(ck), model_cfg=str(d / 'model_config.yaml'),
                           dataset_dir=str(TOKENHMR / 'dataset_dir/evaluation_data'))
    print(f'[load] {run.split("/")[-1]}  {ck.name}', flush=True)
    return m.to(device).eval(), cfg


def geo(A, B):
    R = torch.matmul(A, B.transpose(-1, -2))
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1)))


def main(a):
    from lib.configs import dataset_eval_config
    from lib.datasets import create_dataset
    from lib.utils import recursive_to
    from lib.models.smpl_wrapper import SMPL
    from lib.utils.geometry import aa_to_rotmat

    dev = torch.device(a.device)
    models, cfg = {}, None
    for name, run in RUNS:
        models[name], c = load(run, dev)
        cfg = cfg or c
    smpl = SMPL(**{k.lower(): v for k, v in dict(cfg.SMPL).items()}).to(dev)

    ds_cfg = dataset_eval_config()[a.dataset]
    for k in ('DATASET_FILE', 'IMG_DIR'):
        if k in ds_cfg:
            ds_cfg[k] = os.path.join(str(TOKENHMR / 'dataset_dir/evaluation_data'), ds_cfg[k])
    full = create_dataset(cfg, ds_cfg, train=False)
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
            gt_p = {k: v.float() for k, v in batch['smpl_params'].items()}
            is_aa = batch['smpl_params_is_axis_angle']
            for k in ('global_orient', 'body_pose'):
                gt_p[k] = (aa_to_rotmat(gt_p[k].reshape(-1, 3)).reshape(1, -1, 3, 3)
                           if bool(np.asarray(is_aa[k].cpu()).reshape(-1)[0])
                           else gt_p[k].reshape(1, -1, 3, 3))
            gt_p['betas'] = gt_p['betas'].reshape(1, -1)
            v_gt = smpl(**gt_p, pose2rot=False).vertices[0]
            v_gt = v_gt - v_gt.mean(0, keepdim=True)
            gtR = gt_p['body_pose'][0][:21]
            r = dict(idx=int(idx[i]), v_gt=v_gt.cpu().numpy().astype(np.float16),
                     img=batch['img'][0].cpu().numpy().astype(np.float16),
                     # ground-truth per-joint rotation MAGNITUDE (deg from identity), so
                     # results can be stratified by how far a joint is turned, e.g. the
                     # head rotated away from the body
                     gtmag=geo(gtR, torch.eye(3, device=gtR.device).expand_as(gtR)).cpu().numpy())
            for name in models:
                o = models[name](batch)
                v = o['pred_vertices'][0]; v = v - v.mean(0, keepdim=True)
                r[f'v_{name}'] = v.cpu().numpy().astype(np.float16)
                r[f'pve_{name}'] = float((v - v_gt).norm(dim=-1).mean() * 1000)
                r[f'joint_{name}'] = geo(o['pred_smpl_params']['body_pose'][0][:21], gtR).cpu().numpy()
            rec.append(r)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(idx)}  kept {len(rec)}', flush=True)

    out = {'n': len(rec), 'joints': np.array(JOINTS), 'names': np.array(list(models))}
    for j, r in enumerate(rec):
        for k, v in r.items():
            out[f'{k}_{j}'] = v
    np.savez_compressed(CACHE, **out)
    print(f'[cache] {CACHE}  ({len(rec)} frames)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', type=int, default=400)
    ap.add_argument('--dataset', default='3DPW-TEST')
    ap.add_argument('--device', default='cuda')
    main(ap.parse_args())
