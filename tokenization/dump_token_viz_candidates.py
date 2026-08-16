#!/usr/bin/env python
"""Sweep candidate poses for Figure 4.6 and dump the swap geometry for each.

The original pose was chosen by max bbox diagonal over MOYO, which maximises how
extreme the pose is but not how legible the figure is. This dumps several candidates
(from MOYO and from the everyday AMASS validation subsets) so the clearest one can be
chosen, then re-run render_token_viz.py with TOKVIZ_POSE_IDX to lock it in.

Run from tokenization/ in the thesis-HMR env.
"""
import json
import os

import numpy as np
import torch

import render_token_viz as R          # reuses load_net / best_swap_dv / BODY / ORDER
from utils.rotation_conversions import axis_angle_to_matrix

PAIR = ["FSQ d4", "VQ d4"]
N_CAND = 10
OUT = os.path.join(R.HERE, "output", "token_viz", "pose_candidates.npz")


def candidate_poses(hparams, vallist, n, seed):
    from dataset.dataset_poseVQ import get_dataloader
    torch.manual_seed(seed)
    hparams.DATA.NUM_WORKERS = 4
    hparams.DATA.CACHE_SMPL = False
    saved = hparams.DATA.VALLIST
    hparams.DATA.VALLIST = vallist
    poses = []
    for batch in get_dataloader(hparams, split="val", shuffle=True):
        gp = (batch["gt_pose_body"].float().view(-1, 21, 3, 3) if "gt_pose_body" in batch
              else axis_angle_to_matrix(batch["pose_body_aa"].float().view(-1, 21, 3)))
        poses.append(gp.cpu())
        if sum(p.shape[0] for p in poses) >= n:
            break
    hparams.DATA.VALLIST = saved
    return torch.cat(poses, 0)[:n]


def main():
    runs = {r["label"]: os.path.join(R.HERE, r["ckpt"]) for r in json.load(open(R.STAB))["runs"]}
    _, hp0 = R.load_net(runs["CNN"])
    cands = torch.cat([candidate_poses(hp0, "MOYO", N_CAND // 2, 1),
                       candidate_poses(hp0, "HumanEva", N_CAND // 2, 2)], 0)
    print(f"{cands.shape[0]} candidate poses\n")

    nets = {}
    for lab in PAIR:
        n, _ = R.load_net(runs[lab]); nets[lab] = n.to(R.DEV)

    dump = {"faces": R.FACES}
    rows = []
    for i in range(cands.shape[0]):
        pose = cands[i][None]
        rec = {}
        for lab in PAIR:
            base, dv = R.best_swap_dv(nets[lab], pose)
            dump[f"base_{i}_{lab}"] = base
            dump[f"dv_{i}_{lab}"] = dv
            rec[lab] = dv
        dump[f"pose_{i}"] = pose.numpy()
        f, c = rec["FSQ d4"].max(), rec["VQ d4"].max()
        rows.append((int(i), float(f), float(c), float(f / max(c, 1e-9))))
        print(f"pose {i:2d}  FSQ max {f:5.2f} mm   cosine max {c:5.2f} mm   ratio {f/max(c,1e-9):4.2f}")

    dump["summary"] = np.array(json.dumps(rows))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **dump)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
