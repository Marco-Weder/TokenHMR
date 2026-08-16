"""Per-tokenizer data for the token-visualization figures (Fig 4.5, B.1, B.3).

For each of the eight tokenizers (same set / shared 2048-pose val sample as Table 4.3)
it computes and caches:
  - influence[T, J]: mean per-joint geodesic rotation (deg) induced when each latent
    token is swapped for a random donor's (the token->joint influence map);
  - a subsample of pre-quant token latents + the codebook, for the latent-space scatter.

Plotting is done separately by thesis_figures/plot_influence_maps.py and
plot_latent_scatter.py. Heavy math on CPU (GPU usually busy; model hardcodes .cuda()).

Run from tokenization/ in the thesis-HMR env:
    ~/miniconda3/envs/thesis-HMR/bin/python analyze_token_maps.py

Output: output/token_maps/<label>.npz  (+ index.json)
"""
import json
import os

import numpy as np
import torch

from tokenization.analysis import analyze_latent_pose_info as ali
ali.DEVICE = torch.device("cpu")
# Both tokenizer families keep a module-level SMPL body model on cuda; the CNN decoder
# even runs it inside forward(). Move them to the CPU so the whole pass stays on CPU.
from tokenization.models import transformer_pose_vqvae as _tpv
_tpv.body_model = _tpv.body_model.to("cpu")
from tokenization.models import vanilla_pose_vqvae as _vpv
_vpv.body_model = _vpv.body_model.to("cpu")
from tokenization.analysis.analyze_latent_pose_info import (load_net, get_codebook, _is_cosine, encode_full,
                                      decode_from_codes, pose6d_to_rotmat, geodesic_deg,
                                      JOINT_NAMES)
from tokenization.utils.rotation_conversions import axis_angle_to_matrix
import math

from repro import paths

DEV = torch.device("cpu")


def load_poses_cpu(hparams, n, seed):
    """Balanced val pose sample (n,21,3,3) read straight from the batches on CPU
    (avoids gt_from_batch, which hardcodes .cuda())."""
    from tokenization.dataset.dataset_poseVQ import get_dataloader
    torch.manual_seed(seed)
    hparams.DATA.NUM_WORKERS = min(getattr(hparams.DATA, "NUM_WORKERS", 4), 4)
    hparams.DATA.CACHE_SMPL = False
    ds_list = hparams.DATA.VALLIST.split("_")
    per = math.ceil(n / len(ds_list))
    saved, out = hparams.DATA.VALLIST, []
    for ds in ds_list:
        hparams.DATA.VALLIST = ds
        got = 0
        for batch in get_dataloader(hparams, split="val", shuffle=True):
            if "gt_pose_body" in batch:
                gp = batch["gt_pose_body"].float().view(-1, 21, 3, 3)
            else:
                gp = axis_angle_to_matrix(batch["pose_body_aa"].float().view(-1, 21, 3))
            out.append(gp[:per - got].cpu())
            got += out[-1].shape[0]
            if got >= per:
                break
    hparams.DATA.VALLIST = saved
    return torch.cat(out, 0)[:n]
HERE = os.path.dirname(os.path.abspath(__file__))
STAB = os.path.join(str(paths.TOKENIZER_OUT), "token_stability", "summary.json")
OUTDIR = os.path.join(str(paths.TOKENIZER_OUT), "token_maps")
N_INFL = 256          # poses for the influence estimate (160 decodes each -> keep modest)
N_LAT = 4000          # token-latents to keep for the scatter


def family(net, cosine):
    if str(net.quant).startswith("fsq"):
        return "fsq"
    if cosine:
        return "cos"
    return "conv" if not hasattr(net, "joint_queries") else "tfl2"


@torch.no_grad()
def encode(net, poses, batch=128):
    pres, idxs = [], []
    for i in range(0, poses.shape[0], batch):
        enc = encode_full(net, poses[i:i + batch].to(DEV).float())
        pres.append(enc["pre"].cpu()); idxs.append(enc["idx"].cpu())
    return torch.cat(pres, 0), torch.cat(idxs, 0)


@torch.no_grad()
def influence(net, idx, batch=128):
    T, J = idx.shape[1], net.num_joints
    def dec(ids):
        outs = []
        for i in range(0, ids.shape[0], batch):
            outs.append(pose6d_to_rotmat(net, decode_from_codes(net, ids[i:i + batch].to(DEV))).cpu())
        return torch.cat(outs, 0)
    base = dec(idx)
    g = torch.Generator().manual_seed(0)
    donor = idx[torch.randperm(idx.shape[0], generator=g)]
    infl = torch.zeros(T, J)
    for t in range(T):
        pert = idx.clone(); pert[:, t] = donor[:, t]
        infl[t] = geodesic_deg(base, dec(pert)).mean(0)
    return infl.numpy()                                    # (T, J) deg


def main():
    runs = json.load(open(STAB))["runs"]
    _, hp0 = load_net(os.path.join(HERE, runs[0]["ckpt"]))
    poses = load_poses_cpu(hp0, N_INFL, 0)
    os.makedirs(OUTDIR, exist_ok=True)
    index = []
    for r in runs:
        label = r["label"]
        ckpt = r["ckpt"]
        net, _ = load_net(ckpt); net = net.to(DEV); torch.cuda.empty_cache()
        cosine = _is_cosine(net)
        cb = get_codebook(net).detach().float().cpu()
        fam = family(net, cosine)
        pre, idx = encode(net, poses)
        infl = influence(net, idx)

        # latents for scatter (bound FSQ like analyze_run)
        lat = pre.reshape(-1, cb.shape[1])
        if str(net.quant).startswith("fsq"):
            q = net.quantizer; half = (q._levels // 2).float()
            lat = (q.bound(lat.to(DEV)) / half).cpu()
        g = torch.Generator().manual_seed(0)
        sub = torch.randperm(lat.shape[0], generator=g)[:N_LAT]
        lat = lat[sub].numpy()

        safe = label.replace(" ", "_")
        np.savez(os.path.join(OUTDIR, f"{safe}.npz"), influence=infl, latents=lat,
                 codebook=cb.numpy(), cosine=cosine, d=cb.shape[1], K=cb.shape[0],
                 quant=("FSQ" if fam == "fsq" else "EMA"),
                 metric=("cosine" if cosine else "l2"), family=fam, label=label,
                 joint_names=np.array(JOINT_NAMES))
        index.append({"label": label, "file": f"{safe}.npz", "family": fam,
                      "d": int(cb.shape[1]), "K": int(cb.shape[0]),
                      "quant": ("FSQ" if fam == "fsq" else "EMA"),
                      "metric": ("cosine" if cosine else "l2")})
        print(f"{label:20s} infl mean {infl.mean():.3f} deg  d={cb.shape[1]} K={cb.shape[0]} fam={fam}")
        del net; torch.cuda.empty_cache()

    json.dump({"runs": index, "n_infl": N_INFL}, open(os.path.join(OUTDIR, "index.json"), "w"), indent=2)
    print(f"\nwrote {OUTDIR}/ ({len(index)} tokenizers)")


if __name__ == "__main__":
    main()
