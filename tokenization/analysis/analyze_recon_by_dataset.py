"""Table 4.5: round-trip reconstruction fidelity of the selected FSQ (d=4) tokenizer,
stratified by validation dataset.

Encodes then decodes each held-out pose and measures how far the reconstruction drifts:

  Rot. err [deg]   mean per-joint geodesic angle between GT and reconstructed rotations
  MPJPE [mm]       mean per-joint position error after an SMPL forward pass
  Best/Worst joint the joints with the smallest / largest per-joint rotation error
                   (the range of reconstruction quality across the body)

Runs per validation dataset separately (a fresh balanced sample from each), plus a
pooled "Overall" row. Reuses the encode/decode/joint helpers of
analyze_latent_pose_info; the heavy math runs on the CPU (the GPU is usually busy and
the model / SMPL code hardcodes `.cuda()` at construction). The overall rotation error
should match the FSQ d4 reconstruction of Table 4.3 (~0.54 deg/joint).

Run from tokenization/ in the thesis-HMR env:
    ~/miniconda3/envs/thesis-HMR/bin/python analyze_recon_by_dataset.py

Output: output/recon_by_dataset/summary_all.json (numbers for Table 4.5, all tokenizers).
"""
import json
import os

import numpy as np
import torch

from tokenization.analysis import analyze_latent_pose_info as ali
ali.DEVICE = torch.device("cpu")                            # patch the module global used below
from tokenization.models import transformer_pose_vqvae as tpv
from tokenization.models import vanilla_pose_vqvae as vpv
# BOTH model files hold their own module-level SMPL-H layer, constructed on the GPU. The
# CNN tokenizer decodes through vpv's, so patching only tpv's crashes with a device
# mismatch on the first non-transformer tokenizer.
tpv.body_model = tpv.body_model.to("cpu")
vpv.body_model = vpv.body_model.to("cpu")

from tokenization.analysis.analyze_latent_pose_info import (load_net, encode_full, decode_from_codes,
                                      pose6d_to_rotmat, geodesic_deg, JOINT_NAMES)
from tokenization.utils.rotation_conversions import axis_angle_to_matrix

from repro import paths

HERE = os.path.dirname(os.path.abspath(__file__))
STAB = os.path.join(str(paths.TOKENIZER_OUT), "token_stability", "summary.json")
OUTDIR = os.path.join(str(paths.TOKENIZER_OUT), "recon_by_dataset")
PER_DS = 2048
BODY = tpv.body_model


def all_tokenizers():
    """Every tokenizer of Table 4.3, in that table's order, as (label, ckpt) pairs."""
    out = []
    for r in json.load(open(STAB))["runs"]:
        ck = r["ckpt"]
        out.append((r.get("label", "?"), ck if os.path.isabs(ck) else os.path.join(HERE, ck)))
    return out


@torch.no_grad()
def smpl_joints(rotmat):                                    # (B,21,3,3) -> (B,21,3)
    return BODY(body_pose=rotmat).joints[:, 1:22]


def gt_rotmat(batch):
    if "gt_pose_body" in batch:
        return batch["gt_pose_body"].float().view(-1, 21, 3, 3)
    return axis_angle_to_matrix(batch["pose_body_aa"].float().view(-1, 21, 3))


@torch.no_grad()
def recon_errors(net, hparams, ds, per_ds, batch_size=256):
    """Per-joint rotation (deg) and position (mm) error tensors for one dataset."""
    from tokenization.dataset.dataset_poseVQ import get_dataloader
    hparams.DATA.NUM_WORKERS = min(getattr(hparams.DATA, "NUM_WORKERS", 4), 4)
    hparams.DATA.CACHE_SMPL = False
    saved = hparams.DATA.VALLIST
    hparams.DATA.VALLIST = ds
    rot_errs, pos_errs = [], []
    got = 0
    for batch in get_dataloader(hparams, split="val", shuffle=True):
        gt = gt_rotmat(batch)                               # (B,21,3,3) cpu
        for i in range(0, gt.shape[0], batch_size):
            g = gt[i:i + batch_size]
            enc = encode_full(net, g)
            pred = pose6d_to_rotmat(net, decode_from_codes(net, enc["idx"]))   # (b,21,3,3)
            rot_errs.append(geodesic_deg(g, pred).cpu())                        # (b,21)
            pos_errs.append(((smpl_joints(g) - smpl_joints(pred)).norm(dim=-1) * 1000.0).cpu())
        got += gt.shape[0]
        if got >= per_ds:
            break
    hparams.DATA.VALLIST = saved
    return torch.cat(rot_errs, 0)[:per_ds], torch.cat(pos_errs, 0)[:per_ds]


def summarize(rot, pos):
    pj_rot = rot.mean(0)                                    # (21,)
    pj_pos = pos.mean(0)
    best, worst = int(pj_rot.argmin()), int(pj_rot.argmax())
    return {
        "n": int(rot.shape[0]),
        "rot_deg_mean": float(pj_rot.mean()),
        "mpjpe_mm_mean": float(pj_pos.mean()),
        "best_joint": JOINT_NAMES[best], "best_joint_rot_deg": float(pj_rot[best]),
        "worst_joint": JOINT_NAMES[worst], "worst_joint_rot_deg": float(pj_rot[worst]),
        "per_joint_rot_deg": {JOINT_NAMES[j]: float(pj_rot[j]) for j in range(len(JOINT_NAMES))},
        "per_joint_mpjpe_mm": {JOINT_NAMES[j]: float(pj_pos[j]) for j in range(len(JOINT_NAMES))},
    }


def main():
    # One row per tokenizer of Table 4.3, each stratified by validation dataset. The
    # per-dataset sample is drawn the same way for every tokenizer, so the columns are
    # comparable across rows.
    results = {}
    for label, ckpt in all_tokenizers():
        print(f"\n=== {label}  ({ckpt})")
        try:
            net, hparams = load_net(ckpt)
        except Exception as e:                              # a missing/renamed ckpt must not
            print(f"  SKIP: {type(e).__name__}: {e}")       # abort the other seven
            continue
        net = net.to("cpu")
        datasets = hparams.DATA.VALLIST.split("_")
        per_ds_out, all_rot, all_pos = {}, [], []
        for ds in datasets:
            rot, pos = recon_errors(net, hparams, ds, PER_DS)
            s_ = summarize(rot, pos)
            per_ds_out[ds] = s_
            all_rot.append(rot); all_pos.append(pos)
            print(f"  {ds:10s} rot {s_['rot_deg_mean']:.3f} deg  MPJPE {s_['mpjpe_mm_mean']:.2f} mm")
        overall = summarize(torch.cat(all_rot, 0), torch.cat(all_pos, 0))
        print(f"  {'Overall':10s} rot {overall['rot_deg_mean']:.3f} deg  "
              f"MPJPE {overall['mpjpe_mm_mean']:.2f} mm")
        results[label] = {"ckpt": ckpt, "datasets": per_ds_out, "overall": overall}
        del net

    os.makedirs(OUTDIR, exist_ok=True)
    json.dump({"per_ds": PER_DS, "tokenizers": results},
              open(os.path.join(OUTDIR, "summary_all.json"), "w"), indent=2)
    print(f"\nwrote {OUTDIR}/summary_all.json  ({len(results)} tokenizers)")


if __name__ == "__main__":
    main()
