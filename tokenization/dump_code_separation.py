"""Dump per-code nearest-neighbour distances (rho_k) and per-latent quantization
errors (e(z)) for two tokenizers, so thesis_figures/plot_code_separation.py can draw
the code-separation histogram (Methods fig:cb-separation).

Reuses the exact metric and FSQ-bounding logic of analyze_codebook_geometry.geometry
so the annotated separation ratio matches Table 4.4. CPU only (the GPU is usually
busy and the model code hardcodes .cuda() at construction).

Contrast pair matches Fig 4.7: FSQ d4 vs cosine d4, the two tokenizers at d=4, so
the pair isolates the quantizer and matches the cosine variant used downstream.
(near-duplicate). Run from tokenization/ in the thesis-HMR env:
    ~/miniconda3/envs/thesis-HMR/bin/python dump_code_separation.py

Output: output/codebook_geometry/separation_arrays.npz
"""
import json
import os

import numpy as np
import torch

import visualize_latent_space as vz
from analyze_latent_pose_info import load_net, get_codebook, _is_cosine
from analyze_token_stability import load_pose_set
from analyze_codebook_geometry import encode_shared, DEV, HERE, STAB, OUTDIR, NUM_POSES, SEED

LABELS = ["FSQ d4", "VQ d4"]      # FSQ vs the cosine tokenizer carried into Sec. 4.4 (both d=4)


@torch.no_grad()
def separation_arrays(ckpt, poses):
    net, _ = load_net(ckpt)
    net = net.to(DEV)
    torch.cuda.empty_cache()
    cosine = _is_cosine(net)
    cb = get_codebook(net).detach().float().to(DEV)
    K, d = cb.shape
    pre, idx = encode_shared(net, poses)
    lat_tok = pre.reshape(-1, d)
    if net.quant == "fsq":                                    # bound onto the [-1,1] grid (as in geometry)
        q = net.quantizer
        half = (q._levels // 2).float()
        lat_tok = (q.bound(lat_tok.to(DEV)) / half).cpu()
    idx_flat = idx.reshape(-1)
    usage = torch.bincount(idx_flat, minlength=K)

    cbm = vz._prep(cb, cosine)
    nn_d = vz.code_nn_dist(cbm, cosine)                       # rho_k, per code (all K), quantizer metric
    latm = vz._prep(lat_tok, cosine)
    qd = []
    for i in range(0, latm.shape[0], 8192):
        chunk = latm[i:i + 8192].to(DEV)
        code = cbm[idx_flat[i:i + 8192].to(DEV)]
        if cosine:
            sim = (chunk.double() * code.double()).sum(-1)
            qd.append((2.0 * (1.0 - sim)).clamp(min=0).sqrt().float().cpu())
        else:
            qd.append((chunk - code).norm(dim=-1).cpu())
    quant_d = torch.cat(qd)                                   # e(z), per latent, quantizer metric
    sep = float(nn_d.median()) / max(float(quant_d.median()), 1e-9)
    del net
    torch.cuda.empty_cache()
    return {
        "rho": nn_d.numpy(),
        "rho_used": nn_d[usage > 0].numpy(),
        "err": quant_d.numpy(),
        "metric": "cosine" if cosine else "l2",
        "sep": sep,
        "rho_med": float(nn_d.median()),
        "err_med": float(quant_d.median()),
        "K": int(K), "d": int(d),
    }


def main():
    runs = {r["label"]: r["ckpt"] for r in json.load(open(STAB))["runs"]}
    _, hp0 = load_net(os.path.join(HERE, next(iter(runs.values()))))
    poses = load_pose_set(hp0, NUM_POSES, SEED)
    print(f"shared pose set: {poses.shape[0]} poses (seed {SEED})\n")

    save, meta = {}, {"labels": LABELS, "scalars": {}}
    for i, label in enumerate(LABELS):
        ckpt = runs[label]
        ckpt = ckpt if os.path.isabs(ckpt) else os.path.join(HERE, ckpt)
        r = separation_arrays(ckpt, poses)
        save[f"rho_{i}"] = r["rho"]
        save[f"rhoused_{i}"] = r["rho_used"]
        save[f"err_{i}"] = r["err"]
        meta["scalars"][label] = {k: r[k] for k in ("metric", "sep", "rho_med", "err_med", "K", "d")}
        print(f"{label:20s} metric={r['metric']:6s} K={r['K']:5d} d={r['d']:3d}  "
              f"sep={r['sep']:5.1f}  rho_med={r['rho_med']:.4f}  err_med={r['err_med']:.4f}")

    os.makedirs(OUTDIR, exist_ok=True)
    save["meta"] = np.array(json.dumps(meta))
    out = os.path.join(OUTDIR, "separation_arrays.npz")
    np.savez(out, **save)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
