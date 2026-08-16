"""Codebook geometry for Table 4.4 / Figure 4.4, consistent with Table 4.3.

Runs the SAME eight tokenizers on the SAME 2048 shared val poses (seed 0) that
back Table 4.3 (read straight from output/token_stability/summary.json), so the
utilization column here matches the usage column there by construction. Per
tokenizer it reports the five geometry measures of Section 3.4.1:

  Util. [%]    fraction of the codebook the val poses visit (hard assignment)
  Perplexity   2 ** H(hard code usage)
  Eff. rank    codebook effective rank = exp(spectral entropy of the unit-normalised
               codebook's singular values); how many directions the codes span
  Sep. ratio   median nearest-neighbour code distance / median latent->code
               quantization error, in the quantizer's own metric (>>1 = distinct)
  I(z;k)       distance-based mutual information H(k) - H(k|z) at tau = half the
               median NN code distance (bits)

It also stores H(k) and H(k|z) at that tau for the entropy-decomposition panel of
Figure 4.4. Metric definitions and the FSQ latent bounding are reused verbatim from
visualize_latent_space.analyze_run so the two never drift.

Run from tokenization/ in the thesis-HMR env:
    python analyze_codebook_geometry.py

Output: output/codebook_geometry/summary.json  (consumed by
        thesis_figures/plot_codebook_geometry.py and Table 4.4).
"""
import json
import math
import os

import torch
import torch.nn.functional as F

from tokenization.analysis import visualize_latent_space as vz
from tokenization.analysis.analyze_latent_pose_info import load_net, get_codebook, _is_cosine, encode_full
from tokenization.analysis.analyze_token_stability import load_pose_set

from repro import paths

# The GPU is often occupied by a training run and the model code hardcodes `.cuda()`
# at construction, so we load each net (briefly on GPU) and then move it to the CPU
# and run all encoding / distance / entropy math there. Slower but memory-safe.
DEV = torch.device("cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
STAB = os.path.join(str(paths.TOKENIZER_OUT), "token_stability", "summary.json")
OUTDIR = os.path.join(str(paths.TOKENIZER_OUT), "codebook_geometry")
NUM_POSES, SEED = 2048, 0

def eff_rank(cb):
    """Codebook effective rank: exp(spectral entropy of the unit-normalised codes)."""
    cbn = F.normalize(cb.float(), dim=-1)
    sv = torch.linalg.svdvals(cbn)
    p = (sv ** 2) / (sv ** 2).sum()
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


@torch.no_grad()
def encode_shared(net, poses, batch=256):
    pres, idxs = [], []
    for i in range(0, poses.shape[0], batch):
        enc = encode_full(net, poses[i:i + batch].to(DEV).float())
        pres.append(enc["pre"].cpu())
        idxs.append(enc["idx"].cpu())
    return torch.cat(pres, 0), torch.cat(idxs, 0)


@torch.no_grad()
def geometry(ckpt, poses):
    net, hparams = load_net(ckpt)
    net = net.to(DEV)                                        # off the GPU for the heavy math
    torch.cuda.empty_cache()
    cosine = _is_cosine(net)
    cb = get_codebook(net).detach().float().to(DEV)
    K, d = cb.shape
    pre, idx = encode_shared(net, poses)
    N, T = idx.shape
    lat_tok = pre.reshape(-1, d)
    if net.quant == "fsq":                                   # bound onto the [-1,1] grid (as in analyze_run)
        q = net.quantizer
        half = (q._levels // 2).float()
        lat_tok = (q.bound(lat_tok.to(DEV)) / half).cpu()
    idx_flat = idx.reshape(-1)
    usage = torch.bincount(idx_flat, minlength=K).float()

    cbm = vz._prep(cb, cosine)
    nn_d = vz.code_nn_dist(cbm, cosine)
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
    quant_d = torch.cat(qd)

    tau_ref = float(nn_d.median())
    tau = 0.5 * tau_ref
    ent = vz.soft_assignment_entropies(latm, cbm, cosine, [tau])[tau]
    sep = tau_ref / max(float(quant_d.median()), 1e-9)
    hard_p = usage / usage.sum()
    hard_H = float(-(hard_p * (hard_p + 1e-12).log()).sum()) / math.log(2)

    metric = "cosine" if cosine else "l2"
    is_fsq = str(net.quant).startswith("fsq")
    is_cnn = not hasattr(net, "joint_queries")               # CNN baseline lacks the latent-query attention
    if is_fsq:
        family = "fsq"
    elif cosine:
        family = "cos"
    elif is_cnn:
        family = "conv"
    else:
        family = "tfl2"
    quant_label = "FSQ" if is_fsq else "EMA"
    del net
    torch.cuda.empty_cache()
    return {
        "quant": quant_label, "metric": metric, "K": K, "code_dim": d,
        "utilization_pct": float((usage > 0).float().mean() * 100),
        "perplexity": float(2 ** hard_H),
        "effective_rank": eff_rank(cb.cpu()),
        "separation_ratio": sep,
        "nn_code_dist_median": tau_ref, "quant_err_median": float(quant_d.median()),
        "H_usage_bits": ent["H_usage_bits"], "H_confusion_bits": ent["H_confusion_bits"],
        "I_zk_bits": ent["MI_bits"], "log2K": math.log2(K),
        "family": family,
    }


def main():
    runs = json.load(open(STAB))["runs"]
    ckpts = [r["ckpt"] for r in runs]
    labels = [r["label"] for r in runs]
    _, hp0 = load_net(ckpts[0])                              # shared pose set from the first tokenizer
    poses = load_pose_set(hp0, NUM_POSES, SEED)
    print(f"shared pose set: {poses.shape[0]} poses (seed {SEED})\n")

    out = []
    for ckpt, label in zip(ckpts, labels):
        g = geometry(ckpt, poses)
        g["label"] = label
        out.append(g)
        print(f"{label:20s} util {g['utilization_pct']:5.1f}%  perp {g['perplexity']:6.0f}  "
              f"effrank {g['effective_rank']:5.1f}  sep {g['separation_ratio']:5.1f}x  "
              f"H(k) {g['H_usage_bits']:.2f}  H(k|z) {g['H_confusion_bits']:.2f}  "
              f"I(z;k) {g['I_zk_bits']:.2f} bits")

    os.makedirs(OUTDIR, exist_ok=True)
    json.dump({"num_poses": NUM_POSES, "seed": SEED, "runs": out},
              open(os.path.join(OUTDIR, "summary.json"), "w"), indent=2)
    print(f"\nwrote {OUTDIR}/summary.json")


if __name__ == "__main__":
    main()
