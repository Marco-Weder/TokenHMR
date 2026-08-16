"""Latent-space visualization + distance-based entropy for pose-VQ tokenizers.

Answers the two supervisor questions that analyze_codebook.py only summarizes as
scalars, and adds the pictures:

  A. DISTINCTNESS — are the codes different from each other, relative to how
     precisely latents land on them?
       - inter-code distance distributions (all pairs + nearest neighbor) overlaid
         with the latent->assigned-code (quantization error) distribution: codes
         are "different" exactly when the quantization errors sit well left of the
         nearest-neighbor code distances (separation ratio >> 1).
       - a distance-based entropy decomposition: soft-assign every latent to all
         codes with a Boltzmann distribution over distances,
             p(k|z) = softmax(-d(z, c_k) / tau),
         then
             usage entropy   H(k)   = how evenly the codebook is used
             confusion       H(k|z) = how ambiguous one latent's assignment is
             information     I(z;k) = H(k) - H(k|z), the bits a code carries
         "codes distinct AND all used" is the high-H(k), low-H(k|z) regime; the
         hard-assignment (tau -> 0) limit of H(k) is the usual usage entropy.
       - a Kozachenko-Leonenko differential-entropy estimate of the code cloud
         (only comparable between runs with equal code_dim / metric).

  B. GEOMETRY — what the space looks like. code_dim=2 is plotted directly (plus an
     angular view for cosine models, where codes live on the unit circle), code_dim=4
     as a pairwise scatter matrix, anything higher via PCA fit on the latents with
     the codebook projected into the same basis.

Multi-run: pass --ckpt several times; per-run figures land in
<ckpt_dir>/latent_space_viz/ and a cross-run comparison figure + summary.json go to
--out (default: output/latent_space_comparison/).

Run from the tokenization/ directory:
    python visualize_latent_space.py --ckpt output/<run>/.../best_net.pth [--ckpt ...]
    python visualize_latent_space.py --ckpt ... --num-poses 1024 --quick
"""
import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tokenization.analysis.analyze_latent_pose_info import (DEVICE, load_net, encode_full, get_codebook,
                                      _is_cosine, _pca_reduce)

# Repo-wide categorical palette (matches analyze_latent_pose_info.py figures).
C_BLUE, C_RED, C_ORANGE, C_GREEN, C_PURPLE = '#4C78A8', '#E45756', '#F58518', '#54A24B', '#B279A2'


# --------------------------------------------------------------------------- #
# Distances (always in the quantizer's own metric)                             #
# --------------------------------------------------------------------------- #
def _prep(x, cosine):
    """Metric-consistent representation: unit vectors for cosine, raw for L2/FSQ."""
    return F.normalize(x.float(), dim=-1) if cosine else x.float()


@torch.no_grad()
def _dist(a, b, cosine):
    """(n, d) x (m, d) -> (n, m) distances.

    Cosine models use the chordal distance sqrt(2 * (1 - cos)), in float64: it is a
    true metric on the unit sphere, and neighboring codes can sit ~1e-3 rad apart,
    where fp32 `1 - cos` underflows to 0 (machine eps 1.2e-7).
    """
    if cosine:
        sim = a.double() @ b.double().t()
        return (2.0 * (1.0 - sim)).clamp(min=0).sqrt().float()
    return torch.cdist(a, b)


@torch.no_grad()
def code_nn_dist(cb, cosine, chunk=2048):
    """Per-code distance to its nearest other code, chunked for large K."""
    out = []
    for i in range(0, cb.shape[0], chunk):
        d = _dist(cb[i:i + chunk], cb, cosine)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float('inf')
        out.append(d.min(dim=1).values)
    return torch.cat(out)


@torch.no_grad()
def code_pair_dist_sample(cb, cosine, max_codes=4096, seed=0):
    """Flattened off-diagonal pairwise distances of (a sample of) the codebook."""
    K = cb.shape[0]
    if K > max_codes:
        sel = torch.randperm(K, generator=torch.Generator().manual_seed(seed))[:max_codes]
        cb = cb[sel.to(cb.device)]
    d = _dist(cb, cb, cosine)
    iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1, device=d.device)
    return d[iu[0], iu[1]]


def kl_differential_entropy(nn_d, d_dim):
    """Kozachenko-Leonenko kNN (k=1) differential entropy of the code cloud, in nats.

    H ~ d * mean(log r_nn) + log(N-1) + log V_d + gamma.  Scale-sensitive and only
    comparable between runs with the same code_dim and metric — reported for the
    within-dimension ablations (dim2 vs dim2 rerun etc.), not across dims.
    """
    nn_d = nn_d[nn_d > 0].float()
    if nn_d.numel() < 2:
        return float('nan')
    N = nn_d.numel()
    log_vd = (d_dim / 2) * math.log(math.pi) - math.lgamma(d_dim / 2 + 1)
    return float(d_dim * nn_d.log().mean() + math.log(N - 1) + log_vd + 0.5772156649)


# --------------------------------------------------------------------------- #
# Distance-based entropy decomposition  H(k), H(k|z), I(z;k)                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def soft_assignment_entropies(lat, cb, cosine, taus, chunk=4096):
    """Boltzmann soft assignment p(k|z) = softmax(-d(z, c_k)/tau) for each tau.

    Returns per-tau dicts with H_usage = H(mean_z p(k|z)), H_confusion = mean_z H(p(k|z)),
    and MI = H_usage - H_confusion, all in bits. The distance matrix per chunk is reused
    across taus so large codebooks (FSQ 16k) stay cheap.
    """
    K = cb.shape[0]
    n = lat.shape[0]
    marg = {t: torch.zeros(K, device=cb.device, dtype=torch.float64) for t in taus}
    hcond = {t: 0.0 for t in taus}
    for i in range(0, n, chunk):
        d = _dist(lat[i:i + chunk].to(cb.device), cb, cosine)
        for t in taus:
            p = F.softmax(-d / t, dim=-1)
            marg[t] += p.sum(0).double()
            hcond[t] += float(-(p * (p + 1e-30).log()).sum())
    out = {}
    ln2 = math.log(2)
    for t in taus:
        p = marg[t] / n
        h_use = float(-(p * (p + 1e-30).log()).sum()) / ln2
        h_conf = hcond[t] / n / ln2
        out[t] = {'H_usage_bits': h_use, 'H_confusion_bits': h_conf,
                  'MI_bits': h_use - h_conf}
    return out


# --------------------------------------------------------------------------- #
# Latent collection (encoder pass only — no SMPL forward needed)               #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_latents(net, hparams, num_poses):
    """Pre-quant latents (N, T, d), code indices (N, T) and dataset names over a
    balanced val sample. Same per-dataset loader trick as analyze_latent_pose_info.collect,
    but stops at the encoder: no decode / SMPL, so it is fast."""
    from tokenization.dataset.dataset_poseVQ import get_dataloader
    from tokenization.utils.rotation_conversions import axis_angle_to_matrix

    hparams.DATA.NUM_WORKERS = min(getattr(hparams.DATA, 'NUM_WORKERS', 4), 4)
    hparams.DATA.CACHE_SMPL = False
    ds_list = hparams.DATA.VALLIST.split('_')
    per_ds = math.ceil(num_poses / len(ds_list))

    pres, idxs, names = [], [], []
    saved = hparams.DATA.VALLIST
    for ds in ds_list:
        hparams.DATA.VALLIST = ds
        loader = get_dataloader(hparams, split='val', shuffle=True)
        got = 0
        for batch in loader:
            if 'gt_pose_body' in batch:
                gt_pose = batch['gt_pose_body'].to(DEVICE).float()
            else:
                gt_pose = axis_angle_to_matrix(
                    batch['pose_body_aa'].to(DEVICE).float().view(-1, 21, 3))
            enc = encode_full(net, gt_pose)
            pres.append(enc['pre'].cpu())
            idxs.append(enc['idx'].cpu())
            names.extend([ds] * gt_pose.shape[0])
            got += gt_pose.shape[0]
            if got >= per_ds:
                break
    hparams.DATA.VALLIST = saved
    return torch.cat(pres, 0), torch.cat(idxs, 0), names


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def fig_separation(pair_d, nn_d, quant_d, usage, tag, out_dir):
    """Left: the 'are codes different' picture — three distance distributions.
    Right: sorted code usage (Zipf)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    # The three distributions can sit orders of magnitude apart (tightly packed
    # codebooks): log-spaced bins + per-histogram fraction weights keep all three
    # readable. Exact zeros (latents that coincide with their code at fp32
    # resolution) are excluded and reported in the label instead.
    hi = float(torch.quantile(pair_d, 0.995))
    pos = [x[x > 0].cpu() for x in (pair_d, nn_d, quant_d)]
    lo = max(min(float(p.min()) for p in pos if p.numel()), hi * 1e-9)
    bins = np.logspace(np.log10(lo), np.log10(hi * 1.05), 90)
    for data, v, color, label in [(pair_d, pos[0], C_BLUE, 'all code pairs'),
                                  (nn_d, pos[1], C_ORANGE, 'code -> nearest code'),
                                  (quant_d, pos[2], C_GREEN, 'latent -> assigned code\n(quantization error)')]:
        z = 1.0 - v.numel() / data.numel()
        if z > 0.01:
            label += f'  ({z * 100:.0f}% exactly 0)'
        w = np.full(v.numel(), 1.0 / max(data.numel(), 1))
        ax[0].hist(v.numpy().clip(min=lo), bins=bins, weights=w, alpha=0.5,
                   color=color, label=label)
    ax[0].set_xscale('log')
    ax[0].axvline(float(nn_d.median()), color=C_ORANGE, lw=1, ls='--')
    ax[0].axvline(max(float(quant_d.median()), lo), color=C_GREEN, lw=1, ls='--')
    ax[0].set_xlabel('distance (euclidean; chordal for cosine models)')
    ax[0].set_ylabel('fraction of samples')
    sep = float(nn_d.median() / max(float(quant_d.median()), 1e-9))
    ax[0].set_title(f'Code separation vs quantization noise  (ratio {sep:.1f}x)')
    ax[0].legend(fontsize=8)

    srt = usage.sort(descending=True).values.cpu().numpy()
    ax[1].plot(np.arange(1, len(srt) + 1), np.maximum(srt, 0.5), color=C_BLUE, lw=1.5)
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel('code rank'); ax[1].set_ylabel('assignment count')
    used = int((usage > 0).sum())
    ax[1].set_title(f'Code usage (sorted)  —  {used}/{len(usage)} codes used')
    fig.suptitle(tag, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'code_separation.png'), dpi=130)
    plt.close(fig)
    return sep


def fig_entropy_curve(ent, K, tau_ref, tag, out_dir):
    taus = sorted(ent.keys())
    x = [t / tau_ref for t in taus]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(x, [ent[t]['H_usage_bits'] for t in taus], marker='o', color=C_BLUE,
            label='H(k)  usage entropy')
    ax.plot(x, [ent[t]['H_confusion_bits'] for t in taus], marker='o', color=C_RED,
            label='H(k|z)  assignment confusion')
    ax.plot(x, [ent[t]['MI_bits'] for t in taus], marker='o', color=C_GREEN,
            label='I(z;k) = H(k) - H(k|z)')
    ax.axhline(math.log2(K), ls='--', c='grey', lw=1, label=f'log2(K) = {math.log2(K):.1f} bits')
    ax.set_xscale('log')
    ax.set_xlabel('temperature  τ / median NN code distance')
    ax.set_ylabel('bits per token')
    ax.set_title(f'{tag}\nDistance-based entropy of the latent-code assignment')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'entropy_vs_temperature.png'), dpi=130)
    plt.close(fig)


def _scatter_codes(ax, cb2, usage, label='codebook'):
    used = (usage > 0).cpu().numpy()
    c = np.log10(usage.cpu().numpy() + 1)
    sc = ax.scatter(cb2[used, 0], cb2[used, 1], s=10, c=c[used], cmap='magma',
                    zorder=3, label=f'{label} (used)')
    if (~used).any():
        ax.scatter(cb2[~used, 0], cb2[~used, 1], s=8, facecolors='none',
                   edgecolors='grey', linewidths=0.5, zorder=2, label=f'{label} (dead)')
    return sc


def fig_latent_space(lat_tok, cb, usage, cosine, tag, out_dir, max_pts=60000, seed=0):
    """The geometry picture, laid out by code_dim.

    lat_tok: (n, d) token-level pre-quant latents;  cb: (K, d) codebook.
    Cosine models are plotted in the normalized (unit-sphere) space the quantizer
    actually measures distances in.
    """
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(lat_tok.shape[0], generator=g)[:max_pts]
    lat = _prep(lat_tok[sel], cosine).cpu().numpy()
    cbp = _prep(cb, cosine).cpu().numpy()
    d = lat.shape[1]

    if d == 2:
        ncols = 3 if cosine else 2
        fig, ax = plt.subplots(1, ncols, figsize=(5.4 * ncols, 5))
        ax[0].hexbin(lat[:, 0], lat[:, 1], gridsize=60, cmap='Blues', bins='log')
        ax[0].set_title('encoder latents (log density)')
        ax[1].hexbin(lat[:, 0], lat[:, 1], gridsize=60, cmap='Greys', bins='log')
        sc = _scatter_codes(ax[1], cbp, usage)
        fig.colorbar(sc, ax=ax[1], fraction=0.045, label='log10(usage + 1)')
        ax[1].set_title('codebook over latents')
        ax[1].legend(fontsize=7, loc='upper right')
        if cosine:
            # On the unit circle the only degree of freedom is the angle.
            la = np.arctan2(lat[:, 1], lat[:, 0])
            ca = np.arctan2(cbp[:, 1], cbp[:, 0])
            ax[2].hist(la, bins=180, density=True, color=C_BLUE, alpha=0.6,
                       label='latent angles')
            used = (usage > 0).cpu().numpy()
            ax[2].plot(ca[used], np.full(used.sum(), -0.02), '|', color=C_RED,
                       ms=8, alpha=0.4, label='used code angles')
            ax[2].set_xlabel('angle (rad)'); ax[2].set_ylabel('density')
            ax[2].set_title('angular view (cosine metric: codes live on the circle)')
            ax[2].legend(fontsize=7)
        for a in ax[:2]:
            a.set_xlabel('latent dim 0'); a.set_ylabel('latent dim 1')
    elif d <= 4:
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        for a, (i, j) in zip(axes.ravel(), pairs):
            a.hexbin(lat[:, i], lat[:, j], gridsize=55, cmap='Greys', bins='log')
            sc = _scatter_codes(a, cbp[:, [i, j]], usage)
            a.set_xlabel(f'dim {i}'); a.set_ylabel(f'dim {j}')
        for a in axes.ravel()[len(pairs):]:
            a.axis('off')
        fig.colorbar(sc, ax=axes, fraction=0.02, label='log10(usage + 1)')
        fig.suptitle(f'{tag} — codebook (colored) over encoder latents (grey), all dim pairs')
        fig.savefig(os.path.join(out_dir, 'latent_space.png'), dpi=130)
        plt.close(fig)
        return
    else:
        # PCA basis fit on the latents; codebook projected into the same basis.
        latt = torch.from_numpy(lat)
        mu = latt.mean(0, keepdim=True)
        U, S, V = torch.pca_lowrank(latt - mu, q=2, niter=4)
        lat2 = ((latt - mu) @ V[:, :2]).numpy()
        cb2 = ((torch.from_numpy(cbp) - mu) @ V[:, :2]).numpy()
        var = float((S[:2] ** 2).sum() / (latt - mu).pow(2).sum()) * 100
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 5))
        ax[0].hexbin(lat2[:, 0], lat2[:, 1], gridsize=60, cmap='Blues', bins='log')
        ax[0].set_title(f'encoder latents, PCA of token space ({var:.0f}% var)')
        ax[1].hexbin(lat2[:, 0], lat2[:, 1], gridsize=60, cmap='Greys', bins='log')
        sc = _scatter_codes(ax[1], cb2, usage)
        fig.colorbar(sc, ax=ax[1], fraction=0.045, label='log10(usage + 1)')
        ax[1].set_title('codebook projected into the same PCA basis')
        ax[1].legend(fontsize=7, loc='upper right')
        for a in ax:
            a.set_xlabel('PC1'); a.set_ylabel('PC2')

    fig.suptitle(tag, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'latent_space.png'), dpi=130)
    plt.close(fig)


def fig_token_position(lat_tok, positions, cosine, tag, out_dir, max_pts=60000, seed=0):
    """Latents colored by token position: shows whether the T=160 latent slots carve
    up the space (position-specialized codes) or share one cloud."""
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(lat_tok.shape[0], generator=g)[:max_pts]
    lat = _prep(lat_tok[sel], cosine)
    pos = positions[sel].numpy()
    d = lat.shape[1]
    if d > 2:
        lat2 = _pca_reduce(lat.to(DEVICE), 2).cpu().numpy()
        xl, yl = 'PC1', 'PC2'
    else:
        lat2 = lat.cpu().numpy()
        xl, yl = 'latent dim 0', 'latent dim 1'
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sc = ax.scatter(lat2[:, 0], lat2[:, 1], s=2, c=pos, cmap='viridis', alpha=0.5)
    fig.colorbar(sc, ax=ax, label='token position (0..T-1)')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f'{tag}\nlatents by token position')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'latents_by_token_position.png'), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-run driver                                                               #
# --------------------------------------------------------------------------- #
def analyze_run(ckpt_path, num_poses, out_dir=None):
    net, hparams = load_net(ckpt_path)
    tag = hparams.EXP.NAME
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                                      'latent_space_viz')
    os.makedirs(out_dir, exist_ok=True)
    cosine = _is_cosine(net)
    cb = get_codebook(net).detach().float().to(DEVICE)
    K, d = cb.shape
    print(f'\n=== {tag}  (quant={net.quant}, metric={"cosine" if cosine else "l2"}, '
          f'K={K}, d={d}) -> {out_dir}')

    print('  collecting val latents ...')
    pre, idx, names = collect_latents(net, hparams, num_poses)
    N, T = idx.shape
    lat_tok = pre.reshape(-1, d)                            # (N*T, d)
    if net.quant == 'fsq':
        # FSQ rounds bound(z) onto a normalized [-1, 1] grid; distances to the
        # implicit codebook are only meaningful for the bounded latent, not the
        # unbounded pre-activation.
        q = net.quantizer
        half_width = (q._levels // 2).float()
        lat_tok = (q.bound(lat_tok.to(DEVICE)) / half_width).cpu()
    idx_flat = idx.reshape(-1)
    positions = torch.arange(T).repeat(N)
    usage = torch.bincount(idx_flat, minlength=K).float()

    # --- distances ---
    cbm = _prep(cb, cosine)
    nn_d = code_nn_dist(cbm, cosine)
    pair_d = code_pair_dist_sample(cbm, cosine)
    # quantization error: latent -> its assigned code, in the quantizer metric
    latm = _prep(lat_tok, cosine)
    qd = []
    for i in range(0, latm.shape[0], 8192):
        chunk = latm[i:i + 8192].to(DEVICE)
        code = cbm[idx_flat[i:i + 8192].to(DEVICE)]
        if cosine:
            sim = (chunk.double() * code.double()).sum(-1)
            qd.append((2.0 * (1.0 - sim)).clamp(min=0).sqrt().float().cpu())
        else:
            qd.append((chunk - code).norm(dim=-1).cpu())
    quant_d = torch.cat(qd)

    tau_ref = float(nn_d.median())
    taus = [a * tau_ref for a in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)]
    print('  computing soft-assignment entropies ...')
    ent = soft_assignment_entropies(latm, cbm, cosine, taus)

    sep = fig_separation(pair_d, nn_d, quant_d, usage, tag, out_dir)
    fig_entropy_curve(ent, K, tau_ref, tag, out_dir)
    fig_latent_space(lat_tok, cb, usage, cosine, tag, out_dir)
    fig_token_position(lat_tok, positions, cosine, tag, out_dir)

    e05 = ent[taus[2]]                                      # headline: tau = 0.5 * NN dist
    hard_p = usage / usage.sum()
    hard_H = float(-(hard_p * (hard_p + 1e-12).log()).sum()) / math.log(2)
    stats = {
        'ckpt': ckpt_path, 'tag': tag, 'quant': net.quant,
        'metric': 'cosine' if cosine else 'l2', 'K': K, 'code_dim': d,
        'num_poses': N, 'num_tokens': T,
        'usage': {
            'used_codes': int((usage > 0).sum()), 'utilization_pct': float((usage > 0).float().mean() * 100),
            'hard_usage_entropy_bits': hard_H, 'hard_norm_entropy': hard_H / math.log2(K),
            'perplexity': float(2 ** hard_H),
        },
        'distances': {
            'nn_code_dist_median': tau_ref, 'nn_code_dist_p10': float(nn_d.quantile(0.1)),
            'pair_dist_median': float(pair_d.median()),
            'quant_err_median': float(quant_d.median()),
            'separation_ratio_nn_over_qerr': sep,
            'frac_codes_nn_closer_than_median_qerr':
                float((nn_d < quant_d.median()).float().mean()),
            'codebook_KL_diff_entropy_nats': kl_differential_entropy(nn_d, d),
        },
        'soft_assignment_at_tau_0.5nn': e05,
        'entropy_curve': {f'{t/tau_ref:g}': v for t, v in ent.items()},
    }
    with open(os.path.join(out_dir, 'latent_space_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    u, dd, s5 = stats['usage'], stats['distances'], e05
    print(f"  usage      : {u['used_codes']}/{K} codes | norm-entropy {u['hard_norm_entropy']:.3f} "
          f"| perplexity {u['perplexity']:.0f}")
    print(f"  distances  : NN-code median {dd['nn_code_dist_median']:.4f} | quant-err median "
          f"{dd['quant_err_median']:.4f} | separation {dd['separation_ratio_nn_over_qerr']:.1f}x")
    print(f"  entropy    : H(k) {s5['H_usage_bits']:.2f} | H(k|z) {s5['H_confusion_bits']:.3f} "
          f"| I(z;k) {s5['MI_bits']:.2f} bits (max {math.log2(K):.2f})  [tau = 0.5 x NN dist]")
    del net
    torch.cuda.empty_cache()
    return stats


# --------------------------------------------------------------------------- #
# Cross-run comparison                                                         #
# --------------------------------------------------------------------------- #
def fig_comparison(all_stats, out_dir):
    tags = [s['tag'].replace('tokenization_', '') for s in all_stats]
    x = np.arange(len(tags))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    ax[0, 0].bar(x, [s['usage']['hard_norm_entropy'] for s in all_stats], color=C_BLUE)
    ax[0, 0].set_ylim(0, 1.02); ax[0, 0].axhline(1.0, ls='--', c='grey', lw=1)
    ax[0, 0].set_title('Usage entropy (normalized, 1 = perfectly uniform)')

    mi = [s['soft_assignment_at_tau_0.5nn']['MI_bits'] for s in all_stats]
    mx = [math.log2(s['K']) for s in all_stats]
    ax[0, 1].bar(x, mi, color=C_GREEN, label='I(z;k)')
    ax[0, 1].plot(x, mx, 'v', color='grey', label='log2(K) ceiling')
    ax[0, 1].set_title('Information per token I(z;k), bits  (τ = 0.5 × NN dist)')
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].bar(x, [s['distances']['separation_ratio_nn_over_qerr'] for s in all_stats],
                 color=C_ORANGE)
    ax[1, 0].axhline(1.0, ls='--', c='grey', lw=1)
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_title('Code separation ratio: NN-code dist / quantization error (>1 = distinct)')

    ax[1, 1].bar(x, [s['usage']['utilization_pct'] for s in all_stats], color=C_PURPLE)
    ax[1, 1].set_ylim(0, 102); ax[1, 1].axhline(100, ls='--', c='grey', lw=1)
    ax[1, 1].set_title('Codebook utilization (%)')

    for a in ax.ravel():
        a.set_xticks(x); a.set_xticklabels(tags, rotation=25, ha='right', fontsize=8)
        a.grid(alpha=0.25, axis='y')
    fig.suptitle('Latent-space comparison across tokenizer runs')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'comparison.png'), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', action='append', required=True,
                    help='tokenizer checkpoint; repeat for a multi-run comparison')
    ap.add_argument('--num-poses', type=int, default=2048, help='val poses to encode')
    ap.add_argument('--out', default='output/latent_space_comparison',
                    help='dir for the cross-run summary (per-run figures go next to each ckpt)')
    ap.add_argument('--quick', action='store_true', help='tiny run for a smoke test')
    args = ap.parse_args()
    if args.quick:
        args.num_poses = 256

    all_stats = []
    for ckpt in args.ckpt:
        try:
            all_stats.append(analyze_run(ckpt, args.num_poses))
        except Exception as e:
            print(f'[warn] {ckpt} failed: {type(e).__name__}: {e}')

    if len(all_stats) > 1:
        os.makedirs(args.out, exist_ok=True)
        fig_comparison(all_stats, args.out)
        with open(os.path.join(args.out, 'summary.json'), 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f'\nwrote comparison figure + summary.json to {args.out}/')


if __name__ == '__main__':
    main()
