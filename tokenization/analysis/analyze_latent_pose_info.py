"""How does the latent space store pose information?  (Stage-1 tokenizer analysis)

Complements analyze_codebook.py. That script asks whether the *codebook* is healthy
(usage / geometry / margins). This one asks what pose information the *latent space*
carries and how it is organized, via seven analyses:

  1. INFLUENCE  token -> joint influence map: perturb each of the NUM_TOKENS latent
                tokens and measure which of the 21 body joints move. Answers "where is
                each joint stored" and whether the code is spatially factored or
                entangled/distributed.
  2. RECON      per-joint reconstruction error (rotation deg + position mm), stratified
                by dataset. Answers "which pose information survives the bottleneck".
  3. PROBE      linear probing of the latents (pre- vs post-quantization) for per-joint
                pose. The pre/post gap = information destroyed by quantization; high
                decodability = the space is linearly organized, not just a lookup table.
  4. INTERP     latent-space interpolation smoothness: interpolate between pose pairs and
                measure per-step joint jumps + path straightness. Smooth = pose manifold.
  5. LANGUAGE   structure of the token "language": per-position usage entropy, unique
                codes per position, and inter-token redundancy (adjacent-position MI).
                Yields a bits/pose rate estimate to pair with RECON (rate-distortion).
  6. EMBED      2-D PCA of per-pose latents colored by dataset and by a joint angle, plus
                the codebook colored by usage. The qualitative "is it organized" picture.
  7. ROBUST     decoder sensitivity to latent noise: perturb the latents with increasing
                Gaussian noise and measure MPJPE drift. Steep = the decoder amplifies small
                predictor errors, the mechanism behind "great stage-1 recon, poor in the wild".

No sklearn/umap dependency: PCA and ridge regression are done in torch.

Run from the tokenization/ directory:
    python analyze_latent_pose_info.py --ckpt output/<run>/.../best_net.pth
    python analyze_latent_pose_info.py --ckpt ... --out latent_report --num-poses 4096
    python analyze_latent_pose_info.py --ckpt ... --analyses influence,recon --quick

Outputs: PNG figures + a stats.json into --out (default: <ckpt_dir>/latent_analysis/).
Works for every tokenizer variant (cosine/L2 EMA, FSQ, masked, GNN) because it only uses
the public encode/decode/quantize surface shared by all of them.
"""
import argparse
import json
import math
import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# SMPL body-joint names in body_pose order (model joint 0..20 == SMPL joint 1..21).
JOINT_NAMES = [
    'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle',
    'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head',
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------- #
# Model / codec plumbing (mirrors TransformerTokenizer.encode / forward)      #
# --------------------------------------------------------------------------- #
def resolve_ckpt(ckpt_path):
    """Interpret a tokenizer checkpoint path without relying on the cwd.

    The stage-1 registry stores paths as `output/<run>/.../best_net.pth`,
    relative to `tokenization/`, because the training script wrote them from
    there. Stage-2 configs store absolute paths from the training machine.
    Both resolve here, so no caller needs to be run from a particular place.
    """
    from repro import paths
    from repro.legacy import rewrite_legacy_path

    ckpt_path = rewrite_legacy_path(str(ckpt_path))
    p = Path(ckpt_path)
    if p.is_absolute():
        return str(p)
    for base in (paths.TOKENIZER_OUT.parent, paths.PROJECT_ROOT, Path.cwd()):
        candidate = base / p
        if candidate.exists():
            return str(candidate)
    return str(paths.PROJECT_ROOT / p)


def load_net(ckpt_path):
    from tokenization.train_poseVQ import get_model
    ckpt_path = resolve_ckpt(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hparams = ckpt.get('hparams')
    state_dict = ckpt.get('net', ckpt.get('state_dict'))
    if hparams is None:
        hparams = ckpt['hyper_parameters']['cfg']
        tokenizer_ckpt = hparams.MODEL.get('TOKENIZER_CHECKPOINT_PATH', '')
        if tokenizer_ckpt:
            return load_net(tokenizer_ckpt)
    net = get_model(hparams)
    net.load_state_dict(state_dict, strict=False)
    # Pre-`code_norm` FSQ checkpoints (e.g. transformer_fsq 13-05-2026) were trained without the
    # LayerNorm that TransformerTokenizer now inserts before `to_code`. strict=False leaves it
    # RANDOMLY INITIALISED, and a default LayerNorm still re-normalises, so the encoder is
    # silently corrupted: measured on that checkpoint, only 36% of token IDs match the correct
    # load and reconstruction degrades 0.40 -> 0.87 deg/joint. Restore the trained behaviour.
    import torch.nn as nn
    if isinstance(getattr(net, 'code_norm', None), nn.LayerNorm) \
            and not any(k.endswith('code_norm.weight') for k in state_dict):
        net.code_norm = nn.Identity()
    net.eval().to(DEVICE)
    return net, hparams


def get_codebook(net):
    """Codebook vectors (K, code_dim) for EMA (`codebook`) or FSQ (`implicit_codebook`)."""
    q = net.quantizer
    cb = getattr(q, 'codebook', None)
    if cb is not None and cb.numel() > 0:
        return cb
    return q.implicit_codebook


def _is_cosine(net):
    return getattr(net.quantizer, 'dist_metric', None) == 'cosine'


@torch.no_grad()
def _to_6d_input(net, x):
    """(B, 21, 3, 3) rotmat or (B, 63) -> (B, 21, 6) rot6d, matching the model's front end."""
    from tokenization.models.rotation_utils import matrix_to_rotation_6d
    B = x.shape[0]
    if x.dim() == 2:
        x = x.view(B, net.num_joints, -1)
    if x.shape[-1] == 3 and net.input_joint_dim == 6:
        x = matrix_to_rotation_6d(x)
    return x


@torch.no_grad()
def encode_full(net, gt_pose):
    """Replicate TransformerTokenizer.encode, returning the intermediate tensors.

    Returns dict with:
      pre    (B, T, code_dim)  continuous latents out of `to_code` (pre-quantization)
      post   (B, T, code_dim)  dequantized code vectors (what the decoder receives)
      idx    (B, T)            code indices
    """
    x = _to_6d_input(net, gt_pose)
    B = x.shape[0]
    if hasattr(net, 'joint_queries'):
        xe = net.encoder(x, mask=getattr(net, 'skeleton_mask', None))
        if getattr(net, 'use_kinematic_pe', False):
            xe = xe + net.kinematic_pe_proj(net.lap_pe).unsqueeze(0)
        q = net.cross_attn_down(net.latent_queries.expand(B, -1, -1), context=xe)
        pre = net.to_code(net.code_norm(q))                # (B, T, code_dim)
        flat = net.quantizer.preprocess(pre.permute(0, 2, 1))
        idx = net.quantizer.quantize(flat).view(B, -1)     # (B, T)
        post = net.quantizer.dequantize(idx)               # (B, T, code_dim)
    else:
        xe = net.encoder(x)
        pre = xe.permute(0, 2, 1).contiguous()             # (B, T, code_dim)
        flat = net.quantizer.preprocess(xe)
        idx = net.quantizer.quantize(flat).view(B, -1)     # (B, T)
        post = net.quantizer.dequantize(idx).view(B, -1, net.code_dim)
    return {'pre': pre, 'post': post, 'idx': idx}


@torch.no_grad()
def decode_from_codes(net, idx):
    """Code indices (B, T) -> predicted 6D pose (B, 21, 6)."""
    feat = net.quantizer.dequantize(idx)                   # (B, T, code_dim)
    return _decode_feat(net, feat)


@torch.no_grad()
def decode_from_latent(net, latent):
    """Continuous per-token latent (B, T, code_dim) -> 6D pose, bypassing quantization."""
    return _decode_feat(net, latent)


@torch.no_grad()
def _decode_feat(net, feat):
    B = feat.shape[0]
    if hasattr(net, 'joint_queries'):
        xd = net.decoder(feat)
        jq = net.joint_queries.expand(B, -1, -1)
        xd = net._cross_attn_up_joints(jq, xd)
        return net.decoder_projection(xd)                  # (B, 21, 6)
    return net.decoder(feat.permute(0, 2, 1).contiguous())['pred_pose_body_6d']


@torch.no_grad()
def pose6d_to_rotmat(net, pred_6d):
    from tokenization.models.rotation_utils import rotation_6d_to_matrix
    B = pred_6d.shape[0]
    return rotation_6d_to_matrix(pred_6d.reshape(-1, 6)).view(B, net.num_joints, 3, 3)


@torch.no_grad()
def rotmat_to_joints(rotmat):
    """(B, 21, 3, 3) body pose -> (B, 21, 3) body-joint positions (SMPL joints 1..21)."""
    from tokenization.models.transformer_pose_vqvae import body_model
    out = body_model(body_pose=rotmat.to(DEVICE))
    return out.joints[:, 1:22]


def geodesic_deg(R1, R2):
    """Per-element geodesic angle (degrees) between two (..., 3, 3) rotation batches."""
    Rrel = torch.matmul(R1.transpose(-1, -2), R2)
    tr = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


# --------------------------------------------------------------------------- #
# Data collection                                                             #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect(net, hparams, num_poses):
    """Cache latents, codes, GT/pred poses and joints over a balanced val-set sample.

    The val `ValDataset` merges every val dataset under one merged `dataset_name`, so to keep
    per-sample dataset labels we build a single-dataset loader per entry in VALLIST and draw a
    shuffled, balanced slice from each. Everything is stored on CPU; SMPL is run per batch
    (CACHE_SMPL off) so we never precompute meshes for datasets we only subsample.
    """
    from tokenization.dataset.dataset_poseVQ import get_dataloader
    from tokenization.utils.eval_poseVQ import gt_from_batch
    from tokenization.models.transformer_pose_vqvae import body_model

    hparams.DATA.NUM_WORKERS = min(getattr(hparams.DATA, 'NUM_WORKERS', 4), 4)
    hparams.DATA.CACHE_SMPL = False
    ds_list = hparams.DATA.VALLIST.split('_')
    per_ds = math.ceil(num_poses / len(ds_list))

    store = {k: [] for k in ['pre', 'post', 'idx', 'gt_rotmat', 'pred_rotmat',
                             'gt_jnts', 'pred_jnts']}
    names = []
    saved_vallist = hparams.DATA.VALLIST
    for ds in ds_list:
        hparams.DATA.VALLIST = ds
        loader = get_dataloader(hparams, split='val', shuffle=True)
        got = 0
        for batch in loader:
            gt_pose, _, gt_jnts = gt_from_batch(batch, body_model)   # (B,21,3,3), (B,J,3)
            enc = encode_full(net, gt_pose)
            pred_rotmat = pose6d_to_rotmat(net, decode_from_codes(net, enc['idx']))
            pred_jnts = rotmat_to_joints(pred_rotmat)

            store['pre'].append(enc['pre'].cpu())
            store['post'].append(enc['post'].cpu())
            store['idx'].append(enc['idx'].cpu())
            store['gt_rotmat'].append(gt_pose.cpu())
            store['pred_rotmat'].append(pred_rotmat.cpu())
            store['gt_jnts'].append(gt_jnts[:, 1:22].cpu())
            store['pred_jnts'].append(pred_jnts.cpu())
            names.extend([ds] * gt_pose.shape[0])

            got += gt_pose.shape[0]
            if got >= per_ds:
                break
    hparams.DATA.VALLIST = saved_vallist

    out = {k: torch.cat(v, 0) for k, v in store.items()}
    out['names'] = names
    return out


# --------------------------------------------------------------------------- #
# 1. Token -> joint influence map                                             #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def analysis_influence(net, cache, out_dir, max_poses=512):
    """Perturb each latent token and measure per-joint rotation change (degrees)."""
    # Random subset so the influence map spans all datasets (the cache is grouped by dataset).
    n = cache['idx'].shape[0]
    sel = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:max_poses]
    idx = cache['idx'][sel].to(DEVICE)                     # (B, T)
    B, T = idx.shape
    J = net.num_joints

    base_rot = pose6d_to_rotmat(net, decode_from_codes(net, idx))    # (B, J, 3, 3)
    # Donor tokens come from a RANDOM other pose (val data is unshuffled: adjacent rows are
    # near-duplicate frames of one motion, so a roll-by-1 swap would change almost nothing).
    donor = idx[torch.randperm(B, generator=torch.Generator(device=DEVICE).manual_seed(0),
                               device=DEVICE)]

    infl = torch.zeros(T, J, device=DEVICE)
    for t in range(T):
        pert = idx.clone()
        pert[:, t] = donor[:, t]
        rot = pose6d_to_rotmat(net, decode_from_codes(net, pert))
        infl[t] = geodesic_deg(base_rot, rot).mean(0)      # (J,) mean over poses
    infl = infl.cpu()                                      # (T, J) degrees

    # Per-token specialization: share of a token's total influence on its top joint.
    row_sum = infl.sum(1).clamp(min=1e-8)
    top_share = (infl.max(1).values / row_sum)             # (T,) 1 == fully specialized
    top_joint = infl.argmax(1)                             # (T,)
    # Per-joint concentration: how many tokens supply 80% of a joint's total influence.
    col = infl / infl.sum(0, keepdim=True).clamp(min=1e-8)
    srt = col.sort(0, descending=True).values
    n80 = (srt.cumsum(0) < 0.8).sum(0) + 1                  # (J,)

    # --- heatmap ---
    fig, ax = plt.subplots(figsize=(7, 12))
    im = ax.imshow(infl.numpy(), aspect='auto', cmap='viridis',
                   interpolation='nearest')
    ax.set_xticks(range(J)); ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=7)
    ax.set_xlabel('body joint'); ax.set_ylabel('latent token (0..%d)' % (T - 1))
    ax.set_title('Token → joint influence  (deg rotation change when token swapped)')
    fig.colorbar(im, ax=ax, fraction=0.025, label='mean geodesic Δ (deg)')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, '1_influence_heatmap.png'), dpi=130)
    plt.close(fig)

    # --- specialization histogram ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(top_share.numpy(), bins=30, color='#4C78A8')
    ax[0].set_xlabel('top-joint influence share'); ax[0].set_ylabel('# tokens')
    ax[0].set_title('Token specialization (1.0 = one joint)')
    ax[1].bar(range(J), n80.numpy(), color='#E45756')
    ax[1].set_xticks(range(J)); ax[1].set_xticklabels(JOINT_NAMES, rotation=90, fontsize=7)
    ax[1].set_ylabel('# tokens for 80% influence')
    ax[1].set_title('Per-joint code spread')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, '1_influence_summary.png'), dpi=130)
    plt.close(fig)

    # per-joint: which tokens matter most
    joint_top_tokens = {JOINT_NAMES[j]: infl[:, j].topk(min(5, T)).indices.tolist()
                        for j in range(J)}
    return {
        'tokens': int(T), 'joints': int(J),
        'mean_influence_deg': float(infl.mean()),
        'token_specialization_median': float(top_share.median()),
        'token_specialization_frac_gt_0.5': float((top_share > 0.5).float().mean()),
        'joint_spread_median_tokens_for_80pct': float(n80.float().median()),
        'joint_top_tokens': joint_top_tokens,
        'per_joint_total_influence_deg': {JOINT_NAMES[j]: float(infl[:, j].sum()) for j in range(J)},
    }


# --------------------------------------------------------------------------- #
# 2. Per-joint reconstruction error (stratified by dataset)                   #
# --------------------------------------------------------------------------- #
def analysis_recon(cache, out_dir):
    rot_err = geodesic_deg(cache['gt_rotmat'], cache['pred_rotmat'])     # (N, J)
    pos_err = (cache['gt_jnts'] - cache['pred_jnts']).norm(dim=-1) * 1000.0  # (N, J) mm
    J = rot_err.shape[1]
    names = np.array(cache['names'])
    datasets = sorted(set(names.tolist()))

    per_joint_rot = rot_err.mean(0)                        # (J,)
    per_joint_pos = pos_err.mean(0)

    # stratify position error by dataset
    ds_pos = {ds: float(pos_err[torch.from_numpy(names == ds)].mean()) for ds in datasets}
    ds_rot = {ds: float(rot_err[torch.from_numpy(names == ds)].mean()) for ds in datasets}

    fig, ax = plt.subplots(2, 1, figsize=(10, 9))
    ax[0].bar(range(J), per_joint_pos.numpy(), color='#4C78A8')
    ax[0].set_xticks(range(J)); ax[0].set_xticklabels(JOINT_NAMES, rotation=90, fontsize=7)
    ax[0].set_ylabel('MPJPE (mm)'); ax[0].set_title('Per-joint position error')
    ax[1].bar(range(J), per_joint_rot.numpy(), color='#F58518')
    ax[1].set_xticks(range(J)); ax[1].set_xticklabels(JOINT_NAMES, rotation=90, fontsize=7)
    ax[1].set_ylabel('geodesic error (deg)'); ax[1].set_title('Per-joint rotation error')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, '2_per_joint_error.png'), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = range(len(datasets))
    ax.bar(xs, [ds_pos[d] for d in datasets], color='#54A24B')
    ax.set_xticks(list(xs)); ax.set_xticklabels(datasets, rotation=30, ha='right')
    ax.set_ylabel('MPJPE (mm)'); ax.set_title('Reconstruction error by dataset')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, '2_error_by_dataset.png'), dpi=130)
    plt.close(fig)

    return {
        'mpjpe_mm_mean': float(per_joint_pos.mean()),
        'rot_deg_mean': float(per_joint_rot.mean()),
        'per_joint_mpjpe_mm': {JOINT_NAMES[j]: float(per_joint_pos[j]) for j in range(J)},
        'per_joint_rot_deg': {JOINT_NAMES[j]: float(per_joint_rot[j]) for j in range(J)},
        'mpjpe_mm_by_dataset': ds_pos,
        'rot_deg_by_dataset': ds_rot,
        'worst_joint': JOINT_NAMES[int(per_joint_pos.argmax())],
        'best_joint': JOINT_NAMES[int(per_joint_pos.argmin())],
    }


# --------------------------------------------------------------------------- #
# 3. Linear probing (pre- vs post-quantization) for per-joint pose            #
# --------------------------------------------------------------------------- #
def _pca_reduce(X, q):
    """Center and reduce columns of X (N, D) to q comps via low-rank PCA (torch)."""
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    U, S, V = torch.pca_lowrank(Xc, q=min(q, min(Xc.shape) - 1), niter=4)
    return Xc @ V


def _ridge_r2_per_joint(X, Y, joint_dim, lam=1.0, test_frac=0.2, seed=0):
    """Ridge-regress Y (N, J*joint_dim) from X (N, D); return per-joint R^2 on a held-out split."""
    N = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    ntr = int(N * (1 - test_frac))
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]

    xm, xs = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
    ym = Ytr.mean(0, keepdim=True)
    Xtr, Xte = (Xtr - xm) / xs, (Xte - xm) / xs
    Ytrc = Ytr - ym

    D = Xtr.shape[1]
    A = Xtr.t() @ Xtr + lam * torch.eye(D, device=Xtr.device)
    W = torch.linalg.solve(A, Xtr.t() @ Ytrc)
    pred = Xte @ W + ym

    J = Y.shape[1] // joint_dim
    r2 = []
    for j in range(J):
        c = slice(j * joint_dim, (j + 1) * joint_dim)
        ss_res = ((Yte[:, c] - pred[:, c]) ** 2).sum()
        ss_tot = ((Yte[:, c] - Yte[:, c].mean(0, keepdim=True)) ** 2).sum().clamp(min=1e-8)
        r2.append(float((1 - ss_res / ss_tot).clamp(min=-1.0)))
    return r2


def analysis_probe(net, cache, out_dir, feat_cap=1024):
    from tokenization.models.rotation_utils import matrix_to_rotation_6d
    N = cache['idx'].shape[0]
    J = net.num_joints
    Y = matrix_to_rotation_6d(cache['gt_rotmat'].reshape(-1, 3, 3)).reshape(N, J * 6).to(DEVICE)

    results = {}
    r2_curves = {}
    for tag in ['pre', 'post']:
        X = cache[tag].reshape(N, -1).to(DEVICE).float()
        if X.shape[1] > feat_cap:
            X = _pca_reduce(X, feat_cap)
        r2 = _ridge_r2_per_joint(X, Y, joint_dim=6)
        r2_curves[tag] = r2
        results[f'{tag}_R2_mean'] = float(np.mean(r2))
        results[f'{tag}_R2_per_joint'] = {JOINT_NAMES[j]: r2[j] for j in range(J)}

    gap = [r2_curves['pre'][j] - r2_curves['post'][j] for j in range(J)]
    results['quant_info_loss_R2_mean'] = float(np.mean(gap))
    results['feature_dim_used'] = int(min(cache['pre'].reshape(N, -1).shape[1], feat_cap))

    x = np.arange(J)
    lo = min(min(r2_curves['pre']), min(r2_curves['post']))
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    ax[0].bar(x - 0.2, r2_curves['pre'], width=0.4, label='pre-quant (continuous)', color='#4C78A8')
    ax[0].bar(x + 0.2, r2_curves['post'], width=0.4, label='post-quant (codes)', color='#E45756')
    ax[0].set_xticks(x); ax[0].set_xticklabels(JOINT_NAMES, rotation=90, fontsize=7)
    ax[0].set_ylabel('linear-probe R²'); ax[0].set_ylim(max(0, lo - 0.02), 1.0)
    ax[0].set_title('Per-joint pose decodability from latents (higher = more linearly stored)')
    ax[0].legend()
    ax[1].bar(x, gap, color='#B279A2')
    ax[1].set_xticks(x); ax[1].set_xticklabels(JOINT_NAMES, rotation=90, fontsize=7)
    ax[1].set_ylabel('R² lost to quantization'); ax[1].set_title('pre − post (information destroyed by the codebook)')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, '3_linear_probe_r2.png'), dpi=130)
    plt.close(fig)
    return results


# --------------------------------------------------------------------------- #
# 4. Latent-space interpolation smoothness                                    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def analysis_interp(net, cache, out_dir, n_pairs=32, steps=10):
    N = cache['idx'].shape[0]
    g = torch.Generator().manual_seed(0)
    a = torch.randperm(N, generator=g)[:n_pairs]
    b = torch.randperm(N, generator=g)[:n_pairs]

    # Interpolate the encoder latents (pre-quant); for cosine, renormalize per token
    # so the decoder sees unit-norm codes as in training. FSQ uses the dequantized codes.
    if net.quant == 'fsq':
        La, Lb = cache['post'][a].to(DEVICE), cache['post'][b].to(DEVICE)
        renorm = False
    else:
        La, Lb = cache['pre'][a].to(DEVICE), cache['pre'][b].to(DEVICE)
        renorm = _is_cosine(net)

    alphas = torch.linspace(0, 1, steps, device=DEVICE)
    jnts_path = []            # steps x (P, J, 3)
    for al in alphas:
        L = (1 - al) * La + al * Lb
        if renorm:
            L = F.normalize(L, dim=-1)
        j = rotmat_to_joints(pose6d_to_rotmat(net, decode_from_latent(net, L)))
        jnts_path.append(j.cpu())
    path = torch.stack(jnts_path, 0)                       # (steps, P, J, 3)

    step_jump = (path[1:] - path[:-1]).norm(dim=-1).mean(-1) * 1000.0   # (steps-1, P) mm
    path_len = step_jump.sum(0)                            # (P,) mm
    endpoint = (path[-1] - path[0]).norm(dim=-1).mean(-1) * 1000.0      # (P,) mm
    straightness = (path_len / endpoint.clamp(min=1e-6))   # 1 == perfectly direct
    max_jump = step_jump.max(0).values                     # (P,) mm

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(step_jump.mean(1).numpy(), marker='o', color='#4C78A8')
    ax[0].set_xlabel('interpolation step'); ax[0].set_ylabel('mean joint jump (mm)')
    ax[0].set_title('Per-step motion along latent interpolation')
    ax[1].hist(straightness.numpy(), bins=20, color='#54A24B')
    ax[1].set_xlabel('path length / endpoint distance'); ax[1].set_ylabel('# pairs')
    ax[1].set_title('Interpolation straightness (1.0 = smooth/direct)')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, '4_interpolation.png'), dpi=130)
    plt.close(fig)

    return {
        'n_pairs': int(n_pairs), 'steps': int(steps),
        'straightness_median': float(straightness.median()),
        'straightness_p90': float(straightness.quantile(0.9)),
        'max_step_jump_mm_median': float(max_jump.median()),
        'mean_step_jump_mm': float(step_jump.mean()),
    }


# --------------------------------------------------------------------------- #
# 5. Token "language" structure                                               #
# --------------------------------------------------------------------------- #
def _collapse_topM(col, M):
    """Map a code column (N,) to labels in [0, M]: the M most frequent codes keep distinct
    ids, everything else folds into a single 'other' bucket. Keeps the joint histogram small
    (<=(M+1)^2 bins) so mutual information is estimable from a few thousand samples."""
    counts = torch.bincount(col)
    top = counts.topk(min(M, (counts > 0).sum().item())).indices
    lut = torch.full((counts.numel(),), M, dtype=torch.long)
    lut[top] = torch.arange(top.numel())
    return lut[col], top.numel()


def _norm_mi(a, b, nsym):
    """Normalized MI in [0,1] between two small-alphabet label vectors (nMI = MI/min(Hx,Hy))."""
    joint = a * nsym + b
    pj = torch.bincount(joint, minlength=nsym * nsym).float()
    pj = (pj / pj.sum()).view(nsym, nsym)
    px, py = pj.sum(1), pj.sum(0)
    hx = -(px[px > 0] * px[px > 0].log2()).sum()
    hy = -(py[py > 0] * py[py > 0].log2()).sum()
    m = pj > 0
    mi = (pj[m] * (pj[m] / (px.view(-1, 1).expand_as(pj)[m] * py.view(1, -1).expand_as(pj)[m])).log2()).sum()
    denom = torch.minimum(hx, hy).clamp(min=1e-6)
    return float((mi / denom).clamp(0, 1))


def analysis_language(net, cache, out_dir):
    idx = cache['idx']                                     # (N, T) long
    N, T = idx.shape
    K = net.num_code

    ent_bits = torch.zeros(T)                              # per-position entropy (bits)
    n_unique = torch.zeros(T)
    top_share = torch.zeros(T)
    for t in range(T):
        counts = torch.bincount(idx[:, t], minlength=K).float()
        p = counts / counts.sum().clamp(min=1)
        nz = p[p > 0]
        ent_bits[t] = float(-(nz * nz.log2()).sum())
        n_unique[t] = float((counts > 0).sum())
        top_share[t] = float(p.max())

    # Inter-token redundancy via normalized MI on collapsed top-M symbols (robust to the
    # finite-sample bias that makes a raw 2048x2048 MI meaningless). Compare adjacent token
    # pairs (local structure) vs. random pairs (global): higher adjacent nMI = tokens are
    # locally correlated, so the true bits/pose is well below the independence upper bound.
    M = 15
    lab = torch.empty_like(idx)
    for t in range(T):
        lab[:, t], _ = _collapse_topM(idx[:, t], M)
    g = torch.Generator().manual_seed(0)
    adj = [_norm_mi(lab[:, t], lab[:, t + 1], M + 1) for t in range(T - 1)]
    rnd_pairs = torch.randint(0, T, (2, min(200, T)), generator=g)
    rnd = [_norm_mi(lab[:, i], lab[:, j], M + 1)
           for i, j in zip(rnd_pairs[0].tolist(), rnd_pairs[1].tolist()) if i != j]
    adj_nmi = float(np.mean(adj)) if adj else 0.0
    rnd_nmi = float(np.mean(rnd)) if rnd else 0.0

    rate_bits = float(ent_bits.sum())                      # bits/pose upper bound (indep. tokens)
    unique_seqs = len({tuple(r.tolist()) for r in idx})

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ent_bits.numpy(), color='#4C78A8')
    ax[0].axhline(math.log2(K), ls='--', c='grey', label=f'max = log2(K) = {math.log2(K):.1f}')
    ax[0].set_xlabel('token position'); ax[0].set_ylabel('usage entropy (bits)')
    ax[0].set_title('Per-position code entropy'); ax[0].legend()
    ax[1].hist(n_unique.numpy(), bins=30, color='#B279A2')
    ax[1].set_xlabel('# distinct codes used at a position'); ax[1].set_ylabel('# positions')
    ax[1].set_title(f'Vocabulary per position (K = {K})')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, '5_token_language.png'), dpi=130)
    plt.close(fig)

    return {
        'num_positions': int(T), 'codebook_size': int(K), 'num_poses': int(N),
        'per_position_entropy_bits_mean': float(ent_bits.mean()),
        'per_position_entropy_bits_min': float(ent_bits.min()),
        'per_position_unique_codes_mean': float(n_unique.mean()),
        'per_position_top_code_share_mean': float(top_share.mean()),
        'adjacent_token_norm_MI_mean': adj_nmi,
        'random_pair_norm_MI_mean': rnd_nmi,
        'rate_bits_per_pose_upper_bound': rate_bits,
        'unique_token_sequences': int(unique_seqs),
        'unique_sequence_frac': float(unique_seqs / N),
    }


# --------------------------------------------------------------------------- #
# 6. PCA embedding of latents + codebook                                      #
# --------------------------------------------------------------------------- #
def analysis_embed(net, cache, out_dir):
    N = cache['idx'].shape[0]
    X = cache['pre'].reshape(N, -1).float()
    emb = _pca_reduce(X.to(DEVICE), 2).cpu().numpy()       # (N, 2)

    names = np.array(cache['names'])
    datasets = sorted(set(names.tolist()))
    # knee bend = geodesic angle of L_Knee (model joint 3) from identity, in degrees.
    knee = geodesic_deg(cache['gt_rotmat'][:, 3],
                        torch.eye(3).expand(N, 3, 3)).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap('tab10')
    for i, ds in enumerate(datasets):
        m = names == ds
        ax[0].scatter(emb[m, 0], emb[m, 1], s=5, color=cmap(i % 10), label=ds, alpha=0.5)
    ax[0].legend(markerscale=2, fontsize=8); ax[0].set_title('Per-pose latent PCA  (by dataset)')
    ax[0].set_xlabel('PC1'); ax[0].set_ylabel('PC2')
    sc = ax[1].scatter(emb[:, 0], emb[:, 1], s=5, c=knee, cmap='viridis', alpha=0.6)
    fig.colorbar(sc, ax=ax[1], label='L-knee bend (deg)')
    ax[1].set_title('Per-pose latent PCA  (by knee flexion)')
    ax[1].set_xlabel('PC1'); ax[1].set_ylabel('PC2')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, '6_latent_pca.png'), dpi=130)
    plt.close(fig)

    # codebook embedding colored by usage
    cb = get_codebook(net).detach().float()
    counts = torch.bincount(cache['idx'].reshape(-1), minlength=cb.shape[0]).float()
    used = counts > 0
    if cb.shape[1] > 2 and used.sum() > 3:
        cbe = _pca_reduce(cb[used].to(DEVICE), 2).cpu().numpy()
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sc = ax.scatter(cbe[:, 0], cbe[:, 1], s=6,
                        c=np.log10(counts[used].numpy() + 1), cmap='magma')
        fig.colorbar(sc, ax=ax, label='log10(usage count + 1)')
        ax.set_title('Codebook PCA (used codes, by usage)')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, '6_codebook_pca.png'), dpi=130)
        plt.close(fig)

    return {'used_codes': int(used.sum()), 'total_codes': int(cb.shape[0])}


# --------------------------------------------------------------------------- #
# 7. Latent perturbation robustness                                           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _decode_joints_chunked(net, latent, renorm, chunk=512):
    """Decode per-token latents (B, T, code_dim) -> joint positions (B, J, 3), in chunks."""
    out = []
    for i in range(0, latent.shape[0], chunk):
        L = latent[i:i + chunk].to(DEVICE)
        if renorm:
            L = F.normalize(L, dim=-1)
        j = rotmat_to_joints(pose6d_to_rotmat(net, decode_from_latent(net, L)))
        out.append(j.cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def analysis_robustness(net, cache, out_dir, max_poses=512, seed=0):
    """Decoder sensitivity to latent noise — the mechanism behind in-the-wild fragility.

    Stage-2 feeds the decoder *predicted* latents, never the clean encoder output, so a tokenizer
    whose decoder amplifies small latent errors into large pose errors can reconstruct clean
    training poses almost perfectly (low stage-1 MPJPE) yet fail once a predictor drives it. This
    measures that amplification directly: perturb the (pre-quant) latents with isotropic Gaussian
    noise at increasing magnitude and track how fast MPJPE grows away from the clean decode.

    Noise is whitened — scaled by each latent dimension's own std — so a given sigma means the
    same *relative* perturbation of every active dimension, comparable across models with
    different latent scales AND anisotropy. (A single global RMS would be dominated by a few
    high-variance dims and under-perturb the informative ones, confounding anisotropy with
    sensitivity.) The slope thus pairs with straightness/interp as a fragility measure that links
    a stage-1 property to stage-2 failure.
    """
    n = cache['idx'].shape[0]
    sel = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:max_poses]

    # Same latent surface interp uses: FSQ has no meaningful pre-quant continuum, so perturb the
    # dequantized codes; cosine decodes unit-norm tokens, so renormalize after adding noise.
    if net.quant == 'fsq':
        L0 = cache['post'][sel].clone()
        renorm = False
    else:
        L0 = cache['pre'][sel].clone()
        renorm = _is_cosine(net)

    # Per-dim std (whitening scale): dead dims get ~0 noise, active dims get noise proportional to
    # their own spread, so sigma is anisotropy-invariant. Floor at a fraction of the max to avoid
    # amplifying numerically-dead dimensions.
    std = L0.float().reshape(-1, L0.shape[-1]).std(0)
    std = std.clamp(min=1e-3 * float(std.max()))           # (code_dim,)
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]

    base_j = _decode_joints_chunked(net, L0, renorm)       # (B, J, 3) clean reference
    g = torch.Generator().manual_seed(seed)
    mpjpe = []                                             # mm of drift vs clean decode, per sigma
    for s in sigmas:
        if s == 0.0:
            mpjpe.append(0.0)
            continue
        noise = torch.randn(L0.shape, generator=g) * (s * std)      # broadcast over last dim
        j = _decode_joints_chunked(net, L0 + noise, renorm)
        mpjpe.append(float((j - base_j).norm(dim=-1).mean() * 1000.0))

    # Low-noise sensitivity: mm of MPJPE drift per unit relative-sigma, over the small-sigma regime
    # where a decent predictor operates. Steeper = the decoder magnifies predictor error more.
    lo = [(s, m) for s, m in zip(sigmas, mpjpe) if 0 < s <= 0.2]
    slope = float(np.mean([m / s for s, m in lo])) if lo else 0.0

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(sigmas, mpjpe, marker='o', color='#E45756')
    ax.set_xlabel('latent noise σ  (fraction of latent RMS)')
    ax.set_ylabel('MPJPE drift vs clean decode (mm)')
    ax.set_title('Decoder sensitivity to latent perturbation\n(steeper = more fragile in the wild)')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, '7_robustness.png'), dpi=130)
    plt.close(fig)

    return {
        'n_poses': int(L0.shape[0]),
        'latent_std_mean': float(std.mean()),
        'sigmas': sigmas,
        'mpjpe_drift_mm_by_sigma': {f'{s:g}': m for s, m in zip(sigmas, mpjpe)},
        'low_noise_slope_mm_per_sigma': slope,
        'mpjpe_drift_at_sigma_0.1_mm': mpjpe[sigmas.index(0.1)],
    }


# --------------------------------------------------------------------------- #
ALL = ['influence', 'recon', 'probe', 'interp', 'language', 'embed', 'robustness']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='tokenizer checkpoint (best_net.pth)')
    ap.add_argument('--out', default='', help='output dir (default <ckpt_dir>/latent_analysis)')
    ap.add_argument('--analyses', default='all', help='comma list: ' + ','.join(ALL))
    ap.add_argument('--num-poses', type=int, default=4096, help='val poses to cache')
    ap.add_argument('--influence-poses', type=int, default=512, help='poses for the influence map')
    ap.add_argument('--robustness-poses', type=int, default=512, help='poses for the noise-robustness sweep')
    ap.add_argument('--quick', action='store_true', help='tiny run for a smoke test')
    args = ap.parse_args()

    which = ALL if args.analyses == 'all' else [a.strip() for a in args.analyses.split(',')]
    if args.quick:
        args.num_poses, args.influence_poses, args.robustness_poses = 512, 128, 128
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), 'latent_analysis')
    os.makedirs(out_dir, exist_ok=True)

    print(f'device={DEVICE}  ckpt={args.ckpt}\nout={out_dir}\nanalyses={which}')
    net, hparams = load_net(args.ckpt)
    print(f'model={hparams.ARCH.MODEL_NAME} quant={net.quant} code_dim={net.code_dim} '
          f'K={net.num_code} tokens={net.num_tokens} joints={net.num_joints}')

    print('caching latents/codes over val set ...')
    cache = collect(net, hparams, args.num_poses)
    print(f'cached {cache["idx"].shape[0]} poses across datasets: {sorted(set(cache["names"]))}')

    stats = {'ckpt': args.ckpt, 'model': hparams.ARCH.MODEL_NAME, 'quant': net.quant,
             'code_dim': net.code_dim, 'num_code': net.num_code, 'num_tokens': net.num_tokens,
             'num_poses': int(cache['idx'].shape[0])}
    runners = {
        'influence': lambda: analysis_influence(net, cache, out_dir, args.influence_poses),
        'recon':     lambda: analysis_recon(cache, out_dir),
        'probe':     lambda: analysis_probe(net, cache, out_dir),
        'interp':    lambda: analysis_interp(net, cache, out_dir),
        'language':  lambda: analysis_language(net, cache, out_dir),
        'embed':     lambda: analysis_embed(net, cache, out_dir),
        'robustness': lambda: analysis_robustness(net, cache, out_dir, args.robustness_poses),
    }
    for name in which:
        print(f'\n=== {name} ===')
        stats[name] = runners[name]()
        for k, v in stats[name].items():
            if not isinstance(v, dict):
                print(f'  {k}: {v}')

    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\nwrote {out_dir}/stats.json and figures.')


if __name__ == '__main__':
    main()
