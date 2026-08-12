"""Decoder-aware soft token targets (proposal Part 4).

Hard one-hot token CE treats every non-target code as equally wrong. But many codes decode
to almost the same pose, so hard CE forces the head to make distinctions that do not matter
in pose space -- measured cost on the FSQ runs: +16.6mm EMDB / +8.4mm 3DPW PA-MPJPE against
the pose-loss-only baseline, with the orientation component unchanged.

This module builds the proposal's soft targets:

    C_i    = TopM nearest codes to the GT latent at token position i
    d_i(k) = MPJPE(Decoder(q with position i replaced by k), GT pose)      [mm]
    w_i(k) = softmax(-(d_i(k) - min_k d_i(k)) / tau)      over k in C_i

Targets are computed EXACTLY (real decodes, real MPJPE) but only for a random subset of
token positions each step. Averaging over sampled positions is an unbiased estimator of the
average over all `num_tokens` positions, just noisier -- and exact targets for all 160
positions would cost ~2-5 s/step against a ~0.3 s/step budget. The alternative, the
proposal's offline per-sample table, is not available here: the training mix is a weighted
webdataset with no stable per-sample index.

Note that latent distance is NOT a usable shortcut for d_i(k): measured Spearman against
true decoded damage is 0.655. Encoder-latent softness and decoder-aware softness are
different targets, which is what makes this a distinct experiment from `encode_soft`.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from .token_metrics import _get_gt_encoder
from .geometry import rot6d_to_rotmat

# Module-scoped cache: {(model_path, device): (rest_joints (24,3), parents (24,))}.
# The SMPL skeleton is fixed; only the kinematic chain is needed to turn joint rotations
# into joint positions, so no vertex/shape machinery is ever built.
_SKELETON = {}


def _get_skeleton(smpl_model_path, device):
    key = (smpl_model_path, str(device))
    if key not in _SKELETON:
        import smplx
        m = smplx.SMPL(model_path=smpl_model_path)
        # Rest joints at the mean shape. betas=0 is deliberate: MPJPE here compares two poses
        # of the SAME subject, and is taken root-relative, so shape cancels out.
        rest = (m.J_regressor @ m.v_template).to(torch.float32)          # (24, 3)
        parents = m.parents.to(torch.long)                              # (24,)
        _SKELETON[key] = (rest.to(device), parents.to(device))
    return _SKELETON[key]


def _body_pose_to_joints(body_rotmat, smpl_model_path):
    """(N, 21, 3, 3) body-joint rotations -> (N, 24, 3) root-relative joint positions.

    Joints-only forward: poses the kinematic chain and skips vertex skinning entirely.
    Global orient and the two hand joints are held at identity -- they are not part of the
    tokenized body pose, and a shared constant cannot affect a comparison between two poses.
    """
    from smplx.lbs import batch_rigid_transform

    N = body_rotmat.shape[0]
    rest, parents = _get_skeleton(smpl_model_path, body_rotmat.device)

    full = torch.eye(3, device=body_rotmat.device, dtype=torch.float32)
    full = full.view(1, 1, 3, 3).repeat(N, 24, 1, 1)
    full[:, 1:1 + body_rotmat.shape[1]] = body_rotmat.to(torch.float32)

    joints = rest.unsqueeze(0).expand(N, -1, -1).contiguous()
    posed, _ = batch_rigid_transform(full, joints, parents, dtype=torch.float32)
    return posed - posed[:, [0]]


_POS_WEIGHT_CACHE = {}


def position_sampling_weights(npz_path, num_tokens, device):
    """Per-position sampling probabilities from the token--joint influence map.

    `influence[i, j]` (Equation 3.30, dumped by tokenization/analyze_token_maps.py) is the
    mean rotation joint j undergoes when token position i is swapped. Sampling positions in
    proportion to their raw influence would put almost all of the mass on the wrists, which
    move most in absolute terms simply because they sit at the end of the longest kinematic
    chain. We therefore normalise each joint's column to sum to one before summing over
    joints, so every joint contributes the same total probability and, within a joint, the
    positions that actually drive it are preferred. This is the "uniform across joints"
    part: the weighting says *which position matters for a given joint*, not *which joint
    moves the most millimetres*.

    Returns a (num_tokens,) probability vector summing to 1.
    """
    key = (npz_path, int(num_tokens), str(device))
    if key not in _POS_WEIGHT_CACHE:
        infl = np.load(npz_path)['influence'].astype(np.float64)      # (T, J)
        if infl.shape[0] != int(num_tokens):
            raise ValueError(f'influence map has {infl.shape[0]} positions but the tokenizer '
                             f'has {num_tokens}: {npz_path}')
        col = infl.sum(axis=0, keepdims=True)
        col[col <= 0] = 1.0                       # a joint no token reaches contributes nothing
        w = (infl / col).sum(axis=1)              # each joint contributes equal total mass
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError(f'degenerate influence map: {npz_path}')
        p = w / w.sum()
        # logged once per path so a run's log records whether weighting was actually on
        ess = 1.0 / float((p ** 2).sum())
        print(f'[token-CE] influence-weighted position sampling from {os.path.basename(npz_path)}: '
              f'spread {p.max() / p.min():.1f}x, effective sample size {ess:.0f}/{len(p)}')
        p = torch.as_tensor(p, dtype=torch.float32, device=device)
        _POS_WEIGHT_CACHE[key] = p
    return _POS_WEIGHT_CACHE[key]


@torch.no_grad()
def decoder_aware_soft_targets(gt_bp_rotmat, has_gt, ckpt_path, smpl_model_path,
                               tau=5.0, num_positions=8, num_candidates=16,
                               decode_chunk=1024, generator=None, pos_weights=None, unbiased=False):
    """Build sparse decoder-aware soft CE targets for a batch.

    Args:
        gt_bp_rotmat: (B, >=21, 3, 3) GT body-joint rotation matrices.
        has_gt:       (B,) bool mask of samples with a valid GT body pose.
        ckpt_path:    frozen tokenizer checkpoint (shared with the token metrics).
        smpl_model_path: SMPL model dir, for the joints-only forward.
        tau:          softmax temperature in mm. Larger = flatter targets.
        num_positions: token positions sampled per step (P of num_tokens).
        pos_weights: optional (num_tokens,) sampling distribution over positions, from
            position_sampling_weights(). None samples uniformly without replacement, which
            is what the Part 4 runs used. When given, positions are drawn from it WITH
            replacement (the joint-uniform weights are mild enough that duplicates are rare).
        unbiased: only meaningful with pos_weights. False (default) means the loss becomes a
            weighted objective in which a position counts in proportion to how much it moves
            the body, which is the point of weighting the sampling in the first place. True
            instead returns inverse-propensity weights that undo the sampling bias, so the
            loss still estimates the plain mean over all positions. That variant is unbiased
            but strictly noisier than uniform sampling unless the per-position loss happens
            to correlate with the weights, so it is not the default.
        num_candidates: candidate codes per sampled position (M). NOTE: M caps the target
            entropy at ln(M) -- with M=16 that is 2.77 nats, just under the measured
            predictor entropy of 2.93, so targets cannot be flatter than the model already is.

    Returns:
        (positions, targets, stats), or (None, None, {}) if no sample has a GT pose.
        positions: (P,) long, the sampled token positions.
        targets:   (n_valid, P, nb_code) float, mass only on each position's candidates.
        pos_ips:   (P,) float, 1 / (T * p_i) for weighted sampling and all-ones for uniform.
            Multiplying each sampled position's loss by this keeps the mean over the sample
            an unbiased estimate of the mean over all T positions.
        stats:     telemetry dict of 0-dim tensors.
    """
    device = gt_bp_rotmat.device
    if has_gt is None or not bool(has_gt.any()):
        return None, None, None, {}

    enc = _get_gt_encoder(ckpt_path, device)
    with torch.cuda.amp.autocast(enabled=False):
        gt_in = gt_bp_rotmat[has_gt][:, :enc.num_joints].float()
        code, latent, codebook = enc.encode_latent_and_codes(gt_in)     # (n,T) (n,T,C) (S,C)
        n, T = code.shape
        S = codebook.shape[0]
        P = min(int(num_positions), T)
        M = min(int(num_candidates), S)

        if pos_weights is None:
            positions = torch.randperm(T, generator=generator, device=device)[:P]
            pos_ips = torch.ones(P, device=device)
        else:
            positions = torch.multinomial(pos_weights, P, replacement=True, generator=generator)
            # 1/(T*p_i) is the exact importance weight for sampling with replacement; without
            # replacement the correct weight is the inclusion probability, which has no closed
            # form for unequal probabilities.
            if unbiased:
                pos_ips = 1.0 / (T * pos_weights[positions])
            else:
                pos_ips = torch.ones(P, device=device)

        # Candidate codes: the M nearest to the GT latent at each sampled position.
        # FSQ rounds per dimension, so the encoder's own code is the nearest grid point and
        # is always inside the candidate set.
        d_lat = torch.cdist(latent[:, positions].reshape(n * P, -1), codebook)   # (n*P, S)
        cand = d_lat.topk(M, largest=False).indices.view(n, P, M)               # (n, P, M)

        # One modified sequence per (sample, position, candidate): only position i changes.
        seq = code.view(n, 1, 1, T).repeat(1, P, M, 1)
        seq.scatter_(3, positions.view(1, P, 1, 1).expand(n, P, M, 1), cand.unsqueeze(-1))

        pose6d = enc.decode_code_ids(seq.view(n * P * M, T), chunk=decode_chunk)
        rot = rot6d_to_rotmat(pose6d.reshape(-1, 6)).view(n * P * M, enc.num_joints, 3, 3)

        cand_j = _body_pose_to_joints(rot, smpl_model_path).view(n, P, M, 24, 3)
        gt_j = _body_pose_to_joints(gt_in, smpl_model_path).view(n, 1, 1, 24, 3)

        # d_i(k): MPJPE in mm between the candidate's decoded pose and the GT pose.
        d_mm = (cand_j - gt_j).norm(dim=-1).mean(-1) * 1000.0                   # (n, P, M)
        w = F.softmax(-(d_mm - d_mm.min(-1, keepdim=True).values) / float(tau), dim=-1)

        targets = torch.zeros(n, P, S, device=device, dtype=w.dtype)
        targets.scatter_(2, cand, w)

        # Telemetry. `best_not_encoder` is the fraction of positions where the encoder's own
        # token ID is NOT the best-decoding candidate -- i.e. where hard CE actively points at
        # a worse code. It is the core justification for this loss, so it is logged, not
        # merely asserted.
        ent = -(w * (w + 1e-10).log()).sum(-1)
        best = cand.gather(2, d_mm.argmin(-1, keepdim=True)).squeeze(-1)        # (n, P)
        enc_code = code.gather(1, positions.view(1, P).expand(n, P))
        stats = {
            'token_target_entropy': ent.mean(),
            'token_target_best_mass': w.max(-1).values.mean(),
            'token_target_best_not_encoder': (best != enc_code).float().mean(),
            'token_target_damage_mm': d_mm.mean(),
        }

    return (positions, targets.detach(), pos_ips.detach(),
            {k: v.detach().float() for k, v in stats.items()})
