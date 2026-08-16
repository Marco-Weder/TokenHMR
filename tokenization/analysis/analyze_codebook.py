"""Latent-space health check for a pose-VQ tokenizer checkpoint.

Answers three questions the training-time entropy metrics only partially cover:

  1. USAGE    — are all codes participating?           (norm-entropy, perplexity, utilization)
  2. GEOMETRY — are codes distinct vectors?            (NN cosine similarity, effective rank)
  3. MARGINS  — do latents commit to one code?         (sim(best) - sim(2nd best) on val data)

Entropy alone cannot detect the failure mode where usage is perfectly uniform but the
codes are near-duplicates crowded on a low-dimensional latent manifold (dimensional
collapse) — that is what GEOMETRY and MARGINS measure.

Run from the tokenization/ directory:
    python analyze_codebook.py --ckpt output/<run>/.../best_net.pth
    python analyze_codebook.py --ckpt ... --batches 6 --json stats.json

Geometry needs only the checkpoint. Usage/margins additionally need the val dataset
and a model that can be rebuilt from the checkpoint's hparams (skipped, with a
warning, when the architecture code has drifted from an old checkpoint).
"""
import argparse
import json
import math
import os
import sys
import torch
import torch.nn.functional as F


@torch.no_grad()
def codebook_geometry(codebook):
    """Pairwise-similarity and spectrum stats of the codebook itself (no data needed)."""
    cb = F.normalize(codebook.float(), dim=-1)
    S = cb @ cb.t()
    S.fill_diagonal_(-2.0)
    nn_sim = S.max(dim=1).values
    sv = torch.linalg.svdvals(cb)
    p = (sv ** 2) / (sv ** 2).sum()
    return {
        'K': cb.shape[0], 'd': cb.shape[1],
        'nn_cos_median': float(nn_sim.median()),
        'nn_cos_p95': float(nn_sim.quantile(0.95)),
        'nn_cos_max': float(nn_sim.max()),
        'frac_nn_gt_0.99': float((nn_sim > 0.99).float().mean()),
        'frac_nn_gt_0.95': float((nn_sim > 0.95).float().mean()),
        'codebook_effective_rank': float(torch.exp(-(p * (p + 1e-12).log()).sum())),
    }


@torch.no_grad()
def encode_latents(net, gt_pose):
    """Replicate the tokenizer forward pass up to the pre-quantization latents (B, T, d)."""
    from tokenization.models.rotation_utils import matrix_to_rotation_6d
    x = gt_pose
    B = x.shape[0]
    if x.dim() == 2:
        x = x.view(B, net.num_joints, -1)
    if x.shape[-1] == 3 and net.input_joint_dim == 6:
        x = matrix_to_rotation_6d(x)
    if hasattr(net, 'adjacency'):                                   # GNN tokenizer
        xe = net.encoder(x, net.adjacency)
    else:                                                           # transformer (mask optional)
        xe = net.encoder(x, mask=getattr(net, 'skeleton_mask', None))
        if getattr(net, 'use_kinematic_pe', False):
            xe = xe + net.kinematic_pe_proj(net.lap_pe).unsqueeze(0)
    q = net.cross_attn_down(net.latent_queries.expand(B, -1, -1), context=xe)
    return net.to_code(net.code_norm(q))


@torch.no_grad()
def val_data_stats(net, hparams, codebook, n_batches):
    """Usage, margin, and latent-rank stats over val batches."""
    from tokenization.dataset.dataset_poseVQ import get_dataloader
    from tokenization.utils.eval_poseVQ import gt_from_batch
    from tokenization.models.transformer_pose_vqvae import body_model

    hparams.DATA.NUM_WORKERS = 2
    loader = get_dataloader(hparams, split='val', shuffle=False)
    cbn = F.normalize(codebook.float(), dim=-1)
    K = cbn.shape[0]
    counts = torch.zeros(K, device=cbn.device)
    lats, margins, d1s = [], [], []
    for i, batch in enumerate(loader):
        gt_pose, _, _ = gt_from_batch(batch, body_model)
        lat = encode_latents(net, gt_pose).reshape(-1, cbn.shape[1])
        lats.append(lat)
        z = F.normalize(lat, dim=-1)
        sims = z @ cbn.t()
        top2 = sims.topk(2, dim=-1).values
        counts += torch.bincount(sims.argmax(-1), minlength=K).float()
        margins.append((top2[:, 0] - top2[:, 1]).cpu())
        d1s.append((1.0 - top2[:, 0]).cpu())
        if i + 1 >= n_batches:
            break

    lat = torch.cat(lats).float()
    latc = lat - lat.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(latc / math.sqrt(latc.shape[0]))
    p = (sv ** 2) / (sv ** 2).sum()
    margins = torch.cat(margins)
    d1s = torch.cat(d1s)
    pr = counts / counts.sum()
    H = float(-(pr * (pr + 1e-10).log()).sum())
    return {
        'val_tokens': int(counts.sum()),
        'norm_entropy': H / math.log(K),
        'perplexity': math.exp(H),
        'utilization_pct': float((counts > 0).float().mean() * 100),
        'latent_effective_rank': float(torch.exp(-(p * (p + 1e-12).log()).sum())),
        'latent_dim': lat.shape[1],
        'margin_median': float(margins.median()),
        'margin_p10': float(margins.quantile(0.1)),
        'frac_margin_lt_0.01': float((margins < 0.01).float().mean()),
        'quant_err_median': float(d1s.median()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='tokenizer checkpoint (best_net.pth / latest_checkpoint.pth)')
    ap.add_argument('--batches', type=int, default=6, help='val batches for usage/margin stats')
    ap.add_argument('--json', default='', help='also write stats to this JSON file')
    ap.add_argument('--geometry-only', action='store_true', help='skip the val-data pass')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    codebook = ckpt['net']['quantizer.codebook'].cuda() if torch.cuda.is_available() \
        else ckpt['net']['quantizer.codebook']

    stats = {'ckpt': args.ckpt, 'geometry': codebook_geometry(codebook)}
    g = stats['geometry']
    print(f"\nCodebook geometry  (K={g['K']}, d={g['d']})")
    print(f"  NN cosine similarity  median {g['nn_cos_median']:.4f} | p95 {g['nn_cos_p95']:.4f} | max {g['nn_cos_max']:.4f}")
    print(f"  near-duplicate codes  cos>0.95: {g['frac_nn_gt_0.95']*100:5.1f}% | cos>0.99: {g['frac_nn_gt_0.99']*100:5.1f}%")
    print(f"  effective rank        {g['codebook_effective_rank']:.1f} / {g['d']}")

    if not args.geometry_only:
        try:
            from tokenization.train_poseVQ import get_model
            net = get_model(ckpt['hparams'])
            net.load_state_dict(ckpt['net'], strict=True)
            net.eval()
            if torch.cuda.is_available():
                net.cuda()
            stats['val'] = val_data_stats(net, ckpt['hparams'], codebook, args.batches)
            v = stats['val']
            print(f"\nVal-data stats  ({v['val_tokens']} token assignments)")
            print(f"  usage    norm-entropy {v['norm_entropy']:.3f} | perplexity {v['perplexity']:.0f}/{g['K']} | util {v['utilization_pct']:.1f}%")
            print(f"  latents  effective rank {v['latent_effective_rank']:.1f} / {v['latent_dim']}")
            print(f"  margins  median {v['margin_median']:.5f} | p10 {v['margin_p10']:.5f} | ambiguous(<0.01) {v['frac_margin_lt_0.01']*100:.1f}%")
            print(f"  quant    err median {v['quant_err_median']:.5f}")
        except RuntimeError as e:
            print(f"\n[warn] model rebuild failed (old checkpoint vs current code?) — geometry only.\n       {e}")

    print("\nReading the numbers:")
    print("  usage healthy     : norm-entropy > 0.85, utilization > 90%")
    print("  geometry healthy  : few codes with NN-cos > 0.99, effective rank a decent fraction of d")
    print("  margins healthy   : median margin >> 0 (latents commit to one code)")
    print("  uniform usage + near-duplicate codes = dimensional collapse -> regularize the LATENTS")
    print("  spread codes + low perplexity        = usage collapse       -> classic codebook collapse")

    if args.json:
        json.dump(stats, open(args.json, 'w'), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == '__main__':
    main()
