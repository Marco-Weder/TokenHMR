"""Token ambiguity analysis for TokenHMR (thesis extension, Parts 1 & 2).

Motivation: downstream top-1 token accuracy is low (~20%) yet decoded MPJPE stays reasonable.
This script quantifies whether "wrong" token predictions are actually harmful in pose space.

Part 1 — Wrong-token pose-damage histogram
  For each validation sample let q = encode(GT pose) be the tokenizer-assigned token sequence and
  q_pred = argmax(downstream logits). For every position i where q_pred_i != q_i, decode q and the
  one-token-replaced sequence (q with position i set to q_pred_i) through the frozen tokenizer
  decoder and measure pose_damage_i = MPJPE(decode(q_replace_i), decode(q)). We histogram the
  damage of all wrong tokens and rank token positions by average sensitivity.

Part 2 — Mesh vertex-error maps
  For a benign and a harmful wrong-token event, decode original vs replaced to SMPL meshes and
  color the mesh by per-vertex displacement, showing which body parts a token controls.

Run from the repo root (the dir containing the `tokenhmr/` package and `dataset_dir/`):
  python tokenhmr/analyze_token_ambiguity.py \
      --checkpoint logs/<run>/.../checkpoints/last.ckpt \
      --model_config logs/<run>/.../model_config.yaml \
      --dataset 3DPW-TEST --dataset_dir dataset_dir/evaluation_data \
      --num_batches 20 --out_dir results/ambiguity
"""

import argparse
import json
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Make the sibling `tokenization` package importable (mirrors token_classifier's path setup).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.configs import dataset_eval_config
from lib.datasets import create_dataset
from lib.models import load_tokenhmr
from lib.utils import recursive_to
from lib.utils.geometry import aa_to_rotmat

# Damage histogram bins (mm), matching the supervisor's proposal.
BIN_EDGES = [0.0, 5.0, 10.0, 20.0, 50.0, float('inf')]
BIN_LABELS = ['<5', '5-10', '10-20', '20-50', '>50']


def load_tokenizer(ckpt_path, device):
    """Load the frozen full tokenizer with mesh inference enabled (encode + decode + SMPL)."""
    from tokenization.models.transformer_pose_vqvae import TransformerTokenizer
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    arch = ckpt['hparams'].ARCH
    tok = TransformerTokenizer(arch, mesh_inference=True)
    missing, unexpected = tok.load_state_dict(ckpt['net'], strict=False)
    # Same fix as lib/utils/token_metrics.py: pre-code_norm FSQ checkpoints lack
    # code_norm.{weight,bias}. They were trained without that LayerNorm, so a freshly
    # initialised one would silently change the encoded token IDs. Any other missing or
    # unexpected key is a real mismatch and must still fail loudly.
    if any(k.startswith('code_norm.') for k in missing):
        tok.code_norm = torch.nn.Identity()
        missing = [k for k in missing if not k.startswith('code_norm.')]
    if missing or unexpected:
        raise RuntimeError(f'tokenizer load mismatch for {ckpt_path}: '
                           f'missing={missing} unexpected={unexpected}')
    tok.eval()
    for p in tok.parameters():
        p.requires_grad = False
    return tok.to(device)


@torch.no_grad()
def decode_ids(tok, ids, want_vertices=False):
    """Decode token-id sequences (B, num_tokens) -> body joints (and optionally vertices).

    One-hot -> dequantize_logits is an exact codebook lookup (hard decode), so this decodes the
    discrete sequence through the same frozen decoder the downstream model uses.
    """
    from tokenization.models.transformer_pose_vqvae import rotation_6d_to_matrix, body_model
    B = ids.shape[0]
    onehot = F.one_hot(ids, tok.quantizer.nb_code).float()
    pose6d = tok.decode_logits(onehot)                              # (B, num_joints, 6)
    rotmat = rotation_6d_to_matrix(pose6d.reshape(-1, 6)).view(B, tok.num_joints, 3, 3)
    out = body_model(body_pose=rotmat)
    joints = out.joints[:, 1:tok.num_joints + 1]                   # (B, 21, 3), pelvis excluded
    verts = out.vertices if want_vertices else None
    return joints, verts


def mpjpe_mm(a, b):
    """Per-sample MPJPE in mm between two joint sets (B, J, 3), no alignment (shared canonical frame)."""
    return torch.sqrt(((a - b) ** 2).sum(-1)).mean(-1) * 1000.0    # (B,)


def render_vertex_error(verts, faces, err_mm, path, title, elev=10, azim=70):
    """Save a mesh figure colored by per-vertex displacement (mm)."""
    verts = np.asarray(verts)
    faces = np.asarray(faces)
    face_err = err_mm[faces].mean(axis=1)                          # per-face error for coloring
    vmax = max(float(np.percentile(err_mm, 99)), 1e-6)
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap('turbo')

    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=1.0)
    mesh.set_facecolor(cmap(norm(face_err)))
    mesh.set_edgecolor('none')
    ax.add_collection3d(mesh)

    c = verts.mean(0)
    r = float(np.abs(verts - c).max())
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm); m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.5, label='per-vertex error (mm)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Token ambiguity analysis (Parts 1 & 2)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='3DPW-TEST', help='3DPW-TEST or EMDB')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_batches', type=int, default=20, help='Batches to analyze (<=0: all)')
    parser.add_argument('--out_dir', type=str, default='results/ambiguity')
    parser.add_argument('--benign_thresh', type=float, default=2.0, help='mm; example benign swap')
    parser.add_argument('--harmful_thresh', type=float, default=50.0, help='mm; example harmful swap')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, model_cfg = load_tokenhmr(checkpoint_path=args.checkpoint,
                                     model_cfg=args.model_config, dataset_dir=args.dataset_dir)
    model = model.to(device).eval()
    tok_ckpt = model_cfg.MODEL.TOKENIZER_CHECKPOINT_PATH
    tok = load_tokenizer(tok_ckpt, device)
    from tokenization.models.transformer_pose_vqvae import body_model
    faces = body_model.faces

    dataset_cfg = dataset_eval_config()[args.dataset]
    for key in ('DATASET_FILE', 'IMG_DIR'):
        if key in dataset_cfg:
            dataset_cfg[key] = os.path.join(args.dataset_dir, dataset_cfg[key])
    dataset = create_dataset(model_cfg, dataset_cfg, train=False)
    loader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)

    damages = []                                     # all wrong-token pose-damage values (mm)
    n_tokens = tok.num_tokens if hasattr(tok, 'num_tokens') else None
    pos_dmg_sum = None                               # per-position summed damage (over wrong tokens)
    pos_dmg_cnt = None
    total_tokens = 0
    total_correct = 0
    total_top5 = 0
    benign_example = None                            # (ids_orig, ids_repl, pos, damage)
    harmful_example = None

    n_batches = len(loader) if args.num_batches <= 0 else min(args.num_batches, len(loader))
    for bi, batch in enumerate(tqdm(loader, total=n_batches)):
        if bi >= n_batches:
            break
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        if 'cls_logits' not in out:
            raise RuntimeError('Model has no token head (cls_logits missing).')

        # GT token sequence q (encode GT body pose) over samples with a valid GT pose.
        bp = batch['smpl_params']['body_pose']
        B = bp.shape[0]
        bp = bp.view(B, -1)
        if batch['smpl_params_is_axis_angle']['body_pose'].all():
            gt_bp_rotmat = aa_to_rotmat(bp.reshape(-1, 3)).view(B, -1, 3, 3)
        else:
            gt_bp_rotmat = bp.view(B, -1, 3, 3)
        has_gt = batch['has_smpl_params']['body_pose'].reshape(B, -1)[:, 0] > 0.5
        if not bool(has_gt.any()):
            continue

        logits = out['cls_logits']
        logits = logits[-B:] if logits.shape[0] != B else logits
        logits = logits[has_gt]
        gt_bp_rotmat = gt_bp_rotmat[has_gt]
        b = logits.shape[0]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            q = tok.encode(gt_bp_rotmat[:, :tok.num_joints].float())     # (b, T)
        q_pred = logits.argmax(-1)                                       # (b, T)
        T = q.shape[1]
        if pos_dmg_sum is None:
            pos_dmg_sum = torch.zeros(T, device=device)
            pos_dmg_cnt = torch.zeros(T, device=device)

        # top-1 / top-5 accuracy for context (why top-1 can mislead).
        total_tokens += b * T
        total_correct += int((q_pred == q).sum())
        top5 = logits.topk(5, dim=-1).indices
        total_top5 += int((top5 == q.unsqueeze(-1)).any(-1).sum())

        # Reference decode of the GT token sequence.
        joints_q, _ = decode_ids(tok, q)                                # (b, 21, 3)

        # For each token position, swap to the predicted token and measure the damage.
        for i in range(T):
            mism = q_pred[:, i] != q[:, i]                              # (b,)
            if not bool(mism.any()):
                continue
            q_rep = q.clone()
            q_rep[:, i] = q_pred[:, i]
            joints_rep, _ = decode_ids(tok, q_rep)
            dmg = mpjpe_mm(joints_rep, joints_q)                       # (b,)
            dmg_w = dmg[mism]
            damages.append(dmg_w.detach().cpu())
            pos_dmg_sum[i] += float(dmg_w.sum())
            pos_dmg_cnt[i] += int(mism.sum())

            # Stash a benign and a harmful single-token example for Part 2.
            if benign_example is None:
                cand = torch.where(mism & (dmg < args.benign_thresh))[0]
                if len(cand):
                    s = int(cand[0])
                    benign_example = (q[s].cpu(), q_rep[s].cpu(), i, float(dmg[s]))
            if harmful_example is None:
                cand = torch.where(mism & (dmg > args.harmful_thresh))[0]
                if len(cand):
                    s = int(cand[0])
                    harmful_example = (q[s].cpu(), q_rep[s].cpu(), i, float(dmg[s]))

    if not damages:
        print('No wrong-token events found (empty analysis).')
        return

    damages = torch.cat(damages).numpy()
    # Raw per-event damages, so the thesis figure can be re-binned without re-running the
    # (GPU-bound) sweep. thesis_figures/plot_token_ambiguity.py reads this.
    np.savez_compressed(os.path.join(args.out_dir, 'damages.npz'), damages=damages)
    counts, _ = np.histogram(damages, bins=BIN_EDGES)
    frac = counts / counts.sum()

    # Per-position sensitivity ranking (mean damage over that position's wrong tokens).
    pos_mean = (pos_dmg_sum / pos_dmg_cnt.clamp(min=1)).cpu().numpy()
    pos_cnt = pos_dmg_cnt.cpu().numpy()
    order = np.argsort(-pos_mean)

    stats = {
        'dataset': args.dataset,
        'checkpoint': args.checkpoint,
        'n_wrong_tokens': int(damages.size),
        'top1_accuracy': total_correct / max(total_tokens, 1),
        'top5_accuracy': total_top5 / max(total_tokens, 1),
        'wrong_token_fraction': damages.size / max(total_tokens, 1),
        'damage_mm_mean': float(damages.mean()),
        'damage_mm_median': float(np.median(damages)),
        'damage_mm_p90': float(np.percentile(damages, 90)),
        'histogram_bins_mm': BIN_LABELS,
        'histogram_fraction': frac.tolist(),
        'histogram_counts': counts.tolist(),
        'most_sensitive_positions': [
            {'position': int(p), 'mean_damage_mm': float(pos_mean[p]), 'n_wrong': int(pos_cnt[p])}
            for p in order[:15]
        ],
    }
    with open(os.path.join(args.out_dir, 'ambiguity_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Part 1 figure: pose-damage histogram.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(BIN_LABELS, frac, color='#4C78A8')
    for x, fr in enumerate(frac):
        ax.text(x, fr, f'{fr*100:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Pose damage from wrong token (mm)')
    ax.set_ylabel('Fraction of wrong predictions')
    ax.set_title(f'Wrong-token pose damage — {args.dataset}\n'
                 f'top-1 acc {stats["top1_accuracy"]*100:.1f}%  '
                 f'(mean damage {stats["damage_mm_mean"]:.1f} mm)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'pose_damage_histogram.png'), dpi=130)
    plt.close(fig)

    # Part 2 figures: vertex error maps for a benign and a harmful token swap.
    for name, ex in (('benign', benign_example), ('harmful', harmful_example)):
        if ex is None:
            print(f'No {name} example found (adjust --{name}_thresh).')
            continue
        ids_orig, ids_repl, pos, dmg = ex
        _, verts_o = decode_ids(tok, ids_orig[None].to(device), want_vertices=True)
        _, verts_r = decode_ids(tok, ids_repl[None].to(device), want_vertices=True)
        verr = (torch.sqrt(((verts_r - verts_o) ** 2).sum(-1))[0] * 1000.0).cpu().numpy()
        render_vertex_error(
            verts_o[0].cpu().numpy(), faces, verr,
            os.path.join(args.out_dir, f'vertex_error_{name}.png'),
            title=f'{name}: swap token @pos {pos}  (MPJPE {dmg:.1f} mm, max vert {verr.max():.1f} mm)')

    print(json.dumps(stats, indent=2))
    print(f'\nSaved histogram, vertex-error maps and stats to {args.out_dir}')


if __name__ == '__main__':
    main()
