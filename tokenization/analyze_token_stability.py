"""How good a cross-entropy target is each tokenizer's code index?

Motivation: the downstream CE-only run on the cosine dim-4 tokenizer plateaued near
5% top-1 accuracy. That is not a property of the CE implementation but of the
codebook it is asked to classify into: if the *encoder itself* assigns a different
code to two poses that are visually identical, then near-duplicate training images
carry different labels and the CE target is inconsistent, hence unlearnable. This
script measures that directly, so a tokenizer can be ranked *before* spending a
600k-step downstream run on it.

Four numbers per tokenizer, all on one shared pose set so runs are comparable:

  recon deg     geodesic error per joint of encode->decode. The floor on downstream
                pose accuracy; a tokenizer with a bad floor bottlenecks everything.

  stab@X deg    LABEL STABILITY, the gate on CE. Jitter every joint's axis-angle by
                Gaussian noise of std X deg, re-encode, and report the fraction of
                token IDs that stay the same. Downstream top-1 accuracy is
                effectively upper-bounded by this: an image model cannot resolve a
                label that the encoder itself does not hold fixed under a
                perturbation smaller than the noise in the data.

  used %        fraction of the codebook that the val poses actually visit. A large
                nominal vocabulary that is 90% dead is a harder CE target than a
                small one that is fully used.

  NN-swap deg   REDUNDANCY, whether the ID is meaningful. Replace every token by its
                nearest neighbour in the codebook and measure how far the decoded
                pose moves. Small means many codes are pose-equivalent, so hard CE
                penalizes the model for "right pose, wrong ID"; large means each ID
                carries real pose information.

Usage (from the tokenization/ directory, in the thesis-HMR env):

    python analyze_token_stability.py \
        --ckpt output/tokenization_transformer_cosine_dim4/.../best_net.pth \
        --ckpt output/tokenization_transformer_fsq/.../best_net.pth \
        --label dim4 --label "fsq d4" --latex

Labels are optional and positional w.r.t. --ckpt; the run directory name is used
when they are omitted. Results (markdown, JSON, and a LaTeX booktabs table) land in
--out, default output/token_stability/.

Two measurement notes that matter for citing these numbers:

  * The FSQ d4 checkpoint (13-05-2026) predates the `code_norm` LayerNorm that
    `TransformerTokenizer` now inserts before `to_code` for FSQ runs. Loading it
    with strict=False leaves that LayerNorm randomly initialised, which corrupts
    the encoder and silently invalidates every metric here. This script detects the
    missing weights and restores `nn.Identity()`, i.e. the graph the checkpoint was
    actually trained with, and prints a note when it does so.
  * Nearest neighbours are taken in the quantizer's OWN metric (chordal on the unit
    sphere for cosine runs, L2 for L2/FSQ), computed in float64 for the cosine case
    where neighbouring codes can sit ~1e-3 apart and fp32 `1 - cos` underflows.
    Pass --nn-metric l2 to force plain L2 for every run instead.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from analyze_latent_pose_info import (DEVICE, load_net, get_codebook, _is_cosine,
                                      encode_full, decode_from_codes, pose6d_to_rotmat,
                                      geodesic_deg)
from visualize_latent_space import _dist

DEFAULT_JITTERS = (0.5, 1.0, 2.0)


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_net_compat(ckpt_path):
    """`load_net` plus the pre-`code_norm` FSQ compatibility fix.

    Returns (net, hparams, patched). See the module docstring: without this, an old
    FSQ checkpoint loads an untrained LayerNorm into the encoder and every metric
    below is measured on a corrupted model.
    """
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    keys = list((raw.get('net') or raw.get('state_dict') or {}).keys())
    del raw

    net, hparams = load_net(ckpt_path)
    patched = False
    code_norm = getattr(net, 'code_norm', None)
    if isinstance(code_norm, nn.LayerNorm) and not any(k.endswith('code_norm.weight') for k in keys):
        net.code_norm = nn.Identity()
        net.eval().to(DEVICE)
        patched = True
    return net, hparams, patched


@torch.no_grad()
def load_pose_set(hparams, num_poses, seed):
    """Balanced GT body-pose sample (N, 21, 3, 3) drawn from the val datasets.

    Mirrors `analyze_latent_pose_info.collect`: the val loader merges all datasets
    under one name, so we build one shuffled loader per entry in VALLIST and take an
    equal slice from each. Seeded, so every tokenizer sees the identical poses.
    """
    from dataset.dataset_poseVQ import get_dataloader
    from utils.eval_poseVQ import gt_from_batch
    from models.transformer_pose_vqvae import body_model

    torch.manual_seed(seed)
    hparams.DATA.NUM_WORKERS = min(getattr(hparams.DATA, 'NUM_WORKERS', 4), 4)
    hparams.DATA.CACHE_SMPL = False
    ds_list = hparams.DATA.VALLIST.split('_')
    per_ds = math.ceil(num_poses / len(ds_list))

    saved, out = hparams.DATA.VALLIST, []
    for ds in ds_list:
        hparams.DATA.VALLIST = ds
        got = 0
        for batch in get_dataloader(hparams, split='val', shuffle=True):
            gt_pose, _, _ = gt_from_batch(batch, body_model)     # (B, 21, 3, 3)
            out.append(gt_pose[:per_ds - got].cpu())
            got += out[-1].shape[0]
            if got >= per_ds:
                break
    hparams.DATA.VALLIST = saved
    return torch.cat(out, 0)[:num_poses]


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _encode_batched(net, poses, batch=512):
    return torch.cat([encode_full(net, poses[i:i + batch].to(DEVICE))['idx'].cpu()
                      for i in range(0, poses.shape[0], batch)], 0)


@torch.no_grad()
def _decode_batched(net, ids, batch=512):
    return torch.cat([pose6d_to_rotmat(net, decode_from_codes(net, ids[i:i + batch].to(DEVICE))).cpu()
                      for i in range(0, ids.shape[0], batch)], 0)


@torch.no_grad()
def nearest_code(cb, cosine, chunk=2048):
    """Index of each code's nearest OTHER code, in the quantizer's metric."""
    out = []
    for i in range(0, cb.shape[0], chunk):
        d = _dist(cb[i:i + chunk], cb, cosine)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float('inf')
        out.append(d.argmin(dim=1))
    return torch.cat(out)


@torch.no_grad()
def label_stability(net, poses, ids, deg, seed):
    """Fraction of token IDs unchanged under Gaussian axis-angle jitter of std `deg`.

    The perturbation is additive on the axis-angle vector (std `deg` per component),
    which is what the original ad-hoc measurement used. The realised mean geodesic
    joint rotation is returned alongside, since it is ~sqrt(3) x deg and that is the
    number a figure caption should quote.
    """
    from models.rotation_utils import axis_angle_to_matrix, matrix_to_axis_angle

    g = torch.Generator().manual_seed(seed)
    N, J = poses.shape[0], poses.shape[1]
    aa = matrix_to_axis_angle(poses.reshape(-1, 3, 3)).reshape(N, J, 3)
    aa = aa + torch.randn(aa.shape, generator=g) * math.radians(deg)
    jittered = axis_angle_to_matrix(aa.reshape(-1, 3, 3)).reshape(N, J, 3, 3)

    realised = geodesic_deg(poses.to(DEVICE), jittered.to(DEVICE)).mean().item()
    same = (_encode_batched(net, jittered) == ids).float().mean().item() * 100
    return same, realised


@torch.no_grad()
def measure(net, poses, jitters, nn_metric, seed):
    cosine = _is_cosine(net) if nn_metric == 'quantizer' else False
    cb = get_codebook(net).to(DEVICE).float()
    K = cb.shape[0]

    ids = _encode_batched(net, poses)                                  # (N, T)
    recon = geodesic_deg(poses.to(DEVICE), _decode_batched(net, ids).to(DEVICE)).mean().item()

    nn_idx = nearest_code(cb, cosine).cpu()
    swapped = geodesic_deg(_decode_batched(net, ids).to(DEVICE),
                           _decode_batched(net, nn_idx[ids]).to(DEVICE)).mean().item()

    stab = {}
    for deg in jitters:
        keep, realised = label_stability(net, poses, ids, deg, seed)
        stab[f'{deg:g}'] = {'ids_unchanged_pct': keep, 'realised_joint_rot_deg': realised}

    return {
        'quantizer': 'fsq' if getattr(net, 'quant', '') == 'fsq' else 'ema',
        'metric': 'cosine' if _is_cosine(net) else 'l2',
        'num_codes': K,
        'code_dim': int(cb.shape[1]),
        'num_tokens': int(ids.shape[1]),
        'num_poses': int(ids.shape[0]),
        'recon_deg_per_joint': recon,
        'stability': stab,
        'codes_used': int(torch.unique(ids).numel()),
        'codes_used_pct': 100.0 * torch.unique(ids).numel() / K,
        'nn_swap_deg': swapped,
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def markdown_table(rows, jitters):
    head = (['tokenizer', 'quant', 'codes', 'recon deg']
            + [f'stab@{d:g} deg' for d in jitters] + ['used %', 'NN-swap deg'])
    lines = ['| ' + ' | '.join(head) + ' |',
             '|' + '|'.join(['---'] * len(head)) + '|']
    for r in rows:
        cells = ([r['label'], r['quantizer'], str(r['num_codes']), f"{r['recon_deg_per_joint']:.2f}"]
                 + [f"{r['stability'][f'{d:g}']['ids_unchanged_pct']:.0f}%" for d in jitters]
                 + [f"{r['codes_used_pct']:.0f}%", f"{r['nn_swap_deg']:.2f}"])
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def latex_table(rows, jitters):
    """booktabs table body. Needs \\usepackage{booktabs} in the preamble."""
    cols = 'l l r r ' + 'r ' * len(jitters) + 'r r'
    head = (['Tokenizer', 'Quant.', 'Codes', 'Recon.']
            + [f'Stab.\\,$@{d:g}^\\circ$' for d in jitters] + ['Used', 'NN-swap'])
    units = ([''] * 3 + ['[$^\\circ$/joint]'] + ['[\\%]'] * len(jitters) + ['[\\%]', '[$^\\circ$]'])
    out = [r'\begin{table}[t]', r'  \centering',
           r'  \caption{Codebook quality as a cross-entropy target, measured on '
           f'{rows[0]["num_poses"]}' r' held-out validation poses. Label stability is the '
           r'fraction of token IDs left unchanged by Gaussian axis-angle jitter; NN-swap is the '
           r'pose change caused by replacing every token with its nearest codebook neighbour.}',
           r'  \label{tab:token-target-quality}',
           f'  \\begin{{tabular}}{{{cols.strip()}}}', r'    \toprule',
           '    ' + ' & '.join(head) + r' \\',
           '    ' + ' & '.join(units) + r' \\', r'    \midrule']
    for r in rows:
        cells = ([r['label'].replace('_', r'\_'), r['quantizer'].upper(), str(r['num_codes']),
                  f"{r['recon_deg_per_joint']:.2f}"]
                 + [f"{r['stability'][f'{d:g}']['ids_unchanged_pct']:.0f}" for d in jitters]
                 + [f"{r['codes_used_pct']:.0f}", f"{r['nn_swap_deg']:.2f}"])
        out.append('    ' + ' & '.join(cells) + r' \\')
    out += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    return '\n'.join(out)


def run_name(ckpt_path):
    """`output/<run>/<run>_ID00_<date>/<run>/best_net.pth` -> `<run>`."""
    parts = os.path.normpath(os.path.abspath(ckpt_path)).split(os.sep)
    return parts[parts.index('output') + 1] if 'output' in parts else parts[-2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', action='append', required=True, help='tokenizer checkpoint (repeatable)')
    ap.add_argument('--label', action='append', default=[], help='display name per --ckpt (optional)')
    ap.add_argument('--out', default='output/token_stability', help='output directory')
    ap.add_argument('--num-poses', type=int, default=2048, help='shared val poses to measure on')
    ap.add_argument('--jitter', type=float, action='append', default=[],
                    help=f'jitter std in degrees, repeatable (default {list(DEFAULT_JITTERS)})')
    ap.add_argument('--nn-metric', choices=['quantizer', 'l2'], default='quantizer',
                    help="metric for the nearest-code search (default: the quantizer's own)")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--latex', action='store_true', help='also emit a booktabs table')
    args = ap.parse_args()

    jitters = tuple(args.jitter) if args.jitter else DEFAULT_JITTERS
    labels = list(args.label) + [run_name(p) for p in args.ckpt[len(args.label):]]

    # One pose set for every run, built from the first checkpoint's val config.
    _, hp0, _ = load_net_compat(args.ckpt[0])
    poses = load_pose_set(hp0, args.num_poses, args.seed)
    print(f'Measuring on {poses.shape[0]} shared val poses '
          f'({hp0.DATA.VALLIST}), seed {args.seed}\n')

    rows = []
    for label, ckpt in zip(labels, args.ckpt):
        net, _, patched = load_net_compat(ckpt)
        if patched:
            print(f'  [{label}] checkpoint predates code_norm -> restored nn.Identity()')
        row = measure(net, poses, jitters, args.nn_metric, args.seed)
        row.update(label=label, ckpt=ckpt, code_norm_patched=patched)
        rows.append(row)
        print(f'  [{label}] recon {row["recon_deg_per_joint"]:.2f} deg | '
              f'stab@1 {row["stability"]["1"]["ids_unchanged_pct"]:.0f}% | '
              f'NN-swap {row["nn_swap_deg"]:.2f} deg')
        del net
        torch.cuda.empty_cache()

    os.makedirs(args.out, exist_ok=True)
    md = markdown_table(rows, jitters)
    print('\n' + md)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump({'num_poses': int(poses.shape[0]), 'seed': args.seed,
                   'nn_metric': args.nn_metric, 'jitters': list(jitters), 'runs': rows}, f, indent=2)
    with open(os.path.join(args.out, 'table.md'), 'w') as f:
        f.write(md + '\n')
    if args.latex:
        tex = latex_table(rows, jitters)
        print('\n' + tex)
        with open(os.path.join(args.out, 'table.tex'), 'w') as f:
            f.write(tex + '\n')
    print(f'\nWrote {args.out}/summary.json, table.md' + (', table.tex' if args.latex else ''))


if __name__ == '__main__':
    main()
