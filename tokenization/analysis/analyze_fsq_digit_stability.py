"""Is an FSQ token index a *flat* label, or a factored one with a stability gradient?

Context: `analyze_token_stability.py` measures label stability S(delta) on the packed
FSQ index, the 1920-way (d=4) or 15360-way (d=5) label the downstream cross-entropy
actually predicts. That index is not atomic. FSQ quantizes each channel on its own
integer grid and packs the per-channel levels into one number,

    idx = sum_j  v_j * basis_j,      v_j in [0, L_j),   basis = cumprod(L)[:-1]

so the "flat" label is already a mixed-radix digit string, and each channel's level
v_j is itself an ordered scalar with a radix-2 (bit-plane) decomposition

    v_j = sum_b 2^b * beta_b .

This script asks whether those components behave differently, along two axes:

  stability   how often the component survives a jitter of the input pose. A
              high-order bit flips only when the latent crosses a boundary spaced
              2^b grid steps apart, so it should be far more stable than the LSB.

  damage      how far the decoded pose moves when the component is corrupted, in
              mean per-joint geodesic degrees, the same unit and the same
              all-positions convention as the NN-swap redundancy Delta_NN.

The claim under test is that the two are *inversely* ordered: the stable, learnable
components of the index are the ones that carry the pose, and the unlearnable ones
are the ones that do not matter. If so, a flat softmax over the packed index mixes a
learnable high-damage target with an unlearnable low-damage target and weights them
equally, which is the per-index form of the argument the thesis makes across
tokenizers, and a factored (per-digit or per-bit-plane) head is the fix. Nothing here
requires retraining a tokenizer: the decomposition is a re-reading of indices the
frozen encoder already produces.

Also reported, since it bounds what a factored head can do:

  card / prior / H   number of classes, the bias-only top-1 floor (largest marginal)
                     and the marginal entropy in nats, per component.

  independence gap   product over channels of the per-channel digit stability against
                     the measured joint-index stability. The product is what the joint
                     would be if channel errors were independent; a measured value
                     above it means channels tend to fail together.

Usage (from tokenization/, in the thesis-HMR env):

    python analyze_fsq_digit_stability.py \
        --ckpt output/tokenization_transformer_fsq/.../best_net.pth --label "FSQ d4" \
        --ckpt output/tokenization_transformer_fsq_16k/.../best_net.pth --label "FSQ d5" \
        --latex

FSQ checkpoints only (the decomposition is a property of the scalar grid; a learned
codebook has no digit structure). Results land in --out, default
output/fsq_digit_stability/. The pose set, seed and jitter construction are shared
with `analyze_token_stability.py`, so the "joint index" row of this table reproduces
that script's stab@X column for the same checkpoint.
"""
import argparse
import json
import math
import os
import sys
import torch

from tokenization.analysis.analyze_latent_pose_info import (DEVICE, geodesic_deg, decode_from_latent,
                                      pose6d_to_rotmat)
from tokenization.analysis.analyze_token_stability import (DEFAULT_JITTERS, load_net_compat, load_pose_set,
                                     _encode_batched, _decode_batched, run_name)

# Perturbations whose damage we measure, per channel. "digit" corrupts the whole
# per-channel level, "bit b" flips one bit-plane of it.
FULL_DIGIT = -1


# --------------------------------------------------------------------------- #
# Index <-> digits                                                             #
# --------------------------------------------------------------------------- #
def fsq_levels(net):
    """(levels, basis) of an FSQ tokenizer, or raise if it is not one."""
    q = getattr(net, 'quantizer', None)
    if not hasattr(q, '_levels') or not hasattr(q, '_basis'):
        raise SystemExit('not an FSQ tokenizer: no scalar grid to decompose. '
                         'Use analyze_token_stability.py for learned codebooks.')
    return q._levels.cpu().long(), q._basis.cpu().long()


def to_digits(ids, levels, basis):
    """(N, T) packed indices -> (N, T, d) per-channel levels."""
    return torch.remainder(torch.div(ids.unsqueeze(-1), basis, rounding_mode='floor'), levels)


def to_index(digits, basis):
    """(N, T, d) per-channel levels -> (N, T) packed indices."""
    return (digits * basis).sum(-1)


def n_bits(level):
    """Bit-planes needed to write levels 0..L-1."""
    return max(1, int(math.ceil(math.log2(int(level)))))


# --------------------------------------------------------------------------- #
# Jitter (shared construction with analyze_token_stability.label_stability)     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def jittered_ids(net, poses, deg, seed):
    """Token indices of the pose set after Gaussian axis-angle jitter of std `deg`.

    Mirrors `analyze_token_stability.label_stability` exactly (same generator, same
    additive axis-angle noise), so the joint-index stability computed here is the
    same measurement that produced Table 4.3.
    """
    from tokenization.models.rotation_utils import axis_angle_to_matrix, matrix_to_axis_angle

    g = torch.Generator().manual_seed(seed)
    N, J = poses.shape[0], poses.shape[1]
    aa = matrix_to_axis_angle(poses.reshape(-1, 3, 3)).reshape(N, J, 3)
    aa = aa + torch.randn(aa.shape, generator=g) * math.radians(deg)
    jit = axis_angle_to_matrix(aa.reshape(-1, 3, 3)).reshape(N, J, 3, 3)

    realised = geodesic_deg(poses.to(DEVICE), jit.to(DEVICE)).mean().item()
    return _encode_batched(net, jit), realised


# --------------------------------------------------------------------------- #
# Component definitions                                                        #
# --------------------------------------------------------------------------- #
def component_value(digits, levels, chan, bit):
    """The label a downstream head would predict for one component.

    bit == FULL_DIGIT -> the whole per-channel level v_j (L_j classes)
    bit >= 0          -> bit-plane b of v_j, i.e. (v_j >> b) & 1 (2 classes)
    """
    v = digits[..., chan]
    return v if bit == FULL_DIGIT else torch.bitwise_and(torch.bitwise_right_shift(v, bit), 1)


def corrupt(digits, levels, chan, bit, seed):
    """Corrupt one component at every token position, leaving the rest intact.

    Returns (corrupted digits, fraction of positions actually changed). For a whole
    digit we resample uniformly among the other L_j - 1 levels. For a bit-plane we
    flip that bit, which is skipped where the flip would leave the valid range (the
    L = 6 and L = 5 channels are not powers of two, so their top bit-plane only
    exists for part of the range).
    """
    g = torch.Generator().manual_seed(seed)
    out = digits.clone()
    v = digits[..., chan]
    L = int(levels[chan])

    if bit == FULL_DIGIT:
        # uniform over the other levels: draw an offset in [1, L-1] and wrap
        off = torch.randint(1, L, v.shape, generator=g)
        new = torch.remainder(v + off, L)
        valid = torch.ones_like(v, dtype=torch.bool)
    else:
        new = torch.bitwise_xor(v, 1 << bit)
        valid = new < L
        new = torch.where(valid, new, v)

    out[..., chan] = new
    return out, valid.float().mean().item()


# --------------------------------------------------------------------------- #
# Is the channel axis a coarse-to-fine hierarchy, as in RVQ?                    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _decode_latent_batched(net, latent, batch=512):
    return torch.cat([pose6d_to_rotmat(net, decode_from_latent(net, latent[i:i + batch].to(DEVICE))).cpu()
                      for i in range(0, latent.shape[0], batch)], 0)


@torch.no_grad()
def channel_probe(net, ids, digits, base_pose, levels, basis, eps):
    """Per-channel scale, to test for an RVQ-like coarse-to-fine ordering.

    A residual quantizer is hierarchical by construction: layer 1 holds the coarse
    approximation and later layers hold ever-smaller corrections. FSQ has no such
    construction, its d channels are parallel dimensions of one latent, so any
    apparent ordering could just be an artefact of the level counts (corrupting an
    8-level digit moves further than a 5-level one) rather than a real hierarchy.

    Three measurements separate those:

      digit damage    corrupt the whole level, uniformly among the other L_j - 1.
                      Confounded by L_j, and this is what the main table reports.

      step damage     move the level by exactly one grid step. Removes the "more
                      levels means bigger average jump" confound, but a single step
                      is a larger move in the normalised code for a coarse channel,
                      since the grid spans [-1, 1] whatever L_j is.

      eps damage      add the same eps to the normalised code value of channel j and
                      decode without re-quantizing. Free of both confounds, so this
                      is the decoder's true anisotropy: how much pose sits along
                      channel j per unit of latent. A genuine hierarchy shows a
                      monotone decreasing profile here; parallel channels show a flat
                      one.

    Also returns the encoder's occupancy of each channel (std of the quantized code
    value), since a channel the encoder barely moves along cannot carry much either.
    """
    d = int(levels.numel())
    cb = net.quantizer.implicit_codebook.cpu()
    vecs = cb[ids]                                             # (N, T, d) in [-1, 1]

    out = []
    for c in range(d):
        L = int(levels[c])

        v = digits[..., c]
        stepped = digits.clone()
        stepped[..., c] = torch.where(v + 1 < L, v + 1, v - 1)
        step_dmg = geodesic_deg(
            base_pose, _decode_batched(net, to_index(stepped, basis)).to(DEVICE)).mean().item()

        pert = vecs.clone()
        pert[..., c] = (pert[..., c] + eps).clamp(-1.0, 1.0)
        eps_dmg = geodesic_deg(
            base_pose, _decode_latent_batched(net, pert).to(DEVICE)).mean().item()

        out.append({
            'chan': c,
            'levels': L,
            'grid_step_normalised': 2.0 / max(1, L // 2),
            'step_damage_deg': step_dmg,
            'eps_damage_deg': eps_dmg,
            'code_std': float(vecs[..., c].std()),
            'code_abs_mean': float(vecs[..., c].abs().mean()),
        })
    return {'eps': eps, 'channels': out}


# --------------------------------------------------------------------------- #
# Measurement                                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure(net, poses, jitters, seed, eps=0.1):
    levels, basis = fsq_levels(net)
    d = int(levels.numel())

    ids = _encode_batched(net, poses)                    # (N, T)
    digits = to_digits(ids, levels, basis)               # (N, T, d)
    base_pose = _decode_batched(net, ids).to(DEVICE)     # decoded reference

    # jittered encodings, one per jitter level, shared by every component
    jit = {}
    for deg in jitters:
        jids, realised = jittered_ids(net, poses, deg, seed)
        jit[f'{deg:g}'] = {'digits': to_digits(jids, levels, basis),
                           'ids': jids, 'realised_joint_rot_deg': realised}

    def stats(val, val_jit_by_deg, card):
        counts = torch.bincount(val.reshape(-1), minlength=card).float()
        p = counts / counts.sum().clamp(min=1.0)
        nz = p[p > 0]
        return {
            'cardinality': int(card),
            'classes_used': int((counts > 0).sum()),
            'prior_top1_pct': 100.0 * p.max().item(),
            'marginal_entropy_nats': float(-(nz * nz.log()).sum()),
            'stability': {deg: 100.0 * (val == vj).float().mean().item()
                          for deg, vj in val_jit_by_deg.items()},
        }

    components = []

    # --- the flat label the downstream CE currently predicts -------------------
    K = int(torch.prod(levels))
    row = stats(ids, {deg: j['ids'] for deg, j in jit.items()}, K)
    row.update(name='joint index', chan=None, bit=None, damage_deg=None, valid_pct=100.0)
    components.append(row)

    # --- whole per-channel digits, then their bit-planes ----------------------
    for c in range(d):
        L = int(levels[c])
        for bit in [FULL_DIGIT] + list(range(n_bits(L))):
            val = component_value(digits, levels, c, bit)
            vjit = {deg: component_value(j['digits'], levels, c, bit)
                    for deg, j in jit.items()}
            card = L if bit == FULL_DIGIT else 2
            row = stats(val, vjit, card)

            cd, valid = corrupt(digits, levels, c, bit, seed + 1000 * c + bit)
            dmg = geodesic_deg(base_pose, _decode_batched(net, to_index(cd, basis)).to(DEVICE))
            row.update(name=f'ch{c} digit' if bit == FULL_DIGIT else f'ch{c} bit{bit}',
                       chan=c, bit=(None if bit == FULL_DIGIT else bit),
                       levels=L, damage_deg=dmg.mean().item(), valid_pct=100.0 * valid)
            components.append(row)

    # --- independence check ---------------------------------------------------
    indep = {}
    for deg in jit:
        prod = 1.0
        for r in components:
            if r['chan'] is not None and r['bit'] is None:
                prod *= r['stability'][deg] / 100.0
        indep[deg] = {'product_of_channel_digits_pct': 100.0 * prod,
                      'measured_joint_pct': components[0]['stability'][deg]}

    return {
        'levels': [int(x) for x in levels],
        'channel_probe': channel_probe(net, ids, digits, base_pose, levels, basis, eps),
        'num_codes': K,
        'code_dim': d,
        'num_tokens': int(ids.shape[1]),
        'num_poses': int(ids.shape[0]),
        'realised_joint_rot_deg': {deg: j['realised_joint_rot_deg'] for deg, j in jit.items()},
        'components': components,
        'independence': indep,
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def markdown_table(res, jitters):
    head = (['component', 'classes', 'prior top-1', 'H [nats]']
            + [f'stab@{d:g} deg' for d in jitters] + ['damage deg'])
    lines = ['| ' + ' | '.join(head) + ' |',
             '|' + '|'.join(['---'] * len(head)) + '|']
    for r in res['components']:
        dmg = '--' if r['damage_deg'] is None else f"{r['damage_deg']:.2f}"
        cells = ([r['name'], str(r['cardinality']), f"{r['prior_top1_pct']:.1f}%",
                  f"{r['marginal_entropy_nats']:.2f}"]
                 + [f"{r['stability'][f'{d:g}']:.0f}%" for d in jitters] + [dmg])
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def latex_table(rows_by_label, jitters):
    """booktabs table, one block per tokenizer. Needs \\usepackage{booktabs}."""
    cols = 'l r r ' + 'r ' * len(jitters) + 'r'
    head = (['Component', 'Classes', 'Prior']
            + [f'$S({d:g}^\\circ)$' for d in jitters] + ['Damage'])
    units = ['', '', '[\\%]'] + ['[\\%]'] * len(jitters) + ['[$^\\circ$]']
    out = [r'\begin{table}[htbp]', r'  \centering', r'  \small',
           r'  \caption{Components of the FSQ token index, on the held-out validation '
           r'poses of Table~\ref{tab:token-target-quality}. The packed index is the label '
           r'the downstream cross-entropy predicts; the rows below decompose it into '
           r'per-channel levels and their bit-planes. Damage is the mean per-joint '
           r'geodesic change when that component alone is corrupted at every token '
           r'position, the same convention as $\Delta_{\mathrm{NN}}$. Prior is the '
           r'bias-only top-1 floor.}',
           r'  \label{tab:fsq-digit-stability}',
           r'  \setlength{\tabcolsep}{4pt}',
           f'  \\begin{{tabular}}{{{cols.strip()}}}', r'    \toprule',
           '    ' + ' & '.join(head) + r' \\',
           '    ' + ' & '.join(units) + r' \\']
    for label, res in rows_by_label:
        out += [r'    \midrule', r'    \multicolumn{%d}{l}{\emph{%s}} \\'
                % (3 + len(jitters) + 1, label.replace('_', r'\_'))]
        for r in res['components']:
            dmg = '--' if r['damage_deg'] is None else f"{r['damage_deg']:.2f}"
            cells = ([r['name'], str(r['cardinality']), f"{r['prior_top1_pct']:.1f}"]
                     + [f"{r['stability'][f'{d:g}']:.0f}" for d in jitters] + [dmg])
            out.append('    ' + ' & '.join(cells) + r' \\')
    out += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    return '\n'.join(out)


def plot(rows_by_label, jitter, path):
    """Stability against pose damage, one point per index component."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable, skipping figure')
        return None

    fig, axes = plt.subplots(1, len(rows_by_label), figsize=(5.2 * len(rows_by_label), 4.4),
                             squeeze=False)
    key = f'{jitter:g}'
    for ax, (label, res) in zip(axes[0], rows_by_label):
        for r in res['components']:
            if r['damage_deg'] is None:
                continue
            whole = r['bit'] is None
            ax.scatter(r['stability'][key], r['damage_deg'],
                       s=64 if whole else 40,
                       marker='s' if whole else 'o',
                       c='tab:orange' if whole else 'tab:blue',
                       edgecolor='k', linewidth=0.5, zorder=3)
            ax.annotate(r['name'].replace('ch', '').replace(' digit', 'D').replace(' bit', 'b'),
                        (r['stability'][key], r['damage_deg']),
                        textcoords='offset points', xytext=(4, 3), fontsize=6.5)
        j = res['components'][0]['stability'][key]
        ax.axvline(j, ls='--', c='k', lw=0.8, zorder=1)
        ax.annotate(f'packed index\n{j:.0f}%', (j, ax.get_ylim()[1]), fontsize=7,
                    ha='left', va='top', textcoords='offset points', xytext=(4, -4))
        ax.set_xlabel(f'label stability $S({jitter:g}^\\circ)$ [%]')
        ax.set_ylabel('pose damage when corrupted [$^\\circ$/joint]')
        ax.set_title(label)
        ax.grid(alpha=0.25, zorder=0)
    from matplotlib.lines import Line2D
    axes[0][0].legend(handles=[
        Line2D([], [], marker='s', ls='', c='tab:orange', mec='k', label='whole channel digit'),
        Line2D([], [], marker='o', ls='', c='tab:blue', mec='k', label='bit-plane')],
        fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', action='append', required=True, help='FSQ checkpoint (repeatable)')
    ap.add_argument('--label', action='append', default=[], help='display name per --ckpt')
    ap.add_argument('--out', default='output/fsq_digit_stability', help='output directory')
    ap.add_argument('--num-poses', type=int, default=2048, help='shared val poses')
    ap.add_argument('--jitter', type=float, action='append', default=[],
                    help=f'jitter std in degrees, repeatable (default {list(DEFAULT_JITTERS)})')
    ap.add_argument('--plot-jitter', type=float, default=1.0, help='jitter level used in the figure')
    ap.add_argument('--eps', type=float, default=0.1,
                    help='normalised-code perturbation for the channel anisotropy probe')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--latex', action='store_true')
    args = ap.parse_args()

    jitters = tuple(args.jitter) if args.jitter else DEFAULT_JITTERS
    labels = list(args.label) + [run_name(p) for p in args.ckpt[len(args.label):]]

    _, hp0, _ = load_net_compat(args.ckpt[0])
    poses = load_pose_set(hp0, args.num_poses, args.seed)
    print(f'Measuring on {poses.shape[0]} shared val poses '
          f'({hp0.DATA.VALLIST}), seed {args.seed}\n')

    results, out_rows = {}, []
    for label, ckpt in zip(labels, args.ckpt):
        net, _, patched = load_net_compat(ckpt)
        if patched:
            print(f'  [{label}] checkpoint predates code_norm -> restored nn.Identity()')
        res = measure(net, poses, jitters, args.seed, args.eps)
        res.update(label=label, ckpt=ckpt, code_norm_patched=patched)
        results[label] = res
        out_rows.append((label, res))
        print(f'\n=== {label}  (levels {res["levels"]}, K={res["num_codes"]}) ===')
        print(markdown_table(res, jitters))
        cp = res['channel_probe']
        print('\nchannel probe (RVQ-style coarse-to-fine ordering?)  eps=%g' % cp['eps'])
        print('| chan | levels | grid step | step damage | eps damage | code std |')
        print('|---|---|---|---|---|---|')
        for r in cp['channels']:
            print('| ch%d | %d | %.3f | %.2f | %.2f | %.3f |' % (
                r['chan'], r['levels'], r['grid_step_normalised'],
                r['step_damage_deg'], r['eps_damage_deg'], r['code_std']))

        ind = res['independence'][f'{jitters[0]:g}']
        print(f'\nindependence @{jitters[0]:g} deg: product of channel digits '
              f'{ind["product_of_channel_digits_pct"]:.1f}% vs measured joint '
              f'{ind["measured_joint_pct"]:.1f}%')
        del net
        torch.cuda.empty_cache()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump({'num_poses': int(poses.shape[0]), 'seed': args.seed,
                   'jitters': list(jitters), 'runs': results}, f, indent=2)
    with open(os.path.join(args.out, 'table.md'), 'w') as f:
        for label, res in out_rows:
            f.write(f'### {label} (levels {res["levels"]}, K={res["num_codes"]})\n\n')
            f.write(markdown_table(res, jitters) + '\n\n')
    if args.latex:
        tex = latex_table(out_rows, jitters)
        print('\n' + tex)
        with open(os.path.join(args.out, 'table.tex'), 'w') as f:
            f.write(tex + '\n')
    fig = plot(out_rows, args.plot_jitter, os.path.join(args.out, 'digit_stability_damage.pdf'))
    print(f'\nWrote {args.out}/summary.json, table.md'
          + (', table.tex' if args.latex else '') + (', digit_stability_damage.pdf' if fig else ''))


if __name__ == '__main__':
    main()
