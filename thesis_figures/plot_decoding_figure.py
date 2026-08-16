"""Figure 3.10: the two decoding rules (soft vs hard), on real model outputs.

Two rows:
  Row A -- the mechanism at ONE token position: the predicted distribution p_i over the
           codebook, with the argmax highlighted, plus the two operations it feeds
           (arg max -> a real code; the probability-weighted sum -> a latent that is NOT
           a code).
  Row B -- the consequence for the whole body: the same logit matrix Q decoded both ways
           and rendered, coloured by per-vertex error against the ground-truth fit.

Row B decodes the FULL Q both ways on purpose. Swapping a single position moves only a
few joints (see the token-influence analysis), so a per-position mesh triptych would show
three near-identical bodies -- that is the ambiguity story, not the decoding story.

Runs on CPU by default: the GPU is usually busy with training, and this only needs a
handful of forward passes. Pass --device cuda to override.

Usage (from the tokenhmr/ directory):
  PYOPENGL_PLATFORM=egl ~/miniconda3/envs/thesis-HMR/bin/python \
      thesis_figures/plot_decoding_figure.py --checkpoint <slim.ckpt> --scan 40
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TOKENHMR = HERE.parent
sys.path.insert(0, str(TOKENHMR))
sys.path.insert(0, str(TOKENHMR / 'tokenhmr'))

OUT_PDF = TOKENHMR.parent / 'thesis' / 'images' / 'downstream' / 'decoding.pdf'


# --------------------------------------------------------------------------------------
# mesh rendering (same offscreen pyrender path as visualize_token_effects.py)
# --------------------------------------------------------------------------------------
_RENDERERS = {}


def _render_mesh(vertices, faces, vertex_colors, res=520, azim_deg=0.0):
    import pyrender
    import trimesh

    if res not in _RENDERERS:
        _RENDERERS[res] = pyrender.OffscreenRenderer(viewport_width=res, viewport_height=res,
                                                     point_size=1.0)
    renderer = _RENDERERS[res]

    v = vertices.copy()
    v -= 0.5 * (v.min(0) + v.max(0))
    rot = trimesh.transformations.rotation_matrix(np.radians(azim_deg), [0, 1, 0])
    mesh = trimesh.Trimesh(vertices=v, faces=faces, vertex_colors=vertex_colors, process=False)
    mesh.apply_transform(rot)
    # pyrender's camera looks down -z with +y up; SMPL is +y down, so flip to stand it up.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    extent = float(np.abs(mesh.vertices).max())
    yfov = np.pi / 3.0
    dist = extent / np.tan(yfov / 2.0) * 1.5
    cam_pose = np.eye(4)
    cam_pose[2, 3] = dist
    scene.add(pyrender.PerspectiveCamera(yfov=yfov), pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    lp = np.eye(4)
    lp[:3, 3] = [0.5, 1.0, 2.0]
    scene.add(light, pose=lp)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.copy()
    # crop to the mesh's alpha bounding box (with a small margin) so the body fills the panel
    ys, xs = np.where(color[..., 3] > 0)
    if len(ys):
        m = 6
        y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, color.shape[0])
        x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, color.shape[1])
        color = color[y0:y1, x0:x1]
    return color


def _error_colors(err_mm, vmax):
    """Per-vertex error (mm) -> RGBA, grey where small, hot where large."""
    import matplotlib.cm as cm
    t = np.clip(err_mm / max(vmax, 1e-6), 0.0, 1.0)
    rgba = cm.get_cmap('inferno_r')(0.15 + 0.85 * t)
    rgba[..., 3] = 1.0
    return (rgba * 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
CACHE = Path(__file__).resolve().parent / 'cache' / 'decfig_cache.npz'
CACHE.parent.mkdir(parents=True, exist_ok=True)


def scan(args):
    """Run the model over `args.scan` frames and cache the winning frame's arrays."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from lib.models import load_tokenhmr
    from lib.configs import dataset_eval_config
    from lib.datasets import create_dataset
    from lib.utils import recursive_to
    from lib.models.smpl_wrapper import SMPL
    from lib.utils.geometry import aa_to_rotmat

    device = torch.device(args.device)
    print(f'[load] {args.checkpoint}', flush=True)
    model, cfg = load_tokenhmr(checkpoint_path=args.checkpoint, model_cfg=args.model_config,
                               dataset_dir=args.dataset_dir)
    model = model.to(device).eval()
    head = model.smpl_head.decpose

    smpl_cfg = {k.lower(): v for k, v in dict(cfg.SMPL).items()}
    smpl = SMPL(**smpl_cfg).to(device)
    faces = smpl.faces

    ds_cfg = dataset_eval_config()[args.dataset]
    for key in ('DATASET_FILE', 'IMG_DIR'):
        if key in ds_cfg:
            ds_cfg[key] = os.path.join(args.dataset_dir, ds_cfg[key])
    dataset = create_dataset(cfg, ds_cfg, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    print(f'[scan] {args.scan} frames of {args.dataset} for a legible failure', flush=True)
    best = None
    shortlist = []
    for i, batch in enumerate(loader):
        if i >= args.scan:
            break
        batch = recursive_to(batch, device)
        if not bool(batch['has_smpl_params']['body_pose'].reshape(-1)[0] > 0.5):
            continue
        with torch.no_grad():
            head.decode_mode = 'soft'
            out_s = model(batch)
            head.decode_mode = 'hard'
            out_h = model(batch)

            # GT params may be stored as axis-angle (3DPW/EMDB) or rotation matrices.
            gt_p = {k: v.float() for k, v in batch['smpl_params'].items()}
            is_aa = batch['smpl_params_is_axis_angle']
            for k in ('global_orient', 'body_pose'):
                if bool(np.asarray(is_aa[k].cpu()).reshape(-1)[0]):
                    gt_p[k] = aa_to_rotmat(gt_p[k].reshape(-1, 3)).reshape(1, -1, 3, 3)
                else:
                    gt_p[k] = gt_p[k].reshape(1, -1, 3, 3)
            gt_p['betas'] = gt_p['betas'].reshape(1, -1)
            gt = smpl(**gt_p, pose2rot=False)
            v_gt = gt.vertices[0]
            v_s, v_h = out_s['pred_vertices'][0], out_h['pred_vertices'][0]
            # pelvis-align all three before comparing (translation is the camera's job)
            v_gt = v_gt - v_gt.mean(0, keepdim=True)
            v_s = v_s - v_s.mean(0, keepdim=True)
            v_h = v_h - v_h.mean(0, keepdim=True)
            e_s = (v_s - v_gt).norm(dim=-1) * 1000.0
            e_h = (v_h - v_gt).norm(dim=-1) * 1000.0
            gap = float(e_h.mean() - e_s.mean())
            # Prefer an ARTICULATED ground-truth pose: a standing body makes a dull figure
            # and hides what the decode actually changes. Articulation = mean geodesic angle
            # of the GT body joints away from the rest pose.
            Rgt = gt_p['body_pose'][0]
            tr = Rgt[:, 0, 0] + Rgt[:, 1, 1] + Rgt[:, 2, 2]
            artic = float(torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1))).mean())
            score = gap * min(artic / args.min_artic, 1.0)

        cand = dict(gap=gap, score=score, artic=artic, idx=i,
                    v_gt=v_gt.cpu().numpy(), v_s=v_s.cpu().numpy(), v_h=v_h.cpu().numpy(),
                    e_s=e_s.cpu().numpy(), e_h=e_h.cpu().numpy(),
                    pve_s=float(e_s.mean()), pve_h=float(e_h.mean()),
                    logits=out_s['cls_logits'][-1].float().cpu().numpy())
        shortlist.append(cand)
        shortlist.sort(key=lambda c: -c['score'])
        del shortlist[args.keep:]
        if best is None or score > best['score']:
            best = cand
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{args.scan}  best: gap {best["gap"]:.0f} mm, '
                  f'articulation {best["artic"]:.0f} deg (frame {best["idx"]})', flush=True)

    if best is None:
        raise SystemExit('no frame with GT SMPL params found in the scanned range')
    print(f'[pick] frame {best["idx"]}  soft {best["pve_s"]:.1f} mm  hard {best["pve_h"]:.1f} mm  '
          f'articulation {best["artic"]:.0f} deg', flush=True)

    np.savez(CACHE, v_gt=best['v_gt'], v_s=best['v_s'], v_h=best['v_h'],
             e_s=best['e_s'], e_h=best['e_h'], logits=best['logits'],
             pve_s=best['pve_s'], pve_h=best['pve_h'], idx=best['idx'], faces=faces)
    print(f'[cache] {CACHE}', flush=True)

    # Keep the top `--keep` frames as well, so the figure's frame can be chosen by eye
    # rather than by the score alone (thesis_figures/render_decode_candidates.py).
    multi = {'n': len(shortlist), 'faces': faces}
    for j, c in enumerate(shortlist):
        for k in ('v_gt', 'v_s', 'v_h', 'e_s', 'e_h', 'logits'):
            multi[f'{k}_{j}'] = c[k]
        multi[f'meta_{j}'] = np.array([c['idx'], c['pve_s'], c['pve_h'], c['artic'], c['gap']])
    np.savez(str(CACHE).replace('.npz', '_candidates.npz'), **multi)
    print(f"[cache] {str(CACHE).replace('.npz', '_candidates.npz')}  ({len(shortlist)} candidates)", flush=True)
    for j, c in enumerate(shortlist):
        print(f"  cand {j}: frame {c['idx']:5d}  soft {c['pve_s']:6.1f} mm  hard {c['pve_h']:6.1f} mm  "
              f"artic {c['artic']:4.0f} deg  gap {c['gap']:6.1f}", flush=True)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    d = np.load(CACHE)
    best = {k: d[k] for k in d.files}
    faces = best['faces']
    # ---------------- Row A data: one token position ----------------
    Q = best['logits']                       # (M, K)
    p = np.exp(Q - Q.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    # a TYPICAL position (median entropy). The max-entropy position would be the most
    # extreme case and would overstate how flat the distribution usually is.
    ent = -(p * np.log(p + 1e-12)).sum(-1)
    pos = int(np.argsort(ent)[len(ent) // 2])
    pi = p[pos]
    top = np.argsort(-pi)[:args.topk]

    # ---------------- render ----------------
    def _pad_common(imgs):
        h = max(i.shape[0] for i in imgs)
        w = max(i.shape[1] for i in imgs)
        out = []
        for im in imgs:
            c = np.zeros((h, w, 4), im.dtype)
            y0 = h - im.shape[0]                 # bottom-align: feet on one line
            x0 = (w - im.shape[1]) // 2
            c[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            out.append(c)
        return out

    vmax = float(np.percentile(best['e_h'], 95))
    grey = np.tile(np.array([200, 200, 205, 255], np.uint8), (best['v_gt'].shape[0], 1))
    img_gt = _render_mesh(best['v_gt'], faces, grey, azim_deg=args.azim)
    img_s = _render_mesh(best['v_s'], faces, _error_colors(best['e_s'], vmax), azim_deg=args.azim)
    img_h = _render_mesh(best['v_h'], faces, _error_colors(best['e_h'], vmax), azim_deg=args.azim)
    img_gt, img_s, img_h = _pad_common([img_gt, img_s, img_h])

    # ---------------- figure: one clean strip, distribution then three bodies -------
    fig = plt.figure(figsize=(7.1, 2.55))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.45, 1.0, 1.0, 1.0], wspace=0.06,
                           left=0.085, right=0.995, top=0.80, bottom=0.20)

    K = Q.shape[-1]
    axb = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(top))
    cols = ['#d95f02' if j == 0 else '#a6cee3' for j in range(len(top))]
    axb.bar(xs, pi[top] * 1e3, color=cols, width=0.75)
    axb.axhline(1e3 / K, color='0.4', lw=0.8, ls=(0, (3, 2)))
    axb.text(len(top) - 0.5, 1e3 / K, ' uniform', va='bottom', ha='right',
             fontsize=5.8, color='0.4')
    axb.set_xticks([])
    axb.set_xlabel(f'top {args.topk} of $K={K}$ codes', fontsize=6.8, labelpad=2)
    axb.set_ylabel(r'$p_{ik}\ [\times 10^{-3}]$', fontsize=7)
    axb.tick_params(labelsize=6, length=2)
    axb.set_title('predicted distribution\nat one token position', fontsize=7.5, pad=3)
    for side in ('top', 'right'):
        axb.spines[side].set_visible(False)

    for col, (im, title) in enumerate([
            (img_gt, 'ground truth\n(reference)'),
            (img_s, f'soft decode\n{best["pve_s"]:.0f} mm'),
            (img_h, f'hard decode\n{best["pve_h"]:.0f} mm')]):
        ax = fig.add_subplot(gs[0, col + 1])
        ax.imshow(im)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap='inferno_r', norm=plt.Normalize(0, vmax))
    cax = fig.add_axes([0.545, 0.10, 0.26, 0.030])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('per-vertex error [mm]', fontsize=6, labelpad=1)
    cb.ax.tick_params(labelsize=5.5, length=2)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    png = HERE / 'decoding_preview.png'
    fig.savefig(png, dpi=170, bbox_inches='tight')
    print(f'[write] {OUT_PDF}\n[write] {png}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--model_config',
                    default=str(TOKENHMR / 'logs/tokenhmr_fsq/runs/tokenhmr_fsq_0/model_config.yaml'))
    ap.add_argument('--dataset_dir', default=str(TOKENHMR / 'dataset_dir/evaluation_data'))
    ap.add_argument('--dataset', default='3DPW-TEST')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--scan', type=int, default=40)
    ap.add_argument('--topk', type=int, default=12)
    ap.add_argument('--keep', type=int, default=8,
                    help='how many top-scoring frames to cache for visual selection')
    ap.add_argument('--azim', type=float, default=0.0)
    ap.add_argument('--min_artic', type=float, default=28.0,
                    help='GT articulation (mean deg from rest pose) treated as "fully active"')
    ap.add_argument('--from_cache', action='store_true',
                    help='skip the model and re-plot from the cached scan')
    a = ap.parse_args()
    if not a.from_cache:
        scan(a)
    plot(a)
