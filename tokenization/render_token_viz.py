"""Figure 4.7 (main) and Figure B.2 (appendix): token-displacement mesh maps.

On one shared body pose, for each tokenizer we swap the single most-impactful token
for its nearest codebook neighbour (the per-token counterpart of the redundancy
Delta_NN) and colour each mesh vertex by how far it moved. On a well-separated
codebook (FSQ) this deforms a limb; on a near-duplicate one (cosine) it barely moves.

  - Fig 4.7: FSQ (d=4) vs cosine (d256), same shared vmax -> the contrast.
  - Fig B.2: all eight tokenizers on the same pose and vmax.

Decode/SMPL on CPU (GPU busy); pyrender does the small offscreen render on the GPU.

Run from tokenization/ in the thesis-HMR env:
    ~/miniconda3/envs/thesis-HMR/bin/python render_token_viz.py

Outputs: thesis/images/latent/token_visualization_contrast.pdf
         thesis/images/appendix/tokenviz_all_tokenizers.pdf
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import json
import math

import numpy as np
import torch
import trimesh
import pyrender
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import analyze_latent_pose_info as ali
ali.DEVICE = torch.device("cpu")
import models.transformer_pose_vqvae as _tpv
_tpv.body_model.to("cpu")                                   # in-place: shared object stays cpu
import models.vanilla_pose_vqvae as _vpv
_vpv.body_model.to("cpu")
from analyze_latent_pose_info import (load_net, get_codebook, _is_cosine, encode_full,
                                      decode_from_codes, pose6d_to_rotmat)
from utils.rotation_conversions import axis_angle_to_matrix

DEV = torch.device("cpu")
BODY = _tpv.body_model
FACES = BODY.faces
TURBO = plt.get_cmap("turbo")
HERE = os.path.dirname(os.path.abspath(__file__))
STAB = os.path.join(HERE, "output", "token_stability", "summary.json")
REPO = os.path.dirname(os.path.dirname(HERE))
LAT_OUT = os.path.join(REPO, "thesis", "images", "latent")
APP_OUT = os.path.join(REPO, "thesis", "images", "appendix")
os.makedirs(LAT_OUT, exist_ok=True); os.makedirs(APP_OUT, exist_ok=True)
RES = 340

SHORT = {"CNN": "Conv ($\\ell_2$)", "Transformer tier1": "Transformer ($\\ell_2$)",
         "Transformer cosine": "Cosine $d$256", "Skeleton-masked": "Skeleton-masked",
         "VQ d4": "Cosine $d$4", "VQ d2": "Cosine $d$2", "FSQ d4": "FSQ $d$4", "FSQ d5": "FSQ $d$5"}
ORDER = ["CNN", "Transformer tier1", "Transformer cosine", "Skeleton-masked",
         "VQ d4", "VQ d2", "FSQ d4", "FSQ d5"]


# ----------------------------- rendering (from visualize_token_effects) -----------------------------
_R = {}
def _renderer(res):
    if res not in _R:
        _R[res] = pyrender.OffscreenRenderer(viewport_width=res, viewport_height=res, point_size=1.0)
    return _R[res]


def render_error(vertices, err_mm, vmax, res=RES):
    err = np.clip(err_mm / max(vmax, 1e-6), 0.0, 1.0)
    vcol = (TURBO(err) * 255).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices.copy(), FACES.copy(), vertex_colors=vcol, process=False)
    mesh.apply_translation(-mesh.vertices.mean(axis=0))
    scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    h = float(np.ptp(mesh.vertices[:, 1])); yfov = np.radians(45.0)
    cam = np.eye(4); cam[2, 3] = (h * 0.5) / np.tan(yfov / 2) * 1.4
    scene.add(pyrender.PerspectiveCamera(yfov=yfov), pose=cam)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=cam)
    color, _ = _renderer(res).render(scene, flags=pyrender.RenderFlags.RGBA)
    return color[:, :, :3]


# ----------------------------- decode / swap -----------------------------
@torch.no_grad()
def verts_of(net, idx):
    rot = pose6d_to_rotmat(net, decode_from_codes(net, idx))
    return BODY(body_pose=rot).vertices[0].cpu().numpy()


def nn_index(cb, cosine, used):
    cbf = cb.double()
    if cosine:
        cbf = cbf / cbf.norm(dim=1, keepdim=True).clamp(min=1e-9)
    nn = {}
    for k in torch.unique(used).tolist():
        if cosine:
            s = cbf @ cbf[k]; s[k] = -2.0; nn[k] = int(s.argmax())
        else:
            dd = (cbf - cbf[k]).norm(dim=1); dd[k] = float("inf"); nn[k] = int(dd.argmin())
    return nn


@torch.no_grad()
def best_swap_dv(net, pose):
    cb = get_codebook(net).detach().float().cpu()
    cosine = _is_cosine(net)
    idx = encode_full(net, pose.to(DEV))["idx"]            # (1, T)
    base = verts_of(net, idx)
    nn = nn_index(cb, cosine, idx[0])
    best = (-1.0, None)
    for t in range(idx.shape[1]):
        k = int(idx[0, t]); kk = nn.get(k, k)
        if kk == k:
            continue
        pert = idx.clone(); pert[0, t] = kk
        dv = np.linalg.norm(verts_of(net, pert) - base, axis=1) * 1000.0
        if dv.mean() > best[0]:
            best = (float(dv.mean()), dv)
    return base, best[1]


# ----------------------------- pose pick -----------------------------
def pick_pose(hparams, n_cand=24, seed=1):
    from dataset.dataset_poseVQ import get_dataloader
    torch.manual_seed(seed)
    hparams.DATA.NUM_WORKERS = min(getattr(hparams.DATA, "NUM_WORKERS", 4), 4)
    hparams.DATA.CACHE_SMPL = False
    saved = hparams.DATA.VALLIST; hparams.DATA.VALLIST = "MOYO"      # articulated poses
    poses = []
    for batch in get_dataloader(hparams, split="val", shuffle=True):
        gp = (batch["gt_pose_body"].float().view(-1, 21, 3, 3) if "gt_pose_body" in batch
              else axis_angle_to_matrix(batch["pose_body_aa"].float().view(-1, 21, 3)))
        poses.append(gp.cpu())
        if sum(p.shape[0] for p in poses) >= n_cand:
            break
    hparams.DATA.VALLIST = saved
    poses = torch.cat(poses, 0)[:n_cand]
    with torch.no_grad():
        j = BODY(body_pose=poses).joints[:, 1:22]                    # (n,21,3)
    extent = (j.amax(1) - j.amin(1)).norm(dim=1)                     # bbox diagonal
    return poses[int(extent.argmax())][None]                         # (1,21,3,3)


def main():
    runs = {r["label"]: os.path.join(HERE, r["ckpt"]) for r in json.load(open(STAB))["runs"]}
    _, hp0 = load_net(runs["CNN"])
    pose = pick_pose(hp0)
    print("pose picked")

    data = {}
    for label in ORDER:
        net, _ = load_net(runs[label]); net = net.to(DEV); torch.cuda.empty_cache()
        base, dv = best_swap_dv(net, pose)
        data[label] = (base, dv)
        print(f"{label:20s} max single-token NN-swap displacement: mean {dv.mean():.2f} mm  max {dv.max():.1f} mm")
        del net; torch.cuda.empty_cache()

    vmax = float(np.percentile(data["FSQ d4"][1], 98))               # shared scale from FSQ d4
    print(f"shared vmax = {vmax:.1f} mm")

    # ---- Fig 4.7: FSQ vs cosine contrast ----
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 3.4))
    for a, label in zip(ax, ["FSQ d4", "Transformer cosine"]):
        base, dv = data[label]
        a.imshow(render_error(base, dv, vmax)); a.axis("off")
        a.set_title(f"{SHORT[label]}   (max $\\delta_v$ {dv.max():.0f} mm)", fontsize=9)
    sm = cm.ScalarMappable(cmap=TURBO, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02); cb.set_label("vertex displacement $\\delta_v$ [mm]", fontsize=8.5)
    fig.savefig(os.path.join(LAT_OUT, "token_visualization_contrast.pdf"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(REPO, "tokenhmr", "thesis_figures", "token_viz_contrast_preview.png"), dpi=150, bbox_inches="tight")
    print("wrote token_visualization_contrast.pdf")

    # ---- Fig B.2: all tokenizers ----
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 5.6))
    for a, label in zip(axes.ravel(), ORDER):
        base, dv = data[label]
        a.imshow(render_error(base, dv, vmax)); a.axis("off")
        a.set_title(f"{SHORT[label]}\n(max $\\delta_v$ {dv.max():.0f} mm)", fontsize=8.5)
    sm = cm.ScalarMappable(cmap=TURBO, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02); cb.set_label("vertex displacement $\\delta_v$ [mm]", fontsize=8.5)
    fig.suptitle("Most-impactful single-token nearest-neighbour swap, per tokenizer (shared scale)", fontsize=10.5)
    fig.savefig(os.path.join(APP_OUT, "tokenviz_all_tokenizers.pdf"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(REPO, "tokenhmr", "thesis_figures", "tokenviz_all_preview.png"), dpi=130, bbox_inches="tight")
    print("wrote tokenviz_all_tokenizers.pdf")


if __name__ == "__main__":
    main()
