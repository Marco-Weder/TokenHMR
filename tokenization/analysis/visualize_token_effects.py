"""Part 2 - Token Visualization: what does each pose token control?

The tokenizer squeezes a body pose (21 joints) into 160 discrete tokens. These 160 tokens
are NOT one-per-joint - the encoder mixes everything together, so we cannot just read off
"token 7 = left knee". This script answers the question visually instead:

    Take a body pose, keep every token the same except one, decode both, and look at how far
    each mesh vertex moved. Coloring the body by that per-vertex movement shows exactly which
    body part a token position controls.

Some token swaps barely move the mesh (the token is nearly meaningless for that pose), while
others strongly deform one body region. That contrast is the whole point of Part 2.

It produces three things:
  1. neutral_token_montage.png - start from the rest pose and, for each of the 160 tokens,
     swap it and color the body by per-vertex displacement. All 160 little pictures are tiled
     into one image (and also saved one-by-one in panels/) so you can scan them at a glance.
  2. token_sensitivity_ranking.png + token_sensitivity.json - per-token average displacement
     (mm), MPJPE (mm) and the joint each token affects most, ranked most -> least sensitive.
  3. real_benign_vs_harmful.png - on a few real dataset poses, swap the globally MOST-sensitive
     token (big, localized damage) vs the LEAST-sensitive token (almost no change).

Run it from the tokenization/ directory:
    python visualize_token_effects.py --ckpt output/<run>/.../best_net.pth --out token_viz
    python visualize_token_effects.py --ckpt ... --quick        # fast smoke test
"""

import os
# pyrender must render off-screen on the GPU; set this BEFORE pyrender is imported.
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import sys
import json
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import trimesh
import pyrender

# This script lives next to the other tokenizer-analysis scripts, so make them importable.
# Reuse the model plumbing we already wrote for the latent analysis - no need to repeat it here.
#   load_net          load a tokenizer checkpoint (best_net.pth)  -> (net, hparams)
#   encode_full       pose (B,21,3,3) -> dict with 'idx' (B,160) token ids
#   decode_from_codes token ids (B,160) -> 6D pose (B,21,6)
#   pose6d_to_rotmat  6D pose -> rotation matrices (B,21,3,3)
#   collect           sample real validation poses (returns their token ids)
from tokenization.analysis.analyze_latent_pose_info import (
    load_net, encode_full, decode_from_codes, pose6d_to_rotmat,
    collect, JOINT_NAMES, DEVICE,
)
# The shared SMPL-H body model (turns joint rotations into a 6890-vertex mesh).
from tokenization.models.transformer_pose_vqvae import body_model

# Colormap for the error maps: dark blue = no change, red = large change.
TURBO = plt.get_cmap('turbo')

# The two arms, by joint index (used to find/illustrate tokens that couple both arms at once).
_LEFT_ARM = [JOINT_NAMES.index(n) for n in ('L_Collar', 'L_Shoulder', 'L_Elbow', 'L_Wrist')]
_RIGHT_ARM = [JOINT_NAMES.index(n) for n in ('R_Collar', 'R_Shoulder', 'R_Elbow', 'R_Wrist')]
_NON_ARM = [j for j in range(len(JOINT_NAMES)) if j not in _LEFT_ARM + _RIGHT_ARM]


# --------------------------------------------------------------------------- #
# Step 0: decode a token sequence all the way to a body mesh                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def decode_to_mesh(net, idx):
    """Token ids (B, 160) -> SMPL vertices (B, 6890, 3) and body joints (B, 21, 3), in metres."""
    pose_6d = decode_from_codes(net, idx)             # (B, 21, 6)
    rotmat = pose6d_to_rotmat(net, pose_6d)           # (B, 21, 3, 3)
    out = body_model(body_pose=rotmat.to(DEVICE))
    verts = out.vertices                              # (B, 6890, 3)
    joints = out.joints[:, 1:22]                      # (B, 21, 3), skip the pelvis (joint 0)
    return verts, joints


# --------------------------------------------------------------------------- #
# Step 1: render one mesh, colored by a per-vertex error value                 #
# --------------------------------------------------------------------------- #
# Creating an off-screen renderer is a little expensive, so we make one per image
# resolution and reuse it for every panel.
_RENDERERS = {}


def _get_renderer(res):
    if res not in _RENDERERS:
        _RENDERERS[res] = pyrender.OffscreenRenderer(viewport_width=res, viewport_height=res,
                                                     point_size=1.0)
    return _RENDERERS[res]


def _render_mesh(vertices, faces, vertex_colors, res, azim_deg):
    """Render a mesh with the given per-vertex RGBA colors -> (res, res, 3) uint8 (white background)."""
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy(), vertex_colors=vertex_colors, process=False)

    # The body model already comes out head-up (+y). Optionally spin it about the vertical axis to
    # choose the viewing angle (e.g. --azim 180 for a back view), then centre it so the framing is
    # identical for every render.
    if azim_deg:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(azim_deg), [0, 1, 0]))
    mesh.apply_translation(-mesh.vertices.mean(axis=0))

    # Scene: the mesh + soft ambient light + one head-on light for a little shading.
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    # Camera looking down -z from far enough away to fit the whole (upright) body.
    body_height = float(np.ptp(mesh.vertices[:, 1]))
    yfov = np.radians(45.0)
    distance = (body_height * 0.5) / np.tan(yfov / 2.0) * 1.4         # 1.4 = a little margin
    cam_pose = np.eye(4)
    cam_pose[2, 3] = distance
    scene.add(pyrender.PerspectiveCamera(yfov=yfov), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=cam_pose)

    color, _ = _get_renderer(res).render(scene, flags=pyrender.RenderFlags.RGBA)
    return color[:, :, :3]                                            # drop alpha (bg is white)


def render_mesh_error(vertices, faces, err_mm, vmax, res=256, azim_deg=0.0):
    """Render an SMPL mesh colored by per-vertex error (mm) with the turbo colormap.

    `vmax` is the mm value that maps to full red; sharing it across panels makes them comparable.
    """
    err_norm = np.clip(err_mm / max(vmax, 1e-6), 0.0, 1.0)           # 0 -> dark blue, vmax -> red
    vertex_colors = (TURBO(err_norm) * 255).astype(np.uint8)         # (V, 4) RGBA
    return _render_mesh(vertices, faces, vertex_colors, res, azim_deg)


def render_mesh_plain(vertices, faces, res=256, azim_deg=0.0, rgb=(0.60, 0.62, 0.72)):
    """Render an SMPL mesh in one flat color, to show the actual body pose (not an error map)."""
    rgba = np.array([int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 255], dtype=np.uint8)
    vertex_colors = np.tile(rgba, (vertices.shape[0], 1))
    return _render_mesh(vertices, faces, vertex_colors, res, azim_deg)


# --------------------------------------------------------------------------- #
# Step 2: encode the neutral (rest) pose to its 160 tokens                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def neutral_pose_tokens(net):
    """Encode the rest pose (all 21 joint rotations = identity) -> token ids (1, 160)."""
    identity = torch.eye(3, device=DEVICE).view(1, 1, 3, 3).repeat(1, net.num_joints, 1, 1)
    return encode_full(net, identity)['idx']                          # (1, 160)


# --------------------------------------------------------------------------- #
# Step 3: measure what happens when each token is swapped                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def per_token_effects(net, base_idx, num_alts, seed):
    """Swap each token position to a few random codes and measure the decoded change.

    Returns (all as numpy):
      vert_disp   (T, V)  mean per-vertex displacement (mm) from the base mesh, per position
      sens_mm     (T,)    mean over vertices of vert_disp  = the position's overall sensitivity
      mpjpe_mm    (T,)    mean joint displacement (mm), per position
      joint_disp  (T, J)  mean per-joint displacement (mm), used to name the top affected joint
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    T = base_idx.shape[1]
    nb_code = net.quantizer.nb_code

    # The mesh we compare everything against.
    base_verts, base_joints = decode_to_mesh(net, base_idx)
    base_verts, base_joints = base_verts[0], base_joints[0]           # (V,3), (J,3)
    V, J = base_verts.shape[0], base_joints.shape[0]

    vert_disp = torch.zeros(T, V)
    joint_disp = torch.zeros(T, J)
    for i in range(T):
        # Pick `num_alts` random replacement codes for this position (making sure they differ
        # from the original code so the swap actually does something).
        alts = torch.randint(0, nb_code, (num_alts,), generator=gen, device=DEVICE)
        alts = torch.where(alts == base_idx[0, i], (alts + 1) % nb_code, alts)

        # Build `num_alts` copies of the sequence, each with position i replaced, and decode them.
        seqs = base_idx.repeat(num_alts, 1)                           # (num_alts, T)
        seqs[:, i] = alts
        verts, joints = decode_to_mesh(net, seqs)                     # (num_alts,V,3),(num_alts,J,3)

        # Per-vertex / per-joint distance to the base mesh, averaged over the alternatives, in mm.
        vert_disp[i] = (torch.linalg.norm(verts - base_verts, dim=-1).mean(0) * 1000.0).cpu()
        joint_disp[i] = (torch.linalg.norm(joints - base_joints, dim=-1).mean(0) * 1000.0).cpu()

    sens_mm = vert_disp.mean(dim=1)                                   # (T,)
    mpjpe_mm = joint_disp.mean(dim=1)                                 # (T,)
    return (vert_disp.numpy(), sens_mm.numpy(), mpjpe_mm.numpy(), joint_disp.numpy())


# --------------------------------------------------------------------------- #
# Step 4: the 160-token montage (one panel per token position)                #
# --------------------------------------------------------------------------- #
def _shared_crop_box(images, pad=6, bg_thresh=250):
    """Tight bounding box around the body, shared by all panels (they all render the same mesh).

    The background is white (255); any colored body pixel is < bg_thresh in some channel. Taking the
    darkest value per pixel across every panel gives the full body silhouette to crop to.
    """
    darkest = np.stack(images).min(axis=0)                # (H, W, 3): darkest each pixel gets
    body = np.any(darkest < bg_thresh, axis=2)           # True where the body is
    rows = np.where(body.any(axis=1))[0]
    cols = np.where(body.any(axis=0))[0]
    H, W = body.shape
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, H)
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, W)
    return r0, r1, c0, c1


def build_montage(base_verts, faces, vert_disp, sens_mm, vmax, out_dir, res, azim, cols, title):
    """Render one error-map panel per token, save each separately, and tile them tightly into a grid.

    Every panel shows the SAME body (only the colors change), so we crop them all to one shared box
    and pack them into a compact grid. Each cell has the token index centered in a white strip above
    the mesh and the mean displacement centered in a white strip below it (so no text overlaps the
    body), plus a thin border so the panels read as a clean grid.
    """
    T = vert_disp.shape[0]
    panels_dir = os.path.join(out_dir, 'panels')
    os.makedirs(panels_dir, exist_ok=True)

    # 1. Render (and individually save) every panel.
    images = []
    for i in range(T):
        img = render_mesh_error(base_verts, faces, vert_disp[i], vmax, res=res, azim_deg=azim)
        plt.imsave(os.path.join(panels_dir, f'token_{i:03d}.png'), img)
        images.append(img)

    # 2. Crop all panels to one shared, tight bounding box around the body.
    r0, r1, c0, c1 = _shared_crop_box(images)
    cropped = [img[r0:r1, c0:c1] for img in images]
    ch, cw = cropped[0].shape[:2]

    # 3. Each cell = white header strip (token index) + cropped mesh + white footer strip (mean mm).
    header_h = max(int(round(0.13 * ch)), 18)           # top strip: token index
    footer_h = max(int(round(0.11 * ch)), 16)           # bottom strip: mean displacement
    cell_h = header_h + ch + footer_h
    rows = int(np.ceil(T / cols))
    canvas = np.full((rows * cell_h, cols * cw, 3), 255, dtype=np.uint8)
    for i, panel in enumerate(cropped):
        r, c = divmod(i, cols)
        y = r * cell_h + header_h                       # mesh sits between header and footer
        canvas[y:y + ch, c * cw:(c + 1) * cw] = panel

    # 4. Draw the canvas, centre the token index above and the mean displacement below each mesh,
    #    and outline every cell so the panels read as a clean grid.
    fig_w = cols * 1.05
    fig_h = fig_w * (rows * cell_h) / (cols * cw)        # match the canvas aspect (no stretching)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(canvas)
    ax.set_xlim(0, cols * cw)
    ax.set_ylim(rows * cell_h, 0)
    ax.axis('off')
    for i in range(T):
        r, c = divmod(i, cols)
        x, y = c * cw, r * cell_h
        xc = x + cw / 2
        ax.add_patch(plt.Rectangle((x, y), cw, cell_h, fill=False, edgecolor='#bbbbbb', lw=0.6))
        ax.text(xc, y + header_h / 2, f'{i}', fontsize=8, weight='bold', color='black',
                va='center', ha='center')
        ax.text(xc, y + header_h + ch + footer_h / 2, f'{sens_mm[i]:.0f} mm', fontsize=6.5,
                color='#444444', va='center', ha='center')

    # 5. A slim shared colorbar placed just outside the grid (so it does not shrink the panels).
    sm = plt.cm.ScalarMappable(cmap=TURBO, norm=plt.Normalize(0.0, vmax))
    sm.set_array([])
    cax = ax.inset_axes([1.015, 0.30, 0.015, 0.40])
    fig.colorbar(sm, cax=cax, label='per-vertex displacement (mm)')
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    fig.savefig(os.path.join(out_dir, 'neutral_token_montage.png'), dpi=200,
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Step 5: rank the tokens and write the metrics                                #
# --------------------------------------------------------------------------- #
def save_ranking(sens_mm, mpjpe_mm, joint_disp, out_dir):
    """Save a ranked bar chart + JSON, and return the positions ordered most -> least sensitive."""
    T = len(sens_mm)
    order = np.argsort(-sens_mm)                       # most sensitive first
    top_joint = joint_disp.argmax(axis=1)             # each token's most-moved joint

    ranking = [{'position': int(p),
                'vertex_disp_mm': float(sens_mm[p]),
                'mpjpe_mm': float(mpjpe_mm[p]),
                'top_joint': JOINT_NAMES[int(top_joint[p])]}
               for p in order]
    with open(os.path.join(out_dir, 'token_sensitivity.json'), 'w') as f:
        json.dump({'num_tokens': int(T),
                   'mean_vertex_disp_mm': float(sens_mm.mean()),
                   'mean_mpjpe_mm': float(mpjpe_mm.mean()),
                   'ranking': ranking}, f, indent=2)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(T), sens_mm[order], color='#4C78A8')
    ax.set_xlabel('token position (ranked most -> least sensitive)')
    ax.set_ylabel('mean per-vertex displacement (mm)')
    ax.set_title('Token sensitivity ranking (neutral pose)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'token_sensitivity_ranking.png'), dpi=130)
    plt.close(fig)
    return order


# --------------------------------------------------------------------------- #
# Step 6: benign vs harmful swaps on real dataset poses                        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def benign_vs_harmful(net, hparams, pos_most, pos_least, num_examples, num_real_poses,
                      out_dir, seed, res, azim):
    """For a few real poses, swap the most- vs least-sensitive token and render both error maps."""
    faces = body_model.faces
    panels_dir = os.path.join(out_dir, 'panels')
    os.makedirs(panels_dir, exist_ok=True)

    # Grab some real validation poses (their token sequences) and pick a handful at random.
    cache = collect(net, hparams, num_real_poses)
    idx_all, names = cache['idx'], cache['names']                     # (N, 160), list of names
    pick = torch.randperm(idx_all.shape[0],
                          generator=torch.Generator().manual_seed(seed))[:num_examples].tolist()

    gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    nb_code = net.quantizer.nb_code

    # First pass: compute the per-vertex displacement for both swaps of every picked pose.
    # (We do this before rendering so we can pick a shared color scale that fits the data.)
    records = []                                       # each: (base_verts_np, disp, pos, tag, name, s)
    for s in pick:
        base = idx_all[s:s + 1].to(DEVICE)                            # (1, 160)
        base_verts = decode_to_mesh(net, base)[0][0]                  # (V, 3)
        for pos, tag in [(pos_most, 'harmful'), (pos_least, 'benign')]:
            alt = torch.randint(0, nb_code, (1,), generator=gen, device=DEVICE)
            alt = torch.where(alt == base[0, pos], (alt + 1) % nb_code, alt)
            swapped = base.clone()
            swapped[0, pos] = alt
            verts = decode_to_mesh(net, swapped)[0][0]                # (V, 3)
            disp = (torch.linalg.norm(verts - base_verts, dim=-1) * 1000.0).cpu().numpy()
            records.append((base_verts.cpu().numpy(), disp, pos, tag, names[s], s))

    # Shared color scale: the 99th percentile of the harmful swaps (they dominate the range).
    harmful_disp = np.concatenate([r[1] for r in records if r[3] == 'harmful'])
    vmax = float(np.percentile(harmful_disp, 99)) if harmful_disp.size else 1.0

    # Second pass: render everything and lay the examples out as rows (harmful | benign).
    fig, axes = plt.subplots(len(pick), 2, figsize=(6, 3 * len(pick)), squeeze=False)
    for row, s in enumerate(pick):
        for col, tag in enumerate(['harmful', 'benign']):
            base_verts, disp, pos, _, name, _ = next(r for r in records if r[5] == s and r[3] == tag)
            img = render_mesh_error(base_verts, faces, disp, vmax, res=res, azim_deg=azim)
            plt.imsave(os.path.join(panels_dir, f'real_{name}_{s}_{tag}.png'), img)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'{tag}: token {pos}\n'
                                     f'max {disp.max():.0f}mm  mean {disp.mean():.1f}mm', fontsize=9)
    fig.suptitle('Swapping the most- vs least-sensitive token on real poses', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'real_benign_vs_harmful.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Token coupling: one token controls several distant body parts (attention)   #
# --------------------------------------------------------------------------- #
def pick_coupling_token(joint_disp):
    """Return the token that cleanly couples the two arms (both arms move, little else does).

    joint_disp is (T, 21) mean per-joint displacement per token. A clean bilateral-arm token has a
    large displacement on the left arm AND the right arm (so we take the minimum of the two, which is
    only high when both move) while moving the rest of the body little (so we subtract the non-arm
    mean). This favours tokens like the "both arms" example over ones that just move everything.
    """
    left = joint_disp[:, _LEFT_ARM].mean(axis=1)
    right = joint_disp[:, _RIGHT_ARM].mean(axis=1)
    other = joint_disp[:, _NON_ARM].mean(axis=1)
    both = np.minimum(left, right)                        # both arms have to move
    clean = other < 0.3 * both                            # and the rest of the body barely moves
    if clean.any():                                       # among clean tokens, take the strongest
        return int(np.argmax(np.where(clean, both, -np.inf)))
    return int(np.argmax(both - other))                  # fallback if none are clean


@torch.no_grad()
def visualize_token_coupling(net, base_idx, token, out_dir, res, azim, seed, n_sweep):
    """Show that a single token controls several distant body parts at once.

    We fix the neutral pose and sweep ONE token through a range of code values. Both arms move
    together the whole time - something a per-joint code could never do, but the transformer's global
    attention can. The figure has three rows: the actual bodies, the per-vertex error maps (where the
    body changed), and a bar chart of which joints moved (the two arms, highlighted).
    """
    faces = body_model.faces
    base_verts, base_joints = decode_to_mesh(net, base_idx)
    base_verts, base_joints = base_verts[0], base_joints[0]           # (V,3), (J,3)
    nb_code = net.quantizer.nb_code

    # Try many alternative codes for this token, then keep a spread of them from a moderate change
    # to a big one (so the sweep is informative rather than a random jumble).
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 7)
    pool = torch.randperm(nb_code, generator=gen, device=DEVICE)[:min(80, nb_code)]
    seqs = base_idx.repeat(len(pool), 1)
    seqs[:, token] = pool
    verts, joints = decode_to_mesh(net, seqs)                         # (P,V,3), (P,J,3)
    disp = (torch.linalg.norm(verts - base_verts, dim=-1).mean(1) * 1000.0).cpu().numpy()
    order = np.argsort(disp)
    picks = order[np.linspace(len(order) * 0.4, len(order) - 1, n_sweep).astype(int)]
    sweep_verts = verts[picks].cpu().numpy()                          # (n_sweep, V, 3)
    base_verts_np = base_verts.cpu().numpy()
    sweep_vert_disp = np.linalg.norm(sweep_verts - base_verts_np, axis=2) * 1000.0   # (n_sweep, V)
    sweep_joint_disp = (torch.linalg.norm(joints[picks] - base_joints, dim=-1) * 1000.0).cpu().numpy()
    vmax = float(np.percentile(sweep_vert_disp, 99))

    # Render both rows first (row 0 = actual body, row 1 = colored by displacement from neutral),
    # then crop every panel to one shared box so the bodies fill their cells (no wasted whitespace).
    plain_imgs = [render_mesh_plain(base_verts_np, faces, res=res, azim_deg=azim)]
    err_imgs = [render_mesh_error(base_verts_np, faces, np.zeros(base_verts_np.shape[0]), vmax,
                                  res=res, azim_deg=azim)]
    for k in range(n_sweep):
        plain_imgs.append(render_mesh_plain(sweep_verts[k], faces, res=res, azim_deg=azim))
        err_imgs.append(render_mesh_error(sweep_verts[k], faces, sweep_vert_disp[k], vmax,
                                          res=res, azim_deg=azim))
    r0, r1, c0, c1 = _shared_crop_box(plain_imgs + err_imgs, pad=4)
    plain_imgs = [im[r0:r1, c0:c1] for im in plain_imgs]
    err_imgs = [im[r0:r1, c0:c1] for im in err_imgs]

    ncol = n_sweep + 1                                                # neutral + the sweep
    fig = plt.figure(figsize=(ncol * 1.5, 5.4))
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.3, 1.3, 1.0], hspace=0.06, wspace=0.01)
    last_err_ax = None
    for k in range(ncol):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(plain_imgs[k]); ax.axis('off')
        if k == 0:
            ax.set_title('neutral', fontsize=8)
        ax = fig.add_subplot(gs[1, k]); ax.imshow(err_imgs[k]); ax.axis('off')
        last_err_ax = ax
    sm = plt.cm.ScalarMappable(cmap=TURBO, norm=plt.Normalize(0.0, vmax)); sm.set_array([])
    fig.colorbar(sm, cax=last_err_ax.inset_axes([1.05, 0.0, 0.08, 1.0]), label='mm')

    # Row 2: which joints move when this token changes (arms highlighted in red).
    axbar = fig.add_subplot(gs[2, :])
    per_joint = sweep_joint_disp.mean(axis=0)                         # (21,)
    colors = ['#bbbbbb'] * len(JOINT_NAMES)
    for j in _LEFT_ARM + _RIGHT_ARM:
        colors[j] = '#E45756'
    axbar.bar(range(len(JOINT_NAMES)), per_joint, color=colors)
    axbar.set_xticks(range(len(JOINT_NAMES)))
    axbar.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=6)
    axbar.set_ylabel('mean joint\ndisplacement (mm)', fontsize=8)
    axbar.set_title('Joints moved by this token (red = the two arms)', fontsize=8)

    fig.suptitle(f'One token, two distant body parts: sweeping token {token} moves both arms together',
                 fontsize=11)
    fig.savefig(os.path.join(out_dir, 'token_coupling.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='Part 2 - token visualization (mesh error maps)')
    parser.add_argument('--ckpt', type=str, required=True, help='tokenizer checkpoint (best_net.pth)')
    parser.add_argument('--out', type=str, default=None, help='output dir (default <ckpt_dir>/token_viz)')
    parser.add_argument('--num-alts', type=int, default=8, help='random replacement codes per token')
    parser.add_argument('--num-real-poses', type=int, default=200, help='real val poses to sample')
    parser.add_argument('--num-examples', type=int, default=6, help='real poses shown in the gallery')
    parser.add_argument('--res', type=int, default=256, help='render resolution (pixels)')
    parser.add_argument('--montage-cols', type=int, default=10, help='columns in the token montage')
    parser.add_argument('--montage-title', type=str,
                        default='Per-token mesh vertex-error maps',
                        help='montage title; pass "" to drop it and define labels in the report caption')
    parser.add_argument('--coupling-token', type=int, default=-1,
                        help='token for the coupling demo (-1 = auto-pick a token that moves both arms)')
    parser.add_argument('--coupling-sweep', type=int, default=6,
                        help='number of code variations shown in the coupling sweep')
    parser.add_argument('--azim', type=float, default=0.0, help='view rotation about the vertical axis')
    parser.add_argument('--vmax', type=float, default=None, help='mm value mapped to red (default: auto)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-real', action='store_true', help='skip the real-pose gallery')
    parser.add_argument('--quick', action='store_true', help='fast smoke test (few alts/examples)')
    args = parser.parse_args()

    if args.quick:
        args.num_alts, args.num_examples, args.res, args.num_real_poses = 2, 2, 128, 32

    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt), 'token_viz')
    os.makedirs(out_dir, exist_ok=True)

    # Load the frozen tokenizer.
    net, hparams = load_net(args.ckpt)
    faces = body_model.faces

    # --- neutral pose -> tokens -> per-token effects ---
    print('Encoding the neutral pose and measuring per-token effects...')
    neutral_idx = neutral_pose_tokens(net)
    vert_disp, sens_mm, mpjpe_mm, joint_disp = per_token_effects(net, neutral_idx,
                                                                 args.num_alts, args.seed)
    base_verts = decode_to_mesh(net, neutral_idx)[0][0].cpu().numpy()

    # Shared color scale for all 160 panels.
    vmax = args.vmax if args.vmax is not None else float(np.percentile(vert_disp, 99))

    # --- figure 1: the 160-token montage ---
    print(f'Rendering the {len(sens_mm)}-token montage...')
    build_montage(base_verts, faces, vert_disp, sens_mm, vmax, out_dir, args.res, args.azim,
                  args.montage_cols, args.montage_title)

    # --- figure 2 + metrics: the sensitivity ranking ---
    order = save_ranking(sens_mm, mpjpe_mm, joint_disp, out_dir)
    pos_most, pos_least = int(order[0]), int(order[-1])

    # --- figure: token coupling (one token moves several distant body parts) ---
    ctoken = args.coupling_token if args.coupling_token >= 0 else pick_coupling_token(joint_disp)
    print(f'Rendering the token-coupling demo for token {ctoken}...')
    visualize_token_coupling(net, neutral_idx, ctoken, out_dir, args.res, args.azim, args.seed,
                             args.coupling_sweep)

    # --- figure 3: benign vs harmful on real poses ---
    if not args.skip_real:
        print('Rendering benign-vs-harmful swaps on real poses...')
        benign_vs_harmful(net, hparams, pos_most, pos_least, args.num_examples,
                          args.num_real_poses, out_dir, args.seed, args.res, args.azim)

    # --- short console summary ---
    top_joint = joint_disp.argmax(axis=1)
    print('\nMost sensitive tokens:')
    for p in order[:5]:
        print(f'  token {p:3d}: {sens_mm[p]:6.1f} mm  (mostly moves {JOINT_NAMES[top_joint[p]]})')
    print('Least sensitive tokens:')
    for p in order[-5:]:
        print(f'  token {p:3d}: {sens_mm[p]:6.1f} mm')
    print(f'\nSaved all figures to {out_dir}')


if __name__ == '__main__':
    main()
