#!/usr/bin/env python
"""Render Figure 3.1: the SMPL decomposition strip.

Four panels mirroring the equations in Section 3.1.1:
    (1) mean template  T_bar                    (beta = 0, theta = 0)
    (2) + shape blend shapes  B_S(beta)         (beta != 0, theta = 0)
    (3) + pose blend shapes   B_P(theta)        rest-pose body coloured by |B_P| (turbo)
    (4) linear blend skinning -> M(beta, theta) (full SMPL forward)

Renders the figure for several candidate poses at once (one PNG each) so a good
pose can be picked; the first pose is also saved as the default file wired into
the thesis.

Run with the project env (from the tokenhmr folder):
    ~/miniconda3/envs/thesis-HMR/bin/python thesis_figures/render_smpl_figure.py

Output: <repo>/thesis/images/smpl/smpl_decomposition.png  (+ _p<idx>.png per candidate)
"""
import os
import numpy as np
import torch
import trimesh
import smplx
from smplx.lbs import blend_shapes, batch_rodrigues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Headless Blender instead of pyrender
import bpy
import mathutils
import tempfile

# ---------------------------------------------------------------- config
# This script lives in <repo>/tokenhmr/thesis_figures/ and writes its output into
# the thesis image tree at <repo>/thesis/images/smpl/.
HERE = os.path.dirname(os.path.abspath(__file__))
TOKENHMR = os.path.dirname(HERE)                       # <repo>/tokenhmr
REPO = os.path.dirname(TOKENHMR)                        # <repo>
THESIS = os.path.join(REPO, 'thesis')
MODEL_PATH = os.path.join(TOKENHMR, 'data', 'body_models')
VAL_NPZ = os.path.join(TOKENHMR, 'tokenization',
                       'tokenization_data', 'smplh', 'val', 'val_HumanEva.npz')
OUTDIR = os.path.join(THESIS, 'images', 'smpl')
POSE_IDXS = [800, 1500, 2200, 2900, 3400, 150]   # candidate poses; first is the default
SHAPE_BETAS = np.array([2.2, -1.3, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # a visibly distinct body

RES_X = 800
RES_Y = 800
TURBO = matplotlib.colormaps['turbo']            # match the latent-space displacement colour code

# ---------------------------------------------------------------- SMPL setup
model = smplx.create(MODEL_PATH, model_type='smpl', gender='neutral', num_betas=10, ext='pkl')
faces = model.faces
val = np.load(VAL_NPZ, allow_pickle=True)
betas_t = torch.tensor(SHAPE_BETAS)[None]
zero_pose = torch.zeros(1, 69)
zero_beta = torch.zeros(1, 10)


def verts(betas, pose):
    return model(betas=betas, body_pose=pose, global_orient=torch.zeros(1, 3)).vertices[0].detach().numpy()


def pose_from_val(idx):
    pb = val['pose_body'][idx].astype(np.float32)                 # (63,) = 21 body joints
    return torch.tensor(np.concatenate([pb, np.zeros(6, np.float32)]))[None]  # pad 2 hand joints


def blendshape_offset(betas, body_pose_t):
    """v_shaped + B_P(theta) (rest pose) and per-vertex |B_P| in metres."""
    full_pose = torch.cat([torch.zeros(1, 3), body_pose_t], dim=1)             # (1,72) incl. root
    rot = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)
    pose_feature = (rot[:, 1:, :, :] - torch.eye(3)).view(1, -1)               # (1,207)
    offs = torch.matmul(pose_feature, model.posedirs).view(1, -1, 3)           # (1,V,3)
    v_shaped = model.v_template + blend_shapes(betas, model.shapedirs)
    return (v_shaped + offs)[0].detach().numpy(), np.linalg.norm(offs[0].detach().numpy(), axis=1)


# pose-independent meshes
v_template = verts(zero_beta, zero_pose)
v_shaped = verts(betas_t, zero_pose)
posed = {i: verts(betas_t, pose_from_val(i)) for i in POSE_IDXS}


def get_extents(v):
    c = 0.5 * (v.max(0) + v.min(0))
    return np.abs(v - c).max(0)[:2]

all_extents = [get_extents(v) for v in [v_template, v_shaped] + list(posed.values())]
MAX_X = max(e[0] for e in all_extents)
MAX_Y = max(e[1] for e in all_extents)

YFOV = 0.60
# Distance calculated generously to render safely before auto-cropping
DIST = (max(MAX_X, MAX_Y) * 1.1) / np.tan(YFOV / 2)


# ---------------------------------------------------------------- Headless Blender Renderer
def add_area_light(name, pos, energy, size=2.0):
    """Helper to create studio area lights that track to the origin"""
    light_data = bpy.data.lights.new(name=name, type='AREA')
    light_data.energy = energy
    light_data.size = size
    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    
    light_obj.location = pos
    direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector(pos)
    light_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def crop_image(img_rgba):
    """Auto-crop transparent borders around the rendered mesh."""
    alpha = img_rgba[..., 3]
    y_indices, x_indices = np.where(alpha > 0)
    if len(y_indices) == 0:
        return img_rgba
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    return img_rgba[y_min:y_max+1, x_min:x_max+1]


def render(v, vertex_colors=None):
    """Renders the mesh using Blender Cycles CPU engine and crops empty space."""
    v = v - 0.5 * (v.max(0) + v.min(0))  # centre on bbox
    
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 1. Mesh setup
    mesh = bpy.data.meshes.new('SMPL')
    mesh.from_pydata(v.tolist(), [], faces.tolist())
    mesh.update()
    
    obj = bpy.data.objects.new('SMPL', mesh)
    bpy.context.collection.objects.link(obj)
    for p in mesh.polygons:
        p.use_smooth = True
        
    # 2. Material Setup
    mat = bpy.data.materials.new(name="SMPL_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Roughness'].default_value = 0.8
    bsdf.inputs['Metallic'].default_value = 0.0
    
    if vertex_colors is not None:
        color_attr = mesh.color_attributes.new(name='Color', type='FLOAT_COLOR', domain='POINT')
        colors_flat = (vertex_colors / 255.0).astype(np.float32).flatten().tolist()
        color_attr.data.foreach_set('color', colors_flat)
        
        attr_node = mat.node_tree.nodes.new(type="ShaderNodeAttribute")
        attr_node.attribute_name = "Color"
        mat.node_tree.links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs['Base Color'].default_value = (0.72, 0.73, 0.77, 1.0)
        
    obj.data.materials.append(mat)
    
    # 3. Camera Setup
    cam_data = bpy.data.cameras.new('Camera')
    cam_data.sensor_fit = 'VERTICAL'
    cam_data.angle_y = YFOV
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_obj.location = (0, 0, DIST) 
    
    # 4. Studio Lighting Setup
    add_area_light("KeyLight", pos=(-1.9, 1.9, 1.4), energy=350.0, size=3.0)
    add_area_light("FillLight", pos=(2.3, 0.1, 1.3), energy=80.0, size=5.0)
    add_area_light("Headlamp", pos=(0, 0, DIST), energy=100.0, size=1.0)
    
    # 5. Render Settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.render.resolution_x = RES_X
    bpy.context.scene.render.resolution_y = RES_Y
    bpy.context.scene.render.film_transparent = True
    
    tmp_path = os.path.join(tempfile.gettempdir(), "smpl_render_temp.png")
    bpy.context.scene.render.filepath = tmp_path
    
    with open(os.devnull, 'w') as f:
        os.dup2(f.fileno(), 1)
        bpy.ops.render.render(write_still=True)
        
    img = plt.imread(tmp_path)
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Auto-crop empty pixels around human body
    return crop_image(img_uint8)


# render the two pose-independent panels once
img_template = render(v_template)
img_shaped = render(v_shaped)

# ---------------------------------------------------------------- compose
titles = [r'$\bar{T}$', r'$\bar{T}+B_S(\beta)$', r'$\bar{T}+B_S(\beta)+B_P(\theta)$', r'$M(\beta,\theta)$']
subtitles = ['mean template', 'shape blend shapes', 'pose blend shapes', 'linear blend skinning']
ops = [r'$+\,B_S(\beta)$', r'$+\,B_P(\theta)$', r'$\mathrm{LBS}$']


def compose(pose_idx, out_path):
    v_blend, bp_mag = blendshape_offset(betas_t, pose_from_val(pose_idx))
    vmax_mm = float(bp_mag.max()) * 1000
    bp_colors = (TURBO(Normalize(0.0, bp_mag.max())(bp_mag)) * 255).astype(np.uint8)
    imgs = [img_template, img_shaped, render(v_blend, bp_colors), render(posed[pose_idx])]

    # Slightly increased height (4.0 inches) just to give the colorbar text enough physical pixels
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.25, wspace=0.08)
    
    for ax, im, t, st in zip(axes, imgs, titles, subtitles):
        ax.imshow(im)
        ax.set_title(t, fontsize=16, pad=8)
        ax.axis('off')
        
        # Position subtitle tightly under each panel's image bounds
        p = ax.get_position()
        fig.text(0.5 * (p.x0 + p.x1), 0.14, st, ha='center', va='center', fontsize=12)
        
    for i, op in enumerate(ops):
        x = 0.5 * (axes[i].get_position().x1 + axes[i + 1].get_position().x0)
        fig.text(x, 0.50, r'$\rightarrow$', ha='center', va='center', fontsize=26)
        fig.text(x, 0.60, op, ha='center', va='center', fontsize=13)
        
    # Colorbar right below Panel 3
    p3 = axes[2].get_position()
    cax = fig.add_axes([p3.x0 + 0.01, 0.05, p3.width - 0.02, 0.03])
    sm = plt.cm.ScalarMappable(norm=Normalize(0, vmax_mm), cmap=TURBO); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    
    # Render main label centered below
    cb.set_label(r'$|B_P(\theta)|$', fontsize=10, labelpad=4)
    cb.ax.tick_params(labelsize=9)
    
    # Explicitly append "[mm]" directly to the right of the tick labels
    cax.annotate('[mm]', xy=(1.01, 0), xycoords='axes fraction', 
                 xytext=(3, -9), textcoords='offset points', 
                 va='center', ha='left', fontsize=9)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return vmax_mm


os.makedirs(OUTDIR, exist_ok=True)
for k, idx in enumerate(POSE_IDXS):
    out = os.path.join(OUTDIR, f'smpl_decomposition_p{idx}.png')
    vmax = compose(idx, out)
    if k == 0:
        compose(idx, os.path.join(OUTDIR, 'smpl_decomposition.png'))   # default wired into the thesis
    print(f'pose {idx:5d}  |B_P| max={vmax:5.1f} mm  ->  {os.path.basename(out)}')
print('default = pose', POSE_IDXS[0])