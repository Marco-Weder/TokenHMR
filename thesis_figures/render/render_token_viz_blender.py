#!/usr/bin/env python
"""Figure 4.6: token-displacement mesh maps, rendered in the Figure 3.1 (SMPL) style.

Replaces the earlier pyrender version, which rendered on a black background and put
fully saturated turbo colours on every vertex, so a body whose tokens barely move still
looked strongly coloured. Here the mesh keeps the neutral studio-grey material of
Figure 3.1 and the turbo colour is blended in proportionally to the displacement, so a
vertex that did not move stays grey. That is what makes the contrast between a
well-separated codebook and a redundant one readable at a glance.

Reads the geometry dumped by tokenization/render_token_viz.py (swap_geometry.npz), so
the expensive CPU decode/swap is not repeated here.

Run from the repo's tokenhmr folder in the thesis-HMR env:
    python -m thesis_figures.plot.render_token_viz_blender

Output: thesis/images/latent/token_visualization_contrast.pdf
"""
import json
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
import bpy
import mathutils

from repro import paths

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENHMR = os.path.dirname(HERE)
REPO = os.path.dirname(TOKENHMR)
NPZ = os.path.join(str(paths.TOKENIZER_OUT), "token_viz", "swap_geometry.npz")
OUTDIR = str(paths.figure_out_dir("latent"))
os.makedirs(OUTDIR, exist_ok=True)

PANELS = [("FSQ d4", "FSQ ($d = 4$)"), ("VQ d4", "Cosine ($d = 4$)")]
TURBO = matplotlib.colormaps["turbo"]
BASE_GREY = np.array([0.72, 0.73, 0.77])     # the Figure 3.1 studio-grey material
RES = 900
YFOV = 0.60

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "cm", "font.size": 10.5, "pdf.fonttype": 42})


def blend(t):
    """The mesh's albedo as a function of normalised displacement.

    Plain turbo, exactly as the pose-blend-shape panel of Figure 3.1 colours its mesh
    (render_smpl_figure.py), so the two figures read the same way and the colour bar is
    the ordinary turbo scale.
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return TURBO(t)[..., :3]


BLEND_CMAP = TURBO


def blended_colors(dv, vmax):
    """Turbo blended over the neutral base, with weight given by the displacement.

    A vertex that did not move keeps the plain grey material; one at vmax is fully
    turbo. Without this the whole body is saturated and both panels look alike.
    """
    out = blend(dv / max(vmax, 1e-9))
    return np.concatenate([out, np.ones((len(out), 1))], axis=1)


def add_area_light(name, pos, energy, size):
    d = bpy.data.lights.new(name=name, type="AREA"); d.energy = energy; d.size = size
    o = bpy.data.objects.new(name=name, object_data=d)
    bpy.context.collection.objects.link(o)
    o.location = pos
    o.rotation_euler = (mathutils.Vector((0, 0, 0)) - mathutils.Vector(pos)).to_track_quat("-Z", "Y").to_euler()


def crop(img):
    a = img[..., 3]
    ys, xs = np.where(a > 0)
    if len(ys) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def render(v, faces, colors, dist):
    v = v - 0.5 * (v.max(0) + v.min(0))
    bpy.ops.wm.read_factory_settings(use_empty=True)

    mesh = bpy.data.meshes.new("M")
    mesh.from_pydata(v.tolist(), [], faces.tolist()); mesh.update()
    obj = bpy.data.objects.new("M", mesh)
    bpy.context.collection.objects.link(obj)
    for p in mesh.polygons:
        p.use_smooth = True

    mat = bpy.data.materials.new(name="Mat"); mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.8
    bsdf.inputs["Metallic"].default_value = 0.0
    ca = mesh.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="POINT")
    ca.data.foreach_set("color", colors.astype(np.float32).flatten().tolist())
    attr = mat.node_tree.nodes.new(type="ShaderNodeAttribute"); attr.attribute_name = "Color"
    mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)

    cam_data = bpy.data.cameras.new("C"); cam_data.sensor_fit = "VERTICAL"; cam_data.angle_y = YFOV
    cam = bpy.data.objects.new("C", cam_data); bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam; cam.location = (0, 0, dist)

    add_area_light("Key", (-1.9, 1.9, 1.4), 350.0, 3.0)
    add_area_light("Fill", (2.3, 0.1, 1.3), 80.0, 5.0)
    add_area_light("Head", (0, 0, dist), 100.0, 1.0)

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 24
    sc.render.resolution_x = RES; sc.render.resolution_y = RES
    sc.render.film_transparent = True
    tmp = os.path.join(tempfile.gettempdir(), "tokviz_tmp.png")
    sc.render.filepath = tmp
    with open(os.devnull, "w") as f:
        os.dup2(f.fileno(), 1)
        bpy.ops.render.render(write_still=True)
    return crop((plt.imread(tmp) * 255).astype(np.uint8))


def main():
    z = np.load(NPZ, allow_pickle=True)
    faces = z["faces"]; vmax = float(z["vmax"])

    # one shared camera distance so both bodies are drawn at the same scale
    ext = []
    for lab, _ in PANELS:
        v = z[f"base_{lab}"]; c = 0.5 * (v.max(0) + v.min(0))
        ext.append(np.abs(v - c).max(0)[:2])
    dist = (max(max(e) for e in ext) * 1.1) / np.tan(YFOV / 2)

    imgs, titles = [], []
    for lab, nice in PANELS:
        dv = z[f"dv_{lab}"]
        imgs.append(render(z[f"base_{lab}"], faces, blended_colors(dv, vmax), dist))
        titles.append(f"{nice}\nmax $\\delta_v$ = {dv.max():.1f} mm")
        print(f"{lab:10s} rendered  max dv {dv.max():.2f} mm  mean {dv.mean():.2f} mm")

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9))
    fig.patch.set_facecolor("white")
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im); ax.axis("off")
        ax.set_title(t, fontsize=9.5, pad=5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.18, wspace=0.0)

    cax = fig.add_axes([0.30, 0.08, 0.40, 0.038])
    sm = plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=BLEND_CMAP); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(r"vertex displacement $\delta_v$ [mm]", fontsize=8.5, labelpad=2)
    cb.ax.tick_params(labelsize=8)

    out = os.path.join(OUTDIR, "token_visualization_contrast.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200, facecolor="white")
    fig.savefig(os.path.join(HERE, "token_viz_contrast_preview.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    appendix(z, faces, vmax, dist)


APP_ORDER = ["CNN", "Transformer tier1", "Transformer cosine", "Skeleton-masked",
             "VQ d4", "VQ d2", "FSQ d4", "FSQ d5"]
APP_NICE = {"CNN": "Conv ($\\ell_2$)", "Transformer tier1": "Transformer ($\\ell_2$)",
            "Transformer cosine": "Cosine $d$256", "Skeleton-masked": "Skeleton-masked",
            "VQ d4": "Cosine $d$4", "VQ d2": "Cosine $d$2",
            "FSQ d4": "FSQ $d$4", "FSQ d5": "FSQ $d$5"}


def appendix(z, faces, vmax, dist):
    """Figure B.2: the same swap for every tokenizer, on one shared colour scale."""
    have = [l for l in APP_ORDER if f"dv_{l}" in z]
    if len(have) < len(APP_ORDER):
        print(f"skipping appendix figure: geometry has only {len(have)}/{len(APP_ORDER)} tokenizers")
        return
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.4))
    fig.patch.set_facecolor("white")
    for ax, lab in zip(axes.ravel(), APP_ORDER):
        dv = z[f"dv_{lab}"]
        ax.imshow(render(z[f"base_{lab}"], faces, blended_colors(dv, vmax), dist))
        ax.axis("off")
        ax.set_title(f"{APP_NICE[lab]}\nmax $\\delta_v$ = {dv.max():.1f} mm", fontsize=9)
        print(f"  {lab:20s} max dv {dv.max():.2f} mm")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.12, wspace=0.0, hspace=0.16)
    cax = fig.add_axes([0.34, 0.055, 0.32, 0.022])
    sm = plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=BLEND_CMAP); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(r"vertex displacement $\delta_v$ [mm]", fontsize=8.5, labelpad=2)
    cb.ax.tick_params(labelsize=8)
    out = os.path.join(str(paths.figure_out_dir("appendix")), "tokenviz_all_tokenizers.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
