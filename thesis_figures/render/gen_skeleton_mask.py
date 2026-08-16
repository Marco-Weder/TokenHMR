#!/usr/bin/env python
"""Build Figure 3.8: the skeleton attention mask (fig:skeleton-mask).

One self-contained pipeline:
  1. run SMPL forward for a neutral A-pose, regress the 24 joints;
  2. render a faint front-view body (pyrender/EGL) -> thesis/images/tokenizer/skeleton_mesh.png;
  3. project the joints to that render's frame;
  4. build the 1-hop skeleton attention mask from utils/skeleton.py;
  5. emit the inline TikZ (left: numbered kinematic tree over the faint body;
     right: the 21x21 mask) -> thesis_figures/skeleton_mask_tikz.tex.

The .tex is pasted (inside a \\resizebox) into the fig:skeleton-mask environment in
materialsandmethods.tex. The mesh PNG is referenced by that TikZ. Joints follow the
SMPL numbering; the pelvis (0) and hands (22, 23) are drawn in grey (not tokenized).
Small display nudges declutter genuinely-adjacent joints; the mesh still shows the
true location.

Run with the project env (from the tokenhmr folder; offscreen render needs EGL):
    python -m thesis_figures.plot.gen_skeleton_mask

Outputs:
    <repo>/thesis/images/tokenizer/skeleton_mesh.png
    <repo>/tokenhmr/thesis_figures/skeleton_mask_tikz.tex
"""
import os, sys
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import torch
import trimesh
import pyrender
import smplx
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/tokenhmr/thesis_figures
TOKENHMR = os.path.dirname(HERE)                            # <repo>/tokenhmr
REPO = os.path.dirname(TOKENHMR)                            # <repo>
from tokenization.utils.skeleton import build_skeleton_attention_mask    # noqa: E402

from repro import paths

MODEL_PATH = os.path.join(TOKENHMR, "data", "body_models")
MESH_PNG = os.path.join(str(paths.figure_out_dir("tokenizer")), "skeleton_mesh.png")
TIKZ_OUT = os.path.join(HERE, "skeleton_mask_tikz.tex")
MESH_TEX_PATH = "images/tokenizer/skeleton_mesh.png"        # as referenced from the thesis


# ---------------------------------------------------------------- 1. SMPL forward
def smpl_joints_and_mesh():
    model = smplx.create(MODEL_PATH, model_type="smpl", gender="neutral", batch_size=1)
    bp = torch.zeros(23, 3)                 # A-pose: bring arms down from the T-pose
    bp[15, 2] = -0.98   # L_shoulder (rotate about z)
    bp[16, 2] =  0.98   # R_shoulder
    bp[17, 2] = -0.30   # L_elbow
    bp[18, 2] =  0.30   # R_elbow
    bp[0, 2]  =  0.06   # L_hip slight stance
    bp[1, 2]  = -0.06   # R_hip
    out = model(betas=torch.zeros(1, 10), global_orient=torch.zeros(1, 3),
                body_pose=bp.view(1, 69))
    V = out.vertices[0]
    Jreg = model.J_regressor
    if not torch.is_tensor(Jreg):
        Jreg = torch.tensor(np.array(Jreg.todense()), dtype=torch.float32)
    J = (Jreg @ V).detach().numpy()          # (24, 3)
    V = V.detach().numpy()
    c = J[0].copy()                          # centre on the pelvis
    return (V - c), (J - c), model.faces


# ---------------------------------------------------------------- 2. faint render
def raymond_lights():
    Ls = []
    for phi, theta in [(-np.pi / 6, np.pi / 3), (np.pi / 6, np.pi / 3), (0.0, -np.pi / 9)]:
        z = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        z /= np.linalg.norm(z)
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        M = np.eye(4); M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z
        Ls.append((pyrender.DirectionalLight(color=np.ones(3), intensity=2.2), M))
    return Ls


def render_mesh(V, J, faces):
    xh, yh = np.abs(V[:, 0]).max(), np.abs(V[:, 1]).max()
    Mx, My = xh * 1.10, yh * 1.06            # ortho half-extents (frame the body)
    H = 1400; W = int(round(H * Mx / My))
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.55, 0.55, 0.6])
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.62, 0.68, 0.82, 1.0],
                                             metallicFactor=0.0, roughnessFactor=0.85)
    scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(V, faces, process=False),
                                         material=mat, smooth=True))
    cam = pyrender.OrthographicCamera(xmag=Mx, ymag=My)
    cp = np.eye(4); cp[2, 3] = 2.0
    scene.add(cam, pose=cp)
    for l, mm in raymond_lights():
        scene.add(l, pose=mm)
    r = pyrender.OffscreenRenderer(W, H)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    os.makedirs(os.path.dirname(MESH_PNG), exist_ok=True)
    imageio.imwrite(MESH_PNG, color)
    # joint image fractions (0..1, y up) in the render's ortho frame
    u = (J[:, 0] + Mx) / (2 * Mx)
    v = (J[:, 1] + My) / (2 * My)
    return np.stack([u, v], 1)


# ---------------------------------------------------------------- 3. TikZ
def build_tikz(uv):
    m = build_skeleton_attention_mask(21, 1, True).int().tolist()
    Wc, Hc = 2.95, 6.0                       # mesh box in tikz cm (aspect matches render)
    # small display nudges (fraction units) to declutter dots; mesh shows true location
    nud = {3: (0.0, -0.002), 6: (0.0, -0.014), 9: (0.0, 0.019), 12: (0.0, -0.023),
           13: (0.020, -0.020), 14: (-0.020, -0.020), 15: (0.0, 0.033),
           16: (0.030, 0.0), 17: (-0.030, 0.0),
           7: (0.0, 0.022), 8: (0.0, 0.022), 10: (0.020, 0.0), 11: (-0.020, 0.0),
           22: (0.035, -0.020), 23: (-0.035, -0.020)}

    def xy(i):
        du, dv = nud.get(i, (0.0, 0.0))
        return (uv[i, 0] + du) * Wc, (uv[i, 1] + dv) * Hc

    parents24 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    blue = [(i, p) for i, p in enumerate(parents24) if 1 <= i <= 21 and 1 <= p <= 21]
    grey_stub = [(1, 0), (2, 0), (3, 0), (22, 20), (23, 21)]
    sib = [(1, 2), (1, 3), (2, 3)]
    orange, excluded = {1, 2, 3}, {0, 22, 23}

    L = ["% faint SMPL mesh (context)",
         f"\\node[opacity=0.42,inner sep=0] at ({Wc/2:.3f},{Hc/2:.3f}) "
         f"{{\\includegraphics[width={Wc:.3f}cm]{{{MESH_TEX_PATH}}}}};",
         "% grey stubs to excluded joints"]
    for i, j in grey_stub:
        a, bb = xy(i), xy(j); L.append(f"\\draw[stub] ({a[0]:.3f},{a[1]:.3f}) -- ({bb[0]:.3f},{bb[1]:.3f});")
    L.append("% pelvis-sibling links (dashed)")
    for i, j in sib:
        a, bb = xy(i), xy(j); L.append(f"\\draw[sib] ({a[0]:.3f},{a[1]:.3f}) -- ({bb[0]:.3f},{bb[1]:.3f});")
    L.append("% kinematic-tree bones (21 joints)")
    for i, j in blue:
        a, bb = xy(i), xy(j); L.append(f"\\draw[bone] ({a[0]:.3f},{a[1]:.3f}) -- ({bb[0]:.3f},{bb[1]:.3f});")
    L.append("% joints")
    for i in range(24):
        x, y = xy(i)
        sty = "jx" if i in excluded else ("js" if i in orange else "jt")
        L.append(f"\\node[{sty}] at ({x:.3f},{y:.3f}) {{{i}}};")
    left = "\n".join(L)

    s = 0.30
    R = ["% allowed cells"]
    for i in range(21):
        for j in range(21):
            if m[i][j]:
                x, y = j * s, -i * s
                R.append(f"\\fill[{'selfc' if i == j else 'allow'}] ({x:.3f},{y:.3f}) rectangle ({x+s:.3f},{y-s:.3f});")
    R += ["% grid + frame",
          f"\\draw[gridl] (0.000,0.000) grid[step={s}] ({21*s:.3f},{-21*s:.3f});",
          f"\\draw[thick] (0.000,0.000) rectangle ({21*s:.3f},{-21*s:.3f});",
          f"\\draw[sibbox] (0.000,0.000) rectangle ({3*s:.3f},{-3*s:.3f});",
          "% axis indices (SMPL joint numbers 1..21): columns on top, rows on the left"]
    for k in range(21):
        cx = k * s + s / 2
        R.append(f"\\node[axn,rotate=90,anchor=west] at ({cx:.3f},0.050) {{{k+1}}};")
    for k in range(21):
        cy = -k * s - s / 2
        R.append(f"\\node[axn,anchor=east] at (-0.070,{cy:.3f}) {{{k+1}}};")
    right = "\n".join(R)

    masksz = 21 * s
    tikz = r"""\begin{tikzpicture}[
  font=\small, >={Stealth[length=2.2mm]},
  bone/.style={draw=blue!45!black, line width=1.0pt},
  stub/.style={draw=black!35, line width=0.8pt},
  sib/.style={draw=orange!80!black, dashed, line width=0.9pt},
  jt/.style={circle, draw=blue!55!black, fill=blue!55, text=white, font=\fontsize{4.6}{5}\selectfont, inner sep=0pt, minimum size=0.30cm},
  js/.style={circle, draw=orange!85!black, fill=orange!80!black, text=white, font=\fontsize{4.6}{5}\selectfont, inner sep=0pt, minimum size=0.30cm},
  jx/.style={circle, draw=black!45, fill=black!22, text=black, font=\fontsize{4.6}{5}\selectfont, inner sep=0pt, minimum size=0.28cm},
  allow/.style={fill=blue!45},
  selfc/.style={fill=blue!75!black},
  gridl/.style={draw=black!15, line width=0.2pt},
  sibbox/.style={draw=orange!80!black, line width=1.3pt},
  panttl/.style={font=\small, text=black!90},
  lgt/.style={font=\scriptsize},
  axn/.style={font=\fontsize{5}{6}\selectfont, text=black!70, inner sep=0.5pt},
]
% ---------- LEFT: kinematic tree over SMPL body ----------
\begin{scope}[shift={(0,0)}]
__LEFT__
\end{scope}
% ---------- arrow ----------
\draw[->, line width=1pt] (3.40,2.60) -- node[above,font=\scriptsize]{build mask} node[below,font=\scriptsize]{$m_{ij}$} (4.15,2.60);
% ---------- RIGHT: attention mask (scaled down and shifted) ----------
\begin{scope}[shift={(5.00, 5.00)}, scale=0.76]
__RIGHT__
\node[lgt] at (__MHALF__,0.70) {attends to (key $j$)};
\node[lgt, rotate=90] at (-0.52,__MNEGHALF__) {query $i$};
\end{scope}
% ---------- panel titles ----------
\node[panttl] at (__MESHCX__,-0.55) {Kinematic tree on the body};
\node[panttl] at (__MASKCX__,-0.55) {Attention mask ($21\times21$)};
% ---------- legend (chained relative positioning -> equal 4mm gaps) ----------
\begin{scope}[shift={(-0.8,-1.5)}]
  \node[inner sep=0, minimum size=0.28cm, fill=blue!75!black] (b1) at (0,0) {};
  \node[lgt, right=1mm of b1, inner sep=0] (t1) {self};
  \node[inner sep=0, minimum size=0.28cm, fill=blue!45, right=4mm of t1] (b2) {};
  \node[lgt, right=1mm of b2, inner sep=0] (t2) {bone neighbour};
  \node[inner sep=0, minimum width=0.40cm, minimum height=0.28cm, right=4mm of t2] (b3) {};
  \draw[sib] (b3.west) -- (b3.east);
  \node[lgt, right=1mm of b3, inner sep=0] (t3) {pelvis siblings};
  \node[circle, draw=black!45, fill=black!22, minimum size=0.28cm, inner sep=0pt, right=4mm of t3] (b4) {};
  \node[lgt, right=1mm of b4, inner sep=0] (t4) {excluded};
  \node[inner sep=0, minimum size=0.28cm, draw=black, fill=white, right=4mm of t4] (b5) {};
  \node[lgt, right=1mm of b5, inner sep=0] (t5) {masked ($-\infty$)};
\end{scope}
\end{tikzpicture}"""
    return (tikz.replace("__LEFT__", left).replace("__RIGHT__", right)
            .replace("__MHALF__", f"{masksz/2:.3f}").replace("__MNEGHALF__", f"{-masksz/2:.3f}")
            .replace("__MESHCX__", f"{Wc/2:.3f}").replace("__MASKCX__", f"{5.0+0.76*masksz/2:.3f}"))


if __name__ == "__main__":
    V, J, faces = smpl_joints_and_mesh()
    uv = render_mesh(V, J, faces)
    open(TIKZ_OUT, "w").write(build_tikz(uv))
    print("wrote", MESH_PNG)
    print("wrote", TIKZ_OUT, "(paste inside the \\resizebox of fig:skeleton-mask)")
