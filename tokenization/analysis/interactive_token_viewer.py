"""Interactive token editor for the pose tokenizer.

Opens a little web page where you can pick one of the 160 tokens, replace it with any code value,
and immediately see the effect on the decoded body. Two 3D meshes are shown side by side:

  * LEFT  - the neutral body colored by how far each vertex moved (the same per-vertex error map as
            in the montage), so you can see WHERE the token acts.
  * RIGHT - the actual resulting pose after the swap (a plain-colored body).

Both meshes are drag-to-rotate (turntable around the vertical/human axis) and zoomable. The default
is the neutral pose (no change), so you start from the rest pose and edit one token at a time.

This is a headless machine, so it runs as a small web server (Plotly + Dash) that you open in a
browser. In VS Code the port is forwarded automatically; otherwise open the printed URL.

    python interactive_token_viewer.py --ckpt <best_net.pth or a downstream last-v2.ckpt> [--port 8050]
"""

import os
import sys
import argparse
import random

import numpy as np
import torch

# This script lives next to the other tokenizer tools.
from tokenization.analysis.visualize_token_effects import load_net, neutral_pose_tokens, decode_to_mesh, DEVICE
from tokenization.models.transformer_pose_vqvae import body_model

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

# --------------------------------------------------------------------------- #
# Globals filled in by init() so the Dash callbacks can reach them.           #
# --------------------------------------------------------------------------- #
NET = None
NEUTRAL_IDX = None          # (1, T) token ids of the neutral pose
NEUTRAL_VERTS = None        # (V, 3) neutral-pose vertices on the device
NEUTRAL_VERTS_NP = None     # same, on cpu/numpy (reused for the left mesh)
NEUTRAL_CODES = None        # python list, the neutral code at each token position
FACES = None                # (F, 3) int mesh faces
NB_CODE = None              # codebook size
T = None                    # number of tokens (usually 160)
CENTER = None               # shared centring point so overlaid meshes stay aligned


@torch.no_grad()
def init(ckpt):
    """Load the frozen tokenizer and cache the neutral pose."""
    global NET, NEUTRAL_IDX, NEUTRAL_VERTS, NEUTRAL_VERTS_NP, NEUTRAL_CODES, FACES, NB_CODE, T, CENTER
    NET = load_net(ckpt)[0]
    NEUTRAL_IDX = neutral_pose_tokens(NET)                       # (1, T)
    NEUTRAL_VERTS = decode_to_mesh(NET, NEUTRAL_IDX)[0][0]       # (V, 3)
    NEUTRAL_VERTS_NP = NEUTRAL_VERTS.cpu().numpy()
    NEUTRAL_CODES = NEUTRAL_IDX[0].cpu().tolist()
    FACES = np.asarray(body_model.faces).astype(np.int64)
    NB_CODE = int(NET.quantizer.nb_code)
    T = int(NEUTRAL_IDX.shape[1])
    # Shared centring point (neutral centroid in z-up coords) so the ghost and swapped meshes align.
    oriented = np.stack([NEUTRAL_VERTS_NP[:, 0], -NEUTRAL_VERTS_NP[:, 2], NEUTRAL_VERTS_NP[:, 1]], axis=1)
    CENTER = oriented.mean(axis=0)


@torch.no_grad()
def swap_and_decode(token, code):
    """Replace one token and return (swapped vertices, per-vertex displacement in mm)."""
    seq = NEUTRAL_IDX.clone()
    seq[0, token] = int(code)
    verts = decode_to_mesh(NET, seq)[0][0]                       # (V, 3)
    disp = (torch.linalg.norm(verts - NEUTRAL_VERTS, dim=-1) * 1000.0).cpu().numpy()
    return verts.cpu().numpy(), disp


# --------------------------------------------------------------------------- #
# Plotly helpers                                                              #
# --------------------------------------------------------------------------- #
# Plotly's turntable drag rotates about the world z-axis, so we put the body's vertical axis on z
# (SMPL is y-up). Then a horizontal drag spins the body about its own (human) axis.
SCENE = dict(
    xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
    aspectmode='data', dragmode='turntable', bgcolor='rgba(0,0,0,0)',
)
# Front view (body vertical on z); a horizontal drag then spins around the human axis.
CAMERA = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=-1.9, z=0.15))


def _orient(v):
    """SMPL body is y-up; rotate +90 deg about x so vertical is z (Plotly's up), share the centre."""
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    return v - CENTER


# Bright, even lighting from the front-upper-left (camera sits at -y, so the light is on the same
# side as the viewer). High ambient keeps every part visible; low specular avoids harsh hot spots.
LIGHTING = dict(ambient=0.75, diffuse=0.55, specular=0.1, roughness=0.5, fresnel=0.1)
LIGHTPOS = dict(x=-1.5, y=-3.0, z=2.0)


def _mesh(verts, intensity=None, color=None, cmax=None, opacity=1.0):
    verts = _orient(verts)
    kw = dict(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=FACES[:, 0], j=FACES[:, 1], k=FACES[:, 2],
        flatshading=False, hoverinfo='skip', opacity=opacity,
        lighting=LIGHTING, lightposition=LIGHTPOS,
    )
    if intensity is not None:
        kw.update(intensity=intensity, intensitymode='vertex', colorscale='Turbo',
                  cmin=0.0, cmax=cmax, colorbar=dict(title='mm', thickness=12, len=0.7))
    else:
        kw.update(color=color)
    return go.Mesh3d(**kw)


def _figure(meshes, camera):
    # We set scene.camera to the CURRENT camera (captured from the user's drag) on every update, so
    # re-rendering never moves the view: the specified camera equals where the user already is.
    fig = go.Figure(data=meshes if isinstance(meshes, list) else [meshes])
    fig.update_layout(scene=dict(SCENE, camera=camera or CAMERA), margin=dict(l=0, r=0, t=0, b=0),
                      height=560, uirevision='keep', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig


def build(token, code, camera=None):
    """Return (left figure, right figure, info string) for a given token/code and camera."""
    verts, disp = swap_and_decode(token, code)
    cmax = max(float(disp.max()), 1.0)                           # keep small changes visible
    left = _figure(_mesh(NEUTRAL_VERTS_NP, intensity=disp, cmax=cmax), camera)
    # Right: before (neutral, blue) and after (swapped, orange) overlaid and semi-transparent, so you
    # can see exactly which body parts moved and where they moved to.
    right = _figure([_mesh(NEUTRAL_VERTS_NP, color='#4C78A8', opacity=0.55),
                     _mesh(verts, color='#F58518', opacity=0.55)], camera)
    neutral = NEUTRAL_CODES[token]
    changed = int(code) != int(neutral)
    info = (f"Token {token}   ·   neutral code {neutral} → replacement {int(code)}"
            f"{'' if changed else '   (no change)'}   ·   "
            f"mean {disp.mean():.1f} mm  ·  max {disp.max():.0f} mm")
    return left, right, info


# --------------------------------------------------------------------------- #
# Dash app                                                                    #
# --------------------------------------------------------------------------- #
INDEX = """<!DOCTYPE html>
<html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  body{margin:0;background:#eef1f7;color:#232733;
       font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;}
  .wrap{max-width:1360px;margin:0 auto;padding:0 20px 34px;}
  .card{background:#fff;border-radius:16px;box-shadow:0 2px 12px rgba(30,45,90,.07);}
  .btn{border:none;border-radius:9px;padding:9px 15px;font-size:13px;font-weight:600;
       cursor:pointer;color:#fff;transition:filter .15s;}
  .btn:hover{filter:brightness(1.08);}
  .lab{font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
       color:#8a92a6;margin-bottom:2px;}
  .ptitle{font-size:13px;font-weight:600;color:#4a5163;text-align:center;padding:12px 0 2px;}
  .rc-slider-track{background:#5b7cfa !important;height:5px !important;}
  .rc-slider-rail{height:5px !important;background:#e3e7f0 !important;}
  .rc-slider-handle{border:2px solid #5b7cfa !important;width:16px !important;height:16px !important;
                    margin-top:-6px !important;opacity:1 !important;box-shadow:0 1px 4px rgba(0,0,0,.2) !important;}
  .rc-slider-tooltip-inner{background:#2b2f3a !important;border-radius:6px !important;padding:3px 8px !important;}
</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""


def make_app():
    app = Dash(__name__)
    app.title = 'Pose token editor'
    app.index_string = INDEX

    l0, r0, i0 = build(0, NEUTRAL_CODES[0], camera=CAMERA)
    gcfg = {'displayModeBar': False, 'scrollZoom': True}
    tip = {'placement': 'top', 'always_visible': True}

    def panel(title, gid, fig):
        return html.Div([html.Div(title, className='ptitle'),
                         dcc.Graph(id=gid, figure=fig, config=gcfg, style={'height': '560px'})],
                        className='card', style={'flex': 1, 'padding': '6px 6px 12px'})

    app.layout = html.Div(className='wrap', children=[
        html.Div([
            html.H2('Pose token editor', style={'margin': 0, 'fontWeight': 700}),
            html.Div('Replace a single token and watch the decoded body change in real time.',
                     style={'color': '#7b8296', 'fontSize': '14px', 'marginTop': '4px'}),
        ], style={'textAlign': 'center', 'padding': '26px 0 16px'}),

        html.Div(className='card', style={'display': 'flex', 'alignItems': 'flex-end', 'gap': '22px',
                                          'padding': '18px 24px', 'marginBottom': '14px'}, children=[
            html.Div([html.Div('Token position', className='lab'),
                      dcc.Slider(min=0, max=T - 1, step=1, value=0, id='token', marks=None, tooltip=tip)],
                     style={'flex': 2}),
            html.Div([html.Div('Replacement code', className='lab'),
                      dcc.Slider(min=0, max=NB_CODE - 1, step=1, value=NEUTRAL_CODES[0], id='code',
                                 marks=None, tooltip=tip)], style={'flex': 3}),
            html.Button('🎲  Random', id='rand', n_clicks=0, className='btn', style={'background': '#5b7cfa'}),
            html.Button('↺  Reset', id='reset', n_clicks=0, className='btn', style={'background': '#98a0b3'}),
        ]),

        html.Div(id='info', className='card', children=i0,
                 style={'textAlign': 'center', 'fontFamily': 'ui-monospace,SFMono-Regular,Menlo,monospace',
                        'fontSize': '13.5px', 'color': '#3b4150', 'padding': '11px', 'marginBottom': '14px'}),

        html.Div([panel('Vertex displacement (mm)', 'left', l0),
                  panel('Before (blue)  vs  after (orange)', 'right', r0)],
                 style={'display': 'flex', 'gap': '14px'}),

        dcc.Store(id='cam', data=CAMERA),
    ])

    # Remember the camera wherever the user drags/zooms either graph, so a swap never moves the view.
    @app.callback(Output('cam', 'data'),
                  Input('left', 'relayoutData'), Input('right', 'relayoutData'),
                  prevent_initial_call=True)
    def _capture_cam(left_rl, right_rl):
        rl = {'left': left_rl, 'right': right_rl}.get(ctx.triggered_id)
        if rl and 'scene.camera' in rl:
            return rl['scene.camera']
        return no_update

    # Picking a new token starts from "no change"; the buttons set the replacement code.
    @app.callback(Output('code', 'value', allow_duplicate=True),
                  Input('token', 'value'), prevent_initial_call=True)
    def _reset_on_token(token):
        return NEUTRAL_CODES[int(token)]

    @app.callback(Output('code', 'value', allow_duplicate=True),
                  Input('rand', 'n_clicks'), prevent_initial_call=True)
    def _random_code(_n):
        return random.randrange(NB_CODE)

    @app.callback(Output('code', 'value', allow_duplicate=True),
                  Input('reset', 'n_clicks'), State('token', 'value'), prevent_initial_call=True)
    def _reset_code(_n, token):
        return NEUTRAL_CODES[int(token)]

    # Any token/code change re-decodes; the stored camera is re-applied so the view stays put.
    @app.callback(Output('left', 'figure'), Output('right', 'figure'), Output('info', 'children'),
                  Input('token', 'value'), Input('code', 'value'), State('cam', 'data'),
                  prevent_initial_call=True)
    def _update(token, code, cam):
        return build(int(token), int(code), camera=cam)

    return app


def main():
    parser = argparse.ArgumentParser(description='Interactive token editor (web app)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='tokenizer best_net.pth OR a downstream .ckpt (auto-redirects to its tokenizer)')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8050)
    args = parser.parse_args()

    print('Loading tokenizer...')
    init(args.ckpt)
    print(f'Loaded: {T} tokens, codebook size {NB_CODE}.')
    print(f'\n  Open  http://{args.host}:{args.port}  in your browser (VS Code forwards the port).\n')
    make_app().run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
