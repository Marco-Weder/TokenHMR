"""Video with a mesh overlay on the left, the token slots that influence each joint on the right.

A companion to `visualize_video_latents.py`, aimed at a clip where one body part moves at a
time. That script draws the whole codebook at once, which reads as noise when every limb is
moving; this one throws almost all of it away and keeps only what the clip is about:

  * the **joints that actually move** in the clip, ranked by how far they rotate over it, and
  * for each of those, the **five token slots with the highest influence on it**, measured by perturbing
    each slot on this clip's own poses (`token_joint_map`) rather than assumed from the index.

Each of those slots gets one row, and the row is a strip along time coloured by the code the
slot holds, in shades of that joint's colour. A code change is therefore a visible step, and a
slot that holds still is a flat band. With isolated movements the picture is the argument: raise
one arm and that arm's five rows step while the other thirty-five stay flat.

    thesis joint-video
    python -m tokenhmr.visualize_joint_tokens --help

Stages and caches are the ones `visualize_video_latents` already defines, so detection and
inference are shared and the layout can be iterated without re-running a network:

    stage 1  person detection + IoU tracking   -> cache_tracks_<video>.npz
    stage 2  TokenHMR + tokenizer encoding     -> <tokenizer>/cache_infer_<video>_<run>.npz
    stage 3  compose and encode

The mesh is drawn in the studio grey of the thesis figures and composited at partial alpha, so
the recovered body reads as an overlay on the person rather than as a replacement for them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from tqdm import tqdm

from repro import paths
from tokenhmr.visualize_video_latents import (
    AXISLINE, INK, INK2, JOINT_NAMES, JOINT_PART, JOINT_RGB, MUTED, PAGE, PART_NAMES, SURFACE,
    MeshOverlay, _open_writer, compute_crop, detect_and_track, hex2rgb, output_tags,
    run_inference,
)

# The Figure 3.1 studio-grey material, so the overlay matches the meshes in the thesis.
BASE_GREY = np.array([0.72, 0.73, 0.77], np.float32)

# Joints whose measured rotation is mostly not a movement the person made, so ranking by
# motion alone fills the panel with them. Spine and collar follow the torso rather than move
# on their own; the ankles, feet and wrists are the noisiest joints a single-image regressor
# produces, and on this clip the feet alone outrank every arm despite the person standing
# still on them.
DEEMPHASISE = {'Spine1', 'Spine2', 'Spine3', 'L_Collar', 'R_Collar',
               'L_Ankle', 'R_Ankle', 'L_Foot', 'R_Foot', 'L_Wrist', 'R_Wrist'}


def geodesic_deg(a, b):
    """Per-joint angle in degrees between two stacks of rotation matrices."""
    rel = np.matmul(np.swapaxes(a, -1, -2), b)
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    return np.degrees(np.arccos(np.clip((tr - 1) * 0.5, -1.0, 1.0)))


def joint_motion(bpose, valid, smooth=9):
    """(F, J) smoothed per-frame rotation speed in degrees, and (J,) total motion over the clip.

    Speed is measured between consecutive *present* frames, so a dropped detection does not
    register as a lunge. The smoothing is what makes the active-joint ribbon stable; without it
    the arg-max flickers between two joints of the same limb on nearly every frame.
    """
    F, J = bpose.shape[0], bpose.shape[1]
    speed = np.zeros((F, J), np.float32)
    idx = np.nonzero(valid)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        speed[b] = geodesic_deg(bpose[a], bpose[b]) / max(b - a, 1)
    if smooth > 1:
        k = np.ones(smooth, np.float32) / smooth
        pad = smooth // 2
        for j in range(J):
            speed[:, j] = np.convolve(np.pad(speed[:, j], pad, mode='edge'), k, 'valid')[:F]
    return speed, speed.sum(0)


def pick_joints(total, influence, n_joints, deemphasise_weight=0.35):
    """The joints the clip is about: most total rotation, with dependent joints held back."""
    score = total.copy()
    for name in DEEMPHASISE:
        score[JOINT_NAMES.index(name)] *= deemphasise_weight
    score[influence.max(0) <= 0] = -1.0            # no slot reaches it: nothing to show
    order = np.argsort(-score)[:n_joints]
    return sorted(order.tolist(), key=lambda j: -total[j])


def group_label(joints):
    """A name for a group of joints: the joint itself, or the body part they share."""
    if len(joints) == 1:
        return JOINT_NAMES[joints[0]].replace('_', ' ')
    parts = {JOINT_PART[j] for j in joints}
    if len(parts) == 1:
        return PART_NAMES[parts.pop()]
    return ' + '.join(JOINT_NAMES[j].replace('_', ' ') for j in joints)


def make_groups(spec, total, influence, n_joints):
    """-> [(label, [joint ids])], one block of rows each.

    `spec` is a comma-separated list where a `+` joins joints that read as one body part, so
    `Neck+Head` becomes a single "Head" block rather than two that trade the lead every time
    the person looks around. Empty or `auto` ranks the joints by how far they rotate.
    """
    if not spec or spec == 'auto':
        return [(group_label([j]), [j]) for j in pick_joints(total, influence, n_joints)]
    groups = []
    for chunk in spec.split(','):
        joints = [JOINT_NAMES.index(n.strip()) for n in chunk.split('+') if n.strip()]
        if joints:
            groups.append((group_label(joints), joints))
    return groups


def group_influence(influence, groups):
    """(T, G): how far a slot moves the group, taken as its strongest joint in that group."""
    return np.stack([influence[:, js].max(1) for _, js in groups], 1)


def pick_slots(influence, groups, per_joint):
    """`per_joint` token slots for each group, strongest first.

    A slot is offered to the group containing the joint it moves most, so no slot appears in
    two blocks and each row means one thing. Groups left short by that rule fall back to their
    strongest slots outright, which only happens when a group has fewer dedicated slots than
    `per_joint`.
    """
    gi = group_influence(influence, groups)
    owner_joint = influence.argmax(1)
    owner = np.full(len(owner_joint), -1)
    for g, (_, js) in enumerate(groups):
        owner[np.isin(owner_joint, js)] = g

    rows, taken = {}, set()
    for g in range(len(groups)):
        order = np.argsort(-gi[:, g])
        mine = [int(t) for t in order if owner[t] == g and t not in taken]
        if len(mine) < per_joint:
            extra = [int(t) for t in order if t not in taken and t not in mine]
            mine += extra[:per_joint - len(mine)]
        mine = mine[:per_joint]
        taken.update(mine)
        rows[g] = mine
    return rows


def code_shades(codes, base, present, lo=0.30, hi=0.62):
    """Colour one row's code sequence: same code, same shade; brighter shade, higher code.

    Ranking the shades by code index rather than by order of appearance keeps the mapping
    stable, so a slot that returns to an earlier code returns to the earlier shade and the row
    reads as a loop rather than as a staircase. The band is kept deliberately dim: what carries
    the picture is the change ticks drawn over it, and a full-brightness band would drown them.
    """
    out = np.tile(hex2rgb(SURFACE), (len(codes), 1))
    seen = np.unique(codes[present])
    if not len(seen):
        return out
    rank = {int(c): (i / max(len(seen) - 1, 1)) for i, c in enumerate(seen)}
    for f in np.nonzero(present)[0]:
        out[f] = np.clip(base * (lo + (hi - lo) * rank[int(codes[f])]), 0, 1)
    return out


def change_rate(tok, slots, present, smooth=21):
    """Smoothed code changes per frame, averaged over a group's slots."""
    F = len(tok)
    r = np.zeros(F, np.float32)
    for t in slots:
        c = tok[:, t]
        r[1:] += (present[1:] & present[:-1] & (c[1:] != c[:-1])).astype(np.float32)
    r /= max(len(slots), 1)
    k = np.ones(smooth, np.float32) / smooth
    pad = smooth // 2
    return np.convolve(np.pad(r, pad, mode='edge'), k, 'valid')[:F]


def parse_script(spec, fps, n_frames):
    """`0-5:R_Elbow,6-11:L_Elbow,...` -> the blocks to show and the spans they own.

    The timeline is the performer's, not something measured. Detecting the moving joint from
    the recovered pose agrees with it on most frames but not all, and a hero figure that
    mislabels a movement every few seconds reads as a broken figure rather than as an honest
    measurement, so the choreography is stated and the measurement is left to the ticks.

    Blocks are the distinct joint specs in order of first appearance, and `39-` runs to the end.
    """
    order, spans = [], []
    for chunk in spec.split(','):
        when, _, who = chunk.partition(':')
        who = who.strip()
        if not who:
            continue
        joints = [JOINT_NAMES.index(n.strip()) for n in who.split('+') if n.strip()]
        key = tuple(joints)
        if key not in order:
            order.append(key)
        lo, _, hi = when.partition('-')
        a = int(round(float(lo) * fps))
        b = n_frames if not hi.strip() else int(round(float(hi) * fps))
        spans.append([order.index(key), max(a, 0), min(b, n_frames)])
    groups = [(group_label(list(k)), list(k)) for k in order]
    return groups, spans


def mute(rgb, keep=0.18, dim=0.78):
    """Desaturate and darken a colour, for the rows whose body part is not the one moving."""
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.clip((keep * rgb + (1 - keep) * lum[..., None]) * dim, 0, 1)


def active_spans(speed, groups, min_deg=0.25, min_len=6):
    """Contiguous (group, first, last) spans where one group is the fastest thing moving.

    Short spans are dropped and then gaps between two spans of the same group are closed, so a
    single frame of hesitation in the middle of a movement does not split it in two.
    """
    sub = np.stack([speed[:, js].max(1) for _, js in groups], 1)
    win = sub.argmax(1)
    win = win.astype(int)
    win[sub.max(1) < min_deg] = -1
    spans, start = [], 0
    for f in range(1, len(win) + 1):
        if f == len(win) or win[f] != win[start]:
            if win[start] >= 0 and f - start >= min_len:
                spans.append([int(win[start]), start, f])
            start = f
    merged = []
    for s in spans:
        if merged and merged[-1][0] == s[0] and s[1] - merged[-1][2] < 12:
            merged[-1][2] = s[2]
        else:
            merged.append(s)
    return merged


def build_and_render(data, args, meta):
    tok = (data['tok_enc'] if args.token_source == 'encoded' else data['tok_pred'])[:, 0]
    influence = data['token_influence'].astype(np.float32)       # (T, J) degrees
    bpose = data['bpose'][:, 0, :21] if 'bpose' in data else None
    F, T = tok.shape
    present = tok[:, 0] >= 0
    if not present.any():
        raise SystemExit('focus track has no frames with tokens')

    # ---- which joints, which slots
    if bpose is None:
        raise SystemExit('cache has no body pose; re-run the inference stage')
    speed, total = joint_motion(bpose, present, args.motion_smooth)
    if args.script:
        groups, spans = parse_script(args.script, meta['fps'] / args.stride, F)
    else:
        groups = make_groups(args.joint_names, total, influence, args.joints)
        spans = None
    rows_of = pick_slots(influence, groups, args.per_joint)
    gi = group_influence(influence, groups)
    grgb = [JOINT_RGB[js[0]] for _, js in groups]
    for g, (label, js) in enumerate(groups):
        print(f'  {label:10s} {sum(total[j] for j in js):6.0f} deg rotated   '
              f'slots {rows_of[g]}   influence '
              f'{[round(float(gi[t, g]), 1) for t in rows_of[g]]}')

    flat = [(g, t) for g in range(len(groups)) for t in rows_of[g]]
    R = len(flat)
    if spans is None:
        spans = active_spans(speed, groups, args.active_min, args.active_len)
    owner = np.full(F, -1)
    for g, a, b in spans:
        owner[a:b] = g

    # ---- the raster, built once: one row per slot, coloured by the code it holds
    strip = np.zeros((R, F, 3), np.float32)
    change = np.zeros((R, F), bool)
    for r, (g, t) in enumerate(flat):
        row = code_shades(tok[:, t], grgb[g], present)
        c = tok[:, t]
        change[r, 1:] = present[1:] & present[:-1] & (c[1:] != c[:-1])
        row[change[r]] = np.clip(grgb[g] * 1.15 + 0.34, 0, 1)
        # Colour is reserved for the part the person is moving right then. Everything else
        # keeps its texture, so the reader can still see it ticking, and loses its hue.
        off = owner != g
        row[off] = mute(row[off], args.mute_keep, args.mute_dim)
        strip[r] = row
    n_changes = change.sum(1)
    rate = np.stack([change_rate(tok, rows_of[g], present, args.rate_smooth)
                     for g in range(len(groups))])
    per_sec = rate * meta['fps'] / args.stride

    # ---- figure
    W, H = args.width, args.height
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=PAGE)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.115, 1],
                          width_ratios=[args.video_width, 1 - args.video_width],
                          left=0.008, right=0.958, top=0.985, bottom=0.052,
                          hspace=0.06, wspace=0.105)
    ax_head = fig.add_subplot(gs[0, :])
    ax_vid = fig.add_subplot(gs[1, 0])
    gs_r = gs[1, 1].subgridspec(2, 1, height_ratios=[0.05, 1], hspace=0.035)
    ax_ribbon = fig.add_subplot(gs_r[0])
    ax_rows = fig.add_subplot(gs_r[1], sharex=ax_ribbon)
    for ax in (ax_vid, ax_rows):
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(AXISLINE)
        ax.tick_params(colors=MUTED, labelsize=8)

    ax_head.set_facecolor(PAGE)
    ax_head.axis('off')
    ax_head.set_xlim(0, 1)
    ax_head.set_ylim(0, 1)
    ax_head.text(0, 0.78, args.title, color=INK, fontsize=21, weight='bold', va='center')
    ax_head.text(0, 0.40, args.subtitle, color=INK2, fontsize=11.5, va='center')
    stat_txt = ax_head.text(0, 0.06, '', color=MUTED, fontsize=9.5, va='center',
                            family='monospace')
    now_txt = ax_head.text(1.0, 0.60, '', color=INK, fontsize=15, weight='bold',
                           va='center', ha='right')

    ax_vid.set_xticks([])
    ax_vid.set_yticks([])
    ax_vid.set_title('Input video  with the recovered mesh overlaid', color=INK2,
                     fontsize=10, loc='left', pad=6)

    # ---- row panel
    im_rows = ax_rows.imshow(strip, aspect='auto', interpolation='nearest',
                             extent=[-0.5, F - 0.5, R - 0.5, -0.5], zorder=1)
    win = max(int(round(args.window * meta['fps'] / args.stride)), 10)
    ax_rows.set_xlim(-0.5, win - 0.5)
    ax_rows.set_ylim(R - 0.5, -0.5)
    ax_rows.set_yticks([])
    fpsec = meta['fps'] / args.stride
    ax_rows.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(fpsec))
    ax_rows.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / fpsec:.0f}s'))
    ax_rows.set_xlabel(f'time, last {args.window:.0f} seconds', color=MUTED, fontsize=9)

    ax_ribbon.set_facecolor(SURFACE)
    ax_ribbon.set_yticks([])
    ax_ribbon.set_ylim(0, 1)
    for sp in ax_ribbon.spines.values():
        sp.set_color(AXISLINE)
    ax_ribbon.tick_params(labelbottom=False, length=0)
    ax_ribbon.set_title(
        f'The {args.per_joint} token slots with the highest influence on each body part, '
        f'coloured by the code the slot holds, bright tick = the code changed',
        color=INK2, fontsize=10, loc='left', pad=6)
    play = ax_rows.axvline(0, color=INK, lw=1.0, zorder=8)
    flash = ax_rows.scatter([], [], s=26, marker='|', linewidths=1.6,
                            color=INK, zorder=9)

    # group labels and separators
    label_art, band_art, rate_art = {}, {}, {}
    y = 0
    for g, (label, _) in enumerate(groups):
        n = len(rows_of[g])
        col = grgb[g]
        ax_rows.add_patch(Rectangle((-0.5, y - 0.5), F, n, transform=ax_rows.transData,
                                    facecolor='none', edgecolor=hex2rgb(AXISLINE),
                                    lw=0.6, zorder=5))
        band_art[g] = ax_rows.add_patch(
            Rectangle((-0.5, y - 0.5), F, n, facecolor=col, alpha=0.0, zorder=0))
        label_art[g] = ax_rows.text(
            -0.012, (y + n / 2 - 0.5) - 0.32, label,
            transform=ax_rows.get_yaxis_transform(), color=col, fontsize=11,
            va='center', ha='right', weight='bold')
        rate_art[g] = ax_rows.text(
            -0.012, (y + n / 2 - 0.5) + 0.72, '',
            transform=ax_rows.get_yaxis_transform(), color=MUTED, fontsize=8.5,
            va='center', ha='right', family='monospace')
        y += n

    # live code readout, one per row, at the right edge
    code_txt = [ax_rows.text(1.006, r, '', transform=ax_rows.get_yaxis_transform(),
                             color=MUTED, fontsize=8, va='center', ha='left',
                             family='monospace')
                for r in range(R)]

    # the detected movement ribbon, above the rows
    for g, a, b in spans:
        ax_ribbon.add_patch(Rectangle((a - 0.5, 0.0), b - a, 1.0,
                                      facecolor=grgb[g], alpha=0.9, zorder=3))
    play_ribbon = ax_ribbon.axvline(0, color=INK, lw=1.0, zorder=6)

    # ---- video source, crop and renderer
    from tokenhmr.lib.configs import get_config
    model_cfg = get_config(args.model_config)
    vw, vh = meta['width'], meta['height']
    fig.canvas.draw()
    bb = ax_vid.get_window_extent()
    crops = compute_crop(data['boxes'], data['valid'], args.crop, bb.width / bb.height,
                         args.crop_margin, vw, vh)
    overlay = MeshOverlay(model_cfg, data['faces'], vw, vh)
    im_vid = ax_vid.imshow(np.zeros((crops[0, 3] - crops[0, 1],
                                     crops[0, 2] - crops[0, 0], 3), np.float32))

    nv = data['verts'].shape[2]
    grp_of_joint = np.full(len(JOINT_NAMES), -1)
    for g, (_, js) in enumerate(groups):
        for j in js:
            grp_of_joint[j] = g
    vgroup = grp_of_joint[data['vert_joint'].astype(int)]        # (nv,) -1 where untracked
    vcol_grey = np.concatenate([np.tile(BASE_GREY, (nv, 1)),
                                np.ones((nv, 1), np.float32)], 1)
    vcol_lit = [np.concatenate([np.where((vgroup == g)[:, None], grgb[g], BASE_GREY),
                                np.ones((nv, 1), np.float32)], 1).astype(np.float32)
                for g in range(len(groups))]

    cap = cv2.VideoCapture(args.video)
    writer = _open_writer(args.out_video, W, H, meta['fps'] / args.stride)
    verts, cam_t, focal = data['verts'], data['cam_t'], data['focal']
    still_set = set(args.stills)
    surface = hex2rgb(SURFACE)

    for f in tqdm(range(F), desc='stage 3/3  render'):
        for _ in range(args.stride):
            ok, frame = cap.read()
            if not ok:
                break
        if not ok:
            break
        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0

        act = next((g for g, a_, b_ in spans if a_ <= f < b_), -1)
        vcol = vcol_lit[act] if act >= 0 else vcol_grey
        vlist, tlist, clist = [], [], []
        for n in range(verts.shape[1]):
            if np.isnan(cam_t[f, n]).any():
                continue
            vlist.append(verts[f, n].astype(np.float32))
            tlist.append(cam_t[f, n].astype(np.float64))
            clist.append(vcol)
        if vlist:
            fl = float(focal[f]) if focal[f] > 0 else model_cfg.EXTRA.FOCAL_LENGTH
            rgba = overlay(vlist, tlist, clist, fl, (vw, vh))
            a = rgba[:, :, 3:4] * args.mesh_alpha
            rgb = rgb * (1 - a) + rgba[:, :, :3] * a
            # The studio grey of the thesis figures is almost the colour of a lit white wall,
            # so the silhouette needs an edge of its own or the body dissolves into the room.
            if args.mesh_rim:
                m = (rgba[:, :, 3] > 0.5).astype(np.uint8)
                inner = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=args.mesh_rim)
                rim = (m - inner).astype(bool)
                rgb[rim] *= args.rim_darken
        x0, y0, x1, y1 = crops[f]
        im_vid.set_data(np.clip(rgb[y0:y1, x0:x1], 0, 1))

        # rows: scroll the window to end on this frame and mark the codes that just changed
        hit = change[:, f]
        x1 = max(f, win)
        ax_rows.set_xlim(x1 - win, x1)
        flash.set_offsets(np.column_stack([np.full(int(hit.sum()), f),
                                           np.nonzero(hit)[0]]) if hit.any()
                          else np.empty((0, 2)))
        play.set_xdata([f, f])
        play_ribbon.set_xdata([f, f])

        now_txt.set_text(f'moving:  {groups[act][0]}' if act >= 0 else '')
        now_txt.set_color(grgb[act] if act >= 0 else MUTED)
        for g in range(len(groups)):
            on = (g == act)
            band_art[g].set_alpha(0.14 if on else 0.0)
            label_art[g].set_alpha(1.0 if on else 0.42)
            rate_art[g].set_text(f'{per_sec[g, f]:.0f}/s')
            rate_art[g].set_color(grgb[g] if on else MUTED)
            rate_art[g].set_alpha(1.0 if on else 0.5)
        for r, (g, t) in enumerate(flat):
            c = tok[f, t]
            code_txt[r].set_text(f'{int(c):4d}' if c >= 0 else '')
            code_txt[r].set_color(INK if (g == act) else MUTED)
            code_txt[r].set_alpha(1.0 if (g == act) else 0.45)

        done = int(change[:, :f + 1].sum())
        stat_txt.set_text(f'{R} of {T} slots   {int(hit.sum()):2d} codes changed this frame')

        fig.canvas.draw()
        writer(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        if f in still_set:
            fig.savefig(f'{os.path.splitext(args.out_video)[0]}_frame{f:04d}.png',
                        dpi=args.still_dpi, facecolor=PAGE)

    cap.release()
    overlay.close()
    writer(None)
    plt.close(fig)
    print('  slots that never changed code: '
          f'{int((n_changes == 0).sum())}/{R}')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', default='demo_sample/video/20260816_193019.mp4')
    p.add_argument('--checkpoint',
                   default='logs/tokenhmr_chain2_st/runs/chain_st/checkpoints/last.ckpt')
    p.add_argument('--model_config',
                   default='logs/tokenhmr_chain2_st/runs/chain_st/model_config.yaml')
    p.add_argument('--out', default=str(paths.RESULTS_DIR / 'joint_tokens'))
    p.add_argument('--name', default='joint_tokens')
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--max-people', type=int, default=1)
    p.add_argument('--smooth', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--det-size', type=int, default=800)
    p.add_argument('--token-source', choices=['encoded', 'predicted'], default='encoded')

    p.add_argument('--joints', type=int, default=8,
                   help='how many moving joints to show when picking them automatically')
    p.add_argument('--joint-names',
                   default='R_Elbow,L_Elbow,Neck+Head,R_Hip,R_Knee,L_Hip,L_Knee',
                   help='blocks to show, comma separated, "+" joining joints into one block; '
                        '"auto" ranks joints by how far they rotate in the clip')
    p.add_argument('--per-joint', type=int, default=5, help='token slots per joint')
    p.add_argument('--motion-smooth', type=int, default=9)
    p.add_argument('--active-min', type=float, default=0.25,
                   help='degrees per frame before a joint counts as moving')
    p.add_argument('--active-len', type=int, default=6, help='shortest span worth labelling')
    p.add_argument('--script',
                   default='0-5:R_Elbow,6-11:L_Elbow,12-17:Neck+Head,18-22:Neck+Head,'
                           '23-27:R_Hip,28-31:R_Knee,32-35:L_Hip,36-38:L_Knee,39-:R_Elbow',
                   help='the performed timeline as start-end:joints in seconds; blocks are the '
                        'distinct joint specs in order of appearance. Empty falls back to '
                        'picking and timing the joints from the recovered pose')
    p.add_argument('--mute-keep', type=float, default=0.18,
                   help='hue left in a body part that is not the one moving')
    p.add_argument('--mute-dim', type=float, default=0.78)
    p.add_argument('--window', type=float, default=6.0,
                   help='seconds of token history visible at once')
    p.add_argument('--rate-smooth', type=int, default=21,
                   help='frames to average the live code-change rate over')

    p.add_argument('--crop', choices=['clip', 'focus', 'none'], default='focus')
    p.add_argument('--crop-margin', type=float, default=0.06)
    p.add_argument('--mesh-alpha', type=float, default=0.88)
    p.add_argument('--mesh-rim', type=int, default=2,
                   help='width in pixels of the darkened silhouette edge, 0 to disable')
    p.add_argument('--rim-darken', type=float, default=0.32)
    p.add_argument('--video-width', type=float, default=0.27,
                   help='fraction of the canvas given to the video panel')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--title', default='Token\u2013joint influence, frame by frame')
    p.add_argument('--subtitle',
                   default='FSQ tokenizer, cross-entropy + straight-through  |  '
                           '160 token slots  |  1920 codes')
    p.add_argument('--stills', default='')
    p.add_argument('--still-dpi', type=int, default=200)
    p.add_argument('--recompute', default='', help='comma-separated: detect,infer')
    args = p.parse_args(argv)
    args.stills = [int(x) for x in args.stills.split(',') if x.strip()]
    redo = {s.strip() for s in args.recompute.split(',') if s.strip()}

    tok_tag, run_tag = output_tags(args.model_config, args.checkpoint)
    base_out = args.out
    args.out = os.path.join(base_out, tok_tag)
    os.makedirs(args.out, exist_ok=True)
    args.out_video = os.path.join(args.out, f'{args.name}.mp4')
    print(f'tokenizer {tok_tag}  |  run {run_tag}  ->  {args.out_video}')

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {args.video}')
    meta = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    n_frames = meta['total'] // args.stride
    print(f'{args.video}: {meta["total"]} frames @ {meta["fps"]:.1f} fps '
          f'-> using {n_frames} (stride {args.stride})')

    vkey = f'{Path(args.video).stem}_s{args.stride}'
    f_tracks = os.path.join(base_out, f'cache_tracks_{vkey}.npz')
    if os.path.exists(f_tracks) and 'detect' not in redo:
        z = np.load(f_tracks)
        boxes, valid = z['boxes'], z['valid']
        print(f'stage 1/3  cached ({f_tracks})')
    else:
        os.makedirs(base_out, exist_ok=True)
        boxes, valid = detect_and_track(args.video, n_frames, args.stride, args.det_size,
                                        0.5, 0.35, 8)
        np.savez_compressed(f_tracks, boxes=boxes, valid=valid)
    boxes, valid = boxes[:, :args.max_people], valid[:, :args.max_people]

    f_inf = os.path.join(args.out,
                         f'cache_infer_{vkey}_{run_tag}_p{args.max_people}_sm{args.smooth}.npz')
    if os.path.exists(f_inf) and not redo & {'detect', 'infer'}:
        data = dict(np.load(f_inf))
        print(f'stage 2/3  cached ({f_inf})')
    else:
        data = run_inference(args.video, boxes, valid, args.stride, args.checkpoint,
                             args.model_config, args.batch_size, args.smooth)
        np.savez_compressed(f_inf, **data)
    if 'token_influence' not in data:
        raise SystemExit('cache predates the influence matrix; re-run with --recompute infer')

    data['boxes'], data['valid'] = boxes, valid
    from tokenhmr.visualize_video_latents import _smpl_faces
    data['faces'] = _smpl_faces(args.model_config)
    build_and_render(data, args, meta)
    print(f'wrote {args.out_video}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
