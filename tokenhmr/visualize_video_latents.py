"""Side-by-side video: human mesh recovery on the left, the live token/latent code on the right.

Built for the thesis title slide. Left panel = the input video with the recovered SMPL meshes
overlaid; right panels = what the tokenizer is doing on that same frame, i.e. *which codes are
used* to represent the pose currently on screen.

The right-hand panels are:

  ATLAS       The whole FSQ codebook drawn as an exact 2-D lattice, with the codes used by the
              current pose lit up. This is not a projection: FSQ quantizes each latent channel
              independently, so with levels [L0, L1, L2, L3] the code index factorizes as
                  code = d0 + L0*d1 + L0*L1*d2 + L0*L1*L2*d3.
              For the [8, 8, 6, 5] tokenizer that means `code % 64 = d0 + 8*d1` and
              `code // 64 = d2 + 6*d3`, so `arange(1920).reshape(30, 64)` *is* the lattice --
              every cell is one real code, columns run over the two 8-level channels and rows
              over the 6- and 5-level ones. Lit cells fade out over the next few frames, which
              turns the panel into a live trace of the code stream.
              (For a non-FSQ tokenizer there is no such factorization; the panel then falls back
              to a plain rectangular reshape of the code index, which is an arbitrary ordering.)

  RASTER      The 160 token slots as rows and time as columns -- a "barcode" of the motion.
              A row is bright on the frames where that token's code changed, so stable body
              parts read as dark bands and moving ones flicker. Rows are grouped into one
              labelled block per joint (21 of them, grouped into 8 body parts; --granularity
              part collapses each part to one block), so you can read off which individual joint the
              tokenizer is spending its code changes on.

  TRAJECTORY  The pose's pre-quantization latent (160 x code_dim, flattened) projected to 2-D
              with PCA fit over the whole clip, drawn as a path with a fading comet trail.
              Shows the clip as a curve through the tokenizer's latent space.

Tokens are colored by the joint they control, measured (not assumed) by perturbing each token's
code on this clip's own poses and recording which joints rotate -- see `token_joint_map`. The
recovered mesh is painted per joint with the same colors, so a lit cell in the atlas, a labelled
row block, and a body part in the video all share one color.

Run from the repo root (the dir holding the `tokenhmr/` package, `data/` and `logs/`):

    python tokenhmr/visualize_video_latents.py \
        --video demo_sample/video/gymnasts.mp4 \
        --checkpoint logs/tokenhmr_fsq_ce_only/runs/tokenhmr_fsq_ce_only_0/checkpoints/epoch=9-step=600000.ckpt \
        --model_config logs/tokenhmr_fsq_ce_only/runs/tokenhmr_fsq_ce_only_0/model_config.yaml \
        --out results/title_slide

Results land in a per-tokenizer subfolder of --out, tagged with the tokenizer's training date
(the same tokenizer NAME can cover two different trainings), and are named after the stage-2 run,
so two runs sharing one tokenizer sit side by side:

    results/title_slide/
        cache_tracks_gymnasts_s1.npz                  <- detection: video-only, shared
        transformer_fsq_13-05-2026/
            fsq.mp4              fsq_summary.json          <- FSQ, pose loss only
            fsq_ce_only.mp4      fsq_ce_only_summary.json  <- same tokenizer, token CE
            cache_infer_gymnasts_s1_fsq.npz

Pass --no-subdir to write straight into --out, or --name to override the basename.

Work is staged and cached, so restyling never re-runs a network:
    stage 1  person detection + IoU tracking      -> cache_tracks_<video>.npz
    stage 2  TokenHMR + tokenizer encoding        -> cache_infer_<video>_<name>.npz
    stage 3  panel rendering -> mp4 / stills      (cheap, iterate here)
Delete a cache file (or pass --recompute) to redo a stage.

The GPU is shared with training runs, so the two model stages never live at the same time: the
detector is freed before TokenHMR is loaded. `--det-size` trades detection accuracy for memory.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------------------------------------- #
# Chart chrome, on a dark surface.                                             #
# --------------------------------------------------------------------------- #
INK        = '#ffffff'
INK2       = '#c3c2b7'
MUTED      = '#898781'
SURFACE    = '#1a1a19'
PAGE       = '#0d0d0d'
GRIDLINE   = '#2c2c2a'
AXISLINE   = '#383835'

# Eight body PARTS, each with its own hue; left and right limbs are separate parts, and the head,
# the hands and the feet are parts in their own right, so none of them shares a color with anything
# else. Within a part, a lightness step separates its joints (out along the limb, or left before
# right for the two-joint parts) -- identity never rests on that step, because every raster row
# block is labelled and the mesh is anatomical.
#
# These eight hues are NOT the skill's documented default palette: that palette's eight slots
# cannot clear the all-pairs gates (its aqua and magenta sit at CVD dE 1.6), and with the whole
# body on screen at once every pair is an adjacent pair. They were optimized instead -- 8 hues in
# OKLCH, lightness free inside the dark band, chroma maximized in gamut, maximizing the worst
# all-pairs CVD separation -- and then held to the full gate via the skill's validator on the dark
# surface: lightness band PASS, chroma floor PASS, all-pairs CVD dE 10.4 PASS, all-pairs
# normal-vision dE 15.7 PASS, contrast >= 3:1 PASS.
# The assignment was then chosen to put the largest distances on the pairs a reader actually
# compares: L arm vs R arm and L leg vs R leg land at CVD dE 26.6. It falls out warm-left /
# cool-right for both limbs, which is worth saying out loud in a caption.
PART_GROUPS = [
    # name,     joints (in reading order),                     shades dark -> light
    ('Spine',  ['Spine1', 'Spine2', 'Spine3'],      ['#09819c', '#1499b9', '#3bb1d1']),
    ('Head',   ['Neck', 'Head'],                    ['#ee288d', '#ff5ba3']),
    ('L arm',  ['L_Collar', 'L_Shoulder', 'L_Elbow'], ['#818107', '#989812', '#afb037']),
    ('R arm',  ['R_Collar', 'R_Shoulder', 'R_Elbow'], ['#90019d', '#af11be', '#c838d7']),
    ('Hands',  ['L_Wrist', 'R_Wrist'],              ['#3e60fe', '#2a40ee']),
    ('L leg',  ['L_Hip', 'L_Knee', 'L_Ankle'],      ['#ab0008', '#ce1114', '#e93830']),
    ('R leg',  ['R_Hip', 'R_Knee', 'R_Ankle'],      ['#6c69e2', '#8181fc', '#9a9fff']),
    ('Feet',   ['L_Foot', 'R_Foot'],                ['#218562', '#046e4e']),
]
PART_NAMES  = [g[0] for g in PART_GROUPS]
PART_COLORS = [g[2][len(g[2]) // 2] for g in PART_GROUPS]        # mid shade = the part's hue

# SMPL body-joint names in body_pose order (model joint j == SMPL joint j+1).
JOINT_NAMES = [
    'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle',
    'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head',
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
]


def hex2rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)], dtype=np.float32)


JOINT_PART = np.zeros(len(JOINT_NAMES), int)     # joint -> which of the 8 parts
_JOINT_HEX = [None] * len(JOINT_NAMES)
JOINT_ORDER = []                                 # raster reading order: part by part
for _p, (_nm, _joints, _shades) in enumerate(PART_GROUPS):
    for _j, _shade in zip(_joints, _shades):
        _i = JOINT_NAMES.index(_j)
        JOINT_PART[_i], _JOINT_HEX[_i] = _p, _shade
        JOINT_ORDER.append(_i)
assert len(JOINT_ORDER) == len(JOINT_NAMES), 'every joint must belong to exactly one part'

JOINT_RGB = np.stack([hex2rgb(h) for h in _JOINT_HEX])
PART_RGB = np.stack([hex2rgb(c) for c in PART_COLORS])


# --------------------------------------------------------------------------- #
# Stage 1 -- person detection and greedy IoU tracking                          #
# --------------------------------------------------------------------------- #
def _iou(a, b):
    x0 = np.maximum(a[0], b[:, 0]); y0 = np.maximum(a[1], b[:, 1])
    x1 = np.minimum(a[2], b[:, 2]); y1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def detect_and_track(video, n_frames, stride, det_size, score_thr, iou_thr, max_gap):
    """Per-frame person boxes linked into tracks by greedy IoU matching.

    Returns (boxes, valid): boxes (F, N, 4) and valid (F, N) bool, with N = number of tracks.
    Full tracking (PHALP) is overkill here -- the panels only need a stable identity per person
    so the mesh colors and the token stream stay attached to the same body.
    """
    from pathlib import Path
    import detectron2.data.transforms as T
    from detectron2.config import LazyConfig
    from tokenhmr.lib.utils.utils_detectron2 import DefaultPredictor_Lazy
    import tokenhmr.lib as lib
    cfg_path = Path(lib.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    d2_cfg = LazyConfig.load(str(cfg_path))
    d2_cfg.train.init_checkpoint = ('https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/'
                                    'cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl')
    for i in range(3):
        d2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(d2_cfg)
    # ViTDet-H at its native 1024 needs ~9 GB; shrinking the shortest edge keeps it inside a
    # GPU shared with a training run. Boxes are mapped back to full resolution by detectron2.
    detector.aug = T.ResizeShortestEdge([det_size, det_size], det_size)

    cap = cv2.VideoCapture(video)
    per_frame = []
    for f in tqdm(range(n_frames), desc='stage 1/3  detect'):
        for _ in range(stride):
            ok, frame = cap.read()
            if not ok:
                break
        if not ok:
            break
        with torch.autocast('cuda', dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
            inst = detector(frame)['instances']
        keep = (inst.pred_classes == 0) & (inst.scores > score_thr)
        per_frame.append(inst.pred_boxes.tensor[keep].detach().cpu().numpy().astype(np.float32))
    cap.release()

    del detector
    torch.cuda.empty_cache()

    tracks = []          # each: dict(box=last box, last=frame idx, hits=[(f, box), ...])
    for f, boxes in enumerate(per_frame):
        free = list(range(len(boxes)))
        order = sorted(range(len(tracks)), key=lambda t: -len(tracks[t]['hits']))
        for t in order:
            if not free or f - tracks[t]['last'] > max_gap:
                continue
            cand = np.stack([boxes[i] for i in free])
            ious = _iou(tracks[t]['box'], cand)
            j = int(ious.argmax())
            if ious[j] >= iou_thr:
                i = free.pop(j)
                tracks[t]['box'] = boxes[i]
                tracks[t]['last'] = f
                tracks[t]['hits'].append((f, boxes[i]))
        for i in free:
            tracks.append({'box': boxes[i], 'last': f, 'hits': [(f, boxes[i])]})

    F = len(per_frame)
    tracks = [t for t in tracks if len(t['hits']) >= max(3, 0.05 * F)]
    # Rank by screen presence: a big, long-lived track is the subject of the shot.
    def presence(t):
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for _, b in t['hits']]
        return len(t['hits']) * float(np.mean(areas))
    tracks.sort(key=presence, reverse=True)

    N = len(tracks)
    boxes = np.zeros((F, N, 4), np.float32)
    valid = np.zeros((F, N), bool)
    for n, t in enumerate(tracks):
        for f, b in t['hits']:
            boxes[f, n] = b
            valid[f, n] = True
    print(f'  {N} tracks over {F} frames '
          f'(presence-ranked, track 0 = {valid[:, 0].sum()} frames)' if N else '  no tracks found')
    return boxes, valid


# --------------------------------------------------------------------------- #
# Stage 2 -- TokenHMR inference + tokenizer encoding                           #
# --------------------------------------------------------------------------- #
def _rotmat_to_rot6d(rotmat):
    """(..., 3, 3) -> (..., 6), inverse of this repo's `rot6d_to_rotmat`.

    CAREFUL: this repo's `rot6d_to_rotmat` ends in `torch.stack((b1, b2, b3), dim=-2)`, so the
    basis vectors are the ROWS of the returned matrix -- the standard column version is the
    commented-out line right above it ("Original HMR2.0 model was trained with this line").
    The inverse therefore takes the first two ROWS. Taking the first two columns instead round-
    trips to R-transpose: every joint rotation comes back inverted, which leaves camera position
    untouched and silently ruins only the pose. `_check_rot6d_roundtrip` guards against that.
    """
    return rotmat[..., :2, :].reshape(*rotmat.shape[:-2], 6)


def _check_rot6d_roundtrip():
    from tokenhmr.lib.utils.geometry import rot6d_to_rotmat
    q = torch.linalg.qr(torch.randn(8, 3, 3))[0]
    q = q * torch.sign(torch.linalg.det(q))[:, None, None]
    back = rot6d_to_rotmat(_rotmat_to_rot6d(q).reshape(-1, 6)).reshape(-1, 3, 3)
    err = float((back - q).abs().max())
    if err > 1e-4:
        raise RuntimeError(f'rot6d round-trip is broken (max err {err:.3g}) -- '
                           '_rotmat_to_rot6d no longer matches lib.utils.geometry.rot6d_to_rotmat')


def _smooth_rotations(rotmat, k):
    """Moving average over time in 6-D rotation space, re-orthonormalized.

    rotmat: (F, J, 3, 3) with NaN rows for frames where the person is absent.
    Averaging rotation matrices directly leaves the manifold; averaging the 6-D representation
    and re-running Gram-Schmidt is the cheap standard fix and is exactly the parameterization
    the head already predicts in.
    """
    from tokenhmr.lib.utils.geometry import rot6d_to_rotmat
    if k <= 1:
        return rotmat
    F, J = rotmat.shape[:2]
    r6 = _rotmat_to_rot6d(rotmat)
    out = np.empty_like(r6)
    half = k // 2
    for f in range(F):
        lo, hi = max(0, f - half), min(F, f + half + 1)
        win = r6[lo:hi]
        m = ~np.isnan(win).any(axis=(1, 2))
        out[f] = win[m].mean(0) if m.any() else r6[f]
    t = torch.from_numpy(out.reshape(-1, 6).astype(np.float32))
    return rot6d_to_rotmat(t).reshape(F, J, 3, 3).numpy()


def run_inference(video, boxes, valid, stride, ckpt, model_config, batch_size, smooth,
                  tokens=True):
    """Run TokenHMR on every tracked box and encode each predicted pose into token indices.

    Returns a dict of arrays; NaN / -1 mark frames where a track is absent.

    With ``tokens=False`` the pose is recovered but nothing is encoded. The token
    panels are specific to this thesis's transformer tokenizer, and a comparison
    against a model built on a different one only needs the mesh.
    """
    from tokenhmr.lib.models import load_tokenhmr
    from tokenhmr.lib.datasets.vitdet_dataset import ViTDetDataset
    from tokenhmr.lib.utils import recursive_to
    from tokenhmr.lib.utils.renderer import cam_crop_to_full
    from tokenhmr.lib.utils.token_metrics import _get_gt_encoder

    model, model_cfg = load_tokenhmr(checkpoint_path=ckpt, model_cfg=model_config,
                                     is_train_state=False, is_demo=True)
    model = model.to(DEVICE).eval()
    enc = None
    if tokens:
        tok_path = model_cfg.MODEL.get('TOKENIZER_CHECKPOINT_PATH', '')
        assert tok_path, 'model config has no MODEL.TOKENIZER_CHECKPOINT_PATH'
        enc = _get_gt_encoder(tok_path, DEVICE)

    F, N = valid.shape
    T = int(model_cfg.MODEL.SMPL_HEAD.TOKENIZER.TOKEN_NUM)
    K = int(model_cfg.MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CLASS_NUM)
    D = int(model_cfg.MODEL.SMPL_HEAD.TOKENIZER.TOKEN_CODE_DIM)
    nv = 6890

    verts    = np.full((F, N, nv, 3), np.nan, np.float16)
    cam_t    = np.full((F, N, 3), np.nan, np.float32)
    grot     = np.full((F, N, 1, 3, 3), np.nan, np.float32)
    bpose    = np.full((F, N, 23, 3, 3), np.nan, np.float32)
    betas    = np.full((F, N, 10), np.nan, np.float32)
    tok_enc  = np.full((F, N, T), -1, np.int32)
    tok_pred = np.full((F, N, T), -1, np.int32)
    latent   = np.full((F, T, D), np.nan, np.float32)      # focus track only
    focal    = np.zeros(F, np.float32)

    cap = cv2.VideoCapture(video)
    for f in tqdm(range(F), desc='stage 2/3  TokenHMR'):
        for _ in range(stride):
            ok, frame = cap.read()
            if not ok:
                break
        if not ok:
            break
        idx_n = np.nonzero(valid[f])[0]
        if len(idx_n) == 0:
            continue
        ds = ViTDetDataset(model_cfg, frame, boxes[f, idx_n])
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in dl:
            batch = recursive_to(batch, DEVICE)
            with torch.no_grad():
                out = model(batch)
            B = batch['img'].shape[0]
            pid = batch['personid'].cpu().numpy().astype(int)
            n_slots = idx_n[pid]

            img_size = batch['img_size'].float()
            sfl = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            focal[f] = float(sfl)
            full_t = cam_crop_to_full(out['pred_cam'], batch['box_center'].float(),
                                      batch['box_size'].float(), img_size, sfl)

            bp = out['pred_smpl_params']['body_pose'].float()
            te = tp = lat = None
            with torch.no_grad():
                if enc is not None:
                    te = enc.encode(bp[:, :enc.num_joints])
                # The pre-quantisation latent is a transformer-tokenizer hook. The
                # vanilla convolutional tokenizer has no equivalent, so the latent
                # panels are simply unavailable for it and the rest still works.
                    lat = (enc._encode_continuous(bp[:, :enc.num_joints])
                           if hasattr(enc, '_encode_continuous') else None)
                    tp = out['cls_logits'][-B:].argmax(-1)

            verts[f, n_slots] = out['pred_vertices'].detach().cpu().numpy().astype(np.float16)
            cam_t[f, n_slots] = full_t.detach().cpu().numpy()
            grot[f, n_slots]  = out['pred_smpl_params']['global_orient'].float().cpu().numpy()
            bpose[f, n_slots] = bp.cpu().numpy()
            betas[f, n_slots] = out['pred_smpl_params']['betas'].float().cpu().numpy()
            if te is not None:
                tok_enc[f, n_slots]  = te.cpu().numpy()
            if tp is not None:
                tok_pred[f, n_slots] = tp.cpu().numpy()
            if lat is not None and 0 in n_slots:
                latent[f] = lat[int(np.nonzero(n_slots == 0)[0][0])].cpu().numpy()
    cap.release()

    # Kept for the predicted-vs-encoded statistic, which must compare the head's tokens against
    # the encoding of the pose the head actually predicted -- not against a smoothed edit of it.
    tok_enc_raw = tok_enc.copy()

    if smooth > 1:
        # Cosmetic only: per-frame HMR jitters, and jitter both looks bad and flips tokens that
        # the pose does not really move. Smoothing the pose, then re-encoding, keeps the token
        # stream consistent with the mesh actually shown.
        print(f'  smoothing poses over {smooth} frames and re-encoding')
        _check_rot6d_roundtrip()
        for n in range(N):
            m = valid[:, n]
            if m.sum() < 3:
                continue
            full = np.concatenate([grot[:, n], bpose[:, n]], axis=1)     # (F, 24, 3, 3)
            full = _smooth_rotations(full, smooth)
            grot[:, n], bpose[:, n] = full[:, :1], full[:, 1:]
        with torch.no_grad():
            for f in tqdm(range(F), desc='  re-run SMPL'):
                idx_n = np.nonzero(valid[f])[0]
                if len(idx_n) == 0:
                    continue
                go = torch.from_numpy(grot[f, idx_n]).to(DEVICE)
                bp = torch.from_numpy(bpose[f, idx_n]).to(DEVICE)
                be = torch.from_numpy(betas[f, idx_n]).to(DEVICE)
                so = model.smpl(global_orient=go, body_pose=bp, betas=be, pose2rot=False)
                verts[f, idx_n] = so.vertices.cpu().numpy().astype(np.float16)
                if enc is not None:
                    tok_enc[f, idx_n] = enc.encode(bp[:, :enc.num_joints]).cpu().numpy()
                if 0 in idx_n and hasattr(enc, '_encode_continuous'):
                    i0 = int(np.nonzero(idx_n == 0)[0][0])
                    latent[f] = enc._encode_continuous(bp[:, :enc.num_joints])[i0].cpu().numpy()

    # Token -> body region, and per-vertex region for painting the mesh, both measured here
    # while the tokenizer and SMPL are still loaded.
    if enc is None:
        token_joint = np.zeros((T, 21), np.float32)
    else:
        ref = torch.from_numpy(bpose[valid[:, 0], 0, :enc.num_joints]).to(DEVICE)
        token_joint = token_joint_map(enc, ref)
    lbs = model.smpl.lbs_weights.detach().cpu().numpy()                  # (6890, 24)
    # SMPL joints 0..23 -> body-pose joint index, with the pelvis folded into Spine1 and each
    # hand into its wrist, so every vertex lands on one of the 21 joints the tokens control.
    smpl_to_body = np.concatenate([[JOINT_NAMES.index('Spine1')], np.arange(21),
                                   [JOINT_NAMES.index('L_Wrist'), JOINT_NAMES.index('R_Wrist')]])
    vert_joint = smpl_to_body[lbs.argmax(1)]

    levels = (enc.quantizer._levels.cpu().numpy().tolist()
              if enc is not None and hasattr(enc.quantizer, '_levels') else [])

    del model, enc
    torch.cuda.empty_cache()

    return dict(verts=verts, cam_t=cam_t, tok_enc=tok_enc, tok_enc_raw=tok_enc_raw,
                tok_pred=tok_pred, latent=latent,
                focal=focal, token_joint=token_joint, vert_joint=vert_joint.astype(np.int8),
                levels=np.array(levels, np.int64), nb_code=np.int64(K), num_tokens=np.int64(T),
                smooth=np.int64(smooth))


@torch.no_grad()
def token_joint_map(enc, poses, n_alt=6, max_poses=24, seed=0):
    """Which single joint does each token slot control? (measured, not assumed)

    For every token slot, swap its code for random alternatives on real poses from this clip and
    record which of the 21 joints change local rotation the most. FSQ tokens came out near-local
    in earlier analysis (~1 joint each), so this argmax is stable; it is only used for coloring
    and for grouping the raster rows.
    """
    from tokenization.models.rotation_utils import rotation_6d_to_matrix

    def decode(feat):
        B = feat.shape[0]
        xd = enc.decoder(feat)
        jq = enc.joint_queries.expand(B, -1, -1)
        xd = enc._cross_attn_up_joints(jq, xd)
        p6 = enc.decoder_projection(xd)
        return rotation_6d_to_matrix(p6.reshape(-1, 6)).view(B, -1, 3, 3)

    g = torch.Generator(device='cpu').manual_seed(seed)
    sel = torch.randperm(poses.shape[0], generator=g)[:max_poses]
    poses = poses[sel].to(DEVICE)
    P = poses.shape[0]
    idx = enc.encode(poses)                                    # (P, T)
    T = idx.shape[1]
    base = decode(enc.quantizer.dequantize(idx))               # (P, J, 3, 3)

    nb = enc.quantizer.nb_code
    joint = np.zeros(T, np.int8)
    for t in tqdm(range(T), desc='  token -> joint'):
        alt = idx.repeat_interleave(n_alt, 0)                  # (P*n_alt, T)
        alt[:, t] = torch.randint(0, nb, (P * n_alt,), generator=g).to(DEVICE)
        pert = decode(enc.quantizer.dequantize(alt))
        rel = torch.matmul(base.repeat_interleave(n_alt, 0).transpose(-1, -2), pert)
        cos = ((rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2] - 1) * 0.5).clamp(-1, 1)
        move = torch.rad2deg(torch.acos(cos)).mean(0)          # (J,) mean degrees per joint
        joint[t] = int(move.argmax())
    hit = [(JOINT_NAMES[j], int((joint == j).sum())) for j in JOINT_ORDER]
    print('  tokens per joint: ' + ', '.join(f'{n}={c}' for n, c in hit if c))
    empty = [n for n, c in hit if not c]
    if empty:
        print('  no token claims: ' + ', '.join(empty))
    return joint


# --------------------------------------------------------------------------- #
# Stage 3 -- rendering                                                         #
# --------------------------------------------------------------------------- #
class MeshOverlay:
    """One reusable pyrender offscreen renderer for the whole clip.

    The stock `Renderer.render_rgba_multiple` builds and deletes an OffscreenRenderer per call,
    which dominates runtime over a few hundred frames, and paints every mesh a single flat color.
    This keeps one renderer alive and takes per-vertex colors so the body can wear the region
    palette.
    """

    def __init__(self, model_cfg, faces, width, height):
        import pyrender
        from tokenhmr.lib.utils.renderer import Renderer
        self.pyrender = pyrender
        self.base = Renderer(model_cfg, faces)
        self.faces = faces
        self.r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height,
                                            point_size=1.0)

    def __call__(self, verts_list, cam_t_list, vcolor_list, focal_length, res):
        import trimesh
        scene = self.pyrender.Scene(bg_color=[0, 0, 0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        flip = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        for i, (v, t, c) in enumerate(zip(verts_list, cam_t_list, vcolor_list)):
            m = trimesh.Trimesh(v.astype(np.float64) + t, self.faces.copy(), vertex_colors=c)
            m.apply_transform(flip)
            scene.add(self.pyrender.Mesh.from_trimesh(m), f'mesh_{i}')
        cam = self.pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                             cx=res[0] / 2., cy=res[1] / 2., zfar=1e12)
        cam_node = self.pyrender.Node(camera=cam, matrix=np.eye(4))
        scene.add_node(cam_node)
        self.base.add_point_lighting(scene, cam_node)
        self.base.add_lighting(scene, cam_node)
        from tokenhmr.lib.utils.renderer import create_raymond_lights
        for node in create_raymond_lights():
            scene.add_node(node)
        color, _ = self.r.render(scene, flags=self.pyrender.RenderFlags.RGBA)
        return color.astype(np.float32) / 255.0

    def close(self):
        self.r.delete()


def compute_crop(boxes, valid, mode, aspect, margin, W, H, smooth=31):
    """Per-frame crop rects (F, 4) of constant size, matching `aspect` (= w/h).

    A wide arena shot leaves the bodies a few percent of frame height, which is the wrong picture
    for a title slide -- the mesh is the point. Cropping to the tracked people makes them large
    and, as a side effect, fills the panel exactly instead of letterboxing it. The rect size is
    held constant over the clip so the apparent zoom never breathes.
    """
    F, N = valid.shape
    if mode == 'none' or not valid.any():
        return np.tile(np.array([0, 0, W, H], np.int32), (F, 1))

    if mode == 'focus':
        src = boxes[:, 0]
        ok = valid[:, 0]
    else:                                                  # 'clip': every tracked person
        src = np.zeros((F, 4), np.float32)
        ok = valid.any(1)
        for f in np.nonzero(ok)[0]:
            b = boxes[f, valid[f]]
            src[f] = [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]
    if not ok.any():
        return np.tile(np.array([0, 0, W, H], np.int32), (F, 1))

    idx = np.arange(F)
    for c in range(4):                                     # carry the rect across gaps
        src[:, c] = np.interp(idx, idx[ok], src[ok, c])
    cx = (src[:, 0] + src[:, 2]) / 2
    cy = (src[:, 1] + src[:, 3]) / 2
    if smooth > 1 and mode == 'focus':
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        cx = np.convolve(np.pad(cx, pad, mode='edge'), k, 'valid')[:F]
        cy = np.convolve(np.pad(cy, pad, mode='edge'), k, 'valid')[:F]

    bw = float(np.percentile(src[ok, 2] - src[ok, 0], 90)) * (1 + 2 * margin)
    bh = float(np.percentile(src[ok, 3] - src[ok, 1], 90)) * (1 + 2 * margin)
    if mode == 'clip':      # one rect for the whole clip: hold the union, don't track
        cx[:] = (src[ok, 0].min() + src[ok, 2].max()) / 2
        cy[:] = (src[ok, 1].min() + src[ok, 3].max()) / 2
        bw = (src[ok, 2].max() - src[ok, 0].min()) * (1 + 2 * margin)
        bh = (src[ok, 3].max() - src[ok, 1].min()) * (1 + 2 * margin)

    cw, ch = max(bw, bh * aspect), max(bh, bw / aspect)    # grow to the panel aspect
    cw, ch = min(cw, W), min(ch, H)
    cw, ch = min(cw, ch * aspect), min(ch, cw / aspect)    # and back inside the frame
    cw, ch = int(round(cw)), int(round(ch))

    x0 = np.clip(np.round(cx - cw / 2), 0, W - cw).astype(np.int32)
    y0 = np.clip(np.round(cy - ch / 2), 0, H - ch).astype(np.int32)
    return np.stack([x0, y0, x0 + cw, y0 + ch], 1)


def atlas_shape(levels, nb_code):
    """(rows, cols) such that reshape(nb_code, rows, cols) puts real lattice axes on the axes.

    With FSQ levels [L0..L3] the index is `d0 + L0*d1 + L0*L1*d2 + ...`, so splitting the digits
    into a low group and a high group gives a plain reshape. We split after the digit that lands
    closest to a square-ish, wide panel. Without levels (non-FSQ) the reshape is arbitrary.
    """
    if len(levels) >= 2:
        best, split = None, 1
        for s in range(1, len(levels)):
            cols = int(np.prod(levels[:s]))
            rows = int(np.prod(levels[s:]))
            score = abs(np.log((cols / max(rows, 1)) / 2.0))     # aim for ~2:1 landscape
            if best is None or score < best:
                best, split = score, s
        cols = int(np.prod(levels[:split]))
        return int(np.prod(levels[split:])), cols, split
    side = int(np.ceil(np.sqrt(nb_code)))
    while nb_code % side and side < nb_code:
        side += 1
    return nb_code // side, side, 0


def build_and_render(data, args, video_meta):
    """Compose the panels into one matplotlib figure and stream frames to a video file."""
    tok = data['tok_enc'] if args.token_source == 'encoded' else data['tok_pred']
    tok_focus = tok[:, 0]                                        # (F, T) focus track
    token_joint = data['token_joint'].astype(int)                # (T,) dominant joint per slot
    by_joint = args.granularity == 'joint'
    # `group` is what the raster blocks and the colors key off: one block per joint (21), or one
    # per body part (8).
    group = token_joint if by_joint else JOINT_PART[token_joint]
    group_order = JOINT_ORDER if by_joint else list(range(len(PART_NAMES)))
    group_names = JOINT_NAMES if by_joint else PART_NAMES
    group_rgb = JOINT_RGB if by_joint else PART_RGB
    levels = data['levels'].tolist()
    nb_code = int(data['nb_code'])
    F, T = tok_focus.shape
    rows, cols, split = atlas_shape(levels, nb_code)
    exact = len(levels) >= 2

    present = tok_focus[:, 0] >= 0
    if not present.any():
        raise SystemExit('focus track has no frames with tokens')
    tok_ref = data['tok_enc_raw'][:, 0] if 'tok_enc_raw' in data else data['tok_enc'][:, 0]

    # ---- token ordering for the raster: one block per group, in anatomical reading order
    rank = np.full(len(group_names), len(group_names), int)
    for r, g in enumerate(group_order):
        rank[g] = r
    order = np.argsort(rank[group] * 10000 + np.arange(T), kind='stable')
    group_sorted = group[order]
    blocks = []                                   # (group id, first row, last row + 1)
    start = 0
    for i in range(1, T + 1):
        if i == T or group_sorted[i] != group_sorted[start]:
            blocks.append((int(group_sorted[start]), start, i))
            start = i

    # ---- precomputed raster: bright where a token's code changed on that frame
    changed = np.zeros((T, F), np.float32)
    prev = None
    for f in range(F):
        if tok_focus[f, 0] < 0:
            continue
        if prev is not None:
            changed[:, f] = (tok_focus[f] != prev).astype(np.float32)
        prev = tok_focus[f]
    changed = changed[order]
    idle = 0.30                                                  # dim tint so every row is visible
    w = (idle + (1 - idle) * changed)[..., None]
    raster_rgb = hex2rgb(SURFACE)[None, None] * (1 - w) + group_rgb[group_sorted][:, None, :] * w

    # ---- latent trajectory: PCA over the clip's own pre-quantization latents
    lat = data['latent'].reshape(F, -1)
    ok = ~np.isnan(lat).any(1)
    traj = np.full((F, 2), np.nan, np.float32)
    if ok.sum() > 3:
        X = lat[ok]
        mu = X.mean(0, keepdims=True)
        _, _, V = np.linalg.svd(X - mu, full_matrices=False)
        traj[ok] = ((X - mu) @ V[:2].T).astype(np.float32)

    # ---- figure skeleton
    W, H = args.width, args.height
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=PAGE)
    show_traj = args.layout == 'quad'
    if args.layout == 'duo':
        gs = fig.add_gridspec(2, 2, height_ratios=[0.13, 1], width_ratios=[1.15, 1],
                              left=0.012, right=0.988, top=0.985, bottom=0.03,
                              hspace=0.10, wspace=0.06)
        ax_head = fig.add_subplot(gs[0, :]); ax_vid = fig.add_subplot(gs[1, 0])
        ax_atlas = fig.add_subplot(gs[1, 1]); ax_rast = None; ax_traj = None
    elif args.layout == 'quad':
        gs = fig.add_gridspec(3, 2, height_ratios=[0.13, 1, 0.85], width_ratios=[1.15, 1],
                              left=0.012, right=0.988, top=0.985, bottom=0.045,
                              hspace=0.28, wspace=0.06)
        ax_head = fig.add_subplot(gs[0, :]); ax_vid = fig.add_subplot(gs[1, 0])
        ax_atlas = fig.add_subplot(gs[1, 1]); ax_traj = fig.add_subplot(gs[2, 0])
        ax_rast = fig.add_subplot(gs[2, 1])
    else:                                                        # 'title'
        # Per-joint needs 21 labelled row blocks in the raster, so give it more height.
        gs = fig.add_gridspec(3, 2, height_ratios=[0.13, 1, 0.95 if by_joint else 0.62],
                              width_ratios=[1.15, 1],
                              left=0.012, right=0.988, top=0.985, bottom=0.045,
                              hspace=0.24, wspace=0.06)
        ax_head = fig.add_subplot(gs[0, :]); ax_vid = fig.add_subplot(gs[1:, 0])
        ax_atlas = fig.add_subplot(gs[1, 1]); ax_rast = fig.add_subplot(gs[2, 1])
        ax_traj = None

    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(AXISLINE)
        ax.tick_params(colors=MUTED, labelsize=8)

    # ---- header: title, legend, live stats
    ax_head.set_facecolor(PAGE); ax_head.axis('off')
    ax_head.set_xlim(0, 1); ax_head.set_ylim(0, 1)
    ax_head.text(0, 0.80, args.title, color=INK, fontsize=21, weight='bold', va='center')
    subtitle = args.subtitle + ('   |   shade: joints within a part' if by_joint else '')
    ax_head.text(0, 0.44, subtitle, color=INK2, fontsize=11.5, va='center')
    # Eight parts do not fit on one row beside a long title, so the legend is 2x4 on the right.
    # Per joint each part shows its whole ramp as a strip: the strip is the key to the shading,
    # and the joint names themselves are labelled directly on the raster rows.
    for i, nm in enumerate(PART_NAMES):
        col, row = i % 4, i // 4
        x, y = 0.585 + col * 0.106, 0.82 - row * 0.38
        steps = PART_GROUPS[i][2] if by_joint else [PART_COLORS[i]]
        wid = 0.025 / len(steps)
        for k, cl in enumerate(steps):
            ax_head.add_patch(Rectangle((x + k * wid, y - 0.12), wid, 0.24, color=cl,
                                        transform=ax_head.transAxes))
        ax_head.text(x + 0.025 + 0.007, y, nm, color=INK2, fontsize=9.5, va='center')
    stat_txt = ax_head.text(0.56, 0.08, '', color=INK2, fontsize=10.5, va='center', ha='right',
                            family='monospace')

    # ---- video panel
    ax_vid.set_xticks([]); ax_vid.set_yticks([])
    vw, vh = video_meta['width'], video_meta['height']
    fig.canvas.draw()                                    # settle the layout to read panel aspect
    pos = ax_vid.get_position()
    panel_aspect = (pos.width * W) / (pos.height * H)
    crops = compute_crop(data['boxes'], data['valid'], args.crop, panel_aspect,
                         args.crop_margin, vw, vh)
    cw, ch = int(crops[0, 2] - crops[0, 0]), int(crops[0, 3] - crops[0, 1])
    im_vid = ax_vid.imshow(np.zeros((ch, cw, 3), np.float32), interpolation='bilinear')
    painted = {'joint': 'painted by the joint each token controls',
               'part': 'painted by the body part each token controls',
               'plain': 'recovered SMPL mesh'}[args.mesh_color]
    ax_vid.set_title(f'Input video  {painted}', color=INK, fontsize=11.5, loc='left', pad=7)

    # ---- atlas panel
    heat_rgb = group_rgb[group]                                  # color a code takes when lit
    lattice = hex2rgb(GRIDLINE)[None, :] * np.ones((nb_code, 1), np.float32)
    atlas_img = lattice.reshape(rows, cols, 3)
    im_atlas = ax_atlas.imshow(atlas_img, interpolation='nearest', aspect='auto',
                               vmin=0, vmax=1)
    if exact:
        cols_per = int(np.prod(levels[:max(split - 1, 1)])) if split > 1 else levels[0]
        rows_per = levels[split] if split < len(levels) else rows
        for x in range(cols_per, cols, cols_per):
            ax_atlas.axvline(x - 0.5, color=AXISLINE, lw=0.6, alpha=0.75)
        for y in range(rows_per, rows, rows_per):
            ax_atlas.axhline(y - 0.5, color=AXISLINE, lw=0.6, alpha=0.75)
        dig = ''.join(f'd{i}({l})x' for i, l in enumerate(levels[:split])).rstrip('x')
        dig2 = ''.join(f'd{i}({l})x' for i, l in enumerate(levels[split:], split)).rstrip('x')
        ax_atlas.set_xlabel(dig, color=MUTED, fontsize=9)
        ax_atlas.set_ylabel(dig2, color=MUTED, fontsize=9)
        sub = (f'every cell is one of the {nb_code} FSQ codes -- '
               f'lit = used by this pose, fading = recently used')
    else:
        ax_atlas.set_xlabel('code index (arbitrary reshape -- no product structure)',
                            color=MUTED, fontsize=9)
        sub = f'{nb_code} codes, lit = used by this pose'
    ax_atlas.set_xticks([]); ax_atlas.set_yticks([])
    ax_atlas.set_title('Codebook  ' + sub, color=INK, fontsize=11.5, loc='left', pad=7)

    # ---- raster panel
    if ax_rast is not None:
        im_rast = ax_rast.imshow(raster_rgb, interpolation='nearest', aspect='auto',
                                 vmin=0, vmax=1)
        play = ax_rast.axvline(0, color=INK, lw=1.2, alpha=0.9)
        # Hairline between joints, a heavier rule where the body part changes.
        for (g0, _, hi0), (g1, _, _) in zip(blocks[:-1], blocks[1:]):
            heavy = not by_joint or JOINT_PART[g0] != JOINT_PART[g1]
            ax_rast.axhline(hi0 - 0.5, color=AXISLINE, lw=1.4 if heavy else 0.6,
                            alpha=1.0 if heavy else 0.7)
        # Labels go *inside* the panel: as y-tick labels they render to the left of the axes and
        # overlap the video panel next to it.
        fs = 7.2 if by_joint else 9.5
        for gid, lo, hi in blocks:
            ax_rast.text(0.006, (lo + hi) / 2 - 0.5, group_names[gid],
                         transform=ax_rast.get_yaxis_transform(), color=INK, fontsize=fs,
                         va='center', ha='left',
                         bbox=dict(fc=SURFACE, ec='none', alpha=0.7, pad=1.2))
        ax_rast.set_yticks([])
        ax_rast.set_xlabel('frame', color=MUTED, fontsize=9)
        what = 'joint' if by_joint else 'body part'
        ax_rast.set_title(f"Token stream  {T} token slots x time, grouped by the {what} each "
                          f"token controls -- bright = its code changed",
                          color=INK, fontsize=11.5, loc='left', pad=7)
        ax_rast.tick_params(axis='y', length=0)
    else:
        im_rast = play = None

    # ---- trajectory panel
    if ax_traj is not None:
        ax_traj.plot(traj[:, 0], traj[:, 1], color=AXISLINE, lw=1.0, zorder=1)
        trail = LineCollection([], linewidths=2.0, zorder=2)
        ax_traj.add_collection(trail)
        head, = ax_traj.plot([], [], 'o', ms=9, color=INK, mec=SURFACE, mew=1.5, zorder=3)
        good = ~np.isnan(traj).any(1)
        if good.any():
            pad = 0.06 * np.ptp(traj[good], axis=0).max()
            ax_traj.set_xlim(traj[good, 0].min() - pad, traj[good, 0].max() + pad)
            ax_traj.set_ylim(traj[good, 1].min() - pad, traj[good, 1].max() + pad)
        ax_traj.set_xlabel('PC1', color=MUTED, fontsize=9)
        ax_traj.set_ylabel('PC2', color=MUTED, fontsize=9)
        ax_traj.set_title('Latent trajectory  pre-quantization latent, PCA over the clip',
                          color=INK, fontsize=11.5, loc='left', pad=7)
        ax_traj.grid(color=GRIDLINE, lw=0.6)
        ax_traj.set_axisbelow(True)
    else:
        trail = head = None

    # ---- renderer + per-vertex colors
    from tokenhmr.lib.configs import get_config
    model_cfg = get_config(args.model_config)
    overlay = MeshOverlay(model_cfg, data['faces'], vw, vh)
    vj = data['vert_joint'].astype(int)
    if args.mesh_color == 'plain':
        vcol_focus = np.tile(np.array([[0.65, 0.74, 0.86, 1.0]], np.float32), (len(vj), 1))
    else:
        vrgb = JOINT_RGB[vj] if args.mesh_color == 'joint' else PART_RGB[JOINT_PART[vj]]
        vcol_focus = np.concatenate([vrgb, np.ones((len(vj), 1), np.float32)], 1)
    vcol_other = np.tile(np.array([[0.55, 0.55, 0.56, 1.0]], np.float32), (len(vj), 1))

    # ---- video source + writer
    cap = cv2.VideoCapture(args.video)
    writer = _open_writer(args.out_video, W, H, video_meta['fps'] / args.stride)

    heat = np.zeros(nb_code, np.float32)
    heat_col = np.tile(hex2rgb(GRIDLINE), (nb_code, 1))
    seen = np.zeros(nb_code, bool)
    verts, cam_t, focal = data['verts'], data['cam_t'], data['focal']
    still_set = set(args.stills)

    for f in tqdm(range(F), desc='stage 3/3  render'):
        for _ in range(args.stride):
            ok, frame = cap.read()
            if not ok:
                break
        if not ok:
            break
        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0

        vlist, tlist, clist = [], [], []
        for n in range(verts.shape[1]):
            if np.isnan(cam_t[f, n]).any():
                continue
            vlist.append(verts[f, n].astype(np.float32))
            tlist.append(cam_t[f, n].astype(np.float64))
            clist.append(vcol_focus if n == 0 else vcol_other)
        if vlist:
            fl = float(focal[f]) if focal[f] > 0 else model_cfg.EXTRA.FOCAL_LENGTH
            rgba = overlay(vlist, tlist, clist, fl, (vw, vh))
            a = rgba[:, :, 3:4] * args.mesh_alpha
            rgb = rgb * (1 - a) + rgba[:, :, :3] * a
        x0, y0, x1, y1 = crops[f]          # crop after compositing: the mesh needs full-frame cam
        im_vid.set_data(np.clip(rgb[y0:y1, x0:x1], 0, 1))

        # atlas: decay the trail, then light this frame's codes
        heat *= args.decay
        codes = tok_focus[f]
        if codes[0] >= 0:
            heat[codes] = 1.0
            heat_col[codes] = heat_rgb
            seen[codes] = True
        w = np.clip(heat, 0, 1)[:, None] ** 0.65
        img = lattice * (1 - w) + heat_col * w
        hot = heat >= 0.999
        img[hot] = heat_col[hot] * 0.62 + 0.38                    # bright core for "used now"
        im_atlas.set_data(img.reshape(rows, cols, 3))

        if im_rast is not None:
            shown = raster_rgb.copy()
            shown[:, f + 1:] = shown[:, f + 1:] * 0.16 + hex2rgb(SURFACE) * 0.84
            im_rast.set_data(shown)
            play.set_xdata([f, f])

        if trail is not None:
            lo = max(0, f - args.trail)
            seg = traj[lo:f + 1]
            m = ~np.isnan(seg).any(1)
            if m.sum() > 1:
                p = seg[m]
                segs = np.stack([p[:-1], p[1:]], axis=1)
                al = np.linspace(0.12, 1.0, len(segs))
                trail.set_segments(segs)
                # Neutral ink, not a region hue: the trail is time, not a body part, and the
                # three hues are spoken for.
                trail.set_color([(*hex2rgb(INK), a) for a in al])
                head.set_data([p[-1, 0]], [p[-1, 1]])

        n_used = len(np.unique(codes[codes >= 0]))
        agree = ''
        # Compared against the PRE-smoothing encoding, so the number stays a statement about the
        # token predictor rather than about the cosmetic smoothing.
        if tok_ref[f, 0] >= 0 and data['tok_pred'][f, 0, 0] >= 0:
            agree = ('   predicted == encoded  '
                     f'{100 * (data["tok_pred"][f, 0] == tok_ref[f]).mean():4.0f}%')
        stat_txt.set_text(f'frame {f + 1:4d}/{F}   distinct codes now {n_used:3d}/{T} slots'
                          f'   codes seen {seen.sum():4d}/{nb_code}'
                          f' ({100 * seen.mean():4.1f}%){agree}')

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer(buf)
        if f in still_set:
            fig.savefig(f'{os.path.splitext(args.out_video)[0]}_frame{f:04d}.png',
                        dpi=args.still_dpi, facecolor=PAGE)

    cap.release()
    overlay.close()
    writer(None)
    plt.close(fig)


def _ffmpeg_cmd(path, w, h, fps):
    return ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{w}x{h}', '-r', f'{fps:.4f}', '-i', '-', '-an',
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', path]


def _open_writer(path, w, h, fps):
    """Stream RGB frames to H.264 via ffmpeg; fall back to OpenCV if that does not work.

    H.264 in yuv420p is what PowerPoint/Keynote reliably play, so it is worth preferring, but
    ffmpeg is not always usable -- a snap-packaged ffmpeg, for instance, is confined and cannot
    write outside $HOME. Rather than discover that via a broken pipe halfway through the clip,
    probe it once with a single black frame and pick the writer up front.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    probe_ok = False
    try:
        probe = subprocess.run(_ffmpeg_cmd(path, w, h, fps), input=b'\x00' * (w * h * 3),
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        probe_ok = probe.returncode == 0 and os.path.exists(path)
        if not probe_ok:
            msg = probe.stderr.decode(errors='replace').strip().splitlines()
            print(f'  ffmpeg cannot write {path}: {msg[-1] if msg else "unknown error"}')
    except (FileNotFoundError, OSError) as e:
        print(f'  ffmpeg unavailable ({e})')

    if probe_ok:
        proc = subprocess.Popen(_ffmpeg_cmd(path, w, h, fps), stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        def write(buf):
            if buf is None:
                proc.stdin.close()
                if proc.wait() != 0:
                    print('  ffmpeg: ' + proc.stderr.read().decode(errors='replace').strip())
            else:
                proc.stdin.write(np.ascontiguousarray(buf).tobytes())
        return write

    print('  falling back to OpenCV mp4v (may not play in PowerPoint; '
          're-encode with ffmpeg if needed)')
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    def write(buf):
        if buf is None:
            vw.release()
        else:
            vw.write(np.ascontiguousarray(buf[:, :, ::-1]))
    return write


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', default='demo_sample/video/gymnasts.mp4')
    p.add_argument('--checkpoint', default='logs/tokenhmr_fsq_ce_only/runs/'
                   'tokenhmr_fsq_ce_only_0/checkpoints/epoch=9-step=600000.ckpt')
    p.add_argument('--model_config', default='logs/tokenhmr_fsq_ce_only/runs/'
                   'tokenhmr_fsq_ce_only_0/model_config.yaml')
    p.add_argument('--out', default='results/title_slide', help='output directory; results go into '
                   'a per-tokenizer subfolder of it unless --no-subdir')
    p.add_argument('--no-subdir', dest='subdir', action='store_false', default=True,
                   help='write straight into --out instead of a per-tokenizer subfolder')
    p.add_argument('--name', default='', help="basename for the mp4 / stills "
                   "(default: the stage-2 run name, e.g. 'fsq_ce_only')")

    p.add_argument('--max-frames', type=int, default=0, help='0 = whole clip')
    p.add_argument('--stride', type=int, default=1, help='keep every Nth frame')
    p.add_argument('--max-people', type=int, default=8, help='tracks to render (presence-ranked); '
                   'a panning camera fragments tracks, so allow more than there are people')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--det-size', type=int, default=800, help='detector shortest edge; lower = '
                   'less GPU memory (native is 1024)')
    p.add_argument('--score-thr', type=float, default=0.5)
    p.add_argument('--iou-thr', type=float, default=0.35)
    p.add_argument('--max-gap', type=int, default=8, help='frames a track may go unmatched')
    p.add_argument('--smooth', type=int, default=0, help='temporal pose smoothing window '
                   '(odd, cosmetic; 0 = off, 5 is a good starting point)')

    p.add_argument('--token-source', choices=['encoded', 'predicted'], default='encoded',
                   help="'encoded' = the codes the frozen tokenizer assigns to the pose on "
                        "screen; 'predicted' = the head's own argmax tokens (only meaningful "
                        'for a CE-supervised run)')
    p.add_argument('--layout', choices=['title', 'duo', 'quad'], default='title')
    p.add_argument('--crop', choices=['clip', 'focus', 'none'], default='clip',
                   help="crop the video to the people so the mesh is large and the panel is not "
                        "letterboxed: 'clip' = one fixed rect over the whole clip, "
                        "'focus' = smoothly follows the main track")
    p.add_argument('--crop-margin', type=float, default=0.22, help='padding around the boxes')
    p.add_argument('--granularity', choices=['joint', 'part'], default='joint',
                   help='group and color tokens by the individual joint they control '
                        '(21 blocks) or by body part (8 blocks)')
    p.add_argument('--mesh-color', choices=['joint', 'part', 'plain'], default='joint')
    p.add_argument('--mesh-alpha', type=float, default=1.0)
    p.add_argument('--decay', type=float, default=0.80, help='atlas trail decay per frame')
    p.add_argument('--trail', type=int, default=45, help='trajectory trail length in frames')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--title', default='Exploring Latent Representations for Human Mesh Recovery')
    p.add_argument('--subtitle', default='FSQ tokenizer  |  160 token slots  |  1920 codes')
    p.add_argument('--stills', default='', help='comma-separated frame indices to also save as PNG')
    p.add_argument('--still-dpi', type=int, default=200)
    p.add_argument('--recompute', default='', help='comma-separated stages to redo: detect,infer')

    args = p.parse_args()
    args.stills = [int(x) for x in args.stills.split(',') if x.strip()]
    redo = {s.strip() for s in args.recompute.split(',') if s.strip()}

    tok_tag, run_tag = output_tags(args.model_config, args.checkpoint)
    args.name = args.name or run_tag
    base_out = args.out                                   # detection lives here: model-independent
    if args.subdir:
        args.out = os.path.join(base_out, tok_tag)
    os.makedirs(args.out, exist_ok=True)
    args.out_video = os.path.join(args.out, f'{args.name}.mp4')
    print(f'tokenizer {tok_tag}  |  run {run_tag}  ->  {args.out}/{args.name}.*')

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {args.video}')
    meta = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    n_frames = meta['total'] // args.stride
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    print(f'{args.video}: {meta["total"]} frames @ {meta["fps"]:.1f} fps '
          f'-> using {n_frames} (stride {args.stride})')

    # ---- stage 1 (depends only on the video, so it is shared across models)
    vkey = f'{Path(args.video).stem}_s{args.stride}' + (f'_n{args.max_frames}'
                                                        if args.max_frames else '')
    os.makedirs(base_out, exist_ok=True)
    f_tracks = os.path.join(base_out, f'cache_tracks_{vkey}.npz')
    if os.path.exists(f_tracks) and 'detect' not in redo:
        z = np.load(f_tracks)
        boxes, valid = z['boxes'], z['valid']
        print(f'stage 1/3  cached ({f_tracks})')
    else:
        boxes, valid = detect_and_track(args.video, n_frames, args.stride, args.det_size,
                                        args.score_thr, args.iou_thr, args.max_gap)
        np.savez_compressed(f_tracks, boxes=boxes, valid=valid)
    boxes, valid = boxes[:, :args.max_people], valid[:, :args.max_people]

    # ---- stage 2. Keyed by everything it actually depends on -- the run (two runs can share one
    # tokenizer folder) plus --max-people and --smooth, which change what gets cached. Keying it
    # by --name instead would both recompute per layout variant and silently serve a stale cache
    # after a --smooth change.
    f_inf = os.path.join(args.out,
                         f'cache_infer_{vkey}_{run_tag}_p{args.max_people}_sm{args.smooth}.npz')
    if os.path.exists(f_inf) and not redo & {'detect', 'infer'}:
        data = dict(np.load(f_inf))
        print(f'stage 2/3  cached ({f_inf})')
    else:
        data = run_inference(args.video, boxes, valid, args.stride, args.checkpoint,
                             args.model_config, args.batch_size, args.smooth)
        data['faces'] = _smpl_faces(args.model_config)   # tiny; the renderer needs them
        np.savez_compressed(f_inf, **data)
    if 'faces' not in data:
        data['faces'] = _smpl_faces(args.model_config)
    data['boxes'], data['valid'] = boxes, valid

    # ---- stage 3
    build_and_render(data, args, meta)
    summary = dict(video=args.video, checkpoint=args.checkpoint, frames=int(n_frames),
                   tokenizer=tok_tag, run=run_tag, token_source=args.token_source,
                   layout=args.layout, smooth=args.smooth, crop=args.crop,
                   granularity=args.granularity,
                   nb_code=int(data['nb_code']), num_tokens=int(data['num_tokens']),
                   levels=data['levels'].tolist())
    with open(os.path.join(args.out, f'{args.name}_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'\nwrote {args.out_video}')


def output_tags(model_config, checkpoint):
    """(tokenizer_tag, run_tag) -- the output subfolder and the basename for this run's files.

    The tokenizer tag carries the training DATE, not just the experiment name, because the same
    tokenizer name can cover two different trainings (`with_fnn_blocks` exists as both a 30-04 and
    an 08-05 tokenizer, behind different downstream runs). Name-only folders would silently merge
    them. The run tag keeps two stage-2 runs that share a tokenizer -- e.g. FSQ with and without
    token CE -- from overwriting each other's files inside that one folder.
    """
    from tokenhmr.lib.configs import get_config
    tok_path = get_config(model_config).MODEL.get('TOKENIZER_CHECKPOINT_PATH', '')
    m = re.search(r'([^/]+?)_ID\d+_(\d{2}-\d{2}-\d{4})', tok_path)
    if m:
        tok_tag = f"{re.sub(r'^tokenization_', '', m.group(1))}_{m.group(2)}"
    else:
        tok_tag = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(tok_path).stem) or 'unknown_tokenizer'
    run_tag = Path(checkpoint).parent.parent.name          # .../runs/<run>/checkpoints/<ckpt>
    run_tag = re.sub(r'_\d+$', '', re.sub(r'^tokenhmr_', '', run_tag)) or 'run'
    return tok_tag, run_tag


def _smpl_faces(model_config):
    from tokenhmr.lib.configs import get_config
    import smplx
    cfg = get_config(model_config)
    body = smplx.SMPLLayer(model_path=cfg.SMPL.MODEL_PATH, gender=cfg.SMPL.GENDER, num_betas=10)
    return body.faces.astype(np.int64)


if __name__ == '__main__':
    main()
