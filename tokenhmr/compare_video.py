"""Render two models side by side on the same clip.

Produces the comparison in the README: this thesis's token classifier against
the released TokenHMR checkpoint, on the sample video, with the mesh overlaid on
each frame.

    thesis compare-video                         # defaults reproduce the README
    python -m tokenhmr.compare_video --help      # the same, with every option

Detection and tracking run **once** and both models consume the identical boxes,
crops and frame indices, so the only difference between the two panels is the
pose model. Running two pipelines end to end would differ in detection as well
and the comparison would not mean anything.

Each stage caches to an npz, so restyling the output never re-runs a network:

    stage 1  detect and track           -> cache/compare_tracks_<video>.npz
    stage 2  inference, once per model  -> cache/compare_infer_<video>_<tag>.npz
    stage 3  compose frames, encode

The two panels are drawn as one matplotlib figure rather than two videos joined
afterwards, which guarantees the frames stay aligned.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from repro import paths

CACHE = paths.PROJECT_ROOT / "thesis_figures" / "cache"

# The thesis's best token classifier, and upstream's released model.
DEFAULT_A_CKPT = paths.LOGS_DIR / "tokenhmr_chain2_st/runs/chain_st/checkpoints/last.ckpt"
DEFAULT_A_CFG = paths.LOGS_DIR / "tokenhmr_chain2_st/runs/chain_st/model_config.yaml"
DEFAULT_B_CKPT = paths.DATA_DIR / "checkpoints/tokenhmr_model_latest.ckpt"
DEFAULT_B_CFG = paths.DATA_DIR / "checkpoints/model_config.yaml"
DEFAULT_VIDEO = paths.PROJECT_ROOT / "demo_sample/video/gymnasts.mp4"

MESH_A = (0.36, 0.62, 0.85)   # blue, ours
MESH_B = (0.90, 0.62, 0.30)   # amber, upstream


def _subtitle(run: str) -> str:
    """The model's EMDB PA-MPJPE, read from the manifest so it cannot drift."""
    if not run:
        return ""
    try:
        from repro import manifest

        return f"EMDB PA-MPJPE {manifest.value(run, 'EMDB', 'hard_mode_re'):.1f} mm"
    except Exception:  # noqa: BLE001 - a caption is not worth failing a render for
        return ""


def stage_tracks(args):
    from tokenhmr.visualize_video_latents import detect_and_track

    cache = CACHE / f"compare_tracks_{Path(args.video).stem}.npz"
    if cache.is_file() and not args.recompute_detect:
        d = np.load(cache)
        print(f"tracks: reusing {paths.relative(cache)}")
        return d["boxes"], d["valid"]
    CACHE.mkdir(parents=True, exist_ok=True)
    boxes, valid = detect_and_track(
        args.video, args.max_frames, args.stride, args.det_size,
        args.score_thr, args.iou_thr, args.max_gap,
    )
    np.savez_compressed(cache, boxes=boxes, valid=valid)
    print(f"tracks: wrote {paths.relative(cache)}")
    return boxes, valid


def stage_infer(args, boxes, valid, tag, ckpt, cfg):
    from tokenhmr.visualize_video_latents import run_inference

    cache = CACHE / f"compare_infer_{Path(args.video).stem}_{tag}.npz"
    if cache.is_file() and not args.recompute_infer:
        print(f"inference [{tag}]: reusing {paths.relative(cache)}")
        return dict(np.load(cache, allow_pickle=True))
    # tokens=False: the token panels are specific to this thesis's transformer
    # tokenizer, and the released model uses a different one. Only the mesh is
    # being compared, so nothing is encoded.
    data = run_inference(args.video, boxes, valid, args.stride,
                         str(ckpt), str(cfg), args.batch_size, args.smooth,
                         tokens=False)
    np.savez_compressed(cache, **data)
    print(f"inference [{tag}]: wrote {paths.relative(cache)}")
    return data


def compose(args, boxes, valid, A, B, faces_a, faces_b):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tokenhmr.visualize_video_latents import MeshOverlay

    cap = cv2.VideoCapture(args.video)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_idx = list(range(0, args.max_frames * args.stride, args.stride))
    n = min(len(frames_idx), A["verts"].shape[0], B["verts"].shape[0])

    from tokenhmr.lib.configs import get_config
    cfg_a = get_config(str(args.a_model_config), merge=False)
    # One renderer for both panels. Two live pyrender offscreen contexts fight
    # over EGL, and the SMPL topology is identical between the two models, so a
    # second renderer would buy nothing anyway.
    assert np.array_equal(faces_a, faces_b), "the two models use different mesh topologies"
    overlay = MeshOverlay(cfg_a, faces_a, W, H)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "gymnasts_ours_vs_upstream.mp4"

    # Two panels side by side, so each is half the output width. The figure is
    # that panel height plus explicit bands for the title above and the caption
    # below, otherwise both are drawn outside the canvas and clipped.
    panel_h_px = (args.width / 2.0) * H / W
    head_px, foot_px = 30.0, 30.0
    fig_w = args.width / 100.0
    fig_h = (panel_h_px + head_px + foot_px) / 100.0
    writer = None

    sub_a, sub_b = _subtitle(args.a_run), _subtitle(args.b_run)

    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_idx[k])
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        panels = []
        for data, colour in ((A, MESH_A), (B, MESH_B)):
            vs, ts = [], []
            for j in range(valid.shape[1]):
                v = data["verts"][k, j]
                t = data["cam_t"][k, j]
                if np.isnan(v).any() or np.isnan(t).any():
                    continue
                vs.append(v.astype(np.float64))
                ts.append(t.astype(np.float64))
            if not vs:
                panels.append(rgb)
                continue
            colours = [np.tile(np.array(colour), (v.shape[0], 1)) for v in vs]
            focal = float(np.nanmean(data["focal"][k])) if "focal" in data else 5000.0
            rgba = overlay(vs, ts, colours, focal, (W, H))
            alpha = rgba[..., 3:4]
            panels.append(rgb * (1 - alpha) + rgba[..., :3] * alpha)

        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=100)
        for ax, panel, title, sub in zip(
            axes, panels, (args.a_name, args.b_name), (sub_a, sub_b)
        ):
            ax.imshow(np.clip(panel, 0, 1))
            ax.set_axis_off()
            ax.set_title(title, fontsize=11, pad=8)
            if sub:
                ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top",
                        fontsize=8.5, color="0.35")
        fig.subplots_adjust(left=0.005, right=0.995, wspace=0.012,
                            top=1.0 - head_px / (fig_h * 100.0),
                            bottom=foot_px / (fig_h * 100.0))
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)

        if writer is None:
            h, w = img.shape[:2]
            writer = _open_writer(mp4, w - (w % 2), h - (h % 2), args.fps or fps_in / args.stride)
        writer.stdin.write(img[: img.shape[0] - (img.shape[0] % 2),
                               : img.shape[1] - (img.shape[1] % 2)].tobytes())
        if (k + 1) % 10 == 0:
            print(f"  composed {k + 1}/{n}")

    cap.release()
    overlay.close()
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    print(f"wrote {paths.relative(mp4)}")
    return mp4


def _open_writer(path, w, h, fps):
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def to_gif(mp4: Path, gif: Path, fps: int, width: int, colors: int,
           speed: float = 1.0, dither: bool = True) -> int:
    """Two-pass palette encode. GitHub renders a GIF inline but not an mp4.

    `speed` is an setpts multiplier, so 0.4 plays the clip 2.5x faster; a long clip is
    otherwise too many frames to keep a GIF inside GitHub's comfortable size. Dithering
    costs about a fifth of the file and only pays for itself on photographic content, so
    panels of flat colour are better off without it.
    """
    step = f"setpts={speed}*PTS," if speed != 1.0 else ""
    use = "dither=bayer:bayer_scale=3" if dither else "dither=none"
    vf = (f"{step}fps={fps},scale={width}:-2:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
          f"[b][p]paletteuse={use}")
    subprocess.run(["ffmpeg", "-y", "-i", str(mp4), "-vf", vf, "-loop", "0", str(gif)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return gif.stat().st_size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--out", default=str(paths.RESULTS_DIR / "compare"))
    ap.add_argument("--gif", default=None,
                    help="also write a GIF here (default: docs/media/ in the parent repo)")

    ap.add_argument("--a-name", default="Ours (token classifier)")
    ap.add_argument("--a-checkpoint", default=str(DEFAULT_A_CKPT))
    ap.add_argument("--a-model-config", default=str(DEFAULT_A_CFG))
    ap.add_argument("--a-run", default="chain_st_hard", help="manifest run id for the caption")

    ap.add_argument("--b-name", default="TokenHMR (released)")
    ap.add_argument("--b-checkpoint", default=str(DEFAULT_B_CKPT))
    ap.add_argument("--b-model-config", default=str(DEFAULT_B_CFG))
    ap.add_argument("--b-run", default="", help="manifest run id for the caption, if any")

    ap.add_argument("--max-frames", type=int, default=84)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--det-size", type=int, default=1024)
    ap.add_argument("--score-thr", type=float, default=0.55)
    ap.add_argument("--iou-thr", type=float, default=0.35)
    ap.add_argument("--max-gap", type=int, default=6)

    ap.add_argument("--width", type=int, default=720, help="output width in pixels")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--gif-colors", type=int, default=128)
    ap.add_argument("--recompute-detect", action="store_true")
    ap.add_argument("--recompute-infer", action="store_true")
    args = ap.parse_args(argv)

    for label, p in [("video", args.video),
                     ("A checkpoint", args.a_checkpoint), ("A config", args.a_model_config),
                     ("B checkpoint", args.b_checkpoint), ("B config", args.b_model_config)]:
        if not Path(p).exists():
            print(f"missing {label}: {p}", file=sys.stderr)
            return 2

    from tokenhmr.visualize_video_latents import _smpl_faces

    boxes, valid = stage_tracks(args)
    A = stage_infer(args, boxes, valid, "ours", args.a_checkpoint, args.a_model_config)
    B = stage_infer(args, boxes, valid, "upstream", args.b_checkpoint, args.b_model_config)

    mp4 = compose(args, boxes, valid, A, B,
                  _smpl_faces(args.a_model_config), _smpl_faces(args.b_model_config))

    gif = Path(args.gif) if args.gif else (
        paths.PROJECT_ROOT.parent.parent / "docs" / "media" / "gymnasts_ours_vs_upstream.gif"
    )
    gif.parent.mkdir(parents=True, exist_ok=True)
    size = to_gif(mp4, gif, int(args.fps), args.width, args.gif_colors)
    print(f"wrote {gif}  ({size / 1e6:.1f} MB)")
    if size > 8e6:
        print("  over 8 MB. Re-run with --gif-colors 96, then --fps 10, then --width 640.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
