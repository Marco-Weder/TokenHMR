"""Populate the detection and inference caches for `visualize_joint_tokens`.

The two network stages are the slow part and they do not depend on how the figure is
drawn, so running them once up front lets the layout be iterated on cache hits alone.
`visualize_joint_tokens.py` reads exactly these files and skips both stages when it
finds them.
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tokenhmr.visualize_video_latents import detect_and_track, run_inference  # noqa: E402

VIDEO = "demo_sample/video/20260816_193019.mp4"
CKPT = "logs/tokenhmr_chain2_st/runs/chain_st/checkpoints/last.ckpt"
CFG = "logs/tokenhmr_chain2_st/runs/chain_st/model_config.yaml"
OUT = Path("results/joint_tokens")
STRIDE, MAX_PEOPLE, SMOOTH, DET_SIZE, BATCH = 2, 1, 5, 800, 8


def main() -> int:
    import cv2

    from tokenhmr.visualize_video_latents import output_tags

    tok_tag, run_tag = output_tags(CFG, CKPT)
    sub = OUT / tok_tag
    sub.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    n_frames = total // STRIDE
    print(f"{VIDEO}: {total} frames @ {fps:.1f} fps -> {n_frames} at stride {STRIDE}")

    vkey = f"{Path(VIDEO).stem}_s{STRIDE}"
    f_tracks = OUT / f"cache_tracks_{vkey}.npz"
    if f_tracks.exists():
        z = np.load(f_tracks)
        boxes, valid = z["boxes"], z["valid"]
        print(f"stage 1/2  cached ({f_tracks})")
    else:
        boxes, valid = detect_and_track(VIDEO, n_frames, STRIDE, DET_SIZE, 0.5, 0.35, 8)
        np.savez_compressed(f_tracks, boxes=boxes, valid=valid)
        print(f"stage 1/2  wrote {f_tracks}")
    boxes, valid = boxes[:, :MAX_PEOPLE], valid[:, :MAX_PEOPLE]
    print(f"  tracked frames: {int(valid[:, 0].sum())}/{len(valid)}")

    f_inf = sub / f"cache_infer_{vkey}_{run_tag}_p{MAX_PEOPLE}_sm{SMOOTH}.npz"
    if f_inf.exists():
        print(f"stage 2/2  cached ({f_inf})")
        return 0
    data = run_inference(VIDEO, boxes, valid, STRIDE, CKPT, CFG, BATCH, SMOOTH)
    np.savez_compressed(f_inf, **data)
    print(f"stage 2/2  wrote {f_inf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
