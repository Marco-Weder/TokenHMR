#!/usr/bin/env python
"""Re-tile the per-token mesh error maps into a page-shaped figure (fig:app-token-map).

`tokenization/visualize_token_effects.py` renders one panel per token position and
tiles them 10 wide. With 160 portrait panels that grid comes out far taller than the
text block, so it has to be scaled down to fit the page height and then uses only part
of the text width. This script re-tiles the panels that script already wrote (no GPU,
no re-render) into 11 columns, the count whose grid aspect is closest to the text
block's, which is what maximises the size each panel is finally rendered at.

The colour bar is cropped straight out of the original montage rather than rebuilt, so
the scale is guaranteed to be the one the panels were coloured with.

Inputs (from a visualize_token_effects.py run):
    <run>/token_viz/panels/token_XXX.png
    <run>/token_viz/neutral_token_montage.png   (colour bar only)
    <run>/token_viz/token_sensitivity.json      (per-token mean displacement)

Output: <repo>/thesis/images/appendix/token_map_all_positions.png

Run with the project env (from the tokenhmr folder):
    python -m thesis_figures.plot.plot_token_map_page
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from repro import paths

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENHMR = os.path.dirname(HERE)
REPO = os.path.dirname(TOKENHMR)

RUN = os.path.join(str(paths.TOKENIZER_OUT), "tokenization_transformer_fsq",
    "tokenization_transformer_fsq_ID00_13-05-2026_00-29-18",
    "tokenization_transformer_fsq", "token_viz")
OUT = os.path.join(str(paths.figure_out_dir("appendix")), "token_map_all_positions.png")

COLS = 11                 # chosen so the grid aspect matches the text block, which
                          # is what maximises the rendered panel size (see module docstring)
LABEL_H = 26              # white strip above each panel for the token index
PAD = 3                   # gap between panels
BG = (255, 255, 255)


def shared_crop_box(imgs, thresh=245):
    """One tight box around the body, shared by every panel so they stay aligned."""
    r0, r1, c0, c1 = 10**9, -1, 10**9, -1
    for im in imgs:
        a = np.asarray(im.convert("RGB"))
        nz = (a < thresh).any(axis=2)
        rs, cs = np.where(nz)
        r0, r1 = min(r0, rs.min()), max(r1, rs.max())
        c0, c1 = min(c0, cs.min()), max(c1, cs.max())
    return r0, r1 + 1, c0, c1 + 1


def colourbar_strip(montage_path):
    """Crop the colour bar (gradient + tick labels) out of the original montage."""
    a = np.asarray(Image.open(montage_path).convert("RGB")).astype(int)
    h, w, _ = a.shape
    # the bar is the only tall, strongly saturated column block on the right margin
    sat = (a.max(axis=2) - a.min(axis=2)) > 60
    colcount = sat[:, int(w * 0.85):].sum(axis=0)
    hits = np.where(colcount > h * 0.2)[0]
    if len(hits) == 0:
        return None
    x0 = int(w * 0.85) + hits.min() - 12
    rows = np.where((a[:, x0:] < 245).any(axis=2).any(axis=1))[0]
    return Image.fromarray(a[rows.min():rows.max() + 1, x0:w].astype(np.uint8))


def main():
    paths = sorted(os.path.join(RUN, "panels", f)
                   for f in os.listdir(os.path.join(RUN, "panels"))
                   if f.startswith("token_") and f.endswith(".png"))
    imgs = [Image.open(p) for p in paths]
    n = len(imgs)
    r0, r1, c0, c1 = shared_crop_box(imgs)
    panels = [im.convert("RGB").crop((c0, r0, c1, r1)) for im in imgs]
    pw, ph = panels[0].size

    with open(os.path.join(RUN, "token_sensitivity.json")) as f:
        sens = {e["position"]: e["vertex_disp_mm"] for e in json.load(f)["ranking"]}

    rows = (n + COLS - 1) // COLS
    grid_w = COLS * pw + (COLS + 1) * PAD
    grid_h = rows * (ph + LABEL_H) + (rows + 1) * PAD

    bar = colourbar_strip(os.path.join(RUN, "neutral_token_montage.png"))
    if bar is not None:                       # scale the bar to the grid height
        bw = int(bar.width * (grid_h * 0.8) / bar.height)
        bar = bar.resize((bw, int(grid_h * 0.8)), Image.LANCZOS)

    canvas = Image.new("RGB", (grid_w + (bar.width + 18 if bar else 0), grid_h), BG)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for i, p in enumerate(panels):
        r, c = divmod(i, COLS)
        x = PAD + c * (pw + PAD)
        y = PAD + r * (ph + LABEL_H + PAD)
        draw.text((x + pw / 2, y + LABEL_H / 2), str(i), fill=(0, 0, 0),
                  font=font, anchor="mm")
        canvas.paste(p, (x, y + LABEL_H))
        draw.rectangle([x, y, x + pw - 1, y + LABEL_H + ph - 1],
                       outline=(210, 210, 210))

    if bar is not None:
        canvas.paste(bar, (grid_w + 18, (grid_h - bar.height) // 2))

    canvas.save(OUT, optimize=True)
    print(f"{n} panels, {COLS}x{rows} grid -> {canvas.size}")
    print("mean per-token displacement: "
          f"{np.mean(list(sens.values())):.2f} mm, max {max(sens.values()):.2f} mm")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
