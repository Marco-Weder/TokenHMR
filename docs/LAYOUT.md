# Repository layout

Everything below is inside the fork (`external/tokenhmr/` when seen from the
parent repository). The parent holds documentation, the submodule pin and one
media file; all the code is here.

```
.project-root          anchors PROJECT_ROOT. pyrootutils and repro.paths both use it.
pyproject.toml         the single distribution. `pip install -e .` exposes the packages below.

repro/                 shared runtime, imported by everything else
  paths.py             the one path oracle: explicit argument, then env var, then sentinel
  legacy.py            repoints paths recorded by earlier runs at this checkout
  manifest.py          reads the provenance manifest
  build_manifest.py    rebuilds it from evaluation results already on disk
  golden.py            snapshots and verifies the deterministic analysis outputs
  cli.py               the `thesis` command

manifest/              provenance, version controlled
  eval_index.csv       every measured number, one row per run/dataset/metric
  tokenizers.yaml      the eight stage-1 tokenizers and their checkpoints
  thesis_numbers.yaml  the values the thesis prints, transcribed by hand
  golden/              numeric digests of the analysis outputs

env/                   one coherent environment story, see system-requirements.md
docs/                  this file, REPRODUCE.md, MANIFEST.md, design notes
tools/
  queues/              the long-running training queues
  prepare/             AMASS and MOYO preprocessing

tokenization/          STAGE 1: the pose tokenizer
  train_poseVQ.py      training entry point
  configs/             12 tokenizer configurations
  models/              encoders, decoders and quantizers (l2, cosine, FSQ)
  utils/  options/  dataset/
  analysis/            14 scripts that measure a trained tokenizer
  output/              checkpoints and analysis results (not version controlled)

tokenhmr/              STAGE 2: the image model that predicts tokens
  train.py  eval.py    entry points
  demo.py  track.py    upstream image and video demos
  compare_video.py     the side-by-side video in the README
  visualize_video_latents.py
  analyze_token_ambiguity.py
  lib/                 the model, datasets, losses and Hydra configs

thesis_figures/        figure scripts, split by what they need to run
  plot/                matplotlib only, runs on a laptop
  render/              needs a GPU with EGL, or Blender
  scan/                needs a GPU, caches frame candidates for render/

figures/               where figures land when no LaTeX checkout is present
data/                  body models and released checkpoints  (6 GB)
dataset_dir/           training and evaluation data          (825 GB)
logs/                  stage-2 runs and checkpoints          (586 GB)
results/               evaluation output                     (10 GB)
```

## Two conventions worth knowing

**Nothing depends on the working directory.** Every path comes from
`repro.paths`, which resolves an explicit argument first, then a `THESIS_*`
environment variable, then a default anchored on `.project-root`. Run anything
from anywhere. `thesis paths` prints what resolved.

**Names that look redundant are not.** Stage-1 checkpoints live under
`tokenization/output/<config>/<config>_ID00_<timestamp>/`, and the timestamp is
what distinguishes two trainings of the same configuration. It is never
flattened. Configs refer to tokenizers through the stable aliases in
`tokenization/output/tokenizers/<name>/`, which `thesis manifest
link-tokenizers` creates.

## Where the entry points are

| Task | Command |
| --- | --- |
| Check the install | `thesis doctor` |
| Train a tokenizer | `python -m tokenization.train_poseVQ --cfg tokenization/configs/<name>.yaml` |
| Measure a tokenizer | `python -m tokenization.analysis.analyze_token_stability` |
| Train the image model | `python -m tokenhmr.train experiment=<name>` |
| Evaluate a checkpoint | `python tokenhmr/eval.py --checkpoint ... --exp_name ...` |
| Render a figure | `thesis figure <name>` |
| Look up a number | `thesis manifest check` |

`docs/REPRODUCE.md` maps each thesis table and figure to the command that
produces it.
