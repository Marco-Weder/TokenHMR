# Reproducing the thesis results

Every table and figure in the thesis, with the command that produces it.

Commands assume the `thesis-HMR` environment and an editable install (see the
[README](../../../README.md)). They run from any working directory, since every path is
resolved from the repository root rather than from the shell's location:

```bash
conda activate thesis-HMR
thesis doctor          # check the environment, data and provenance first
```

Offscreen mesh rendering needs `PYOPENGL_PLATFORM=egl`.

## Before anything else: what is already recorded

Most numbers do not need to be recomputed. The manifest holds every measurement the
thesis reports, built from evaluation outputs already on disk:

```bash
thesis manifest check        # every published value against every measurement
thesis manifest tokenizers   # the eight stage-1 tokenizers and their checkpoints
thesis manifest runs         # all 83 downstream runs
thesis golden check          # the deterministic analyses, value by value
```

`thesis manifest check` passing means the code still produces the numbers in the
document. Start there, and recompute only what you actually want to change.

## Order of work

1. **Stage 1**, train the tokenizers. The ablation needs all eight; everything downstream
   needs only the FSQ `d = 4` one.
2. **Tokenizer analysis** (Sections 4.2 and 4.3). CPU, reads stage-1 checkpoints.
3. **Stage 2**, train the image models. The expensive part: each from-scratch arm is
   200k steps, roughly 14 h on one GPU.
4. **Downstream analysis and figures** (Section 4.4).

Steps 2 and 4 only need re-running if you have changed what they read.

## Stage 1: tokenizers

```bash
python -m tokenization.train_poseVQ --cfg tokenization/configs/<config>.yaml
```

Output lands in `tokenization/output/<experiment>/<experiment>_ID00_<timestamp>/`, with
`best_net.pth` and a resumable `latest_checkpoint.pth`. The timestamp is deliberate: it is
what distinguishes two trainings of the same configuration, so it is never flattened.

Give the result a stable name before using it downstream:

```bash
thesis manifest link-tokenizers
```

which creates `tokenization/output/tokenizers/<name>/best_net.pth`. The experiment configs
refer to tokenizers through those names, so nothing has to be repointed by hand.

| Config | Thesis row |
| --- | --- |
| `tokenizer_amass_moyo_original.yaml` | Convolutional baseline, EMA `l2` |
| `tokenizer_amass_moyo_transformer.yaml` | Transformer, EMA `l2` and EMA cosine `d256` |
| `tokenizer_amass_moyo_transformer_masked.yaml` | Skeleton-masked, EMA cosine |
| `tokenizer_amass_moyo_transformer_dim4.yaml` | Transformer, EMA cosine `d4` |
| `tokenizer_amass_moyo_transformer_dim2.yaml` | Transformer, EMA cosine `d2` |
| `tokenizer_amass_moyo_fsq.yaml` | Transformer, FSQ `d4` (carried forward) |
| `tokenizer_amass_moyo_fsq_16k.yaml` | Transformer, FSQ `d5` |

`calculate_codebook_utilization.py` needs the `results_<VALLIST>.pkl` a validation pass
writes (~89 MB, not version controlled). The geometry script below reports utilisation
too, so it is rarely needed.

## Tokenizer ablation and latent-space analysis

Run the stability analysis first: it establishes the shared pose set the others reuse.

| Thesis artefact | Command | Notes |
| --- | --- | --- |
| Tab. 4.3 stability, `Δ_NN`, usage | `python -m tokenization.analysis.analyze_token_stability` | writes `output/token_stability/summary.json` |
| Tab. 4.4 codebook geometry | `python -m tokenization.analysis.analyze_codebook_geometry` | 8 tokenizers, 2048 val poses, seed 0. CPU |
| Tab. B.1 reconstruction by dataset | `python -m tokenization.analysis.analyze_recon_by_dataset` | writes `output/recon_by_dataset/summary_all.json` |
| Fig. 4.3 recon/stability frontier | `thesis figure recon-frontier` | reads the tokenizer registry, not a hard-coded list |
| Fig. 4.4 code separation | `python -m tokenization.analysis.dump_code_separation` then `thesis figure code-separation` | dump writes `output/codebook_geometry/separation_arrays.npz` |
| Fig. 4.5 and Fig. B.2 influence maps | `python -m tokenization.analysis.analyze_token_maps` then `thesis figure influence-maps` | analysis writes `output/token_maps/<label>.npz` |
| Fig. B.4 latent scatter | same `.npz`, then `thesis figure latent-scatter` | |
| Fig. 4.6 and Fig. B.3 token displacement | `python -m tokenization.analysis.render_token_viz` then `thesis figure token-viz` | the first decodes on CPU and caches the geometry, the second renders in Blender |
| Fig. B.1 all 160 token positions | `python -m tokenization.analysis.visualize_token_effects` then `thesis figure token-map-page` | the page script re-tiles panels the first rendered; no GPU |

Single-run exploratory versions, superseded by the drivers above but kept:
`tokenization/analysis/analyze_codebook.py`, `visualize_latent_space.py`,
`analyze_latent_pose_info.py` (the last is also the shared library the others build on).

## Stage 2: the image model

```bash
python -m tokenhmr.train experiment=<name>
```

| Thesis artefact | Experiment config | Run name |
| --- | --- | --- |
| Tab. 4.9 row 1, token CE only, no gate | `tokenhmr_additive` (phase A) | `chain_nogate` |
| Tab. 4.9 row 2, `+` label-purity gate | `tokenhmr_additive` (phase B) | `chain_gate` |
| Tab. 4.9 row 3, `+` straight-through | `tokenhmr_additive` (phase D) | `chain_st` |
| Tab. 4.9 row 4, `+` Gumbel sampling | `tokenhmr_additive` (phase E) | `chain_gumbel` |
| Tab. 4.10 rows 2-4, other tokenizers | `tokenhmr_additive` with a different tokenizer | `cetok_tfl2`, `cetok_cnn`, `cetok_cosd4` |
| Tab. 4.7 pose-supervised baselines | `tokenhmr_fsq`, `tokenhmr_transformer_*` | see `thesis manifest runs` |
| Tab. B.2 continuous reference | `tokenhmr_continuous_hmr2` | `continuous_400k` |
| Tab. B.5 decoder-aware targets | `tokenhmr_decoder_aware` | `dec_aware_w_st` |

The queues run a whole study end to end and evaluate each arm as it finishes. They are
resumable: an arm whose run directory already holds a checkpoint is skipped.

```bash
bash tools/queues/run_additive_queue.sh        # Tab. 4.9
bash tools/queues/run_cetokenizer_queue.sh     # Tab. 4.10
bash tools/queues/run_continuous_baseline.sh   # Tab. B.2
```

Evaluation:

```bash
python tokenhmr/eval.py \
    --checkpoint logs/<task>/runs/<run>/checkpoints/last.ckpt \
    --model_config logs/<task>/runs/<run>/model_config.yaml \
    --dataset EMDB --decode_mode hard --exp_name <name>
```

`--exp_name` is required and names the results directory. Output goes to
`results/release/<name>/eval_regression.csv`, holding `(hard_|soft_)mode_re`,
`mode_mpjpe` and `mode_pve` for PA-MPJPE, MPJPE and PVE, plus `token_top1_acc`,
`token_top5_acc` and `token_pred_entropy`. Rebuild the manifest afterwards with
`thesis manifest build`.

A run trained on another machine re-evaluates here unchanged: the absolute paths its
saved config recorded are repointed as the config is read.

## Downstream analysis and figures

| Thesis artefact | Command | Notes |
| --- | --- | --- |
| Fig. 4.8 token ambiguity | `python -m tokenhmr.analyze_token_ambiguity` then `thesis figure token-ambiguity` | analysis dumps `results/ambiguity_gate/damages.npz` |
| Fig. 4.7 decoding mechanism | `thesis figure decoding` then `python -m thesis_figures.render.render_decode_candidates --sheet` / `--pick J` | the sheet lets the frame be chosen by eye |
| Fig. B.5 what tokenization buys | `python -m thesis_figures.scan.scan_token_benefit` then `thesis figure token-benefit` | scan spreads frames across the whole test set |
| Fig. B.6 qualitative results | `python -m thesis_figures.scan.scan_qualitative` then `thesis figure qualitative` | `--stats`, `--sheet --by <criterion>`, `--picks` |
| Qualitative decode comparison | `python -m thesis_figures.render.render_decode_comparison` then `python -m thesis_figures.plot.make_decode_figure` | not a thesis figure; a browsing set with a `ranking.csv` |

## Methods figures

| Thesis artefact | Command | Output |
| --- | --- | --- |
| Fig. 3.1 SMPL decomposition | `thesis figure smpl` | `images/smpl/` (needs Blender / `bpy`) |
| Fig. 3.8 skeleton attention mask | `thesis figure skeleton-mask` | `images/tokenizer/` and the inline TikZ in `thesis_figures/render/skeleton_mask_tikz.tex` |

Figures are written into the LaTeX checkout if one is present next to this repository,
and into `figures/` otherwise. `--out-dir` overrides both, and `thesis paths` prints what
would be used.

## The video comparison

```bash
thesis compare-video
```

Renders this thesis's token classifier against the released TokenHMR model on
`demo_sample/video/gymnasts.mp4`. Detection and tracking run once and both models consume
the identical boxes and frames, so the only difference between the panels is the pose
model. Writes an mp4 and a GIF for the README.

## Interactive tools

Not thesis artefacts, but useful for inspecting a tokenizer:

- `python -m tokenization.analysis.interactive_token_viewer` — a web page to replace a
  single token and watch the decoded body change.
- `python -m tokenhmr.visualize_video_latents` — video of the recovered mesh beside the
  live FSQ codes.

## Known gaps

None outstanding. The four gaps this file previously listed (three projected rows of
Tab. 4.10, the decoder-aware table, the continuous-regression reference, and the
qualitative appendix figure) are all measured or produced, and `thesis manifest check`
verifies the reported values against the measurements.

One dependency remains awkward. The stage-1 analyses read the pose data through the config
that trained the tokenizer, so they need AMASS and MOYO present even when only re-deriving
a table. The manifest exists partly so that is avoidable: if you only need the numbers,
read them from `manifest/` rather than recomputing.
