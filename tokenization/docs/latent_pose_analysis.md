# Latent-space pose analysis

*How is the latent space storing pose information?* — the question from the supervisor
meeting. This is the diagnostic layer above codebook health: `analyze_codebook.py` asks
whether the **codebook** is healthy (usage / geometry / margins);
[`analyze_latent_pose_info.py`](../analyze_latent_pose_info.py) asks what pose information
the **latent space** carries and how it is organized.

One self-contained script, six analyses, runs on any tokenizer checkpoint (cosine/L2 EMA,
FSQ, masked, GNN) because it only touches the shared `encode` / `decode` / `quantize`
surface. No sklearn/umap — PCA and ridge regression are done in torch.

## Run

```bash
cd external/tokenhmr/tokenization
python analyze_latent_pose_info.py --ckpt output/<run>/.../best_net.pth
# options:
#   --out DIR            output dir           (default <ckpt_dir>/latent_analysis)
#   --analyses a,b,c     subset of the six    (default all)
#   --num-poses 4096     val poses to cache   (balanced across the 5 val datasets)
#   --influence-poses 512  poses for the influence map (the one O(tokens) heavy loop)
#   --quick              tiny smoke-test run
```

Outputs `stats.json` + nine PNGs into `--out`. Cross-checkpoint comparison = run it on each
variant and compare the JSONs / figures (the interesting axis for the thesis).

## The six analyses

| # | key | figure(s) | what it measures |
|---|-----|-----------|------------------|
| 1 | `influence` | `1_influence_*` | **token → joint influence.** Swap each of the 160 latent tokens for a random other pose's token and measure per-joint rotation change (deg). Answers *where is each joint stored* and whether the code is spatially factored or entangled. |
| 2 | `recon` | `2_*` | **per-joint reconstruction error** (position mm + rotation deg), stratified by dataset. Answers *which pose information survives the bottleneck*. |
| 3 | `probe` | `3_linear_probe_r2` | **linear probe** of the latents for per-joint pose, pre- vs post-quantization. Pre/post gap = information the codebook destroys; high R² = the space is linearly organized, not a lookup table. |
| 4 | `interp` | `4_interpolation` | **interpolation smoothness.** Interpolate between pose pairs in the latent, measure per-step joint jump + path straightness. Smooth/straight = a pose manifold. |
| 5 | `language` | `5_token_language` | **token "language" structure.** Per-position usage entropy, distinct codes per position, and inter-token redundancy (normalized MI, adjacent vs random pairs). Gives a bits/pose rate to pair with recon. |
| 6 | `embed` | `6_*` | **2-D PCA** of per-pose latents colored by dataset and by knee flexion, plus the codebook colored by usage. The qualitative "is it organized" picture. |

### Reading the numbers
- **influence:** `token_specialization_median` near 1 and a low `joint_spread_median_tokens_for_80pct`
  ⇒ spatially factored (each token ≈ one joint). Near 0 / high spread ⇒ distributed/entangled.
- **probe:** `quant_info_loss_R2_mean` ≈ 0 ⇒ quantization barely costs anything (the key number
  for the dim-256 → dim-4 question). Per-joint gap shows *which* joints the codebook hurts.
- **language:** `adjacent_token_norm_MI_mean` ≈ `random_pair_norm_MI_mean` ⇒ tokens are
  ~independent, so `rate_bits_per_pose_upper_bound` (Σ per-position entropy) is a tight rate
  estimate. Adjacent ≫ random ⇒ local redundancy, true rate is lower.
- **interp:** `straightness_median` near 1 ⇒ smooth/direct latent paths.

## Worked example — dim-4 cosine run (`tokenization_transformer_cosine_dim4`)

5120 val poses balanced over HumanEva/HDM05/SFU/MPI-Mosh/MOYO, 512 for the influence map.

- **Spatially factored.** 93% of tokens put >50% of their influence on a single joint
  (median specialization 0.74); ~9 tokens cover 80% of a joint. The influence heatmap is
  sparse and structured, with the highest per-token influence on distal joints
  (elbows / wrists / shoulders / neck).
- **Near-lossless quantization.** Linear-probe R² = 0.992 pre-quant vs 0.987 post-quant —
  a **0.005** mean R² drop from quantizing to a 4-dim code. Pose is stored almost linearly.
  L_Elbow loses the most (~0.016); feet the least. This directly supports "dim-4 is enough".
- **Extremes are what's discarded.** MOYO (yoga) reconstructs at **3.85 mm** vs ~1 mm for the
  others; wrists are the worst joints (~2.8 mm), hips/spine near-perfect. In the latent PCA
  MOYO occupies its own region and PC1 tracks knee flexion smoothly (0°→160°).
- **Smooth + high-rate code.** Interpolation straightness ≈ 1.14 (direct paths). Per-position
  entropy ≈ 8.3 of 11 bits, 2046/2048 codes used, adjacent-token nMI (0.076) ≈ random-pair
  nMI (0.066) ⇒ tokens near-independent ⇒ ~1326 bits/pose, essentially all sequences unique.

Foot **rotation** error ≈ 0° while foot **position** error is ~1.3 mm is expected, not a bug:
the toe joints are barely articulated in mocap (GT ≈ identity), so position error there is
driven by ankle/knee upstream.

## Notes / caveats
- Dataset stratification builds one single-dataset loader per `VALLIST` entry (the merged
  `ValDataset` carries only one merged `dataset_name`), sampling a balanced shuffled slice
  from each; SMPL runs per batch (`CACHE_SMPL=False`) so we don't precompute meshes we discard.
- Influence uses a random donor pose per token (not the adjacent frame — val data is
  unshuffled, so neighbours are near-duplicate frames and would show ~0 influence).
- Inter-token MI collapses each position to its top-15 codes + "other" before estimating, to
  avoid the finite-sample blow-up of a raw 2048×2048 joint histogram.
- For large `code_dim` (e.g. the 256-dim run) the probe/PCA features are PCA-reduced to
  ≤1024 dims first; dim-4 (160×4 = 640) is used directly.
