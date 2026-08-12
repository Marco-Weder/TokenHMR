# Latent-space visualization & distance-based entropy (`visualize_latent_space.py`)

Addresses the supervisor's two questions: **are the codes very different from each
other**, and **can we compute an entropy from the latent distribution over distances**.

## The distance-based entropy ("is this possible?" — yes)

Every encoder latent `z` is soft-assigned to all codes with a Boltzmann distribution
over distances in the quantizer's own metric,

```
p(k|z) = softmax( -d(z, c_k) / tau )
```

which yields three quantities (reported in bits, per token):

| quantity | meaning | healthy |
|---|---|---|
| `H(k)` — usage entropy | entropy of the marginal `E_z[p(k|z)]`; its `tau -> 0` limit is the classic hard usage entropy / perplexity | close to `log2(K)` |
| `H(k|z)` — confusion | mean entropy of one latent's assignment; high = neighboring codes are indistinguishable at scale `tau` | low |
| `I(z;k) = H(k) - H(k|z)` | information a discrete code actually carries about its latent | close to `log2(K)` |

"Codes distinct AND all used" is exactly the high-`H(k)`, low-`H(k|z)` regime.
`tau` is swept in units of the median nearest-neighbor code distance
(`entropy_vs_temperature.png`); headline numbers use `tau = 0.5 x NN dist`.

Complementary direct evidence (`code_separation.png`): histograms of all-pair code
distances, nearest-neighbor code distances, and latent-to-assigned-code distances
(quantization error). Codes are "different" when the quantization-error mass sits
left of the NN-code distances — summarized by the **separation ratio**
`median(NN code dist) / median(quant err)`.

Numerics: cosine models use the chordal distance `sqrt(2(1-cos))` in float64 —
neighboring codes can be ~1e-3 apart, where fp32 `1 - cos` underflows to 0. FSQ
latents are mapped through `bound(z)/half_width` into the codes' normalized [-1,1]
grid space before measuring anything.

## Usage

```bash
# from tokenization/, env thesis-HMR
python visualize_latent_space.py --ckpt output/<run>/.../best_net.pth [--ckpt ...]
```

Per-run figures + `latent_space_stats.json` go to `<ckpt_dir>/latent_space_viz/`;
with several `--ckpt` a comparison figure + `summary.json` go to
`output/latent_space_comparison/`. Geometry figures adapt to the code dimension:
d=2 direct scatter (+ angular view for cosine), d<=4 all dim pairs, else PCA fit on
the latents with the codebook projected into the same basis.

## Results (2026-07-15, 2048 val poses x 160 tokens, tau = 0.5 x NN dist)

| run | K | d | util % | norm H(k) | separation | H(k\|z) bits | I(z;k) bits | log2 K |
|---|---|---|---|---|---|---|---|---|
| Noise0_warmrestart (l2, 256d) | 2048 | 256 | 94.2 | 0.898 | 10.6x | 1.93 | **8.76** | 11.0 |
| masked (cosine, 256d) | 2048 | 256 | 100.0 | 0.917 | 6.6x | 4.33 | 6.07 | 11.0 |
| cosine_dim4 | 2048 | 4 | 99.8 | 0.939 | 3.4x | 3.62 | 6.94 | 11.0 |
| cosine_dim2 | 2048 | 2 | 99.8 | 0.927 | 6.7x | 1.91 | 8.34 | 11.0 |
| fsq [8,8,6,5] | 1920 | 4 | 85.2 | 0.713 | 1.8x | 4.90 | 4.68 | 10.9 |
| fsq_16k [8,8,8,6,5] | 15360 | 5 | 23.2 | 0.516 | 7.4x | 5.47 | 5.16 | 13.9 |

Readings:

- **All EMA runs pass the distinctness test** (separation > 1): latents land several
  times closer to their assigned code than the nearest other code sits.
- **dim2 is a circular ruler.** The cosine metric leaves 1 degree of freedom; all
  2048 codes line up on the unit circle ~0.1 deg apart (median NN chordal 0.0017).
  Codes are *not* far apart in absolute terms — but the encoder is so precise
  (32% of latents coincide with their code at fp32 resolution) that assignments
  still carry 8.3 bits. Distinctness here is a statement about encoder precision,
  not code geometry.
- **dim4** codes tile a curved 3-manifold of the unit 3-sphere; separation drops to
  3.4x and confusion triples vs dim2 — the extra dimensions are used, but codes
  crowd locally.
- **FSQ latents saturate the grid boundary** (tanh bound): corner/edge codes absorb
  most mass, interior codes starve. fsq_16k uses 23% of its codebook (perplexity
  145 of 15360) and both FSQ runs carry the least information per token.
- Connection to the stage-2 token top-1/top-5 accuracy
  (`lib/utils/token_metrics.py`): `H(k)` is the entropy of the target the predictor
  must hit, and high `H(k|z)` means GT tokens are intrinsically ambiguous (an
  imperceptibly different pose flips the token) — so low top-1 with good mesh
  metrics is expected for runs with high confusion (fsq, dim4, masked).
