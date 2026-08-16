# Skeleton-masked transformer — the GNN as an attention mask

Second take on the "skeleton inductive bias" experiment. The first take was a dedicated GNN
architecture ([`gnn_tokenizer.md`](gnn_tokenizer.md)), which showed overfitting at ~5k iterations
(val loss rising while train loss falls). This version gets the same inductive bias **without a new
architecture**: the existing transformer tokenizer, with joint self-attention masked to the
kinematic tree.

Files: mask builder [`utils/skeleton.py`](../utils/skeleton.py) (`build_skeleton_attention_mask`),
mask plumbing [`tokenhmr/lib/models/components/pose_transformer.py`](../../tokenhmr/lib/models/components/pose_transformer.py),
model flags [`models/transformer_pose_vqvae.py`](../models/transformer_pose_vqvae.py),
config [`configs/tokenizer_amass_moyo_transformer_masked.yaml`](../configs/tokenizer_amass_moyo_transformer_masked.yaml).

---

## 1. The one-sentence idea

A transformer layer whose attention is restricted to the edges of a graph **is** a Graph Attention
Network (GAT) layer — so instead of maintaining `gnn_pose_vqvae.py`, we mask the attention scores of
`TransformerTokenizer` with the skeleton adjacency: before the softmax, every joint-pair that is not
bone-connected gets `-inf`.

```
dots = q @ k.T * scale
dots = dots.masked_fill(~skeleton_mask, -inf)    # <- the entire "GNN"
attn = softmax(dots)
```

Compared to the dedicated GNN this has two big advantages:

1. **Zero new parameters.** The masked run has *byte-for-byte the same parameter set* as the
   256-dim transformer baseline (`tokenizer_amass_moyo_transformer.yaml`, run
   `tokenization_transformer_with_fnn_blocks`); CODE_DIM 256 also matches the GNN run. Any
   difference in the val curves is attributable to the mask alone — a perfectly controlled
   experiment.
2. **Learned, per-head edge weights.** A GCN aggregates neighbours with fixed weights
   (`A_norm @ H`); masked attention learns *how much* each neighbour matters, per head and per pose.

## 2. The mask

`build_skeleton_attention_mask(num_joints=21, n_hops=1, connect_pelvis_siblings=True)` returns a
boolean `(21, 21)` matrix, `True` = may attend:

- **Edges**: undirected parent links from `SMPLH_PARENTS_21` (same single source of truth as the
  GNN and the kinematic PE).
- **Pelvis siblings**: the root pelvis is not one of the 21 joints, so the raw tree is actually
  **three disconnected components** (left leg, right leg, upper body). L_Hip (0), R_Hip (1) and
  Spine1 (2) all attach to the pelvis, so they are connected pairwise to restore one body graph.
  (The GraphConv GNN did *not* do this — its legs never talked to the torso inside the encoder.)
- **Self-loops**: always on; no attention row is ever fully masked (no softmax NaNs).
- **`n_hops`**: closure to the n-hop neighbourhood. With `DEPTH: 2` encoder layers the receptive
  field is `2 × n_hops` bones; the config grids over `[1, 2]` (pure GNN vs. the 4-hop receptive
  field of the `GNN_LAYERS: 4` run).

Where the mask applies — only where tokens *are joints*:

| stage                                   | tokens         | masked? |
|-----------------------------------------|----------------|---------|
| encoder self-attention                  | 21 joints      | ✅ |
| cross-attn down (queries + self-attn)   | 160 latents    | ❌ (latents aren't joints) |
| decoder self-attention                  | 160 latents    | ❌ |
| cross-attn up: joint queries' self-attn | 21 joints      | ✅ (the decoder-side "GNN") |
| cross-attn up: queries → latent context | 21 × 160       | ❌ |

## 3. Do we still need positional encodings?

Two different things were called "PE", and the mask only replaces one of them:

1. **Structure ("who is connected to whom") — replaced.** The kinematic Laplacian PE
   (`USE_KINEMATIC_PE`) *soft*-encoded the skeleton into the joint features so attention could use
   it. The mask *hard*-codes the same information. The masked config therefore sets
   `USE_KINEMATIC_PE: False`.
2. **Identity ("which joint am I") — still needed.** The learned per-joint `pos_embedding` is an ID
   badge, not an order encoding. A masked transformer shares one set of weights across all joints,
   so a joint is only distinguishable through its input value and its connectivity pattern — and the
   skeleton has an **exact left↔right graph symmetry** (swap all L/R joints and the graph maps to
   itself). Connectivity can never tell L_Knee from R_Knee. Without identity, a pose X and its
   mirror X' produce identical per-joint features up to the L/R permutation, and the (permutation-
   invariant) cross-attention readout then yields **latent(X) == latent(mirror(X)) exactly**:
   mirror-pose pairs collapse onto the same tokens and cannot both be reconstructed. The config
   keeps `USE_JOINT_PE: True` (21×512 ≈ 10k params); flip it to False to measure the collapse as an
   ablation.

## 4. Backwards compatibility

Everything defaults to the old behaviour:

- `pose_transformer.py`: `mask` / `self_attn_mask` / `use_pos_embedding` are optional kwargs;
  `mask=None` paths are unchanged, so TokenHMR's `smpl_head` / `token_head` and all existing
  checkpoints are unaffected.
- `TransformerTokenizer`: `USE_SKELETON_MASK` defaults to False; the mask buffer is
  `persistent=False`, so state-dict keys are unchanged and pre-mask checkpoints still load with
  `strict=True`.
- `TransformerDecodeTokens` reads the flags from the checkpoint's `hparams.ARCH`, so full-TokenHMR
  inference with a masked tokenizer checkpoint applies the same mask automatically.

## 5. Running the comparison

```bash
cd tokenization
# 1-hop mask (pure GNN: attend to bone neighbours only)
python train_poseVQ.py --cfg configs/tokenizer_amass_moyo_transformer_masked.yaml --cfg_id 0
# 2-hop mask (attend within 2 bones)
python train_poseVQ.py --cfg configs/tokenizer_amass_moyo_transformer_masked.yaml --cfg_id 1
```

Compare in wandb against `tokenization_transformer_with_fnn_blocks` (unmasked 256-dim baseline,
same params) and the `tokenization_gnn` run (also 256-dim). The question the experiment answers:
does the skeleton bias close the val/train gap the GNN showed from ~5k iterations, without the
GNN's optimization problems?
