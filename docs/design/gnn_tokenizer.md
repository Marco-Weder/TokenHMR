# GNN pose tokenizer — how it works

A graph-neural-network alternative to the transformer in the pose VQ-VAE tokenizer. This note is
meant to be read top-to-bottom to explain the idea, the architecture, and the design trade-off.

Files: model [`models/gnn_pose_vqvae.py`](../models/gnn_pose_vqvae.py),
skeleton helper [`utils/skeleton.py`](../utils/skeleton.py),
config [`configs/tokenizer_amass_moyo_gnn.yaml`](../configs/tokenizer_amass_moyo_gnn.yaml).

---

## 1. The one-sentence idea

The tokenizer compresses a body pose (21 joint rotations) into discrete tokens and reconstructs it.
The **transformer** mixes the 21 joints with self-attention, so every joint can attend to every
other joint (a *fully-connected* graph). The **GNN** instead mixes joints with **message passing
constrained to the human skeleton**: a joint only exchanges information with the joints it is
physically connected to (its parent/children in the kinematic tree).

> Transformer = "every joint talks to every joint."
> GNN = "every joint talks only to the joints it's bone-connected to."

This bakes human anatomy into the architecture as an *inductive bias*, instead of asking attention
to learn it from data.

---

## 2. The skeleton graph

The graph nodes are the **21 SMPL-H body joints** (the root pelvis is excluded). The edges come
straight from the kinematic tree the project already uses for its kinematic positional encoding,
now centralized in [`utils/skeleton.py`](../utils/skeleton.py):

```
SMPLH_PARENTS_21 = [-1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 11, 12, 13, 15, 16, 17, 18]
```

Entry `i` is the parent of joint `i` (`-1` = attaches to the root pelvis). Drawing the edges gives
the body tree (indices are joint IDs; `pelvis` is the excluded root):

```
                 pelvis (root)
        ┌────────────┼────────────┐
        0            1            2          (3 joints attach to the pelvis)
        │            │            │
        3            4            5
        │            │            │
        6            7            8 ───────┐ ─────────┐        (joint 8 is a hub: 3 children)
        │            │            │        │          │
        9           10           11       12         13
                                  │        │          │
                                 14       15         16
                                          │          │
                                         17         18
                                          │          │
                                         19         20
   └── leg-like chains ──┘     └──── spine → hub → 3 limb chains ────┘
```

That is **18 undirected edges** over 21 nodes — extremely sparse compared to the transformer's
21×21 all-to-all attention.

### From tree to the GCN propagation matrix

`build_skeleton_adjacency()` turns this tree into the standard GCN propagation matrix
(Kipf & Welling, 2017):

```
A_norm = D^{-1/2} (A + I) D^{-1/2}
```

- `A` = symmetric adjacency (1 where two joints share a bone).
- `+ I` = self-loops, so a joint keeps its own information.
- `D^{-1/2}(·)D^{-1/2}` = symmetric normalization, so message magnitudes don't blow up at
  high-degree joints (like the hub at index 8).

`A_norm` is a fixed `21×21` buffer (not learned) — it *is* the skeleton.

---

## 3. The graph-convolution layer (the only genuinely new module)

One layer is a **transformer block with self-attention swapped for skeleton message passing** —
two pre-norm residual sub-blocks, applied to per-joint features `h` of shape `(B, 21, dim)`:

```
# message sub-block (analogue of  x + attn(LN(x)) )
x   = LayerNorm(h)
msg = A_norm @ neigh_lin(x)               # gather messages from skeletal neighbours
h   = h + mp_proj(GELU(self_lin(x) + msg))

# FFN sub-block (identical to the transformer's)
h   = h + FFN(LayerNorm(h))               # FFN = Linear(dim → mlp_dim) → GELU → Linear(mlp_dim → dim)
```

- `self_lin(x)` — what a joint computes from **its own** features.
- `A_norm @ neigh_lin(x)` — the **message** each joint receives, aggregated over its bone-neighbours.
- `mp_proj` — the message sub-block's output projection (the analogue of attention's `to_out`).
- `FFN` — reused verbatim from the transformer (`FeedForward`, `mlp_dim = FFN_MULT × WIDTH`), so the
  GNN's per-joint capacity matches the transformer's.

This is deliberately the **same block structure as the transformer** (pre-norm message sub-block +
pre-norm FFN sub-block, both with clean additive residuals), so the only thing that changes between
the two models is *token-mixing = attention vs. graph convolution*. That makes the comparison clean.

> **Why not the simpler `h + GELU(self_lin(x) + msg)`?** An earlier version used exactly that —
> one op, GELU as the *last* step on the residual branch, no output projection, no FFN. It trained
> ~9× worse than the CNN/transformer and *diverged* on validation after ~6k iterations (the GELU
> biases every residual update positive, and with no FFN the per-joint capacity was a fraction of
> the transformer's). The two-sub-block form above fixes both.

Stacking `GNN_LAYERS` layers lets information propagate **hop-by-hop along the limbs** (after `k`
layers a joint has "heard from" everything within `k` bones of it). `GNN_LAYERS=4` is the default.

### Per-joint positional embedding

Before the first layer, `GNNEncoder` adds a **learned per-joint positional embedding**
(`pos_embedding`, shape `(1, 21, WIDTH)`) to the projected input — the analogue of the transformer
encoder's learned `pos_embedding`. This is essential, not cosmetic: the SMPL skeleton is
**bilaterally symmetric** and many joints are *topologically identical* (e.g. L_Knee and R_Knee both
have one hip parent and one ankle child, same degree). With weights shared across nodes and no
positional signal, a GNN literally cannot tell those joints apart (the graph-automorphism problem),
so it conflates their codes. The positional embedding gives every joint a distinct identity and
breaks the symmetry.

---

## 4. Full data flow (matched bottleneck)

We keep everything except the joint-mixing identical to the transformer, so the token budget and
codebook are unchanged. `<< GNN` marks the two new stages; everything else is reused as-is.

```
input pose            (B, 21, 6)     6D rotation per body joint
  └─ GNN encoder      (B, 21, W)     << skeleton message passing
  └─ cross-attn down  (B, 160, W)    reused: 21 joints → 160 latent tokens
  └─ to_code          (B, 160, d)    reused: project to code dim (d = CODE_DIM, or len(FSQ_LEVELS))
  └─ quantize         (B, 160, d)    reused: EMA or FSQ codebook
  └─ lift to width    (B, 160, W)    Linear(d → W)
  └─ cross-attn up    (B, 21, W)     reused: 160 tokens → 21 joints
  └─ GNN decoder      (B, 21, W)     << skeleton message passing
  └─ linear out       (B, 21, 6)     reused: 6D rotation → SMPL-H mesh
```

### Reused vs. new

| Component                       | Status | Notes |
|---------------------------------|--------|-------|
| Encoder joint-mixing            | **NEW**   | self-attention → skeleton GNN |
| Decoder joint-mixing            | **NEW**   | self-attention → skeleton GNN |
| Cross-attention down/up (21↔160)| reused | the compression bottleneck, unchanged |
| Quantizer (EMA / FSQ)           | reused | same codebook, same `loss`/`perplexity` outputs |
| 6D-rotation I/O, SMPL-H mesh, losses | reused | identical training loop |

Because `NUM_TOKENS`, the codebook, and the loss are byte-for-byte the same as the transformer, a
GNN run vs. a transformer run is an **apples-to-apples** test of "does skeleton message passing
encode poses as well as full attention?"

---

## 5. Design decision: where do the discrete tokens come from?

**Why this decision even exists.** The tokenizer has to turn **21 joints** into a sequence of
**discrete tokens**, and "joint" and "token" are not the same thing. The transformer deliberately
*decouples* them: cross-attention **pools** the 21 joints into a learned set of 160 latent tokens
(and back again on the way out). A GNN, on the other hand, naturally lives in **joint-space** — its
message passing is *defined on* the 21-node skeleton graph. There is no skeleton over 160 abstract
tokens, so a GNN cannot directly produce them. That forces a choice about how to get from joints to
tokens:

- **Matched bottleneck (implemented).** Keep the transformer's cross-attention pooling (21 ⇄ 160)
  and use the GNN *only* for the joint-mixing stages. The latent stays at 160 tokens with the same
  codebook, so a GNN run and a transformer run differ in **exactly one thing**: how joints are mixed
  (skeleton message passing vs. attention). That's a controlled experiment.
- **Pure graph-native (alternative).** Drop cross-attention entirely and quantize **one token per
  joint** → 21 tokens. The whole model becomes a single graph: encode on the skeleton, quantize each
  node, decode on the skeleton. Conceptually cleaner, and every token maps to a specific joint — but
  the latent is now 21 tokens instead of 160, a *different compression budget*, so it is no longer a
  like-for-like comparison with the transformer.

| Aspect | **Matched bottleneck** (implemented) | **Pure graph-native** (alternative) |
|---|---|---|
| Latent size | 160 tokens (same as transformer) | 21 tokens (one per joint) |
| Joints → tokens | learned cross-attention pooling | one-to-one (token *is* a joint) |
| Is it a *pure* GNN? | No — attention still in the bottleneck | Yes — graph end-to-end |
| Compression budget vs. transformer | identical | different (fewer tokens) |
| Comparison validity | **controlled** — only joint-mixing changes | **confounded** — architecture *and* budget change together |
| Token interpretability | tokens are abstract / distributed | each token = one named joint (great for latent analysis) |
| Downstream (TokenHMR) | drop-in, no changes | needs a 21-token classifier |
| Effort from here | done | small change |

**For the supervisor — the trade-off is *fair comparison* vs *clean story*:**
- *Matched bottleneck* answers **"does skeleton message passing beat attention, holding everything
  else equal?"**
- *Pure graph-native* answers **"how good is a fully-skeletal tokenizer?"** — a different (also valid)
  question, and its per-joint tokens are attractive for the latent-space analysis in the thesis.

They are **not mutually exclusive**: we can keep the matched version as the controlled baseline *and*
add the pure variant as a second contribution.

> **Related note — positional encoding.** The transformer offers `USE_KINEMATIC_PE` (Laplacian
> eigenvectors of the skeleton) *on top of* its learned `pos_embedding`. The GNN does not use the
> Laplacian variant — the skeleton already *is* the graph the convolutions run on, so the
> *connectivity* is built in by construction. It does, however, keep a **learned per-joint
> `pos_embedding`** (see §3): connectivity alone is not enough to give symmetric joints distinct
> identities, so the learned embedding is required to break the skeleton's bilateral symmetry.

---

## 6. How to run

**Shape/contract smoke test (CPU, no data needed)** — instantiate the model, push random poses
through, and check the output shapes match the transformer's contract. Verified output:

```
adjacency shape: (21, 21)   symmetric: True   self-loops: True
off-diagonal edges match kinematic tree: True   (18 undirected edges)
[fsq]        pred_pose_body_6d: (4, 21, 6) | perplexity ≈ 20 | encode(): (4, 16)
[ema_reset]  pred_pose_body_6d: (4, 21, 6) | perplexity ≈ 64 | encode(): (4, 16)
```

**Train the GNN tokenizer** (needs the AMASS pose data under `DATA.DATA_ROOT`):

```bash
cd tokenization
python train_poseVQ.py --cfg configs/tokenizer_amass_moyo_gnn.yaml
```

Select the architecture purely from the config: `ARCH.MODEL_NAME: gnn` (vs `transformer`), with
`ARCH.GNN_LAYERS` controlling the number of graph-conv layers. To compare head-to-head, run the same
config with `MODEL_NAME: transformer` — every other setting (tokens, codebook, losses) is identical.
