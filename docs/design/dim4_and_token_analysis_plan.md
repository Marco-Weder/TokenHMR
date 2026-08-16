# Plan: 4-dim cosine VQ ablation + codebook token analysis

## Context

After the supervisor meeting, two threads came out of comparing the **FSQ `[8,8,6,5]`** run
against the earlier **cosine-distance transformer** run:

1. **The dimension question.** FSQ `[8,8,6,5]` has a code width of `len([8,8,6,5]) = 4`,
   yet it matches the cosine-distance run, which uses a code width of **256**
   (`CODE_DIM`). The cosine run encodes each pose as **160 tokens** (`NUM_TOKENS`),
   each token a 256-dim code drawn from a **2048-entry** codebook (`NB_CODE`).
   The supervisor asked: *would the cosine run reach the same result with a code width
   of only 4?* → run the cosine VQ with `CODE_DIM = 4`, everything else fixed.

2. **Token analysis.** Save **accuracy, entropy, and probability-distance** for the runs
   (to wandb), to characterise how the codebook is used. His "**high = good, low = 1-hot**"
   remark is about the **entropy of the code-usage distribution**: high entropy = codes
   used uniformly (healthy); low entropy → usage collapses onto a single code (a one-hot
   distribution, entropy 0 = codebook collapse). A natural follow-up is whether
   **regularizing the latent space** (pushing usage toward uniform) helps.

Terminology note: "160 tokens × 256 dimensions" = `NUM_TOKENS=160` (latent length) and
`CODE_DIM=256` (per-token code width). The codebook holds `NB_CODE=2048` distinct codes.
Only `CODE_DIM` changes in this experiment.

---

## Experiment 1 — cosine VQ with a 4-dim codebook (ready to run)

A new config is already created:
[`configs/tokenizer_amass_moyo_transformer_dim4.yaml`](../configs/tokenizer_amass_moyo_transformer_dim4.yaml)

It is a byte-for-byte clone of `tokenizer_amass_moyo_transformer.yaml` with a single change:

```yaml
CODE_DIM: [4]     # was [256]
```

Everything else is identical: `QUANTIZER: 'ema_reset'`, `DIST_METRIC: 'cosine'`,
`NB_CODE: [2048]`, `NUM_TOKENS: 160`. The projection
`to_code = nn.Linear(WIDTH, CODE_DIM)` in
[`models/transformer_pose_vqvae.py:174`](../models/transformer_pose_vqvae.py#L174)
automatically becomes `Linear(512 → 4)`, and `QuantizeEMAReset(nb_code=2048, code_dim=4,
dist_metric='cosine')` quantizes on the unit sphere in 4-D.

**One model-code change was required:** the EMA path previously asserted
`CODE_DIM % 8 == 0`. This guardrail was spurious — `code_dim` is only the bottleneck
projection width and the decoder's input embedding dim (embedded back up to `width` before
any attention), so heads never operate on `code_dim` (the same reason FSQ runs at dim 4).
The assertion was removed in
[`models/transformer_pose_vqvae.py:121-126`](../models/transformer_pose_vqvae.py#L121-L126);
this only relaxes a constraint, so every existing config still builds unchanged.

**Run:**
```bash
cd external/tokenhmr/tokenization
python train_poseVQ.py --cfg configs/tokenizer_amass_moyo_transformer_dim4.yaml
```

**Why this is a clean ablation:** the only variable changed vs. the original cosine run is
the codebook dimension (256 → 4). `NB_CODE=2048` is kept (≈ FSQ's 1920 codes), so the
comparison to FSQ stays fair on code count too.

---

## Experiment 2 — token-analysis metrics (entropy, probability distance, accuracy)

### What's already there
`perplexity` (= `exp(entropy)` of the per-batch code-usage distribution) is **already
computed and logged** to wandb for every quantizer: `tr/curr_perplexity`,
`val/curr_perplexity`
([`quantize_cnn.py:43-51`](../models/quantize_cnn.py#L43-L51),
[`fsq.py:130-134`](../models/fsq.py#L130-L134)). wandb is fully wired up
(project `TokenHMR-Transformer`,
[`train_poseVQ.py:93-101`](../train_poseVQ.py#L93-L101)).

### Metrics to add
Computed from the per-batch `code_idx` (usage histogram `p`, `N = nb_code` codes):

| Metric (wandb key)            | Formula                                  | Reading |
|-------------------------------|------------------------------------------|---------|
| `*/entropy`                   | `H(p) = -Σ p·log p` (nats)               | high = uniform usage (good); 0 = one-hot collapse |
| `*/norm_entropy`              | `H(p) / log(N)` ∈ [0,1]                  | 1 = perfectly uniform; 0 = one-hot. ← the "high good, low 1-hot" axis |
| `*/kl_to_uniform`             | `KL(p‖uniform) = log(N) − H(p)`          | "probability distance"; 0 = uniform, large = collapse (complement of entropy) |
| `*/codebook_util`             | `(code_count > 0).sum() / N · 100`       | % of codes actually used (accumulate over eval set) |
| `*/pck`                       | fraction of joints within a mm threshold | reconstruction "accuracy" |

`entropy`, `norm_entropy`, `kl_to_uniform`, and `codebook_util` are **codebook** metrics
(from `code_idx`). `pck` is a **reconstruction-accuracy** metric (from the decoded joints).

### Implementation (low-risk, uniform across quantizers)

1. **Shared util** — add `codebook_usage_stats(code_idx, nb_code) -> dict` (e.g. in
   `utils/utils_model.py`) returning `{entropy, norm_entropy, kl_to_uniform, codebook_util}`.
   This mirrors the existing perplexity math (reuse the `prob` already computed there).

2. **Expose from the quantizer** — in each quantizer `forward`
   (`QuantizeEMAReset`, `QuantizeReset`, `QuantizeEMA`, `Quantizer`, `FSQQuantizer`),
   stash `self.codebook_stats = codebook_usage_stats(code_idx, self.nb_code)` next to the
   existing perplexity computation. Keep the `(x_d, commit_loss, perplexity)` return
   signature unchanged so nothing downstream breaks. Add a `get_codebook_stats()` helper on
   the tokenizer (`TransformerTokenizer` / GNN / vanilla) that returns its quantizer's
   `codebook_stats`.

3. **PCK accuracy** — add `calculate_pck(gt_jnts, pred_jnts, thresh_mm)` next to the other
   `calculate_*_reconstruction_error` helpers in
   [`utils/eval_poseVQ.py:70-78`](../utils/eval_poseVQ.py#L70-L78). Suggest logging two
   thresholds (e.g. 25 mm and 50 mm); make the threshold(s) a small `EXP`/`LOSS` config key.

4. **Log in the loops** — extend `err_list` (train +
   [`train_poseVQ.py:217-232`](../train_poseVQ.py#L217-L232); val +
   [`eval_poseVQ.py:123-181`](../utils/eval_poseVQ.py#L123-L181)) with the new keys, reading
   codebook stats via `net.get_codebook_stats()` and PCK from `output['pred_body_joints']`
   vs `gt_jnts`. They flow into the existing `wandb.log(...)` calls automatically.
   For `codebook_util` in eval, **accumulate `code_count` across all val batches** and
   compute utilization once at the end (per-batch undercounts) — see the standalone
   [`calculate_codebook_utilization.py`](../calculate_codebook_utilization.py) for reference logic.

---

## Next step (documented, not yet implemented) — latent-space regularization

Once Experiment 1 + the metrics land, read `norm_entropy` / `kl_to_uniform`. If usage is
peaky (low entropy), add a **codebook-entropy regularizer** that maximizes `H(p)`
(equivalently minimizes `kl_to_uniform`), gated by a new `LOSS.ENTROPY_REG_WT` config key,
added to the total loss at [`train_poseVQ.py:200-204`](../train_poseVQ.py#L200-L204).
Then re-run and compare reconstruction + usage to test whether regularizing helps. Deferred
on purpose: decide based on the measured entropy rather than adding it blind.

---

## Verification

1. **Config sanity:** start the dim-4 run; confirm the log shows the quantizer built as
   `code_dim=4, nb_code=2048` and that `to_code` is `Linear(512, 4)`; let it run a few
   hundred iters and confirm loss decreases and perplexity is non-trivial (codebook not
   collapsed at `1.0`).
2. **Metrics:** confirm the new keys appear in wandb for both `tr/*` and `val/*`, that
   `entropy ≈ log(perplexity)` (cross-check), `norm_entropy ∈ [0,1]`, and
   `kl_to_uniform ≈ log(N) − entropy`.
3. **Comparison:** plot dim-4 cosine vs. FSQ `[8,8,6,5]` vs. the original 256-dim cosine on
   `val/curr_mesh_recons`, `val/curr_jnt_recons`, `pck`, and the usage metrics to answer the
   supervisor's question directly.
```
