"""Stage-2 token-analysis metrics for TokenHMR.

These are logged during HMR training/validation to characterise the *token predictor*
(the image -> token classifier), as opposed to the frozen tokenizer's codebook:

  token_top1_acc / token_top5_acc  -- prediction accuracy vs the GT tokens (the GT SMPL
                                      body pose encoded through the frozen tokenizer).
  token_pred_entropy               -- mean entropy of the predicted per-token softmax
                                      (predictor confidence; high = uncertain).
  token_usage_entropy / _util /    -- entropy, utilisation %, and KL-to-uniform of the
  token_usage_kl_uniform              distribution of argmax-predicted tokens (does the
                                      predictor collapse onto a few tokens?). "kl_uniform"
                                      is the probability-distance-from-uniform.

The GT-token encoder is a frozen full tokenizer loaded once and cached at module scope, so
it is never added to the Lightning module (no checkpoint bloat, never optimised).
"""

import torch

from tokenization.utils.utils_model import codebook_usage_stats

# Module-scoped cache: {ckpt_path: tokenizer (frozen, on the active device)}.
_GT_ENCODER = {}


def _get_gt_encoder(ckpt_path, device):
    enc = _GT_ENCODER.get(ckpt_path)
    if enc is None:
        # Imported lazily; the tokenization package is on sys.path via token_classifier.
        import torch.nn as nn
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        arch = ckpt['hparams'].ARCH
        # Same dispatch key the tokenizer's own trainer uses (train_poseVQ.get_model): the
        # checkpoint records the architecture that produced it, so the GT encoder is always the
        # class that was trained. Hardcoding the transformer here made every CNN-tokenizer run
        # die on the strict-key check below.
        model_name = getattr(arch, 'MODEL_NAME', 'transformer')
        if model_name in ('vanilla', 'vanilla-v1'):
            from tokenization.models.vanilla_pose_vqvae import VanillaTokenizer as Tokenizer
        elif model_name == 'transformer':
            from tokenization.models.transformer_pose_vqvae import TransformerTokenizer as Tokenizer
        else:
            raise NotImplementedError(
                f'GT encoder for ARCH.MODEL_NAME={model_name!r} not implemented ({ckpt_path})')
        enc = Tokenizer(arch, mesh_inference=False)  # encode() needs no SMPL
        missing, unexpected = enc.load_state_dict(ckpt['net'], strict=False)
        # Pre-code_norm FSQ checkpoints lack code_norm.{weight,bias}: they were trained without
        # that LayerNorm, so reproduce the trained encoder exactly by replacing it with Identity.
        # Leaving the freshly-initialised LayerNorm in place would silently corrupt the GT token
        # IDs (encode() feeds through code_norm), poisoning the CE targets. Any OTHER missing or
        # unexpected key is a real mismatch and must fail loudly.
        if any(k.startswith('code_norm.') for k in missing):
            enc.code_norm = nn.Identity()
            missing = [k for k in missing if not k.startswith('code_norm.')]
        if missing or unexpected:
            raise RuntimeError(f'GT encoder load mismatch for {ckpt_path}: '
                               f'missing={missing} unexpected={unexpected}')
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        _GT_ENCODER[ckpt_path] = enc
    if next(enc.parameters()).device != device:
        enc = enc.to(device)
        _GT_ENCODER[ckpt_path] = enc
    return enc


@torch.no_grad()
def compute_token_metrics(cls_logits_softmax, gt_body_pose_rotmat, has_gt, ckpt_path, gt_tok=None):
    """Return a dict of 0-dim tensors (on `cls_logits_softmax.device`) for wandb logging.

    cls_logits_softmax:  (N, token_num, token_class_num) predicted per-token softmax.
    gt_body_pose_rotmat: (B, >=21, 3, 3) GT body-joint rotation matrices (or None).
    has_gt:              (B,) bool mask of samples with a valid GT body pose (or None).
    ckpt_path:           tokenizer checkpoint, used to build the GT-token encoder.
    gt_tok:              (B, token_num) precomputed GT token IDs (or None to encode here).
                         Lets the caller share one encode with a token-CE loss.
    """
    device = cls_logits_softmax.device
    p = cls_logits_softmax.float()
    token_class_num = p.shape[-1]

    metrics = {}

    # Predictor confidence: mean entropy of the per-token softmax (nats).
    ent = -(p * (p + 1e-10).log()).sum(-1)            # (N, token_num)
    metrics['token_pred_entropy'] = ent.mean()

    # Predicted-token usage distribution (collapse / probability distance from uniform).
    pred_tok = p.argmax(-1)                           # (N, token_num)
    counts = torch.bincount(pred_tok.reshape(-1), minlength=token_class_num).float()
    usage = codebook_usage_stats(counts, token_class_num)
    metrics['token_usage_entropy'] = torch.tensor(usage['entropy'], device=device)
    metrics['token_usage_util'] = torch.tensor(usage['codebook_util'], device=device)
    metrics['token_usage_kl_uniform'] = torch.tensor(usage['kl_to_uniform'], device=device)

    # Token-prediction accuracy vs GT tokens (only over samples with a real GT pose).
    if gt_tok is not None or (gt_body_pose_rotmat is not None and has_gt is not None and bool(has_gt.any())):
        if gt_tok is None:
            enc = _get_gt_encoder(ckpt_path, device)
            gt_in = gt_body_pose_rotmat[:, :enc.num_joints].to(device).float()  # (B, 21, 3, 3)
            with torch.cuda.amp.autocast(enabled=False):
                gt_tok = enc.encode(gt_in)            # (B, token_num)
        # Align IEF iterations: predictions are cat'd over iters; keep the final B rows.
        B = gt_tok.shape[0]
        pp = p[-B:] if p.shape[0] != B else p
        pred = pp.argmax(-1)
        mask = has_gt if has_gt is not None else torch.ones(B, dtype=torch.bool, device=device)
        gt_tok, pred, pp = gt_tok[mask], pred[mask], pp[mask]
        metrics['token_top1_acc'] = (pred == gt_tok).float().mean()
        top5 = pp.topk(5, dim=-1).indices            # (n, token_num, 5)
        metrics['token_top5_acc'] = (top5 == gt_tok.unsqueeze(-1)).any(-1).float().mean()

    return {k: v.detach().float() for k, v in metrics.items()}
