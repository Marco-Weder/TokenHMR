"""Graph Neural Network (GNN) pose tokenizer — a drop-in alternative to the transformer.

The transformer tokenizer (`transformer_pose_vqvae.TransformerTokenizer`) treats the 21 body
joints as a fully-connected set and mixes them with self-attention. This module instead mixes
joints with **skeleton-constrained message passing**: each joint only exchanges information with
its anatomical neighbours in the SMPL-H kinematic tree.

To make this a clean "attention vs. GNN" comparison we keep the *matched bottleneck*: only the two
joint-mixing stages (encoder + decoder) become graph convolutions. The cross-attention down/up
sampling, the 160-token latent, the quantizer (EMA or FSQ) and the output projection are reused
unchanged from the transformer, so `NUM_TOKENS`, the codebook and the loss are identical.

    input pose           (B, 21, 6)
      -> GNN encoder      (B, 21, W)      << graph message passing on the skeleton
      -> cross-attn down  (B, 160, W)     reused
      -> to_code          (B, 160, d)     reused
      -> quantize         (B, 160, d)     reused
      -> lift to width    (B, 160, W)     Linear(d -> W)
      -> cross-attn up    (B, 21, W)      reused
      -> GNN decoder      (B, 21, W)      << graph message passing on the skeleton
      -> linear out       (B, 21, 6)      reused
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

from .quantize_cnn import QuantizeEMAReset
from .fsq import FSQQuantizer
from .rotation_utils import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle

# Reuse the (skeleton) adjacency builder so the GNN and the transformer's kinematic PE share
# the exact same kinematic tree.
from utils.skeleton import build_skeleton_adjacency

# Import the cross-attention components the same way transformer_pose_vqvae does (bypassing
# tokenhmr/lib/models/__init__.py to avoid a circular import). We deliberately do NOT import
# from transformer_pose_vqvae itself: that module loads the SMPL-H body model at import time,
# and we want this module to import cleanly for shape/CPU tests with mesh_inference=False.
_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tokenhmr/lib/models'))
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
from components.pose_transformer import CrossAttention, TransformerCrossAttn, FeedForward  # noqa: E402


def _make_cross_attn(dim, context_dim, heads, depth, mlp_dim, dropout):
    """Down/up cross-attention module — identical to the transformer's helper.

    depth>=2 -> stacked Perceiver-style block; depth==1 -> bare CrossAttention.
    """
    if depth <= 1:
        return CrossAttention(
            dim=dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim // heads,
            dropout=dropout,
        )
    return TransformerCrossAttn(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim // heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        context_dim=context_dim,
    )


def step_multiplier_mapping():
    return {0: 1e-2, 1: 5e-2, 2: 1e-1, 3: 1e-1, 4: 5e-1, 5: 5e-1}


# Lazily-loaded SMPL-H body model (only needed when mesh_inference=True). Loading it is
# deferred so that importing this module / running a shape test needs no body-model files.
_BODY_MODEL = None


def _get_body_model():
    global _BODY_MODEL
    if _BODY_MODEL is None:
        from smplx import SMPLHLayer
        current_dir = os.path.dirname(os.path.realpath(__file__))
        body_model_path = os.path.join(current_dir, '..', '..', 'data/body_models', 'smplh')
        bm = SMPLHLayer(body_model_path, num_betas=10, ext='pkl')
        _BODY_MODEL = bm.cuda() if torch.cuda.is_available() else bm
    return _BODY_MODEL


class GraphConvLayer(nn.Module):
    """A transformer block with self-attention replaced by skeleton message passing.

    Structurally identical to one layer of the transformer's `Transformer` (two pre-norm residual
    sub-blocks), so the GNN-vs-transformer comparison is apples-to-apples — only the token-mixing
    operator differs:

        h = h + mp_proj(GELU(self_lin(LN(h)) + A_norm @ neigh_lin(LN(h))))   # message sub-block
        h = h + FFN(LN(h))                                                    # FFN sub-block

    The message sub-block is the analogue of `x + attn(LN(x))`: `self_lin` is the joint's own
    update, `neigh_lin` followed by the fixed `A_norm` aggregation is the message from connected
    joints, and `mp_proj` is the output projection (the transformer attention's `to_out`). The FFN
    is reused verbatim from the transformer so per-joint capacity matches.

    Note (deliberate, not a bug): we do *not* wrap the message op in the imported `PreNorm`. Its
    non-adaptive (LayerNorm) path forwards only ``**kwargs`` to the wrapped fn, so the positional
    ``adjacency`` argument would be silently dropped. The explicit LayerNorms below avoid that.
    """

    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.mp_norm = nn.LayerNorm(dim)
        self.self_lin = nn.Linear(dim, dim)
        self.neigh_lin = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.mp_proj = nn.Linear(dim, dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim)

    def forward(self, h, adjacency):
        # h: (B, J, dim);  adjacency: (J, J)
        x = self.mp_norm(h)
        msg = torch.einsum('ij,bjd->bid', adjacency, self.neigh_lin(x))
        h = h + self.mp_proj(self.act(self.self_lin(x) + msg))   # message sub-block
        h = h + self.ff(self.ff_norm(h))                          # FFN sub-block
        return h


class GNNEncoder(nn.Module):
    """Embed per-joint features to `width`, add a learned per-joint positional embedding, then
    run `depth` graph-conv layers on the skeleton.

    The positional embedding gives every joint a distinct identity. Without it the GNN cannot
    break the skeleton's bilateral symmetry — topologically identical joints (e.g. L_Knee vs
    R_Knee: one hip parent, one ankle child) would be indistinguishable under the shared weights,
    collapsing their codes. This mirrors the transformer encoder's learned `pos_embedding`.

    Used for both the encoder (in_dim = joint rotation dim) and the decoder GNN (in_dim = width).
    Output shape: (B, num_joints, width).
    """

    def __init__(self, in_dim, width, depth, num_joints, mlp_dim):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, width)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_joints, width) * 0.02)
        self.layers = nn.ModuleList([GraphConvLayer(width, mlp_dim) for _ in range(depth)])

    def forward(self, x, adjacency):
        h = self.input_proj(x) + self.pos_embedding
        for layer in self.layers:
            h = layer(h, adjacency)
        return h


class GNNTokenizer(nn.Module):
    """GNN pose tokenizer. Same constructor signature and forward contract as
    `TransformerTokenizer`, so it is a drop-in replacement in `train_poseVQ.get_model`.
    """

    def __init__(self, arch_params=None, input_joint_dim=6, output_joint_dim=6,
                 mesh_inference=True, add_noise=False):
        super().__init__()
        self.num_joints = arch_params.NB_JOINTS if hasattr(arch_params, 'NB_JOINTS') else 21
        self.width = arch_params.WIDTH
        self.quant = arch_params.QUANTIZER
        self.rot_type = arch_params.ROT_TYPE

        # FSQ overrides code_dim / num_code from FSQ_LEVELS; the EMA path keeps the configured
        # CODE_DIM / NB_CODE. Identical to the transformer so checkpoints/quantizers line up.
        if self.quant == 'fsq':
            self._fsq_levels = list(arch_params.FSQ_LEVELS)
            self.code_dim = len(self._fsq_levels)
            self.num_code = int(np.prod(self._fsq_levels))
        else:
            self.code_dim = arch_params.CODE_DIM[0] if isinstance(arch_params.CODE_DIM, list) else arch_params.CODE_DIM
            self.num_code = arch_params.NB_CODE[0] if isinstance(arch_params.NB_CODE, list) else arch_params.NB_CODE
        self.input_joint_dim = input_joint_dim
        self.output_joint_dim = output_joint_dim

        self.mesh_inference = mesh_inference
        self.add_noise = add_noise
        self.step_multiplier_mapping = step_multiplier_mapping()
        if self.add_noise:
            from utils.skeleton import get_smplx_body_parts
            self.smplx_body_parts = get_smplx_body_parts()

        self.num_tokens = getattr(arch_params, 'NUM_TOKENS', 160)
        # Number of graph-conv layers; deeper = larger skeletal receptive field. Falls back to
        # DEPTH so a GNN config that only sets DEPTH still works.
        gnn_layers = int(getattr(arch_params, 'GNN_LAYERS', getattr(arch_params, 'DEPTH', 4)))

        # Cross-attention runs at `width`; the EMA code_dim must also satisfy the %8 head rule.
        if self.quant != 'fsq':
            assert self.code_dim % 8 == 0, f'CODE_DIM={self.code_dim} must be divisible by 8'
        assert self.width % 8 == 0, f'WIDTH={self.width} must be divisible by 8'

        # Tier-1 capacity knobs reused from the transformer config (default to single-block / 1x).
        ffn_mult      = int(getattr(arch_params, 'FFN_MULT', 1))
        n_down_blocks = int(getattr(arch_params, 'N_DOWN_BLOCKS', 1))
        n_up_blocks   = int(getattr(arch_params, 'N_UP_BLOCKS', 1))
        _cross_dropout = float(getattr(arch_params, 'CROSS_ATTN_DROPOUT', 0.0))

        # Fixed normalized skeleton adjacency (the GNN's graph). Registered as a buffer so it
        # moves to GPU with the module. The skeleton already encodes the kinematic structure, so
        # the transformer's USE_KINEMATIC_PE is intentionally not used here.
        self.register_buffer('adjacency', build_skeleton_adjacency(self.num_joints))

        # Per-joint message+FFN layers run at `width` with an `ffn_mult`× FFN, matching the
        # transformer block so the only architectural difference is attention vs. message passing.
        mlp_dim = ffn_mult * self.width

        # 1. ENCODER — graph message passing over the joints, at `width`.
        self.encoder = GNNEncoder(self.input_joint_dim, self.width, gnn_layers, self.num_joints, mlp_dim)

        # 2. DOWNSAMPLE — reused cross-attn (21 -> num_tokens), queries at `width`.
        self.latent_queries = nn.Parameter(torch.randn(1, self.num_tokens, self.width))
        self.cross_attn_down = _make_cross_attn(
            dim=self.width, context_dim=self.width, heads=8,
            depth=n_down_blocks, mlp_dim=ffn_mult * self.width, dropout=_cross_dropout,
        )

        # 2b. Bottleneck projection to code_dim (reused; LayerNorm only for FSQ).
        self.code_norm = nn.LayerNorm(self.width) if self.quant == 'fsq' else nn.Identity()
        self.to_code = nn.Linear(self.width, self.code_dim)

        # 3. QUANTIZER (reused).
        if self.quant == 'fsq':
            self.quantizer = FSQQuantizer(levels=self._fsq_levels)
        else:
            self.quantizer = QuantizeEMAReset(
                self.num_code, self.code_dim,
                dist_metric=getattr(arch_params, 'DIST_METRIC', 'l2'),
            )

        # 4. DECODER side — lift quantized tokens to width, cross-attn up to joints, then GNN.
        self.to_decoder_width = nn.Linear(self.code_dim, self.width)
        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints, self.width))
        self.cross_attn_up = _make_cross_attn(
            dim=self.width, context_dim=self.width, heads=8,
            depth=n_up_blocks, mlp_dim=ffn_mult * self.width, dropout=_cross_dropout,
        )
        self.decoder = GNNEncoder(self.width, self.width, gnn_layers, self.num_joints, mlp_dim)

        self.decoder_projection = nn.Linear(self.width, self.output_joint_dim)

    def _prep_input(self, x):
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.view(batch_size, self.num_joints, -1)
        if x.shape[-1] == 3 and self.input_joint_dim == 6:
            x = matrix_to_rotation_6d(x)
        return x

    def encode(self, x):
        batch_size = x.shape[0]
        x = self._prep_input(x)

        x_encoder = self.encoder(x, self.adjacency)
        queries = self.latent_queries.expand(batch_size, -1, -1)
        x_encoder = self.cross_attn_down(queries, context=x_encoder)
        x_encoder = self.to_code(self.code_norm(x_encoder))  # (B, num_tokens, code_dim)

        x_encoder = x_encoder.permute(0, 2, 1).contiguous()
        x_encoder = self.quantizer.preprocess(x_encoder)
        code_idx = self.quantizer.quantize(x_encoder)
        return code_idx.view(batch_size, -1)

    def decode_logits(self, logits):
        batch_size = logits.shape[0]
        decode_feat = self.quantizer.dequantize_logits(logits)   # (B, num_tokens, code_dim)
        x = self.to_decoder_width(decode_feat)                   # (B, num_tokens, width)
        j_queries = self.joint_queries.expand(batch_size, -1, -1)
        x = self.cross_attn_up(j_queries, context=x)             # (B, num_joints, width)
        x = self.decoder(x, self.adjacency)                      # GNN over joints
        return self.decoder_projection(x)

    def forward(self, x, global_step=None):
        batch_size = x.shape[0]
        x = self._prep_input(x)

        if self.training and self.add_noise and global_step is not None:
            step = global_step // 5000
            noise_multiplier = float(self.step_multiplier_mapping[step]) if step <= 5 else 0.5
            noised_samples = np.random.randint(low=0, high=batch_size - 1, size=batch_size // 2)
            mask_part = np.random.randint(len(self.smplx_body_parts.keys()))
            masked_joints = self.smplx_body_parts[mask_part]
            noise = torch.cuda.FloatTensor(1).uniform_() * noise_multiplier
            x = x.clone()
            for s_idx in noised_samples:
                x[s_idx, masked_joints] += noise

        # 1. Encode — graph message passing on the skeleton.
        x_encoder = self.encoder(x, self.adjacency)                      # (B, num_joints, width)
        queries = self.latent_queries.expand(batch_size, -1, -1)
        x_encoder = self.cross_attn_down(queries, context=x_encoder)     # (B, num_tokens, width)
        x_encoder = self.to_code(self.code_norm(x_encoder))              # (B, num_tokens, code_dim)

        # 2. Quantize.
        x_encoder_chan = x_encoder.permute(0, 2, 1)
        x_quantized_chan, loss, perplexity = self.quantizer(x_encoder_chan)
        x_quantized = x_quantized_chan.permute(0, 2, 1)                  # (B, num_tokens, code_dim)

        # 3. Decode — lift to width, cross-attn up to joints, graph message passing.
        x_decoder = self.to_decoder_width(x_quantized)                  # (B, num_tokens, width)
        j_queries = self.joint_queries.expand(batch_size, -1, -1)
        x_decoder = self.cross_attn_up(j_queries, context=x_decoder)    # (B, num_joints, width)
        x_decoder = self.decoder(x_decoder, self.adjacency)            # (B, num_joints, width)
        pred_pose_6d = self.decoder_projection(x_decoder)               # (B, num_joints, 6)

        output = {}
        if self.rot_type == 'rot6d':
            pred_pose_rotmat = rotation_6d_to_matrix(pred_pose_6d.reshape(-1, 6)).view(batch_size, self.num_joints, 3, 3)
        output.update({'pred_pose_body_6d': pred_pose_6d, 'pred_pose_body_rotmat': pred_pose_rotmat})

        if self.mesh_inference:
            pred_pose_aa = matrix_to_axis_angle(pred_pose_rotmat.view(-1, 3, 3)).view(batch_size, 3 * self.num_joints)
            pred_body_mesh = _get_body_model()(body_pose=pred_pose_rotmat)
            output.update({'pred_pose_body_aa': pred_pose_aa, 'pred_body_mesh': pred_body_mesh,
                           'pred_body_vertices': pred_body_mesh.vertices, 'pred_body_joints': pred_body_mesh.joints})
        return output, loss, perplexity


class GNNDecodeTokens(nn.Module):
    """Decoder-only wrapper for GNNTokenizer — mirror of `TransformerDecodeTokens`.

    Reconstructs only the decoder-side modules (quantizer, to_decoder_width, cross_attn_up,
    decoder GNN, decoder_projection) from a checkpoint's `hparams.ARCH` and loads the matching
    weights. `forward(logits)` maps soft codebook logits (B, num_tokens, num_codes) to
    (B, num_joints, 6) 6D pose. Needed only for full-TokenHMR inference, not for training the
    tokenizer.
    """

    def __init__(self, ckpt_path: str = '', mesh_inference: bool = False):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        arch = ckpt['hparams'].ARCH

        self.num_joints = getattr(arch, 'NB_JOINTS', 21)
        self.width = arch.WIDTH
        self.num_tokens = getattr(arch, 'NUM_TOKENS', 160)

        self.quant = getattr(arch, 'QUANTIZER', 'ema_reset')
        if self.quant == 'fsq':
            self._fsq_levels = list(arch.FSQ_LEVELS)
            self.code_dim = len(self._fsq_levels)
            self.num_code = int(np.prod(self._fsq_levels))
        else:
            self.code_dim = arch.CODE_DIM[0] if isinstance(arch.CODE_DIM, list) else arch.CODE_DIM
            self.num_code = arch.NB_CODE[0] if isinstance(arch.NB_CODE, list) else arch.NB_CODE

        gnn_layers    = int(getattr(arch, 'GNN_LAYERS', getattr(arch, 'DEPTH', 4)))
        ffn_mult      = int(getattr(arch, 'FFN_MULT', 1))
        n_up_blocks   = int(getattr(arch, 'N_UP_BLOCKS', 1))
        _cross_dropout = float(getattr(arch, 'CROSS_ATTN_DROPOUT', 0.0))

        self.register_buffer('adjacency', build_skeleton_adjacency(self.num_joints))

        if self.quant == 'fsq':
            self.quantizer = FSQQuantizer(levels=self._fsq_levels)
        else:
            self.quantizer = QuantizeEMAReset(self.num_code, self.code_dim)

        self.to_decoder_width = nn.Linear(self.code_dim, self.width)
        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints, self.width))
        self.cross_attn_up = _make_cross_attn(
            dim=self.width, context_dim=self.width, heads=8,
            depth=n_up_blocks, mlp_dim=ffn_mult * self.width, dropout=_cross_dropout,
        )
        self.decoder = GNNEncoder(self.width, self.width, gnn_layers, self.num_joints, ffn_mult * self.width)
        self.decoder_projection = nn.Linear(self.width, 6)

        self._load_weights(ckpt['net'])

    def _load_weights(self, state_dict: dict):
        decoder_prefixes = (
            'quantizer.',
            'to_decoder_width.',
            'joint_queries',
            'cross_attn_up.',
            'decoder.',
            'decoder_projection.',
        )
        filtered = {k: v for k, v in state_dict.items()
                    if any(k.startswith(p) for p in decoder_prefixes)}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        decoder_missing = [k for k in missing if any(k.startswith(p) for p in decoder_prefixes)]
        if decoder_missing:
            print(f'[GNNDecodeTokens] WARNING: missing decoder keys: {decoder_missing}')
        if unexpected:
            print(f'[GNNDecodeTokens] WARNING: unexpected keys: {unexpected}')

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.shape[0]
        decode_feat = self.quantizer.dequantize_logits(logits)   # (B, num_tokens, code_dim)
        x = self.to_decoder_width(decode_feat)
        j_queries = self.joint_queries.expand(batch_size, -1, -1)
        x = self.cross_attn_up(j_queries, context=x)
        x = self.decoder(x, self.adjacency)
        return self.decoder_projection(x)
