import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# SMPL-H kinematic tree for 21 body joints (root excluded). Index 0 = L_Hip, ..., 20 = R_Wrist.
# -1 means parent is the root pelvis. Used only when USE_KINEMATIC_PE=True.
_SMPLH_PARENTS_21 = [-1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 11, 12, 13, 15, 16, 17, 18]


def _build_kinematic_laplacian_pe(num_joints: int = 21) -> torch.Tensor:
    """Laplacian-eigenvector positional encoding for the SMPL kinematic tree.

    Returns (num_joints, num_joints): each row is the PE of one joint.
    Uses the full eigenvector basis (no truncation, no trivial-eigvec drop).
    """
    parents = _SMPLH_PARENTS_21[:num_joints]
    A = torch.zeros(num_joints, num_joints)
    for i, p in enumerate(parents):
        if p >= 0:
            A[i, p] = 1.0
            A[p, i] = 1.0
    deg = A.sum(dim=1).clamp(min=1.0)
    d_inv_sqrt = torch.diag(deg.pow(-0.5))
    L = torch.eye(num_joints) - d_inv_sqrt @ A @ d_inv_sqrt
    _, eigvecs = torch.linalg.eigh(L)
    return eigvecs


from .quantize_cnn import QuantizeEMAReset
from .fsq import FSQQuantizer
from .rotation_utils import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle

from tokenization.utils.skeleton import build_skeleton_attention_mask

# Import the submodule directly rather than through tokenhmr.lib.models, whose
# __init__ imports the heads, which import this module back. Naming the submodule
# skips that parent __init__ and so avoids the cycle.
from tokenhmr.lib.models.components.pose_transformer import (
    TransformerEncoder,
    TransformerCrossAttn,
    CrossAttention,
)

# --- SMPL Instantiation ---
from smplx import SMPLHLayer
smpl_type = 'smplh'
current_dir = os.path.dirname(os.path.realpath(__file__))
body_model_path = os.path.join(current_dir, '..', '..', 'data/body_models', smpl_type)
body_model = eval(f'{smpl_type.upper()}Layer')(body_model_path, num_betas=10, ext='pkl')
body_model = body_model.cuda() if torch.cuda.is_available() else body_model


def step_multiplier_mapping():
    return {0: 1e-2, 1: 5e-2, 2: 1e-1, 3: 1e-1, 4: 5e-1, 5: 5e-1}


def _make_cross_attn(dim, context_dim, heads, depth, mlp_dim, dropout):
    """Down/up cross-attention module. depth>=2 → stacked Perceiver-style block;
    depth==1 → bare CrossAttention (matches the previous architecture exactly)."""
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


class TransformerTokenizer(nn.Module):
    def __init__(self, arch_params=None, input_joint_dim=6, output_joint_dim=6, mesh_inference=True, add_noise=False):
        super().__init__()
        self.num_joints = arch_params.NB_JOINTS if hasattr(arch_params, 'NB_JOINTS') else 21
        self.width = arch_params.WIDTH
        self.depth = arch_params.DEPTH
        self.quant = arch_params.QUANTIZER
        self.rot_type = arch_params.ROT_TYPE

        # FSQ overrides code_dim / num_code from FSQ_LEVELS; the EMA path keeps the
        # configured CODE_DIM / NB_CODE. Doing this before any module construction so
        # `self.to_code` and the decoder pick up the right code_dim.
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
            from tokenization.utils.skeleton import get_smplx_body_parts
            self.smplx_body_parts = get_smplx_body_parts()

        self.num_tokens = getattr(arch_params, 'NUM_TOKENS', 160)
        self.n_heads = getattr(arch_params, 'N_HEADS', 8)
        self.dim_head = getattr(arch_params, 'DIM_HEAD', 64)

        # No constraint on code_dim: it is only the bottleneck projection width
        # (to_code: width -> code_dim) and the decoder's input embedding dim, which is
        # embedded back up to `width` before any attention. Heads always operate on `width`,
        # never on code_dim. This is why FSQ works at code_dim = len(levels) (typ. 3-7), and
        # it lets the EMA path use small code_dim too (e.g. 4, to match FSQ [8,8,6,5]).
        assert self.width % 8 == 0, f'WIDTH={self.width} must be divisible by 8'

        # Tier-1 capacity knobs (default to old single-block / 1× behaviour for compat).
        ffn_mult       = int(getattr(arch_params, 'FFN_MULT', 1))
        n_down_blocks  = int(getattr(arch_params, 'N_DOWN_BLOCKS', 1))
        n_up_blocks    = int(getattr(arch_params, 'N_UP_BLOCKS', 1))

        self.use_kinematic_pe = bool(getattr(arch_params, 'USE_KINEMATIC_PE', False))
        if self.use_kinematic_pe:
            lap_pe = _build_kinematic_laplacian_pe(self.num_joints)  # (J, J)
            self.register_buffer('lap_pe', lap_pe)
            # Encoder runs at `width` now → kinematic PE projects to width.
            self.kinematic_pe_proj = nn.Linear(self.num_joints, self.width, bias=False)

        _dropout          = float(getattr(arch_params, 'DROPOUT', 0.0))
        _emb_dropout      = float(getattr(arch_params, 'EMB_DROPOUT', 0.0))
        _emb_dropout_type = str(getattr(arch_params, 'EMB_DROPOUT_TYPE', 'drop'))
        _cross_dropout    = float(getattr(arch_params, 'CROSS_ATTN_DROPOUT', 0.0))

        # Skeleton-masked attention ("GNN via masking"): joint self-attention restricted to the
        # kinematic tree, turning each attention layer into a GAT layer. Adds zero parameters.
        # USE_JOINT_PE=False additionally drops the encoder's learned per-joint identity
        # embedding (ablation; note the skeleton's L/R symmetry means the mask alone cannot
        # distinguish mirror joints — see docs/masked_transformer.md).
        self.use_skeleton_mask = bool(getattr(arch_params, 'USE_SKELETON_MASK', False))
        self.use_joint_pe = bool(getattr(arch_params, 'USE_JOINT_PE', True))
        if self.use_skeleton_mask:
            assert _emb_dropout == 0.0, \
                'USE_SKELETON_MASK is incompatible with EMB_DROPOUT: DropTokenDropout removes tokens, misaligning the (J, J) mask'
            n_hops = int(getattr(arch_params, 'SKELETON_MASK_HOPS', 1))
            # persistent=False: deterministic from config; keeps state_dict keys unchanged so
            # pre-mask checkpoints still load with strict=True.
            self.register_buffer('skeleton_mask',
                                 build_skeleton_attention_mask(self.num_joints, n_hops=n_hops),
                                 persistent=False)
        else:
            self.skeleton_mask = None
        # Up-path self-attn runs over the 21 joint queries, so the mask applies there too —
        # but only TransformerCrossAttn (n_up_blocks >= 2) has self-attn; bare CrossAttention doesn't.
        self.mask_up_self_attn = self.use_skeleton_mask and n_up_blocks > 1

        # 1. ENCODER — runs at `width` (matches CNN baseline's full-width processing).
        self.encoder = TransformerEncoder(
            num_tokens=self.num_joints,
            token_dim=self.input_joint_dim,
            dim=self.width,
            depth=self.depth,
            heads=8,
            mlp_dim=ffn_mult * self.width,
            dropout=_dropout,
            emb_dropout=_emb_dropout,
            emb_dropout_type=_emb_dropout_type,
            use_pos_embedding=self.use_joint_pe,
        )

        # 2. DOWNSAMPLE — stacked cross-attn at `width`, queries also at `width`.
        self.latent_queries = nn.Parameter(torch.randn(1, self.num_tokens, self.width))
        self.cross_attn_down = _make_cross_attn(
            dim=self.width,
            context_dim=self.width,
            heads=8,
            depth=n_down_blocks,
            mlp_dim=ffn_mult * self.width,
            dropout=_cross_dropout,
        )

        # 2b. Project to code_dim only at the bottleneck (just before the quantizer).
        # FSQ-only: normalize before bottleneck to prevent encoder saturation into tanh bounds
        # (which caused perplexity collapse 1000 -> 90 in the [8,8,6,5] run). EMA path stays
        # byte-for-byte identical because nn.Identity is a no-op.
        self.code_norm = nn.LayerNorm(self.width) if self.quant == 'fsq' else nn.Identity()
        self.to_code = nn.Linear(self.width, self.code_dim)

        # 3. QUANTIZER (still at code_dim).
        if self.quant == 'fsq':
            self.quantizer = FSQQuantizer(levels=self._fsq_levels)
        else:
            self.quantizer = QuantizeEMAReset(
                self.num_code, self.code_dim,
                dist_metric=getattr(arch_params, 'DIST_METRIC', 'l2'),
            )

        # 4. DECODER — runs at width, embeds quantized code_dim → width via to_token_embedding.
        self.decoder = TransformerEncoder(
            num_tokens=self.num_tokens,
            token_dim=self.code_dim,
            dim=self.width,
            depth=self.depth,
            heads=8,
            mlp_dim=ffn_mult * self.width,
            dropout=_dropout,
            emb_dropout=_emb_dropout,
        )

        # 5. UPSAMPLE — stacked cross-attn at width.
        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints, self.width))
        self.cross_attn_up = _make_cross_attn(
            dim=self.width,
            context_dim=self.width,
            heads=8,
            depth=n_up_blocks,
            mlp_dim=ffn_mult * self.width,
            dropout=_cross_dropout,
        )

        self.decoder_projection = nn.Linear(self.width, self.output_joint_dim)

    def _cross_attn_up_joints(self, queries, context):
        """Up cross-attn; skeleton mask on the joint queries' self-attention when enabled."""
        if self.mask_up_self_attn:
            return self.cross_attn_up(queries, context=context, self_attn_mask=self.skeleton_mask)
        return self.cross_attn_up(queries, context=context)

    def _encode_continuous(self, x):
        """Pose -> continuous pre-quantization latent (B, num_tokens, code_dim)."""
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.view(batch_size, self.num_joints, -1)

        if x.shape[-1] == 3 and self.input_joint_dim == 6:
            x = matrix_to_rotation_6d(x)

        x_encoder = self.encoder(x, mask=self.skeleton_mask)
        if self.use_kinematic_pe:
            kin_pe = self.kinematic_pe_proj(self.lap_pe).unsqueeze(0)  # (1, J, width)
            x_encoder = x_encoder + kin_pe

        queries = self.latent_queries.expand(batch_size, -1, -1)
        x_encoder = self.cross_attn_down(queries, context=x_encoder)
        return self.to_code(self.code_norm(x_encoder))  # (B, num_tokens, code_dim)

    def encode(self, x):
        x_encoder = self._encode_continuous(x)
        batch_size = x_encoder.shape[0]
        x_encoder = x_encoder.permute(0, 2, 1).contiguous()
        x_encoder = self.quantizer.preprocess(x_encoder)
        code_idx = self.quantizer.quantize(x_encoder)
        return code_idx.view(batch_size, -1)

    def encode_soft(self, x, tau=0.01):
        """Pose -> soft codebook-assignment targets (B, num_tokens, nb_code).

        Softmax over the similarity of the continuous latent to every code, at
        temperature `tau`. In crowded codebooks (e.g. 2048 cosine codes in 4-d, where
        near-duplicate codes make the argmax ID almost arbitrary) this spreads target
        mass over the functionally-equivalent codes, giving a token-CE loss a learnable
        target. argmax(encode_soft(x)) == encode(x) for both distance metrics.
        """
        if isinstance(self.quantizer, FSQQuantizer):
            raise NotImplementedError('encode_soft needs an explicit codebook (EMA VQ); FSQ is not supported')
        z = self._encode_continuous(x)                    # (B, T, C)
        codebook = self.quantizer.codebook               # (S, C)
        if self.quantizer.dist_metric == 'cosine':
            sim = F.normalize(z, dim=-1) @ F.normalize(codebook, dim=-1).t()
        else:  # l2: similarity = -||z - e||^2
            sim = 2 * z @ codebook.t() - (z ** 2).sum(-1, keepdim=True) - (codebook ** 2).sum(-1)
        return (sim / tau).softmax(-1)

    def encode_latent_and_codes(self, x):
        """Pose -> (code_idx, latent, codebook), with `latent` in the codebook's own space.

        Lets callers measure code-to-latent distances without knowing which quantizer is in
        use. FSQ's codebook lives in the grid-normalized space reached via `bound()`, whereas
        an EMA-VQ codebook is compared against the raw latent directly.
        """
        z = self._encode_continuous(x)                        # (B, T, C)
        if isinstance(self.quantizer, FSQQuantizer):
            half_width = (self.quantizer._levels // 2).to(z.dtype)
            latent = self.quantizer.bound(z) / half_width
            codebook = self.quantizer.implicit_codebook
        else:
            latent = z
            codebook = self.quantizer.codebook
        return self.encode(x), latent, codebook

    def _decode_feat(self, decode_feat):
        """Shared decoder tail: (N, num_tokens, code_dim) latent -> (N, num_joints, out_dim)."""
        x_decoder = self.decoder(decode_feat)
        j_queries = self.joint_queries.expand(decode_feat.shape[0], -1, -1)
        x_decoder = self._cross_attn_up_joints(j_queries, x_decoder)
        return self.decoder_projection(x_decoder)

    def decode_logits(self, logits):
        decode_feat = self.quantizer.dequantize_logits(logits)  # (B, num_tokens, code_dim)
        return self._decode_feat(decode_feat)

    def decode_code_ids(self, code_idx, chunk=1024):
        """Decode hard token IDs (N, num_tokens) without materializing one-hot weights.

        `decode_logits` would need an (N, num_tokens, nb_code) tensor, which for the candidate
        sweeps in decoder-aware token supervision is enormous (4096 x 160 x 1920 floats is
        ~7.5 GiB and OOMs). Indexing the codebook is the same computation for one-hot weights.
        Chunked because the decoder's (N, num_tokens, width) activations dominate memory.
        """
        if isinstance(self.quantizer, FSQQuantizer):
            codebook = self.quantizer.implicit_codebook
        else:
            codebook = self.quantizer.codebook
        return torch.cat([self._decode_feat(codebook[ids]) for ids in code_idx.split(chunk)])

    def forward(self, x, global_step=None):
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.view(batch_size, self.num_joints, -1)

        if x.shape[-1] == 3 and self.input_joint_dim == 6:
            x = matrix_to_rotation_6d(x)

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

        # 1. Encode (full width; skeleton mask restricts joint self-attention when enabled)
        x_encoder = self.encoder(x, mask=self.skeleton_mask)
        if self.use_kinematic_pe:
            kin_pe = self.kinematic_pe_proj(self.lap_pe).unsqueeze(0)  # (1, J, width)
            x_encoder = x_encoder + kin_pe
        queries = self.latent_queries.expand(batch_size, -1, -1)
        x_encoder = self.cross_attn_down(queries, context=x_encoder)  # (B, num_tokens, width)
        x_encoder = self.to_code(self.code_norm(x_encoder))            # (B, num_tokens, code_dim)

        # 2. Quantize
        x_encoder_chan = x_encoder.permute(0, 2, 1)
        x_quantized_chan, loss, perplexity = self.quantizer(x_encoder_chan)
        x_quantized = x_quantized_chan.permute(0, 2, 1)

        # 3. Decode + Upsample (decoder embeds code_dim → width internally)
        x_decoder = self.decoder(x_quantized)
        j_queries = self.joint_queries.expand(batch_size, -1, -1)
        x_decoder = self._cross_attn_up_joints(j_queries, x_decoder)
        pred_pose_6d = self.decoder_projection(x_decoder)

        output = {}
        if self.rot_type == 'rot6d':
            pred_pose_rotmat = rotation_6d_to_matrix(pred_pose_6d.reshape(-1, 6)).view(batch_size, self.num_joints, 3, 3)
        output.update({'pred_pose_body_6d': pred_pose_6d, 'pred_pose_body_rotmat': pred_pose_rotmat})

        if self.mesh_inference:
            pred_pose_aa = matrix_to_axis_angle(pred_pose_rotmat.view(-1, 3, 3)).view(batch_size, 3 * self.num_joints)
            pred_body_mesh = body_model(body_pose=pred_pose_rotmat)
            output.update({'pred_pose_body_aa': pred_pose_aa, 'pred_body_mesh': pred_body_mesh,
                           'pred_body_vertices': pred_body_mesh.vertices, 'pred_body_joints': pred_body_mesh.joints})
        return output, loss, perplexity


class TransformerDecodeTokens(nn.Module):
    """Decoder-only wrapper for TransformerTokenizer — used by TokenHMR's TokenClassfier
    at inference (mirror of VanillaDecodeTokens for the CNN baseline).

    Reads architecture hyperparameters from the checkpoint's `hparams.ARCH` and reconstructs
    only the decoder-side modules (quantizer, decoder, joint_queries, cross_attn_up,
    decoder_projection). Compatible with both:
      - "simple transformer" checkpoints (FFN_MULT=1, N_UP_BLOCKS=1 — bare CrossAttention)
      - Tier-1 checkpoints                (FFN_MULT>=2, N_UP_BLOCKS>=2 — stacked TransformerCrossAttn)

    `forward(logits)` accepts soft codebook logits `(B, num_tokens, num_codes)` from the
    TokenClassfier and returns `(B, num_joints, 6)` 6D pose, matching VanillaDecodeTokens.
    """

    def __init__(self, ckpt_path: str = '', mesh_inference: bool = False):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        arch = ckpt['hparams'].ARCH

        self.num_joints = getattr(arch, 'NB_JOINTS', 21)
        self.width      = arch.WIDTH
        self.depth      = arch.DEPTH
        self.num_tokens = getattr(arch, 'NUM_TOKENS', 160)

        # Mirror the TransformerTokenizer quantizer branch — FSQ overrides
        # code_dim / num_code from FSQ_LEVELS so the decoder embedding shape matches the ckpt.
        self.quant = getattr(arch, 'QUANTIZER', 'ema_reset')
        if self.quant == 'fsq':
            self._fsq_levels = list(arch.FSQ_LEVELS)
            self.code_dim = len(self._fsq_levels)
            self.num_code = int(np.prod(self._fsq_levels))
        else:
            self.code_dim = arch.CODE_DIM[0] if isinstance(arch.CODE_DIM, list) else arch.CODE_DIM
            self.num_code = arch.NB_CODE[0]  if isinstance(arch.NB_CODE,  list) else arch.NB_CODE

        # Same defaults as TransformerTokenizer (1 = old simple-transformer behaviour).
        ffn_mult       = int(getattr(arch, 'FFN_MULT', 1))
        n_up_blocks    = int(getattr(arch, 'N_UP_BLOCKS', 1))
        _dropout       = float(getattr(arch, 'DROPOUT', 0.0))
        _emb_dropout   = float(getattr(arch, 'EMB_DROPOUT', 0.0))
        _cross_dropout = float(getattr(arch, 'CROSS_ATTN_DROPOUT', 0.0))

        # Skeleton mask — mirror TransformerTokenizer so masked checkpoints decode identically.
        # Old checkpoints lack the keys → flags default off → unchanged behaviour.
        self.use_skeleton_mask = bool(getattr(arch, 'USE_SKELETON_MASK', False))
        if self.use_skeleton_mask:
            n_hops = int(getattr(arch, 'SKELETON_MASK_HOPS', 1))
            self.register_buffer('skeleton_mask',
                                 build_skeleton_attention_mask(self.num_joints, n_hops=n_hops),
                                 persistent=False)
        else:
            self.skeleton_mask = None
        self.mask_up_self_attn = self.use_skeleton_mask and n_up_blocks > 1

        if self.quant == 'fsq':
            self.quantizer = FSQQuantizer(levels=self._fsq_levels)
        else:
            self.quantizer = QuantizeEMAReset(self.num_code, self.code_dim)

        self.decoder = TransformerEncoder(
            num_tokens=self.num_tokens,
            token_dim=self.code_dim,
            dim=self.width,
            depth=self.depth,
            heads=8,
            mlp_dim=ffn_mult * self.width,
            dropout=_dropout,
            emb_dropout=_emb_dropout,
        )

        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints, self.width))
        self.cross_attn_up = _make_cross_attn(
            dim=self.width,
            context_dim=self.width,
            heads=8,
            depth=n_up_blocks,
            mlp_dim=ffn_mult * self.width,
            dropout=_cross_dropout,
        )
        self.decoder_projection = nn.Linear(self.width, 6)

        self._load_weights(ckpt['net'])

    def _load_weights(self, state_dict: dict):
        """Filter the full tokenizer state_dict down to decoder-side keys and load them."""
        decoder_prefixes = (
            'quantizer.',
            'decoder.',
            'joint_queries',
            'cross_attn_up.',
            'decoder_projection.',
        )
        filtered = {k: v for k, v in state_dict.items()
                    if any(k.startswith(p) for p in decoder_prefixes)}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        decoder_missing = [k for k in missing if any(k.startswith(p) for p in decoder_prefixes)]
        if decoder_missing:
            print(f'[TransformerDecodeTokens] WARNING: missing decoder keys: {decoder_missing}')
        if unexpected:
            print(f'[TransformerDecodeTokens] WARNING: unexpected keys: {unexpected}')

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Soft-decode token logits to 6D pose.

        Args:
            logits: (B, num_tokens, num_codes) softmax weights from TokenClassfier.
        Returns:
            (B, num_joints, 6) 6D rotation predictions for the body joints.
        """
        batch_size = logits.shape[0]
        decode_feat = self.quantizer.dequantize_logits(logits)  # (B, num_tokens, code_dim)
        x_decoder = self.decoder(decode_feat)
        j_queries = self.joint_queries.expand(batch_size, -1, -1)
        if self.mask_up_self_attn:
            x_decoder = self.cross_attn_up(j_queries, context=x_decoder, self_attn_mask=self.skeleton_mask)
        else:
            x_decoder = self.cross_attn_up(j_queries, context=x_decoder)
        return self.decoder_projection(x_decoder)
