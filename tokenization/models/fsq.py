# -*- coding: utf-8 -*-
"""PyTorch port of Finite Scalar Quantization (FSQ).

Mentzer et al., 2023 — "Finite Scalar Quantization: VQ-VAE Made Simple".
Reference (JAX) implementation:
    https://colab.research.google.com/github/google-research/google-research/blob/master/fsq/fsq.ipynb

Drop-in replacement for VQ that mirrors the public surface of
`quantize_cnn.QuantizeEMAReset` so the rest of the tokenizer (TransformerTokenizer,
TransformerDecodeTokens) can swap quantizers via a single config flag.

Original Apache-2.0 license header from the JAX impl is reproduced below.

    Copyright 2023 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn

from tokenization.utils.utils_model import codebook_usage_stats


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradients."""
    return z + (torch.round(z) - z).detach()


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantizer.

    Each input channel is bounded via tanh and rounded to one of `Lᵢ` integers,
    yielding an implicit codebook of size `∏Lᵢ`. No commitment / codebook losses,
    no EMA, no codebook resets — codebook utilization is ~100% by construction.

    Public surface mirrors `QuantizeEMAReset` so it's a drop-in replacement:
        forward((B, d, T))     -> ((B, d, T), commit_loss=0, perplexity)
        preprocess((B, d, T))  -> (B*T, d)
        quantize((N, d))       -> (N,) long indices in [0, ∏Lᵢ)
        dequantize((N,))       -> (N, d)
        dequantize_logits      -> (..., num_codes) @ codebook -> (..., d)
    """

    def __init__(self, levels: List[int], eps: float = 1e-3):
        super().__init__()
        assert len(levels) >= 1, "FSQ levels must be a non-empty list"
        assert all(int(l) >= 2 for l in levels), f"each level must be >= 2, got {levels}"

        self.eps = eps
        self.code_dim = len(levels)
        self.nb_code = int(np.prod(levels))

        levels_t = torch.tensor(levels, dtype=torch.long)
        basis = torch.tensor(
            np.concatenate(([1], np.cumprod(np.asarray(levels)[:-1]))).astype(np.int64),
            dtype=torch.long,
        )
        self.register_buffer("_levels", levels_t, persistent=True)
        self.register_buffer("_basis", basis, persistent=True)

        # Implicit codebook (∏Lᵢ, d) — precomputed as a buffer so
        # `dequantize_logits(logits) = logits @ codebook` is one matmul.
        all_indices = torch.arange(self.nb_code, dtype=torch.long)
        codebook = self._indices_to_codes_impl(all_indices, levels_t, basis)
        self.register_buffer("implicit_codebook", codebook, persistent=True)

        # Codebook-usage stats from the most recent forward (read by the train/eval loops).
        self.last_code_counts = None
        self.codebook_stats = {}

    # --- core FSQ ops --------------------------------------------------------

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound z so that round(f(z)) lands on an integer in [-⌊L/2⌋, ⌊L/2⌋]."""
        levels = self._levels.to(z.dtype)
        half_l = (levels - 1) * (1 - self.eps) / 2
        offset = torch.where(self._levels % 2 == 1, torch.zeros_like(levels), 0.5 * torch.ones_like(levels))
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def _quantize_to_grid(self, z: torch.Tensor) -> torch.Tensor:
        """Round bounded z to integer grid, then renormalize to [-1, 1]."""
        quantized = round_ste(self.bound(z))
        half_width = (self._levels // 2).to(z.dtype)
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        """Map normalized [-1, 1] codes back to integer range [0, L-1]."""
        half_width = (self._levels // 2).to(zhat_normalized.dtype)
        return zhat_normalized * half_width + half_width

    def _scale_and_shift_inverse(self, zhat_int: torch.Tensor) -> torch.Tensor:
        half_width = (self._levels // 2).to(zhat_int.dtype)
        return (zhat_int - half_width) / half_width

    @staticmethod
    def _indices_to_codes_impl(indices: torch.Tensor, levels: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Buffer-free helper used during __init__ to build the implicit codebook."""
        idx = indices.unsqueeze(-1)
        codes_int = torch.remainder(torch.div(idx, basis, rounding_mode="floor"), levels)
        half_width = (levels // 2).to(torch.float32)
        codes_int = codes_int.to(torch.float32)
        return (codes_int - half_width) / half_width

    # --- public surface (matches QuantizeEMAReset) ---------------------------

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # NCT -> NTC -> [N*T, C]
        x = x.permute(0, 2, 1).contiguous()
        return x.view(-1, x.shape[-1])

    def quantize(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Return long indices in [0, nb_code)."""
        zhat = self._quantize_to_grid(x_flat)
        zhat_int = self._scale_and_shift(zhat)
        # round to clean integers before basis-mixing (guards against any FP drift)
        zhat_int = torch.round(zhat_int).to(torch.long)
        return (zhat_int * self._basis).sum(dim=-1)

    def dequantize(self, code_idx: torch.Tensor) -> torch.Tensor:
        """Indices -> code vectors. Equivalent to F.embedding against implicit_codebook."""
        return self.implicit_codebook[code_idx]

    def dequantize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Soft decode: (..., num_codes) softmax weights -> (..., d)."""
        return torch.matmul(logits, self.implicit_codebook)

    @torch.no_grad()
    def _compute_perplexity(self, code_idx: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(code_idx, minlength=self.nb_code).to(torch.float32)
        prob = counts / counts.sum().clamp(min=1.0)
        return torch.exp(-(prob * (prob + 1e-10).log()).sum())

    def forward(self, x: torch.Tensor):
        """x: (B, d, T) -> (x_d: (B, d, T), commit_loss: 0-tensor, perplexity)."""
        input_3d = x.dim() == 3
        if input_3d:
            N, d, T = x.shape
            x = self.preprocess(x)

        x_d = self._quantize_to_grid(x)  # (B*T, d), STE through round

        with torch.no_grad():
            code_idx = self.quantize(x)
            perplexity = self._compute_perplexity(code_idx)
            counts = torch.bincount(code_idx.reshape(-1), minlength=self.nb_code).float()
            self.last_code_counts = counts
            self.codebook_stats = codebook_usage_stats(counts, self.nb_code)

        if input_3d:
            x_d = x_d.view(N, T, d).permute(0, 2, 1).contiguous()

        commit_loss = torch.zeros((), device=x.device, dtype=x_d.dtype)
        return x_d, commit_loss, perplexity
