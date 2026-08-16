"""Analyses of a trained tokenizer: stability, codebook geometry, token influence.

These modules import one another. ``analyze_latent_pose_info`` is the shared
library the rest build on, and several scripts set its module-level ``DEVICE``
to CPU before calling it, because the model classes place themselves on the GPU
at construction and the analyses are run while the GPU is busy training.
"""
