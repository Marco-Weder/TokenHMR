"""Make configs saved by earlier runs portable.

Every training run writes a ``model_config.yaml`` next to its checkpoints, and
the ones produced during this thesis recorded absolute paths from the machine
they were trained on. Rewriting those files in place would destroy provenance,
so the paths are rewritten as they are read instead. This is what lets a
finished run be re-evaluated on any machine without retraining it.

The rewrite is deliberately noisy: it logs the first time it fires, so a config
that needs it can never look like one that does not.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from repro.paths import PROJECT_ROOT

log = logging.getLogger(__name__)

# Roots this project has lived under. Order matters only in that the longest
# match should win, which the sort below guarantees.
_LEGACY_ROOTS = (
    "/home/marco/Exploring-Latent-Representations-for-Human-Mesh-Recovery/external/tokenhmr",
    "/home/marco/thesis-HMR/external/tokenhmr",
    "/home/marco/thesis-HMR",
)

# Directory names that only ever appear at the top of this project. A config
# value starting with one of these was written to be read from the project root,
# so it is anchored there rather than at whatever the working directory happens
# to be.
_PROJECT_RELATIVE = (
    "data/",
    "dataset_dir/",
    "logs/",
    "results/",
    "tokenization/output/",
    "tokenization_data/",
)

_warned = False


def anchor_relative(value: Any) -> Any:
    """Anchor a project-relative path at the project root."""
    if not isinstance(value, str) or not value or os.path.isabs(value):
        return value
    if value.startswith(_PROJECT_RELATIVE):
        return os.path.join(str(PROJECT_ROOT), value)
    return value


def rewrite_legacy_path(value: Any) -> Any:
    """Repoint an absolute path recorded on another machine at this checkout.

    Non-string values and paths that do not start with a known legacy root are
    returned unchanged, so this is safe to apply to anything.
    """
    global _warned
    if not isinstance(value, str):
        return value

    for root in sorted(_LEGACY_ROOTS, key=len, reverse=True):
        if value.startswith(root):
            tail = value[len(root):].lstrip("/")
            rewritten = os.path.join(str(PROJECT_ROOT), tail) if tail else str(PROJECT_ROOT)
            if not _warned:
                log.warning(
                    "Rewriting paths recorded on another machine (%s -> %s). "
                    "This is expected for configs saved by the thesis runs.",
                    root,
                    PROJECT_ROOT,
                )
                _warned = True
            return rewritten
    return value


def rewrite_cfg(cfg) -> None:
    """Apply :func:`rewrite_legacy_path` to every string leaf of a yacs CfgNode.

    Mutates in place, so call it before the node is frozen.
    """
    for key in list(cfg.keys()):
        value = cfg[key]
        if hasattr(value, "keys"):
            rewrite_cfg(value)
        elif isinstance(value, str):
            cfg[key] = anchor_relative(rewrite_legacy_path(value))
        elif isinstance(value, (list, tuple)):
            rewritten = [anchor_relative(rewrite_legacy_path(v)) for v in value]
            cfg[key] = type(value)(rewritten)
