"""Single source of truth for every path in this project.

Nothing here depends on the current working directory. Resolution order is
always the same, and it is the only order used anywhere in the codebase:

    1. an explicit argument passed by the caller (a ``--out-dir`` flag)
    2. an environment variable
    3. a default derived from the ``.project-root`` sentinel file

Import this instead of computing ``os.path.dirname(os.path.dirname(__file__))``.
"""

from __future__ import annotations

import os
from pathlib import Path

_SENTINEL = ".project-root"


def _find_root() -> Path:
    env = os.environ.get("THESIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / _SENTINEL).is_file():
            return parent
    # repro/ sits directly under the project root, so this is the right
    # fallback if the sentinel was deleted.
    return here.parent.parent


PROJECT_ROOT = _find_root()


def _dir(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _dir("THESIS_DATA_DIR", PROJECT_ROOT / "data")
DATASET_DIR = _dir("THESIS_DATASET_DIR", PROJECT_ROOT / "dataset_dir")
LOGS_DIR = _dir("THESIS_LOGS_DIR", PROJECT_ROOT / "logs")
RESULTS_DIR = _dir("THESIS_RESULTS_DIR", PROJECT_ROOT / "results")
TOKENIZER_OUT = _dir("THESIS_TOKENIZER_OUT", PROJECT_ROOT / "tokenization" / "output")
MANIFEST_DIR = PROJECT_ROOT / "manifest"
FIGURES_DIR = PROJECT_ROOT / "figures"

BODY_MODELS = DATA_DIR / "body_models"
RELEASE_DIR = RESULTS_DIR / "release"


def resolve(path: str | os.PathLike) -> Path:
    """Interpret ``path`` relative to the project root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def relative(path: str | os.PathLike) -> str:
    """Express ``path`` relative to the project root when it sits inside it.

    Used when writing provenance, so that a manifest committed on one machine
    is meaningful on another.
    """
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def figure_out_dir(section: str, override: str | os.PathLike | None = None) -> Path:
    """Where a figure script should write ``images/<section>/``.

    The LaTeX sources live in a separate repository that a clone of this one
    does not have. The last branch keeps figure scripts working there, and the
    ``is_dir`` guards are what stop a missing checkout from being papered over
    with an empty directory tree.
    """
    if override:
        out = Path(override).expanduser().resolve() / section
        out.mkdir(parents=True, exist_ok=True)
        return out

    tex_root = os.environ.get("THESIS_TEX_ROOT")
    if tex_root and (Path(tex_root).expanduser() / "images").is_dir():
        out = Path(tex_root).expanduser().resolve() / "images" / section
        out.mkdir(parents=True, exist_ok=True)
        return out

    sibling = PROJECT_ROOT.parent / "thesis"
    if (sibling / "images").is_dir():
        out = sibling / "images" / section
        out.mkdir(parents=True, exist_ok=True)
        return out

    out = FIGURES_DIR / section
    out.mkdir(parents=True, exist_ok=True)
    return out


def add_figure_args(parser):
    """Add the output-location flag every figure script shares."""
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write images/<section>/ into. Defaults to the LaTeX "
        "checkout if one is present, otherwise <project root>/figures/.",
    )
    return parser


def describe() -> str:
    """Human-readable dump of every resolved path, for `thesis paths`."""
    rows = [
        ("PROJECT_ROOT", PROJECT_ROOT),
        ("DATA_DIR", DATA_DIR),
        ("DATASET_DIR", DATASET_DIR),
        ("LOGS_DIR", LOGS_DIR),
        ("RESULTS_DIR", RESULTS_DIR),
        ("TOKENIZER_OUT", TOKENIZER_OUT),
        ("MANIFEST_DIR", MANIFEST_DIR),
        ("FIGURES_DIR", FIGURES_DIR),
        ("BODY_MODELS", BODY_MODELS),
    ]
    width = max(len(name) for name, _ in rows)
    lines = []
    for name, path in rows:
        mark = "ok     " if Path(path).exists() else "MISSING"
        lines.append(f"  {mark}  {name:<{width}}  {path}")
    return "\n".join(lines)
