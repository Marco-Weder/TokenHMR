"""Read the provenance manifest.

The manifest is the answer to "which checkpoint produced this number". It is
built by :mod:`repro.build_manifest` from evaluation results that already
exist, and it is version controlled so that a clone can resolve a tokenizer
label or look up a reported value without first re-running anything.

Two entry points matter to the rest of the codebase:

    tokenizers()                    the eight stage-1 tokenizers
    value(run, dataset, metric)     a measured downstream number

:func:`value` cross-checks against ``thesis_numbers.yaml`` when the artifact is
one the thesis prints, so a figure can never silently render a number that
differs from the published one.
"""

from __future__ import annotations

import csv
import functools
import os
from pathlib import Path

from repro.paths import MANIFEST_DIR, TOKENIZER_OUT, relative, resolve

EVAL_INDEX = MANIFEST_DIR / "eval_index.csv"
TOKENIZERS = MANIFEST_DIR / "tokenizers.yaml"
THESIS_NUMBERS = MANIFEST_DIR / "thesis_numbers.yaml"

# Stable directory name per tokenizer label. These are the names the alias tree
# and the thesis both use, so a reader can match one to the other.
ALIAS_SLUG = {
    "CNN": "cnn-l2-d256",
    "Transformer tier1": "transformer-l2-d256",
    "Transformer cosine": "transformer-cosine-d256",
    "Skeleton-masked": "skeleton-masked-cosine-d256",
    "VQ d4": "transformer-cosine-d4",
    "VQ d2": "transformer-cosine-d2",
    "FSQ d4": "transformer-fsq-d4",
    "FSQ d5": "transformer-fsq-d5",
}


def _load_yaml(path: Path):
    """Parse the small subset of YAML the manifest uses.

    Deliberately dependency-free: reading provenance must not require the
    training environment to be installed.
    """
    try:
        import yaml  # noqa: PLC0415

        return yaml.safe_load(path.read_text())
    except ImportError:
        pass

    root: dict = {}
    stack: list[tuple[int, object]] = [(-1, root)]
    current_list: list | None = None
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            line = line[2:]
            if not isinstance(parent, list):
                continue
            item: dict = {}
            parent.append(item)
            stack.append((indent, item))
            parent = item
            current_list = parent
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        target = stack[-1][1]
        if val == "":
            child: object = []
            # A key with no value introduces either a list or a mapping; peek is
            # not available here, so a mapping is created lazily on first use.
            child = _Pending()
            target[key] = child
            stack.append((indent, child))
        else:
            target[key] = _scalar(val)
    return _resolve_pending(root)


class _Pending(dict):
    """A node whose kind (list or mapping) is decided by its first child."""

    def append(self, item):
        self.setdefault("__list__", []).append(item)


def _resolve_pending(node):
    if isinstance(node, _Pending):
        if "__list__" in node:
            return [_resolve_pending(x) for x in node["__list__"]]
        return {k: _resolve_pending(v) for k, v in node.items()}
    if isinstance(node, dict):
        return {k: _resolve_pending(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_pending(x) for x in node]
    return node


def _scalar(text: str):
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text in ("null", "~", ""):
        return None
    if text in ("true", "false"):
        return text == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


@functools.lru_cache(maxsize=1)
def tokenizers() -> list[dict]:
    """The stage-1 tokenizers, newest registry first, alias slug attached."""
    if not TOKENIZERS.is_file():
        raise FileNotFoundError(
            f"{relative(TOKENIZERS)} is missing. Run `python -m repro.build_manifest`."
        )
    data = _load_yaml(TOKENIZERS)
    out = []
    for entry in data.get("tokenizers", []):
        entry = dict(entry)
        entry["slug"] = ALIAS_SLUG.get(entry.get("label", ""), "")
        entry["alias"] = relative(TOKENIZER_OUT / "tokenizers" / entry["slug"] / "best_net.pth")
        # YAML reads the jitter keys 0.5/1/2 as numbers; the callers ask for
        # them by the string the thesis uses.
        entry["stability_pct"] = {
            str(k): v for k, v in (entry.get("stability_pct") or {}).items()
        }
        out.append(entry)
    return out


def tokenizer(label: str) -> dict:
    for entry in tokenizers():
        if entry["label"] == label or entry["slug"] == label:
            return entry
    known = ", ".join(e["label"] for e in tokenizers())
    raise KeyError(f"unknown tokenizer {label!r}. Known labels: {known}")


def tokenizer_checkpoint(label: str) -> Path:
    """Absolute path to a tokenizer checkpoint, alias first, real path second."""
    entry = tokenizer(label)
    alias = resolve(entry["alias"])
    if alias.exists():
        return alias
    return resolve(entry["checkpoint"])


@functools.lru_cache(maxsize=1)
def _rows() -> list[dict]:
    if not EVAL_INDEX.is_file():
        raise FileNotFoundError(
            f"{relative(EVAL_INDEX)} is missing. Run `python -m repro.build_manifest`."
        )
    with EVAL_INDEX.open(newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["superseded"] == "0"]


@functools.lru_cache(maxsize=1)
def _published() -> dict:
    if not THESIS_NUMBERS.is_file():
        return {}
    data = _load_yaml(THESIS_NUMBERS) or {}
    out = {}
    for artifact, entries in (data.get("artifacts") or {}).items():
        for entry in entries or []:
            key = (entry.get("run"), entry.get("dataset"), entry.get("metric"))
            out[key] = {
                "value": entry.get("value"),
                "artifact": artifact,
                "status": entry.get("status", "measured"),
                # Half the last printed digit, so a table printed to one decimal
                # is not reported as drifting from a two-decimal measurement.
                "tolerance": entry.get("tolerance"),
            }
    return out


def value(run: str, dataset: str, metric: str, tolerance: float = 0.005) -> float:
    """A measured number, checked against the value the thesis prints.

    Raises if the manifest and the thesis disagree. That is the point: a figure
    regenerated after a code change must either match what was published or
    fail loudly enough to be noticed.
    """
    hits = [
        r for r in _rows()
        if r["exp_name"] == run and r["dataset"] == dataset and r["metric_name"] == metric
    ]
    if not hits:
        raise KeyError(f"no manifest row for run={run!r} dataset={dataset!r} metric={metric!r}")
    measured = float(hits[-1]["metric_value"])

    published = _published().get((run, dataset, metric))
    if published is not None and published["status"] == "measured":
        expected = published["value"]
        if expected is not None:
            tol = published["tolerance"] or tolerance
            if abs(float(expected) - measured) > tol:
                raise ValueError(
                    f"{published['artifact']}: {run}/{dataset}/{metric} is {measured} in "
                    f"the manifest but {expected} in the thesis (tolerance {tol}). "
                    f"Resolve the discrepancy before regenerating this figure."
                )
    return measured


def check_published() -> list[str]:
    """Verify every published value against the manifest. Returns the failures."""
    failures = []
    for (run, dataset, metric), entry in sorted(_published().items()):
        if entry["status"] != "measured":
            continue
        try:
            value(run, dataset, metric)
        except (KeyError, ValueError) as exc:
            failures.append(str(exc))
    return failures


def runs() -> list[str]:
    return sorted({r["exp_name"] for r in _rows()})


def link_tokenizers(dry_run: bool = False) -> list[str]:
    """Create the stable alias tree for the stage-1 checkpoints.

    The real checkpoints sit in timestamped directories, and the timestamp is
    what distinguishes two trainings of the same config, so it is never renamed.
    These symlinks give each tokenizer a name that a config can reference.
    """
    made = []
    for entry in tokenizers():
        target = resolve(entry["checkpoint"])
        link_dir = TOKENIZER_OUT / "tokenizers" / entry["slug"]
        link = link_dir / "best_net.pth"
        if not target.exists():
            made.append(f"skip  {entry['slug']:32s} (checkpoint not present)")
            continue
        if dry_run:
            made.append(f"would link  {entry['slug']:32s} -> {relative(target)}")
            continue
        link_dir.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(target, link_dir))
        made.append(f"linked  {entry['slug']:32s} -> {relative(target)}")
    return made
