"""Snapshot and verify the analysis outputs that must not move.

The reorganisation touches imports, path resolution and model construction. Any
of those could in principle change a number that is already printed in the
thesis. This records digests of the current outputs so a later re-run can be
compared against them.

    python -m repro.golden record     # snapshot the current outputs
    python -m repro.golden check      # re-hash and diff against the snapshot

Only deterministic CPU artifacts are tracked. Rendered meshes are deliberately
excluded: rasterisation is not bit-reproducible across driver versions, so
hashing a render produces false alarms rather than evidence.

Numeric JSON is compared value by value at a tolerance, not by file hash, so
that a formatting change is not reported as a numerical regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from repro.paths import PROJECT_ROOT, TOKENIZER_OUT, relative

GOLDEN_DIR = PROJECT_ROOT / "manifest" / "golden"

# Deterministic, seed-0, CPU. Compared numerically.
NUMERIC = [
    TOKENIZER_OUT / "codebook_geometry" / "summary.json",
    TOKENIZER_OUT / "recon_by_dataset" / "summary_all.json",
    TOKENIZER_OUT / "token_stability" / "summary.json",
]

TOLERANCE = 1e-6


def flatten(obj, prefix: str = "") -> dict[str, float]:
    """Every numeric leaf of a JSON document, keyed by its path."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def digest_numeric(path: Path) -> dict:
    data = json.loads(path.read_text())
    values = flatten(data)
    canonical = json.dumps({k: round(v, 9) for k, v in sorted(values.items())})
    return {
        "n_values": len(values),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "values": {k: round(v, 9) for k, v in sorted(values.items())},
    }


def record() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    recorded, missing = [], []
    for path in NUMERIC:
        if not path.is_file():
            missing.append(relative(path))
            continue
        name = f"{path.parent.name}__{path.stem}.json"
        (GOLDEN_DIR / name).write_text(json.dumps(digest_numeric(path), indent=1))
        recorded.append(relative(path))

    index = {
        "note": "Numeric digests of deterministic CPU analysis outputs, recorded "
        "before the repository reorganisation. Renders are excluded on purpose.",
        "recorded": recorded,
        "missing": missing,
        "tolerance": TOLERANCE,
    }
    (GOLDEN_DIR / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    for item in recorded:
        print(f"recorded  {item}")
    for item in missing:
        print(f"MISSING   {item}  (not snapshotted)")
    return 0


def check() -> int:
    if not (GOLDEN_DIR / "index.json").is_file():
        print("no golden snapshot found, run `python -m repro.golden record` first")
        return 2
    failures = 0
    for path in NUMERIC:
        name = f"{path.parent.name}__{path.stem}.json"
        stored = GOLDEN_DIR / name
        if not stored.is_file():
            continue
        if not path.is_file():
            print(f"MISSING   {relative(path)}")
            failures += 1
            continue
        want = json.loads(stored.read_text())["values"]
        got = digest_numeric(path)["values"]
        drifted = [
            (k, want[k], got[k])
            for k in want.keys() & got.keys()
            if abs(want[k] - got[k]) > TOLERANCE
        ]
        only_before = sorted(want.keys() - got.keys())
        only_after = sorted(got.keys() - want.keys())
        if drifted or only_before or only_after:
            failures += 1
            print(f"CHANGED   {relative(path)}")
            for k, a, b in drifted[:10]:
                print(f"            {k}: {a} -> {b}")
            if len(drifted) > 10:
                print(f"            ... and {len(drifted) - 10} more")
            for k in only_before[:5]:
                print(f"            missing now: {k}")
            for k in only_after[:5]:
                print(f"            new: {k}")
        else:
            print(f"ok        {relative(path)}  ({len(got)} values)")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["record", "check"])
    args = ap.parse_args(argv)
    return record() if args.action == "record" else check()


if __name__ == "__main__":
    sys.exit(main())
