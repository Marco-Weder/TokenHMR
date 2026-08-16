"""Build the provenance manifest from results that already exist on disk.

Nothing is re-run and nothing under ``results/`` is modified. The evaluation
CSVs are the primary record of every number this thesis reports, so they are
read and never written.

    python -m repro.build_manifest [--out manifest/]

Produces:
    manifest/eval_index.csv   one row per (run, dataset, metric), normalised
    manifest/tokenizers.yaml  the eight stage-1 tokenizers and their checkpoints
    manifest/build_report.txt what was found, and what was missing

Two properties of the source data drive the design:

* The CSVs are append-only and lock-guarded, so re-evaluating a checkpoint
  stacks a second set of rows on top of the first. Reading one naively gives
  whichever row happens to come first. Rows are therefore deduplicated by
  timestamp, and the superseded ones are kept and flagged rather than dropped.
* A handful of rows carry absolute paths from this machine. Those are
  normalised here, in the generated file, and left alone at the source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from repro.legacy import rewrite_legacy_path
from repro.paths import PROJECT_ROOT, RESULTS_DIR, TOKENIZER_OUT, relative, resolve

FIELDS = [
    "exp_name",
    "run_dir",
    "experiment_config",
    "checkpoint_path",
    "ckpt_bytes",
    "ckpt_fingerprint",
    "decode_mode",
    "dataset",
    "metric_name",
    "metric_value",
    "iters_done",
    "timestamp",
    "results_file",
    "superseded",
]


def fingerprint(path: Path, chunk: int = 1 << 20) -> tuple[int, str]:
    """Size plus a hash of the first and last MiB.

    A full hash of 80 checkpoints averaging 2.7 GB is hours of IO and protects
    against nothing that this does not: the failure being guarded against is a
    file swapped for a different one, not a targeted collision.
    """
    if not path.is_file():
        return 0, ""
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk:
            fh.seek(max(0, size - chunk))
            h.update(fh.read(chunk))
    return size, h.hexdigest()[:16]


def run_dir_of(checkpoint: str) -> str:
    """logs/<task>/runs/<run>/checkpoints/x.ckpt -> logs/<task>/runs/<run>"""
    p = Path(checkpoint)
    if p.parent.name == "checkpoints":
        return relative(p.parent.parent)
    return relative(p.parent)


def experiment_config_of(run_dir: Path) -> str:
    """Recover `experiment=<name>` from the Hydra overrides the run saved."""
    overrides = run_dir / ".hydra" / "overrides.yaml"
    if not overrides.is_file():
        return ""
    for line in overrides.read_text().splitlines():
        line = line.strip().lstrip("-").strip()
        if line.startswith("experiment="):
            return line.split("=", 1)[1]
    return ""


def decode_mode_of(metric_name: str, exp_name: str) -> str:
    if metric_name.startswith("hard_"):
        return "hard"
    if metric_name.startswith("soft_"):
        return "soft"
    if exp_name.endswith("_hard"):
        return "hard"
    if exp_name.endswith("_soft"):
        return "soft"
    return ""


def collect_rows(report: list[str]) -> list[dict]:
    csv_paths = sorted(RESULTS_DIR.rglob("eval_regression*.csv"))
    report.append(f"evaluation CSVs found: {len(csv_paths)}")

    raw: list[dict] = []
    for path in csv_paths:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if not row.get("metric_name"):
                    continue
                if row.get("error"):
                    continue
                raw.append({**row, "_file": relative(path)})

    ckpt_cache: dict[str, tuple[int, str]] = {}
    cfg_cache: dict[str, str] = {}
    rows: list[dict] = []

    for r in raw:
        ckpt = relative(rewrite_legacy_path(r["checkpoint_path"]))
        rdir = run_dir_of(ckpt)
        if ckpt not in ckpt_cache:
            ckpt_cache[ckpt] = fingerprint(resolve(ckpt))
        if rdir not in cfg_cache:
            cfg_cache[rdir] = experiment_config_of(resolve(rdir))
        size, fp = ckpt_cache[ckpt]
        rows.append(
            {
                "exp_name": r["exp_name"],
                "run_dir": rdir,
                "experiment_config": cfg_cache[rdir],
                "checkpoint_path": ckpt,
                "ckpt_bytes": size,
                "ckpt_fingerprint": fp,
                "decode_mode": decode_mode_of(r["metric_name"], r["exp_name"]),
                "dataset": r["dataset"],
                "metric_name": r["metric_name"],
                "metric_value": r["metric_value"],
                "iters_done": r.get("iters_done", ""),
                "timestamp": r["timestamp"],
                "results_file": r["_file"],
                "superseded": 0,
            }
        )

    # Newest row wins per (exp_name, dataset, metric, checkpoint); the rest are
    # kept and flagged, because an overwritten measurement is still evidence of
    # what was measured when.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["exp_name"], row["dataset"], row["metric_name"], row["checkpoint_path"])
        groups[key].append(row)

    superseded = 0
    for group in groups.values():
        group.sort(key=lambda r: r["timestamp"])
        for row in group[:-1]:
            row["superseded"] = 1
            superseded += 1

    report.append(f"metric rows: {len(rows)} ({superseded} superseded by a later evaluation)")
    rows.sort(key=lambda r: (r["exp_name"], r["dataset"], r["metric_name"], r["timestamp"]))
    return rows


def report_gaps(rows: list[dict], report: list[str]) -> None:
    """Name the run directories that produced no usable metric row.

    Silently skipping these would turn an unevaluated run into an apparently
    complete manifest.
    """
    release = RESULTS_DIR / "release"
    if not release.is_dir():
        return
    have = {r["exp_name"] for r in rows}
    missing = sorted(d.name for d in release.iterdir() if d.is_dir() and d.name not in have)
    report.append(f"run directories under results/release: {len(list(release.iterdir()))}")
    if missing:
        report.append(f"directories with no usable metric row ({len(missing)}):")
        report.extend(f"    {name}" for name in missing)


def build_tokenizers(report: list[str]) -> dict:
    """Stage-1 registry, lifted out of the two gitignored analysis summaries."""
    stability_path = TOKENIZER_OUT / "token_stability" / "summary.json"
    recon_path = TOKENIZER_OUT / "recon_by_dataset" / "summary_all.json"
    if not stability_path.is_file():
        report.append(f"WARNING: {relative(stability_path)} missing, tokenizers.yaml not built")
        return {}

    stability = json.loads(stability_path.read_text())
    recon_mm: dict[str, float] = {}
    if recon_path.is_file():
        recon = json.loads(recon_path.read_text())
        for label, entry in recon.get("tokenizers", {}).items():
            overall = entry.get("datasets", {}).get("overall") or entry.get("overall")
            if isinstance(overall, dict) and "mpjpe_mm_mean" in overall:
                recon_mm[label] = round(overall["mpjpe_mm_mean"], 4)
    else:
        report.append(f"note: {relative(recon_path)} missing, no reconstruction MPJPE recorded")

    out = {
        "num_poses": stability.get("num_poses"),
        "seed": stability.get("seed"),
        "nn_metric": stability.get("nn_metric"),
        "tokenizers": [],
    }
    for run in stability.get("runs", []):
        label = run.get("label", "")
        out["tokenizers"].append(
            {
                "label": label,
                "quantizer": run.get("quantizer"),
                "metric": run.get("metric"),
                "code_dim": run.get("code_dim"),
                "num_codes": run.get("num_codes"),
                "num_tokens": run.get("num_tokens"),
                "checkpoint": relative(resolve(rewrite_legacy_path(run.get("ckpt", "")))),
                "recon_deg_per_joint": round(run.get("recon_deg_per_joint", 0.0), 4),
                "recon_mpjpe_mm": recon_mm.get(label),
                "stability_pct": {
                    jitter: round(entry["ids_unchanged_pct"], 3)
                    for jitter, entry in run.get("stability", {}).items()
                },
                "codes_used_pct": round(run.get("codes_used_pct", 0.0), 3),
                "nn_swap_deg": round(run.get("nn_swap_deg", 0.0), 4),
            }
        )
    report.append(f"stage-1 tokenizers recorded: {len(out['tokenizers'])}")
    return out


def dump_yaml(obj, indent: int = 0) -> str:
    """Minimal YAML writer, so the manifest has no import-time dependency."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{pad}{key}:")
                lines.append(dump_yaml(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {scalar(value)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                body = dump_yaml(item, indent + 1).splitlines()
                lines.append(f"{pad}- {body[0].strip()}")
                lines.extend(body[1:])
            else:
                lines.append(f"{pad}- {scalar(item)}")
    return "\n".join(lines)


def scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    return f'"{text}"' if (text == "" or any(c in text for c in ':#{}[],&*?|<>=!%@`"')) else text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="output directory (default manifest/)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out).resolve() if args.out else PROJECT_ROOT / "manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: list[str] = [f"project root: {PROJECT_ROOT}"]
    rows = collect_rows(report)
    report_gaps(rows, report)

    index = out_dir / "eval_index.csv"
    with index.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    tokenizers = build_tokenizers(report)
    if tokenizers:
        header = (
            "# Stage-1 tokenizer registry, generated by `python -m repro.build_manifest`.\n"
            "# Source: tokenization/output/{token_stability,recon_by_dataset}/, which are\n"
            "# not version controlled. This file is, so a clone can resolve the eight\n"
            "# tokenizer labels without first re-running the stability analysis.\n"
        )
        (out_dir / "tokenizers.yaml").write_text(header + dump_yaml(tokenizers) + "\n")

    (out_dir / "build_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nwrote {relative(index)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
