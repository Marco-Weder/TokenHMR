# Provenance

The manifest answers two questions that were previously answerable only by
reading 81 CSV files: **which checkpoint produced this number**, and **does the
code still produce the number the thesis prints**.

## The three files

| File | Written by | Regenerable |
| --- | --- | --- |
| `manifest/eval_index.csv` | `thesis manifest build` | yes, deterministically |
| `manifest/tokenizers.yaml` | `thesis manifest build` | yes |
| `manifest/thesis_numbers.yaml` | by hand, from the thesis | **no** |

The split matters. The first two are derived from results on disk and can be
rebuilt at any time. The third is the published record, typed from the tables in
the document. Deriving it from the CSVs would make the two agree by
construction, which is exactly what it exists to test.

## eval_index.csv

One row per (run, dataset, metric), with the checkpoint that produced it, the
Hydra experiment config the run used, a checkpoint fingerprint, and the source
file. Built from `results/**/eval_regression.csv` without re-running anything.

Three details decide whether it is correct:

- **The source CSVs are append-only.** They are written under a file lock, so
  re-evaluating a checkpoint stacks another set of rows on top of the first.
  Reading one naively returns whichever row happens to come first, which may be
  months stale. Rows are deduplicated by timestamp, and superseded ones are kept
  with `superseded=1` rather than dropped, because an overwritten measurement is
  still evidence of what was measured when.
- **The experiment config is recovered from `<run_dir>/.hydra/overrides.yaml`.**
  Nothing else records which configuration produced a given run.
- **Runs with no evaluation are named, not skipped.** Silently dropping them
  would turn an unevaluated run into an apparently complete record.

## tokenizers.yaml

The eight stage-1 tokenizers: quantizer, code dimension, codebook size,
checkpoint, reconstruction error, label stability and redundancy.

This file replaces an ordering dependency that used to be undocumented. Five
analysis scripts read their checkpoint paths out of
`tokenization/output/token_stability/summary.json`, which is not version
controlled and only exists after `analyze_token_stability.py` has run. The same
information is now in a tracked file, so a clone can resolve a tokenizer label
before running anything.

Each tokenizer also gets a stable alias under
`tokenization/output/tokenizers/<name>/best_net.pth`. Create them with:

```bash
thesis manifest link-tokenizers
```

The real checkpoints stay in their timestamped directories. That timestamp is
what distinguishes two trainings of the same configuration, so it is never
flattened away. One of them matters: the experiment config named
`tokenhmr_transformer_tier1` points at the **cosine** d256 tokenizer, not the
l2 one its name suggests. The aliases are keyed to tokenizer identity, so this
is visible rather than implied.

## thesis_numbers.yaml and the check

Every value the thesis prints, with the run, dataset and metric it came from,
and a tolerance of half the last printed digit.

`repro.manifest.value()` reads a number from `eval_index.csv` and raises if it
disagrees with the published value. Figure scripts call it instead of carrying
hard-coded numbers, so a figure regenerated after a code change either matches
the thesis or fails loudly enough to notice.

```bash
thesis manifest check      # all published values against all measurements
```

If it fails, the thesis is the record of what was published and the manifest is
the record of what was measured. Find out which is wrong before changing either.

## The golden snapshot

`manifest/golden/` holds numeric digests of the deterministic CPU analysis
outputs, recorded before the repository was reorganised.

```bash
thesis golden check
```

compares the current outputs value by value. It covers 2472 numbers across the
codebook geometry, per-dataset reconstruction and token stability analyses.

Renders are deliberately excluded. Rasterisation is not bit-reproducible across
driver versions, so hashing a rendered image produces false alarms rather than
evidence.

## Rebuilding

```bash
thesis manifest build     # from results already on disk, nothing is re-run
```

Safe at any time. It never writes to `results/`, `logs/` or
`tokenization/output/`.
