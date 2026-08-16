# Where things moved

The thesis and the earlier documentation name paths from before the repository was
reorganised. This table maps each to its current location. Nothing was renamed for its own
sake: script basenames are unchanged, and so are the names of the directories holding
checkpoints and results, because configs and evaluation records refer to them.

| Named as | Now at |
| --- | --- |
| `tokenization/analyze_*.py` | `tokenization/analysis/analyze_*.py` |
| `tokenization/visualize_*.py` | `tokenization/analysis/visualize_*.py` |
| `tokenization/dump_*.py` | `tokenization/analysis/dump_*.py` |
| `tokenization/render_token_viz.py` | `tokenization/analysis/render_token_viz.py` |
| `tokenization/calculate_codebook_utilization.py` | `tokenization/analysis/calculate_codebook_utilization.py` |
| `tokenization/interactive_token_viewer.py` | `tokenization/analysis/interactive_token_viewer.py` |
| `tokenization/train_poseVQ.py` | unchanged |
| `tokenization/docs/` | `docs/design/` |
| `tokenization/scripts/` | `tools/prepare/` |
| `thesis_figures/plot_*.py` | `thesis_figures/plot/plot_*.py` |
| `thesis_figures/make_decode_figure.py` | `thesis_figures/plot/make_decode_figure.py` |
| `thesis_figures/render_*.py` | `thesis_figures/render/render_*.py` |
| `thesis_figures/gen_skeleton_mask.py` | `thesis_figures/render/gen_skeleton_mask.py` |
| `thesis_figures/scan_*.py` | `thesis_figures/scan/scan_*.py` |
| `thesis_figures/skeleton_mask_tikz.tex` | `thesis_figures/render/skeleton_mask_tikz.tex` |
| `run_additive_queue.sh` and the other queues | `tools/queues/` |
| `tokenhmr/setup.py` | deleted, replaced by `pyproject.toml` at the root |
| `requirements.txt`, `stable_thesis_requirements.txt` (parent repo) | `env/` |
| `run_tokenhmr_demo.py`, `run_tokenhmr_track.py` (parent repo) | `tools/` |
| `frankenstein_test_original_tokenhmr.py` (parent repo) | `tools/tokenizer_roundtrip_check.py` |

Unchanged, and deliberately so: `logs/`, `results/`, `tokenization/output/`,
`dataset_dir/`, `data/`, and the timestamped `<config>_ID00_<timestamp>/` directories
inside `tokenization/output/`. Twenty-one experiment configs, eighty evaluation records
and fifty-four saved run configs refer to those names, and the timestamp is what
distinguishes two trainings of the same configuration.

## Invocation also changed

Scripts are modules now, so they run from any directory:

| Before | Now |
| --- | --- |
| `cd tokenization && python analyze_token_stability.py` | `python -m tokenization.analysis.analyze_token_stability` |
| `python thesis_figures/plot_recon_frontier.py` | `thesis figure recon-frontier` |
| `~/miniconda3/envs/thesis-HMR/bin/python …` | `python …`, with the environment activated |

`thesis paths` prints every location the project resolves.
