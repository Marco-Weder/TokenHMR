"""The `thesis` command: check an install, inspect provenance, run a figure.

Deliberately small. Everything that produces a result is still run as
``python -m <module>``, so the commands in the documentation name the same
scripts the thesis appendix does. This exists for the things that are awkward
to express that way: telling a reader whether their environment is usable, and
answering "which checkpoint produced this number".
"""

from __future__ import annotations

import argparse
import importlib
import runpy
import shutil
import subprocess
import sys

from repro import paths


def cmd_paths(args) -> int:
    print(f"project root  {paths.PROJECT_ROOT}\n")
    print(paths.describe())
    print("\nOverride any of these with the matching THESIS_* environment variable.")
    print(f"Figures would be written to: {paths.figure_out_dir('<section>')}")
    return 0


def _check_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report, never raise
        return False, f"{type(exc).__name__}: {exc}"
    return True, getattr(mod, "__version__", "")


def cmd_doctor(args) -> int:
    problems = 0

    print("interpreter")
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info[:2] == (3, 10)
    print(f"  {'ok     ' if ok else 'WARNING'}  python {py} (3.10 expected)")
    problems += 0 if ok else 1

    print("\npackages")
    required = ["torch", "pytorch_lightning", "hydra", "yacs", "smplx", "numpy", "matplotlib"]
    optional = {
        "pyrender": "mesh renders (figures)",
        "bpy": "Blender renders (Figures 3.1, 4.6, 4.7)",
        "detectron2": "person detection (demo, video comparison)",
        "phalp": "PHALP tracking demo only",
        "trimesh": "mesh handling",
        "wandb": "training logs",
    }
    for name in required:
        ok, detail = _check_import(name)
        print(f"  {'ok     ' if ok else 'MISSING'}  {name:20s} {detail}")
        problems += 0 if ok else 1
    for name, why in optional.items():
        ok, detail = _check_import(name)
        print(f"  {'ok     ' if ok else 'absent '}  {name:20s} {detail or ''}  ({why})")

    print("\nthis project")
    for name in ["repro.manifest", "tokenization.analysis.analyze_latent_pose_info",
                 "tokenhmr.lib.configs", "thesis_figures.plot"]:
        ok, _ = _check_import(name)
        print(f"  {'ok     ' if ok else 'FAILED '}  {name}")
        problems += 0 if ok else 1

    print("\ngpu")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ok       {torch.cuda.get_device_name(0)}")
        else:
            print("  absent   no CUDA device; analyses run on CPU, renders and training will not")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED   {exc}")
        problems += 1

    print("\nexternal tools")
    for tool, why in [("ffmpeg", "video comparison"), ("blender", "mesh figures")]:
        found = shutil.which(tool)
        print(f"  {'ok     ' if found else 'absent '}  {tool:20s} {found or ''}  ({why})")

    print("\ndata")
    print(paths.describe())

    print("\nprovenance")
    try:
        from repro import manifest

        n = len(manifest.runs())
        failures = manifest.check_published()
        print(f"  ok       manifest holds {n} runs and {len(manifest.tokenizers())} tokenizers")
        if failures:
            print(f"  FAILED   {len(failures)} published values disagree with the manifest")
            for f in failures[:5]:
                print(f"             {f}")
            problems += 1
        else:
            print("  ok       every published value agrees with the manifest")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED   {exc}")
        problems += 1

    print(f"\n{'no problems found' if not problems else f'{problems} problem(s) found'}")
    return 1 if problems else 0


def cmd_manifest(args) -> int:
    from repro import build_manifest, manifest

    if args.action == "build":
        return build_manifest.main([])
    if args.action == "check":
        failures = manifest.check_published()
        if failures:
            print(f"{len(failures)} disagreement(s) between the thesis and the manifest:")
            for f in failures:
                print(f"  {f}")
            return 1
        print(f"all {len(manifest._published())} published values agree with the manifest")
        return 0
    if args.action == "tokenizers":
        for t in manifest.tokenizers():
            print(f"  {t['label']:20s} {t['slug']:30s} recon {t['recon_mpjpe_mm']:>5.2f} mm  "
                  f"S(1deg) {t['stability_pct']['1']:>4.0f}%")
        return 0
    if args.action == "runs":
        for r in manifest.runs():
            print(" ", r)
        return 0
    if args.action == "link-tokenizers":
        for line in manifest.link_tokenizers(dry_run=args.dry_run):
            print(" ", line)
        return 0
    return 2


def cmd_golden(args) -> int:
    from repro import golden

    return golden.main([args.action])


FIGURES = {
    "recon-frontier": "thesis_figures.plot.plot_recon_frontier",
    "codebook-geometry": "thesis_figures.plot.plot_codebook_geometry",
    "code-separation": "thesis_figures.plot.plot_code_separation",
    "influence-maps": "thesis_figures.plot.plot_influence_maps",
    "latent-scatter": "thesis_figures.plot.plot_latent_scatter",
    "token-map-page": "thesis_figures.plot.plot_token_map_page",
    "token-ambiguity": "thesis_figures.plot.plot_token_ambiguity",
    "additive-chain": "thesis_figures.plot.plot_additive_chain",
    "decoding": "thesis_figures.plot.plot_decoding_figure",
    "smpl": "thesis_figures.render.render_smpl_figure",
    "skeleton-mask": "thesis_figures.render.gen_skeleton_mask",
    "token-viz": "thesis_figures.render.render_token_viz_blender",
    "qualitative": "thesis_figures.render.render_qualitative",
    "token-benefit": "thesis_figures.render.render_token_benefit",
}


def cmd_figure(args) -> int:
    if args.name not in FIGURES:
        print("unknown figure. Available:")
        for key, module in sorted(FIGURES.items()):
            print(f"  {key:20s} {module}")
        return 2
    module = FIGURES[args.name]
    sys.argv = [module] + args.rest
    runpy.run_module(module, run_name="__main__")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="thesis", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("paths", help="print every resolved path and whether it exists").set_defaults(
        func=cmd_paths
    )
    sub.add_parser("doctor", help="check the environment, data and provenance").set_defaults(
        func=cmd_doctor
    )

    m = sub.add_parser("manifest", help="inspect or rebuild the provenance manifest")
    m.add_argument("action", choices=["build", "check", "tokenizers", "runs", "link-tokenizers"])
    m.add_argument("--dry-run", action="store_true")
    m.set_defaults(func=cmd_manifest)

    g = sub.add_parser("golden", help="snapshot or verify the deterministic analysis outputs")
    g.add_argument("action", choices=["record", "check"])
    g.set_defaults(func=cmd_golden)

    f = sub.add_parser("figure", help="render one thesis figure")
    f.add_argument("name", nargs="?", default="")
    f.add_argument("rest", nargs=argparse.REMAINDER)
    f.set_defaults(func=cmd_figure)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
