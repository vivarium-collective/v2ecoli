#!/usr/bin/env python3
"""v2e-compare — run a comparison study or investigation directly from YAML.

  v2e-compare study <name|path> [--ray] [--out DIR] [--render-only]
  v2e-compare run   [investigation|path] [--ray] [--out DIR] [--render-only]

The study / investigation YAML is the single source of truth (no manifest JSON).
`study` runs one condition (genuine process-wrapped vEcoli vs v2ecoli baseline,
matched-initial-state, for the study's seeds x generations) and materializes its
gating tests; `run` does the same for every study in an investigation. Sims run
serial+local by default; --ray (or V2E_MODE=ray) fans seeds/studies out for the
mini. --render-only reuses existing stores under --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts._compare import runner  # noqa: E402
from scripts._compare.materialize import materialize_study as _materialize  # noqa: E402

DEFAULT_INVEST = "whole-cell-model-comparison"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="v2e-compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a whole investigation from its YAML")
    pr.add_argument("investigation", nargs="?", default=DEFAULT_INVEST,
                    help="investigation name or path (default: %(default)s)")
    pr.add_argument("--out", default="out/report")
    pr.add_argument("--ray", action="store_true")
    pr.add_argument("--render-only", action="store_true")

    ps = sub.add_parser("study", help="run a single study from its YAML")
    ps.add_argument("name", help="study name or path")
    ps.add_argument("--out", default="out/report")
    ps.add_argument("--ray", action="store_true")
    ps.add_argument("--render-only", action="store_true")

    psc = sub.add_parser("scaffold", help="materialize an investigation's studies "
                                          "(declare report_card modules; no sims)")
    psc.add_argument("investigation", nargs="?", default=DEFAULT_INVEST)

    pi = sub.add_parser("init", help="scaffold a comparison investigation from a reference repo + configs")
    pi.add_argument("--reference", required=True, help="reference model repo path")
    pi.add_argument("--configs", required=True, help="comma-separated conditions/paths, or a dir of *.json")
    pi.add_argument("-o", "--name", default="whole-cell-model-comparison")
    pi.add_argument("--out-root", default="workspace/investigations")

    args = ap.parse_args(argv)

    if args.cmd == "scaffold":
        # Build specs from `comparison.configs[]` directly (same source
        # run/init use). load_investigation() now also supports configs[]
        # investigations (it delegates to specs_from_configs() internally
        # when `comparison.configs[]` is present), so this could call it
        # too -- kept explicit here since scaffold has no legacy-members:
        # use case to fall back to.
        from scripts._compare.study_spec import _context, _invest_dir, specs_from_configs
        ctx = _context(_invest_dir(args.investigation))
        specs = specs_from_configs(ctx)
        for spec in specs:
            _materialize(spec)
        print(f"scaffolded {len(specs)} studies in {args.investigation}")
        return 0

    if args.cmd == "init":
        from scripts._compare.scaffold import scaffold_investigation
        from scripts._compare.study_spec import _context, _invest_dir, specs_from_configs
        cfgs = ([str(p) for p in sorted(Path(args.configs).glob("*.json"))]
                if Path(args.configs).is_dir() else args.configs.split(","))
        p = scaffold_investigation(name=args.name, reference_repo=args.reference,
                                   configs=cfgs, out_root=args.out_root)
        # NOTE: a freshly scaffolded investigation only has a
        # `comparison.configs[]` block and no `members:` entry, so build
        # specs directly from the scaffolded configs, the same way
        # runner.run_investigation() (and load_investigation(), which now
        # also delegates to specs_from_configs() when configs[] is present)
        # does.
        ctx = _context(_invest_dir(str(p.parent)))
        specs = specs_from_configs(ctx)
        for spec in specs:
            # materialize_study() rewrites an EXISTING study.yaml's report_cards
            # + tests, preserving other keys -- it does not create the file. A
            # freshly scaffolded investigation has no per-study study.yaml yet
            # (specs_from_configs() points at the top-level registry
            # workspace/studies/<name>/study.yaml), so seed a minimal stub
            # before materializing it.
            study_path = Path(spec.study_path)
            if not study_path.exists():
                study_path.parent.mkdir(parents=True, exist_ok=True)
                stub = {"schema_version": 4, "name": spec.name,
                        "investigation": spec.invest_name, "condition": spec.condition,
                        "comparison": {"seeds": spec.seeds, "generations": spec.gens,
                                      "config": spec.config}}
                study_path.write_text(yaml.safe_dump(stub, sort_keys=False, allow_unicode=True))
            _materialize(spec)
        print(f"scaffolded {p} ({len(specs)} studies)")
        return 0

    mode = "ray" if args.ray else "serial"
    if args.cmd == "run":
        return runner.run_investigation(args.investigation, out=args.out, mode=mode,
                                        render_only=args.render_only)
    spec = runner.load_study(args.name)
    return runner.run_study(spec, out=args.out, mode=mode,
                            render_only=args.render_only)


if __name__ == "__main__":
    raise SystemExit(main())
