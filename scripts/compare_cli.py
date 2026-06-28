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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts._compare import runner  # noqa: E402

DEFAULT_INVEST = "v2ecoli-vecoli-comparison"


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

    args = ap.parse_args(argv)
    mode = "ray" if args.ray else "serial"
    if args.cmd == "run":
        return runner.run_investigation(args.investigation, out=args.out, mode=mode,
                                        render_only=args.render_only)
    spec = runner.load_study(args.name)
    return runner.run_study(spec, out=args.out, mode=mode,
                            render_only=args.render_only)


if __name__ == "__main__":
    raise SystemExit(main())
