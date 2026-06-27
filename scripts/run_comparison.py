#!/usr/bin/env python3
"""Manifest-driven comparison runner — the single, AI-free entry point.

Reads a comparison manifest (the new schema: {v2ecoli/vecoli: repo+commit,
defaults.cards, configs:[{config, cards}]}), runs BOTH engines for each config
with the run shape (seeds/gens) read FROM that vEcoli config, writes both
engines' zarr to <out>/<condition>/, then renders the modular report (overview +
a section per config showing its assigned cards).

Usage:
  V2E_VECOLI_DIR=/path/to/vEcoli \\
  .venv/bin/python scripts/run_comparison.py comparison.5cond_1x4.json --out out/report

  # render only (skip sims; assemble the report from existing stores under --out):
  .venv/bin/python scripts/run_comparison.py comparison.5cond_1x4.json --out out/report --render-only

Per-engine wiring mirrors run_local_4x4x5.sh: v2ecoli runs on its own ParCa cache
with matched-initial-state against the upstream simData; genuine vEcoli runs on
the upstream ParCa cache via the vivarium-process engine. Both write
v2ecoli_seed*.zarr / vecoli_seed*.zarr into the SAME per-condition dir.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts._compare.config_adapter import (  # noqa: E402
    config_run_shape, resolve_vecoli_config_local)

PER_GEN_STEPS = 15000  # per-generation tick budget (non-binding cap; division-driven)


def condition_of(cfg_path: str, fork: str) -> str:
    """Condition name for a config: its `condition` field (config = source of
    truth) when the vEcoli fork is known, else the filename stem with a leading
    'cond_' and a trailing scale suffix (_1x4/_4x4) stripped."""
    if fork:
        try:
            cond = resolve_vecoli_config_local(cfg_path, fork).get("condition")
            if cond:
                return cond
        except Exception:  # noqa: BLE001
            pass
    stem = os.path.splitext(os.path.basename(cfg_path))[0]
    if stem.startswith("cond_"):
        stem = stem[len("cond_"):]
    return re.sub(r"_\d+x\d+$", "", stem)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="comparison manifest JSON")
    ap.add_argument("--out", default="out/report", help="output dir (per-condition stores + report)")
    ap.add_argument("--v2-cache", default="out/cache_full", help="v2ecoli ParCa cache")
    ap.add_argument("--ve-cache", default="out/compare_harness/vecoli_parca", help="upstream vEcoli ParCa cache")
    ap.add_argument("--mode", default="serial", help="run_comparison_ensemble --mode (serial|ray)")
    ap.add_argument("--render-only", action="store_true", help="skip sims; render from existing stores")
    args = ap.parse_args(argv)

    spec = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fork = os.environ.get("V2E_VECOLI_DIR", "")
    py = str(REPO / ".venv/bin/python")
    ref_sd = f"{args.ve_cache}/simData.cPickle"
    configs = spec.get("configs", [])
    if not configs:
        sys.exit(f"manifest {args.manifest} has no configs")

    max_seeds = 1
    for entry in configs:
        cfg = entry["config"]
        cond = condition_of(cfg, fork)
        seeds, gens = config_run_shape(cfg, fork) if fork else (1, 1)
        max_seeds = max(max_seeds, seeds)
        out_c = f"{args.out}/{cond}"
        if args.render_only:
            print(f"[render-only] {cond}: seeds={seeds} gens={gens}")
            continue
        print(f">>> {cond}: seeds={seeds} gens={gens} -> {out_c}", flush=True)
        cap = str(gens * PER_GEN_STEPS)
        # v2ecoli engine (its own ParCa cache + matched-initial-state)
        subprocess.run([py, "scripts/run_comparison_ensemble.py",
                        "--composite", "v2ecoli", "--condition", cond,
                        "--cache-dir", args.v2_cache, "--n-seeds", str(seeds),
                        "--max-generations", str(gens), "--max-steps", cap,
                        "--chunk", "60", "--mode", args.mode,
                        "--match-initial-state", "--match-vecoli-simdata", ref_sd,
                        "--out-root", out_c], cwd=REPO, check=True)
        # genuine vEcoli engine (upstream ParCa cache, vivarium-process)
        subprocess.run([py, "scripts/run_comparison_ensemble.py",
                        "--composite", "vecoli", "--condition", cond,
                        "--cache-dir", args.ve_cache, "--n-seeds", str(seeds),
                        "--max-generations", str(gens), "--max-steps",
                        str(PER_GEN_STEPS), "--chunk", "60", "--mode", args.mode,
                        "--vecoli-source", "vivarium-process",
                        "--out-root", out_c], cwd=REPO, check=True)

    # render the modular report (overview + per-config assigned cards)
    subprocess.run([py, "scripts/comparison_report_card.py",
                    "--manifest", args.manifest, "--out", args.out,
                    "--local-pbg-seeds", str(max_seeds)], cwd=REPO, check=True)
    print(f"\nreport: {args.out}/standardized_comparison_report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
