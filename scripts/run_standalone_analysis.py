"""Standalone post-hoc analysis for a Ray/xarray-dispatched simulation.

sms-api's default GovCloud simulation dispatch (scripts/run_phase0_xarray_ensemble.py)
writes one seed_NN/summary.json + seed_NN/store.zarr per seed under an S3 experiment
prefix. sms-api's own standalone-analysis endpoint (POST /simulations/{id}/analysis)
was built exclusively for a different, legacy simulation pipeline (vEcoli-private's
Nextflow-based parca/variant_sim_data/TSV output) and cannot read this dispatch's
output at all. This script builds the row records vEcoli's ported multiseed
AnalysisStep classes expect (v2ecoli.workflow.analysis.ANALYSIS_REGISTRY) directly
from this dispatch's own summary.json files, runs the requested module(s), and
writes results back to S3 -- independent of the LineageProcess/batch_baseline_runner
pipeline (which already supports analyses end-to-end but isn't what this dispatch
path invokes).

This dispatch script runs a single generation per seed with no cell division, so
`divided`/`division_time` are always False/0.0 here -- an honest reflection of what
it actually simulates, not a stand-in for multi-generation lineage data.

Usage:
    python scripts/run_standalone_analysis.py \
        --out-uri s3://bucket/vecoli-output/<experiment_id> \
        --n-seeds 2 \
        --modules '{"multiseed": {"doubling_time_distribution": {}}}' \
        --analysis-name my-analysis
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _aws_cp(src: str, dst: str) -> None:
    subprocess.run(
        ["aws", "s3", "cp", src, dst],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def build_multiseed_rows(out_uri: str, n_seeds: int, tmp: Path) -> list[dict[str, Any]]:
    """One row per seed, read from that seed's own summary.json."""
    rows = []
    for seed in range(n_seeds):
        local = tmp / f"seed_{seed:02d}_summary.json"
        try:
            _aws_cp(f"{out_uri.rstrip('/')}/seed_{seed:02d}/summary.json", str(local))
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode()[:200] if e.stderr else str(e)
            raise RuntimeError(f"seed {seed}: no summary.json under {out_uri} -- {stderr}") from e
        summary = json.loads(local.read_text())
        rows.append({
            "variant": 0,
            "lineage_seed": summary.get("seed", seed),
            "generation": 1,
            "agent_id": "0",
            "divided": False,
            "division_time": 0.0,
            "final_dry_mass": float(summary.get("dry_mass_fg", 0.0)),
        })
    return rows


ROW_BUILDERS: dict[str, Callable[[str, int, Path], list[dict[str, Any]]]] = {
    "multiseed": build_multiseed_rows,
}


def run(out_uri: str, n_seeds: int, modules: dict[str, dict[str, Any]],
        analysis_name: str, tmp: Path) -> dict[str, Any]:
    from bigraph_schema import allocate_core

    import v2ecoli.workflow.analyses  # noqa: F401 -- populates ANALYSIS_REGISTRY
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY

    core = allocate_core()
    outdir = f"{out_uri.rstrip('/')}/analyses/{analysis_name}"
    written: list[str] = []
    errors: list[dict[str, str]] = []

    for scale, entries in modules.items():
        builder = ROW_BUILDERS.get(scale)
        if builder is None:
            errors.append({"scale": scale, "error": f"no row builder for scale {scale!r}"})
            continue
        try:
            rows = builder(out_uri, n_seeds, tmp)
        except RuntimeError as e:
            errors.append({"scale": scale, "error": str(e)})
            continue
        for name in entries:
            cls = ANALYSIS_REGISTRY.get(name)
            if cls is None:
                errors.append({"name": name, "error": f"unknown analysis {name!r}"})
                continue
            if cls.scale != scale:
                errors.append({"name": name, "error": f"{name} is scale={cls.scale!r}, not {scale!r}"})
                continue
            try:
                result = cls({}, core=core).analyze(rows)
            except Exception as e:  # noqa: BLE001 -- surface any analysis failure in the manifest
                errors.append({"name": name, "error": f"{type(e).__name__}: {e}"})
                continue
            local_out = tmp / f"{name}.json"
            local_out.write_text(json.dumps(result, indent=2))
            dest = f"{outdir}/{name}.json"
            _aws_cp(str(local_out), dest)
            written.append(dest)

    status = "done" if written and not errors else ("failed" if errors and not written else "partial")
    manifest = {
        "analysis_name": analysis_name, "modules": modules,
        "written": written, "errors": errors, "status": status,
    }
    manifest_local = tmp / "_manifest.json"
    manifest_local.write_text(json.dumps(manifest, indent=2))
    _aws_cp(str(manifest_local), f"{outdir}/_manifest.json")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", help="JSON file with out_uri/n_seeds/modules/analysis_name "
                                          "-- avoids shell-quoting the modules JSON on the CLI")
    p.add_argument("--out-uri", help="s3:// experiment output prefix")
    p.add_argument("--n-seeds", type=int)
    p.add_argument("--modules", help='JSON: {"scale": {"name": {}}}')
    p.add_argument("--analysis-name")
    args = p.parse_args()

    if args.config_file:
        cfg = json.loads(Path(args.config_file).read_text())
        out_uri, n_seeds = cfg["out_uri"], int(cfg["n_seeds"])
        modules, analysis_name = cfg["modules"], cfg["analysis_name"]
    else:
        missing = [f for f in ("out_uri", "n_seeds", "modules", "analysis_name")
                   if getattr(args, f) is None]
        if missing:
            p.error(f"missing required arguments (or use --config-file): {missing}")
        out_uri, n_seeds = args.out_uri, args.n_seeds
        modules, analysis_name = json.loads(args.modules), args.analysis_name

    with tempfile.TemporaryDirectory() as td:
        manifest = run(out_uri, n_seeds, modules, analysis_name, Path(td))
    print(json.dumps(manifest, indent=2))
    if manifest["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
