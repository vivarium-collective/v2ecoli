"""Standalone post-hoc analysis for a Ray-dispatched simulation.

Two dispatch shapes reach this script, and it routes each analysis to whichever
of v2ecoli's two analysis-class families it belongs to (same
v2ecoli.workflow.analysis.ANALYSIS_REGISTRY for both -- distinguished by
``issubclass(cls, Analysis)``):

  * scripts/run_phase0_xarray_ensemble.py's single-generation dispatch writes
    only seed_NN/summary.json + seed_NN/store.zarr (no parquet). Its analyses
    are ``AnalysisStep`` subclasses (e.g. multiseed's doubling_time_distribution)
    -- this script builds row records straight from summary.json and calls
    ``cls({}, core=core).analyze(rows)`` (unchanged from before).
  * The multi-generation batch dispatch (BatchBaselineRunner, run through
    sms_api.compose.run_pbg's generic runner -- see viva-api backlog items
    26/27) uses emitter="both", which ALSO writes hive-parquet under the same
    S3 prefix. This unlocks the
    DuckDB-based ``Analysis`` family -- the cd1/ptools omics suite
    (ptools_rna, cd1_metabolomics, etc.), the actual original backlog target
    ("reproduce CD1 datasets"), never reachable via standalone analysis before.
    These are routed straight to v2ecoli.workflow.analysis_runner.run_analyses(),
    the SAME function the local batch_baseline flush path already uses
    end-to-end (sweep_dir may be an s3:// URI -- DuckDB reads the parquet in
    place via httpfs) -- no new analysis logic, just a real caller for GovCloud.

sms-api's own standalone-analysis endpoint (POST /simulations/{id}/analysis) was
built exclusively for a different, legacy simulation pipeline (vEcoli-private's
Nextflow-based parca/variant_sim_data/TSV output) and cannot read either of
these dispatch shapes' output at all -- this script is what actually runs.

The single-generation dispatch has no cell division, so `divided`/
`division_time` are always False/0.0 for THAT path's rows -- an honest
reflection of what it actually simulates, not a bug. The multi-generation path
has real division data (the batch dispatch's own real gap-2 fix).

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


def _aws_sync(src: str, dst: str) -> None:
    subprocess.run(
        ["aws", "s3", "sync", src, dst],
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


def run_duckdb_analyses(
    out_uri: str, scale: str, entries: dict[str, Any], tmp: Path, outdir: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Route DuckDB-based ``Analysis`` names (one scale's worth) straight to
    v2ecoli.workflow.analysis_runner.run_analyses() -- the same function the
    local batch_baseline flush path already runs end-to-end. ``out_uri`` is the
    sweep's S3 prefix; run_analyses reads its hive-parquet via DuckDB httpfs in
    place and writes analysis.json/viz/ptools locally, which we then sync up.
    """
    from v2ecoli.workflow import analysis_runner

    written: list[str] = []
    errors: list[dict[str, str]] = []
    local_out = tmp / "duckdb_out" / scale
    try:
        analysis_runner.run_analyses(
            sweep_dir=out_uri, analysis_options={scale: entries}, out_dir=str(local_out),
        )
    except Exception as e:  # noqa: BLE001 -- surface any analysis failure in the manifest
        errors.append({"scale": scale, "error": f"{type(e).__name__}: {e}"})
        return written, errors
    _aws_sync(str(local_out), outdir)
    # run_analyses() writes ONE analysis.json covering every name in `entries`
    # (plus viz/ptools per-group files) -- one manifest entry, not one per name.
    written.append(f"{outdir}/analysis.json")
    return written, errors


def run(out_uri: str, n_seeds: int, modules: dict[str, dict[str, Any]],
        analysis_name: str, tmp: Path) -> dict[str, Any]:
    from bigraph_schema import allocate_core

    import v2ecoli.workflow.analyses  # noqa: F401 -- populates ANALYSIS_REGISTRY
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis

    core = allocate_core()
    outdir = f"{out_uri.rstrip('/')}/analyses/{analysis_name}"
    written: list[str] = []
    errors: list[dict[str, str]] = []

    for scale, entries in modules.items():
        # Split this scale's requested names by which analysis-class family they
        # belong to -- the two families need entirely different inputs (row
        # records vs. a DuckDB connection over the sweep's parquet).
        duckdb_entries: dict[str, Any] = {}
        step_names: list[str] = []
        for name in entries:
            cls = ANALYSIS_REGISTRY.get(name)
            if cls is None:
                errors.append({"name": name, "error": f"unknown analysis {name!r}"})
                continue
            if cls.scale != scale:
                errors.append({"name": name, "error": f"{name} is scale={cls.scale!r}, not {scale!r}"})
                continue
            if issubclass(cls, Analysis):
                duckdb_entries[name] = entries[name]
            else:
                step_names.append(name)

        if duckdb_entries:
            w, e = run_duckdb_analyses(out_uri, scale, duckdb_entries, tmp, outdir)
            written.extend(w)
            errors.extend(e)

        if not step_names:
            continue
        builder = ROW_BUILDERS.get(scale)
        if builder is None:
            errors.append({"scale": scale, "error": f"no row builder for scale {scale!r}"})
            continue
        try:
            rows = builder(out_uri, n_seeds, tmp)
        except RuntimeError as e:
            errors.append({"scale": scale, "error": str(e)})
            continue
        for name in step_names:
            cls = ANALYSIS_REGISTRY[name]
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
