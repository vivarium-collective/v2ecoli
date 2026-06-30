"""Driver + CLI for v2ecoli workflow sweeps.

Loads a vEcoli-style JSON config, builds the meta-composite, and ticks it
until every branch reports ``complete`` (or a global sim-time cap is hit).
Saves the sweep document as a .pbg for inspection.

    v2ecoli-workflow --config configs/two_generations.json --out out/two_gen
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from process_bigraph import Composite

from v2ecoli.core import build_core
from v2ecoli.workflow.config import load_config_with_inheritance
from v2ecoli.workflow.meta_composite import (
    build_meta_composite, register_workflow_processes)
from v2ecoli.workflow.variants import expand_branches


def _maybe_flush(config: dict, out_dir: str, result: dict) -> dict:
    """Run the post-sim flush. Never raises. Runs when an owning study is
    resolvable OR when analysis_options are present (ad-hoc analyses place into
    out_dir/viz)."""
    import os
    from v2ecoli.workflow.flush import resolve_owning_study, run_flush
    try:
        ws_root = config.get("ws_root") or os.getcwd()
        has_analyses = any((config.get("analysis_options") or {}).values())
        if resolve_owning_study(out_dir, config, ws_root) is None and not has_analyses:
            return result
        result["flush"] = run_flush(out_dir, config, ws_root)
    except Exception as e:  # noqa: BLE001 — flush failures must not fail the run
        result["flush"] = {"placed": [], "skipped": [], "error": f"{type(e).__name__}: {e}"}
    return result


def _all_complete(composite) -> bool:
    branches = composite.state.get("branches", {})
    return bool(branches) and all(
        b.get("complete") for b in branches.values())


def _resolve_parallel(config: dict[str, Any], n_branches: int) -> str | None:
    """Resolve the multi-seed parallel mode.

    DEFAULT: parallelize across seeds (one worker per seed) whenever a sweep
    has more than one branch — ``run_seeds_parallel`` falls back to sequential
    automatically if Ray is not installed, so the default is always safe. Set
    ``config['parallel']`` to ``null`` / ``false`` / ``"sequential"`` to force
    the single-process meta-composite, or to ``"ray"`` to be explicit.
    """
    if "parallel" in config and config["parallel"] not in ("ray", "default"):
        p = config["parallel"]
        if p in (None, False, "", "none", "null", "sequential", "off"):
            return None
        return str(p)
    return "ray" if n_branches > 1 else None


def _run_seed_worker(seed, *, config, max_sim_time):
    """Top-level Ray worker: run ONE seed's lineage to completion, emitting to
    the sweep's shared hive ``out_dir`` (distinct ``seed=`` partition per
    worker, so workers never collide). Returns that seed's branch summary.

    Must stay module-level + rely only on its args (Ray re-imports the module
    in a fresh process and serializes ``config`` per call).
    """
    sub = dict(config)
    sub["n_init_sims"] = 1
    sub["lineage_seed"] = int(seed)
    sub["skip_baseline"] = False
    sub["parallel"] = None                 # the worker itself is sequential
    sub["analysis_options"] = {}           # analyses run once, on the driver
    res = _run_sweep_sequential(
        sub, max_sim_time=max_sim_time, pbg_out=None,
        write_summary=False, run_analysis=False)
    return res["branches"]


def run_workflow(config: dict[str, Any], *, max_sim_time: float = 1e9,
                 pbg_out: str | None = None) -> dict[str, Any]:
    """Build and run the sweep for ``config``. Returns a result dict.

    Multi-seed sweeps parallelize across seeds BY DEFAULT (one worker process
    per seed via :func:`run_seeds_parallel`), each worker emitting to the
    sweep's shared hive ``out_dir``; analyses then run once on the driver over
    the whole sweep. Single-branch sweeps, variant sweeps, and
    ``config['parallel'] in (null/false/'sequential')`` use the single-process
    meta-composite. See :func:`_resolve_parallel`.

    Keys in the returned dict:
      ``complete``   – True if every branch finished before the sim-time cap.
      ``elapsed``    – sim-time (sequential) or wall-time (parallel) consumed.
      ``timed_out``  – True if the cap was hit before all branches completed.
      ``branches``   – per-branch summary dicts.
    """
    branches = expand_branches(config)
    mode = _resolve_parallel(config, len(branches))
    # Only seed-split when branches differ ONLY by seed (no variant grid); a
    # variant sweep keeps the single-process meta-composite for now.
    has_variants = bool(config.get("variants"))
    if mode and len(branches) > 1 and not has_variants:
        return _run_sweep_parallel(config, branches, mode,
                                   max_sim_time=max_sim_time)
    return _run_sweep_sequential(config, max_sim_time=max_sim_time,
                                 pbg_out=pbg_out)


def _run_sweep_parallel(config: dict[str, Any], branches, mode: str, *,
                        max_sim_time: float) -> dict[str, Any]:
    """Fan the sweep's seeds out across worker processes, then aggregate +
    write ``summary.json`` and run analyses once over the shared ``out_dir``."""
    from v2ecoli.library.parallel_seeds import run_seeds_parallel

    seeds = [spec.seed for spec in branches]
    pr = run_seeds_parallel(
        seeds, _run_seed_worker, mode=mode,
        num_threads=config.get("parallel_threads"),
        run_kwargs={"config": dict(config), "max_sim_time": max_sim_time})

    branch_result: dict[str, dict] = {}
    for r in pr.results:
        if r:
            branch_result.update(r)
    complete = bool(branch_result) and all(
        v.get("complete") for v in branch_result.values())

    out_dir = config.get("out_dir") or "out/workflow"
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({k: rv.get("summary") or {} for k, rv in branch_result.items()},
                  f, indent=2, default=str)

    result = {
        "complete": complete,
        "elapsed": pr.wall_s,
        "timed_out": not complete,
        "branches": branch_result,
        "parallel": {"mode": pr.mode, "n_workers": pr.n_seeds,
                     "wall_s": pr.wall_s},
    }
    analysis_options = config.get("analysis_options") or {}
    if any((analysis_options or {}).values()):
        # analyses are now driven by the post-sim flush (run_flush's analysis
        # kind), which runs run_analyses once and places outputs into the study
        # report dir. We only record where analysis.json will be written.
        result["analysis"] = os.path.join(out_dir, "analysis.json")
    result = _maybe_flush(config, out_dir, result)
    return result


def _run_sweep_sequential(config: dict[str, Any], *, max_sim_time: float = 1e9,
                          pbg_out: str | None = None,
                          write_summary: bool = True,
                          run_analysis: bool = True) -> dict[str, Any]:
    """Single-process meta-composite run (all branches ticked together).

    ``write_summary`` / ``run_analysis`` are turned off when this is invoked as
    a per-seed worker (the driver writes the combined summary + runs analyses
    once over the whole sweep)."""
    core = build_core()
    register_workflow_processes(core)

    doc = build_meta_composite(config)
    composite = Composite(doc, core=core)

    dt = float(config.get("time_step", 1.0))
    elapsed = 0.0
    while not _all_complete(composite) and elapsed < max_sim_time:
        composite.run(dt)
        elapsed += dt

    if pbg_out:
        from v2ecoli.pbg import save_pbg_doc
        save_pbg_doc(composite.state, pbg_out)

    branches = composite.state.get("branches", {})
    complete = _all_complete(composite)

    # The "summary" output (a "map" type) doesn't persist back to composite
    # state via the standard update path — extract summaries directly from
    # the live LineageProcess instances instead.
    proc_summaries: dict[str, dict] = {}
    for path_tuple, edge in composite.process_paths.items():
        inst = edge.get("instance")
        if inst is not None and hasattr(inst, "_summaries"):
            # path_tuple is e.g. ('branches', 'variant=0/seed=0', 'lineage')
            if len(path_tuple) >= 2 and path_tuple[0] == "branches":
                branch_key = path_tuple[1]
                proc_summaries[branch_key] = {"generations": list(inst._summaries)}

    branch_result = {
        k: {
            "complete": v.get("complete"),
            "summary": proc_summaries.get(k, v.get("summary") or {}),
        }
        for k, v in branches.items()
    }

    out_dir = config.get("out_dir") or "out/workflow"
    os.makedirs(out_dir, exist_ok=True)
    if write_summary:
        import json
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump({k: rv["summary"] for k, rv in branch_result.items()},
                      f, indent=2, default=str)

    result = {
        "complete": complete,
        "elapsed": elapsed,
        "timed_out": not complete,
        "branches": branch_result,
    }

    analysis_options = config.get("analysis_options") or {}
    if run_analysis and any((analysis_options or {}).values()):
        # analyses are now driven by the post-sim flush (run_flush's analysis
        # kind), which runs run_analyses once and places outputs into the study
        # report dir. We only record where analysis.json will be written.
        result["analysis"] = os.path.join(out_dir, "analysis.json")
    result = _maybe_flush(config, out_dir, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a workflow config JSON.")
    parser.add_argument("--out", default=None,
                        help="Output dir for parquet + sweep.pbg (default: out/<experiment_id>).")
    parser.add_argument("--build-only", action="store_true",
                        help="Build + save the .pbg without running.")
    parser.add_argument("--max-sim-time", type=float, default=1e9,
                        help="Global SIM-TIME safety cap (seconds), not wall-clock.")
    args = parser.parse_args()

    config = load_config_with_inheritance(args.config)
    exp = config.get("experiment_id") or "workflow"
    out_dir = args.out or os.path.join("out", exp)
    config.setdefault("out_dir", os.path.join(out_dir, "parquet"))
    os.makedirs(out_dir, exist_ok=True)
    pbg_out = os.path.join(out_dir, "sweep.pbg")

    if args.build_only:
        core = build_core()
        register_workflow_processes(core)
        composite = Composite(build_meta_composite(config), core=core)
        from v2ecoli.pbg import save_pbg_doc
        save_pbg_doc(composite.state, pbg_out)
        print(f"Built {len(composite.state['branches'])} branches -> {pbg_out}")
        return

    result = run_workflow(config, max_sim_time=args.max_sim_time, pbg_out=pbg_out)
    n = len(result["branches"])
    done = sum(1 for b in result["branches"].values() if b["complete"])
    if not result["complete"]:
        import warnings
        warnings.warn(
            f"Sweep '{exp}' hit the sim-time cap before all branches completed "
            f"({done}/{n} done, {result['elapsed']:.0f} s sim).")
    print(f"Sweep '{exp}': {done}/{n} branches complete in {result['elapsed']:.0f} s sim.")
    print(f"Saved: {pbg_out}")


if __name__ == "__main__":
    main()
