"""``build_parca_sim_composite`` + ``v2ecoli-workflow-run`` — the milestone
workflow: ParCa cache -> per-seed ``ecoli_baseline`` runs under
``CompositeTask`` -> a gating verdict, all driven by
``process_bigraph.workflow.run_workflow`` (``LocalRunner`` by default).

The composite this builds is a plain Step network (no time-scheduled
Processes), so ``LocalRunner.run()`` (``composite.run(0.0)``) cascades it to
completion in one call — see ``process_bigraph.workflow.backend`` module docs.

::

    parca (ParcaBundleStep)
      -> sim_data_ref  --[_ArtifactRefFile]--> sim_data_path (file)
                                                    |
    seeds (list) --------------------------------> sims (CompositeTask, scatter=seed)
                                                    |
                                              sims_results {seed: path}
                                                    |
                                        [_SeedResultPaths] -> result_paths (list)
                                                    |
                                        results (viva_superpowers.ResultsStep)
                                                    |
                                        report (SimGateTest) -> verdict
                                                    |
                                                 bridge

Import ``v2ecoli`` first (module docstring convention shared with
``v2ecoli.steps.parca_bundle``): a pre-existing ``process_bigraph/emitter.py``
circular import is avoided by importing v2ecoli before ``process_bigraph``
touches it.

The ParCa cache dir defaults to ``DEFAULT_CACHE_DIR`` (a canonical-checkout
path), overridable via the ``V2ECOLI_PARCA_CACHE_DIR`` env var or the CLI's
``--cache-dir`` -- portable across machines/CI where that literal path
doesn't exist.
"""
from __future__ import annotations

import v2ecoli  # noqa: F401 -- import first, see module docstring

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Iterable, Optional

from process_bigraph import Composite, allocate_core
from process_bigraph.composite import Step
from process_bigraph.workflow import run_workflow
from process_bigraph.workflow.tasks import CompositeTask

from v2ecoli.steps.parca_bundle import ParcaBundleStep
from viva_superpowers import TestStep, ResultsStep, TestBuilder, check, value

#: Absolute default ParCa cache directory (a pre-built bundle -- see
#: ``ParcaBundleStep``'s module docs). Never resolved relative to cwd: the
#: milestone integration test/CLI may run from a neutral directory.
#: Overridable via ``V2ECOLI_PARCA_CACHE_DIR`` (e.g. on another machine/CI).
#: The default is a portable per-user cache location (not a hardcoded
#: developer path -- issue #131), still absolute and cwd-independent.
DEFAULT_CACHE_DIR = os.environ.get(
    "V2ECOLI_PARCA_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "v2ecoli", "parca_cache"))


# ── small adapter Steps (glue between ParcaBundleStep / CompositeTask /
#    ResultsStep -- none of these three packages know about each other) ──

class _ArtifactRefFile(Step):
    """Write an :class:`~process_bigraph.artifacts.ArtifactRef` dict (as
    produced by ``ParcaBundleStep``) out to a JSON file, and hand back its
    path.

    ``CompositeTask``'s ``artifact_params`` ports are read as file paths
    (``_load_ref_hash`` / the ``--artifact PORT=REF.json`` subprocess flag
    both ``open()`` the value), whereas ``ParcaBundleStep`` emits the ref as
    a live dict (its own port is ``_is_file: True`` only as a Nextflow
    channel-declaration convention, not a runtime guarantee -- see
    ``process_bigraph.nextflow``). This Step is the one-line bridge.
    """

    config_schema = {"out_path": "string"}

    def inputs(self):
        return {"ref": "tree"}

    def outputs(self):
        return {"path": {"_type": "string", "_is_file": True}}

    def update(self, state):
        ref = state.get("ref") or {}
        out_path = self.config["out_path"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(ref, fh)
        return {"path": out_path}


class _SeedResultPaths(Step):
    """Adapt ``CompositeTask``'s ``{seed_key: result_path}`` scatter output
    into the flat ``paths`` list ``viva_superpowers.ResultsStep`` expects.
    """

    def inputs(self):
        return {"results": "tree"}

    def outputs(self):
        return {"paths": "tree"}

    def update(self, state):
        results = state.get("results") or {}
        paths = [results[key] for key in sorted(results)]
        return {"paths": paths}


class SimGateTest(TestStep):
    """Minimal gating Test over a real ``ResultsHandle``.

    Unlike ``viva_superpowers.TestStep``'s default ``StudyContext`` contract
    (``build(study)``), this test reads the run's emitted records directly off
    the ``ResultsHandle`` ``ResultsStep`` produced -- the real per-seed
    ``CompositeTask`` output, never a fixture. It grades the emitted-record
    count with ``check()`` and emits a ``report_card_verdict/v2`` doc: the
    ``overall`` verdict gates, and the ``emitted_records`` axis carries a signed
    ``margin`` (the agent-feedback gradient).
    """

    name = "sim_gate"

    def inputs(self):
        return {"results": "tree"}

    def update(self, state, interval=None):
        handle = state.get("results")
        rows = handle.records() if hasattr(handle, "records") else (handle or [])
        n = len(rows)
        axis = check(
            "emitted_records", "Emitted records", observed=n,
            expected=value(1, op=">="), severity="hard",
            detail=(f"{n} emitted rows across the run" if n
                    else "no emitted rows found -- sim(s) may have failed to emit"))
        verdict = TestBuilder(model_ref="sim_gate").add("Run", axis).build()
        status = verdict["overall"]
        html = (
            f"<html><body><h1>sim_gate: {status}</h1>"
            f"<p>{n} emitted rows across the run</p></body></html>")
        return {"view": html, "data": verdict}


def _outer_core():
    """A bare core with just the links this workflow's outer Step network
    needs. Deliberately NOT ``v2ecoli.core.build_core()``: the outer
    composite never runs ``ecoli_baseline`` in-process (``CompositeTask``
    shells out to ``run_composite --build``, which resolves its own core via
    the generator's ``core_extensions``) -- it only orchestrates.
    """
    core = allocate_core()
    core.register_link("ParcaBundleStep", ParcaBundleStep)
    core.register_link("ArtifactRefFile", _ArtifactRefFile)
    core.register_link("CompositeTask", CompositeTask)
    core.register_link("SeedResultPaths", _SeedResultPaths)
    core.register_link("ResultsStep", ResultsStep)
    core.register_link("SimGateTest", SimGateTest)
    return core


def build_parca_sim_composite(
    *,
    seeds: Iterable[int],
    parca_mode: str = "fixture",
    generator: str = "ecoli_baseline",
    overrides: Optional[dict] = None,
    steps: float = 2700,
    outdir: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Composite:
    """Build the milestone workflow composite (unbuilt/unrun).

    ``parca`` (a pre-built ParCa bundle -> a content-addressed ``sim_data``
    ref) feeds ``sims`` (one ``ecoli_baseline`` build per seed, scattered by
    ``CompositeTask``), whose per-seed results feed a real
    ``ResultsStep`` -> gating ``TestStep`` Evaluate tail. The bridge exposes
    ``verdict`` (the gate's graded ``report_card_verdict/v2`` doc: ``overall``
    + per-axis ``margin``) and, for inspection, the raw per-seed ``results``
    paths.
    """
    outdir = os.path.abspath(outdir)
    artifact_root = os.path.join(outdir, ".pbg", "artifacts")
    sim_data_ref_path = os.path.join(artifact_root, "sim_data_ref.json")

    state: dict[str, Any] = {
        "seeds": list(seeds),
        "sim_data_ref": {},
        "sim_data_path": "",
        "sims_results": {},
        "results_handle": {},
        "report_view": "",
        "simulation_id": "",
        "verdict": {},

        "parca": {
            "_type": "step",
            "address": "local:ParcaBundleStep",
            "config": {
                "mode": parca_mode,
                "cpus": 1,
                "condition": None,
                "bundle_dir": cache_dir,
            },
            "inputs": {},
            "outputs": {"sim_data": ["sim_data_ref"]},
        },
        "sim_data_file": {
            "_type": "step",
            "address": "local:ArtifactRefFile",
            "config": {"out_path": sim_data_ref_path},
            "inputs": {"ref": ["sim_data_ref"]},
            "outputs": {"path": ["sim_data_path"]},
        },
        "sims": {
            "_type": "step",
            "address": "local:CompositeTask",
            "config": {
                "generator": generator,
                "import": ["v2ecoli"],
                "overrides": dict(overrides or {}),
                "artifact_params": {"cache_dir": "sim_data"},
                "scatter_param": "seed",
                "steps": float(steps),
                "provision": [],
                "artifact_root": artifact_root,
                "allow_in_memory_emitter": False,
            },
            "inputs": {"seed": ["seeds"], "cache_dir": ["sim_data_path"]},
            "outputs": {"results": ["sims_results"]},
        },
        "result_paths_step": {
            "_type": "step",
            "address": "local:SeedResultPaths",
            "config": {},
            "inputs": {"results": ["sims_results"]},
            "outputs": {"paths": ["result_paths"]},
        },
        "results": {
            "_type": "step",
            "address": "local:ResultsStep",
            "config": {},
            "inputs": {
                "paths": ["result_paths"],
                "sim_data_ref": ["sim_data_ref"],
                "simulation_id": ["simulation_id"],
            },
            "outputs": {"results": ["results_handle"]},
        },
        "report": {
            "_type": "step",
            "address": "local:SimGateTest",
            "config": {},
            "inputs": {"results": ["results_handle"]},
            "outputs": {"view": ["report_view"], "data": ["verdict"]},
        },
    }

    document = {
        "state": state,
        "bridge": {"outputs": {
            "verdict": ["verdict"],
            "results": ["sims_results"],
        }},
        "parallel_steps": True,
    }
    return Composite(document, core=_outer_core())


def _git_sha(repo_dir: Optional[str] = None) -> Optional[str]:
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        proc = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="v2ecoli-workflow-run",
        description=(
            "Run the ParCa -> per-seed ecoli_baseline -> gating verdict "
            "milestone workflow under a process_bigraph workflow backend."))
    p.add_argument("--seeds", type=int, required=True,
                   help="Number of seeds to scatter over (seeds 0..N-1).")
    p.add_argument("--parca-mode", default="fixture")
    p.add_argument("--steps", type=float, default=2700)
    p.add_argument("--backend", default="local")
    p.add_argument("--outdir", required=True)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--generator", default="ecoli_baseline")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    composite = build_parca_sim_composite(
        seeds=range(args.seeds),
        parca_mode=args.parca_mode,
        generator=args.generator,
        steps=args.steps,
        outdir=args.outdir,
        cache_dir=args.cache_dir,
    )

    result = run_workflow(
        composite, backend=args.backend, outdir=args.outdir,
        code_version={"v2ecoli": _git_sha()})

    print(result.outputs.get("verdict"))
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
