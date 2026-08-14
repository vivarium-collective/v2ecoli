"""Tests for ``build_parca_sim_composite`` / ``v2ecoli-workflow-run`` -- the
MILESTONE workflow: a pre-built ParCa cache -> per-seed ``ecoli_baseline``
runs under ``CompositeTask`` -> a real ``ResultsStep`` -> gating
``ReportCardStep`` Evaluate tail, all driven through
``process_bigraph.workflow.run_workflow`` (``LocalRunner``).

(a) is a pure unit test over the built (unrun) ``Composite`` document -- no
cache needed. (b)-(d) actually RUN the workflow end to end against the real
ParCa cache at ``v2ecoli.workflow.build.DEFAULT_CACHE_DIR`` and are skipped
if that cache isn't present locally.
"""
from __future__ import annotations

import v2ecoli  # noqa: F401 -- import first, see v2ecoli.workflow.build docstring

import json
import os
from unittest import mock

import pytest

from v2ecoli.workflow.build import DEFAULT_CACHE_DIR, build_parca_sim_composite, main


pytestmark = pytest.mark.fast

_HAS_CACHE = os.path.isdir(DEFAULT_CACHE_DIR)
_skip_no_cache = pytest.mark.skipif(
    not _HAS_CACHE, reason=f"no ParCa cache at {DEFAULT_CACHE_DIR}")

# NOTE: an earlier revision of this suite xfailed the four run-based tests
# below over a `--state-out` serialization crash inside the installed
# `bigraph_schema==1.6.0` (v2ecoli's string-_data LabeledArray types --
# monomer_counts_vec/rna_init_event_per_cistron_vec -- couldn't render).
# Fixed upstream: process-bigraph commit d051665 makes `run_composite
# --state-out` best-effort (writes a marker document + warns instead of
# raising on a serialization failure), so the sim subprocess now exits 0
# whenever the real run succeeded. These are real (non-xfail) assertions
# again.


# ── (a) unit: composite document shape, no run ──────────────────────────

def test_build_parca_sim_composite_wires_scatter_task_and_bridge(tmp_path):
    composite = build_parca_sim_composite(
        seeds=[0, 1], steps=2, outdir=str(tmp_path))

    sims_state = composite.state["sims"]
    address = sims_state["address"]
    address_str = address if isinstance(address, str) else (
        f"{address['protocol']}:{address['data']}")
    assert address_str == "local:CompositeTask"
    sims_config = dict(sims_state["config"])
    assert sims_config["scatter_param"] == "seed"
    assert sims_config["artifact_params"] == {"cache_dir": "sim_data"}
    assert sims_config["generator"] == "ecoli_baseline"
    assert sims_config["steps"] == 2.0

    assert composite.config["parallel_steps"] is True

    bridge_outputs = composite.bridge.get("outputs", {})
    assert "verdict" in bridge_outputs


def test_build_parca_sim_composite_seeds_store_is_a_list(tmp_path):
    composite = build_parca_sim_composite(
        seeds=[0, 1, 2], steps=1, outdir=str(tmp_path))
    assert composite.state["seeds"] == [0, 1, 2]


# ── (b) MILESTONE integration: real run, real verdict ───────────────────

@_skip_no_cache
def test_milestone_workflow_runs_real_sims_and_produces_a_gating_verdict(tmp_path):
    outdir = str(tmp_path / "run")
    rc = main([
        "--seeds", "2", "--parca-mode", "fixture", "--steps", "5",
        "--backend", "local", "--outdir", outdir,
        "--cache-dir", DEFAULT_CACHE_DIR,
    ])
    assert rc == 0

    # F4: per-seed RESULT DIRECTORIES, not just dict entries.
    artifact_root = os.path.join(outdir, ".pbg", "artifacts")
    prov_path = os.path.join(outdir, ".pbg", "work", "ecoli_baseline", "provenance.json")
    assert os.path.isfile(prov_path), f"no provenance at {prov_path}"
    provenance = json.loads(open(prov_path).read())
    assert set(provenance) == {"0", "1"}
    for key, entry in provenance.items():
        result_dir = os.path.join(artifact_root, entry["address"], "results")
        assert os.path.isdir(result_dir), f"seed {key}: expected a results dir at {result_dir}"


@_skip_no_cache
def test_milestone_workflow_verdict_status_and_real_results_handle(tmp_path):
    outdir = str(tmp_path / "run")
    composite = build_parca_sim_composite(
        seeds=[0, 1], parca_mode="fixture", steps=5, outdir=outdir,
        cache_dir=DEFAULT_CACHE_DIR)

    from process_bigraph.workflow import run_workflow
    result = run_workflow(composite, backend="local", outdir=outdir)

    assert result.status == "ok"
    verdict = result.outputs["verdict"]
    assert verdict["status"] in {"pass", "fail", "warn"}
    # The card's verdict summary must be driven by REAL emitted rows (the
    # actual CompositeTask/ecoli_baseline output), never a fixture stand-in.
    assert "checks" in verdict and verdict["checks"]
    assert verdict["checks"][0]["name"] == "emitted_records"


def _sim_launch_count(spy) -> int:
    """How many of ``spy``'s captured ``subprocess.run`` calls actually
    launched a ``run_composite`` sim subprocess.

    The mock target (``process_bigraph.workflow.tasks.subprocess.run``)
    patches the ``run`` attribute on the singleton ``subprocess`` module --
    ``tasks.subprocess`` IS ``build.subprocess`` IS the one process-wide
    module object -- so it also intercepts unrelated ``subprocess.run``
    calls elsewhere in the same process, e.g. ``v2ecoli.workflow.build._git_sha``'s
    ``git rev-parse HEAD`` lookup inside ``main()``. Count only calls whose
    argv actually invokes ``process_bigraph.run_composite`` (the real sim
    launch ``CompositeTask._run_match`` issues) so an unrelated subprocess
    call never masquerades as a sim (re)run.
    """
    count = 0
    for call in spy.call_args_list:
        cmd = call.args[0] if call.args else call.kwargs.get("args")
        if cmd and any("process_bigraph.run_composite" in str(part) for part in cmd):
            count += 1
    return count


# ── (c) cache hit: second identical run, near-zero subprocess launches ──

@_skip_no_cache
def test_milestone_workflow_second_identical_run_is_a_cache_hit(tmp_path):
    outdir = str(tmp_path / "run")
    argv = [
        "--seeds", "2", "--parca-mode", "fixture", "--steps", "5",
        "--backend", "local", "--outdir", outdir,
        "--cache-dir", DEFAULT_CACHE_DIR,
    ]
    assert main(argv) == 0

    with mock.patch(
            "process_bigraph.workflow.tasks.subprocess.run",
            wraps=__import__("subprocess").run) as spy:
        assert main(argv) == 0
        launches = _sim_launch_count(spy)
        assert launches == 0, (
            f"expected zero sim subprocess launches on a cache-hit rerun, "
            f"got {launches} (raw call_count={spy.call_count})")

    prov_path = os.path.join(outdir, ".pbg", "work", "ecoli_baseline", "provenance.json")
    provenance = json.loads(open(prov_path).read())
    assert provenance["0"]["cache_hit"] is True
    assert provenance["1"]["cache_hit"] is True


# ── (d) cache miss: changed steps re-runs the sims ───────────────────────

@_skip_no_cache
def test_milestone_workflow_changed_steps_is_a_cache_miss(tmp_path):
    outdir = str(tmp_path / "run")
    base_argv = [
        "--seeds", "2", "--parca-mode", "fixture",
        "--backend", "local", "--outdir", outdir,
        "--cache-dir", DEFAULT_CACHE_DIR,
    ]
    assert main(base_argv + ["--steps", "5"]) == 0

    with mock.patch(
            "process_bigraph.workflow.tasks.subprocess.run",
            wraps=__import__("subprocess").run) as spy:
        assert main(base_argv + ["--steps", "7"]) == 0
        launches = _sim_launch_count(spy)
        assert launches == 2, (
            f"expected a real subprocess per seed on a steps-changed cache "
            f"miss, got {launches} (raw call_count={spy.call_count})")

    prov_path = os.path.join(outdir, ".pbg", "work", "ecoli_baseline", "provenance.json")
    provenance = json.loads(open(prov_path).read())
    assert provenance["0"]["cache_hit"] is False
    assert provenance["1"]["cache_hit"] is False
