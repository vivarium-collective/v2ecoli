"""Phase B, Task 5: prove the comparison runs THROUGH the
investigation-as-composite substrate -- not a real e2e (the worker is
stubbed), but the wiring/ordering: materialize -> write native files -> the
substrate's ``run_investigation_composite`` runs ``parca`` before every
config study, then the investigation-level ``comparison_matrix`` analysis
after all config studies, with dependency order coming entirely from the
composite graph the substrate builds off ``pipeline_gate.prerequisites`` +
the investigation's ``analyses:`` wiring (see
``vivarium_workbench.lib.investigation_execution.build_investigation_composite``).

Requires ``process_bigraph``/``bigraph_schema`` from the substrate worktree on
PYTHONPATH (v2ecoli venv):

    PYTHONPATH=/Users/eranagmon/code/vivarium-workbench--inv-composite \\
        .venv/bin/python -m pytest tests/test_phase_b_substrate_integration.py -v

Fixture style (``_comparison_block``/``_materialize``) reused verbatim from
``test_write_native_investigation.py`` (Task 3) so this file materializes +
writes exactly the same shape that file already asserts field-by-field --
this file's job is only to prove the SUBSTRATE executes it in the right
order, not to re-assert the written YAML shape.

The real paired e2e (candidate + reference actually simulating) is a
DEFERRED follow-up tracked on the mini (design doc "Testing" section) -- not
attempted here.
"""
from __future__ import annotations

import pytest

# The vivarium-workbench substrate is an ORCHESTRATOR of v2ecoli, so v2ecoli
# does not (and must not, to avoid a dependency cycle) depend on it. This test
# only runs when the substrate is importable -- i.e. its worktree is on
# PYTHONPATH (see module docstring). In v2ecoli CI it is absent, so pytest
# skips the whole module at collection instead of erroring.
pytest.importorskip(
    "vivarium_workbench",
    reason="Phase B substrate integration requires the vivarium-workbench "
    "process-bigraph substrate on PYTHONPATH; see module docstring.",
)

import v2ecoli.workflow.analyses  # noqa: F401,E402 -- registers comparison_cards/comparison_matrix
from v2ecoli.workflow.comparison_materialize import (  # noqa: E402
    materialize_comparison, write_native_investigation)
from v2ecoli.workflow.parca_study import PARCA_STUDY_NAME  # noqa: E402

from vivarium_workbench.lib.investigation_execution import (  # noqa: E402
    run_investigation_composite)

INVEST_SLUG = "wcm-comparison"


def _comparison_block(ve_cache: str, configs=None) -> dict:
    return {
        "reference": {"repo": "/fake/vecoli-fork", "kind": "vecoli"},
        "v2_cache": "out/cache_full",
        "ve_cache": ve_cache,
        "defaults": {"seeds": 1, "generations": 1, "cards": ["summary", "standard"]},
        "configs": configs or [{"name": "basal", "condition": "basal"}],
    }


def _write(tmp_path, configs=None):
    """Materialize + write the native investigation into ``tmp_path`` (the
    workspace root) -- same pattern as ``test_write_native_investigation.py``.
    Returns the workspace root ``run_investigation_composite`` runs against."""
    ve_cache = tmp_path / "vecoli_parca"
    ve_cache.mkdir()
    (ve_cache / "simData.cPickle").write_bytes(b"fake-simdata")
    block = _comparison_block(str(ve_cache), configs=configs)
    materialized = materialize_comparison(block, invest_name="test-comparison")
    workspace = tmp_path / "workspace"
    result = write_native_investigation(materialized, workspace, INVEST_SLUG)
    return workspace, result


def _recorder(calls: list):
    """The ``run_study_fn`` stub hook: records the study slug it was called
    for and returns the minimal reply shape ``run_investigation_composite``
    / ``StudyStep`` expect, with no real worker/pool involved."""
    def _fn(workspace, study_slug):
        calls.append(study_slug)
        return {
            "run_refs": [{"run_id": study_slug, "status": "completed"}],
            "verdict": {"overall": "within_tol"},
        }
    return _fn


def _recorder_with_comparison_verdict(calls: list):
    """Like ``_recorder`` but the reply also carries a per-study
    ``analyses.comparison_cards.verdict`` -- the store-data-flow refactor's
    canonical source (design: docs/superpowers/specs/2026-08-02-store-
    dataflow-refactor-design.md) -- DISTINCT from the top-level conclusion
    ``verdict``, so a test using this stub proves
    ``InvestigationAnalysisStep`` extracts the comparison verdict
    specifically, not just whatever sits under ``"verdict"``."""
    def _fn(workspace, study_slug):
        calls.append(study_slug)
        return {
            "run_refs": [{"run_id": study_slug, "status": "completed"}],
            "verdict": {"overall": "cnc"},
            "analyses": {
                "comparison_cards": {
                    "verdict": {"overall": "within_tol", "cfg": study_slug},
                },
            },
        }
    return _fn


def test_matrix_receives_config_verdicts_from_wired_stores(tmp_path, monkeypatch):
    """Store data-flow proof (design: docs/superpowers/specs/2026-08-02-
    store-dataflow-refactor-design.md, integration test): the investigation-
    level ``comparison_matrix`` analysis is dispatched with
    ``config["config_verdicts"]`` assembled from each config study's WIRED
    result store (``state["study_<slug>"]["analyses"]["comparison_cards"]
    ["verdict"]``, per ``InvestigationAnalysisStep.update``/
    ``_extract_study_verdict``) -- not the raw ``run_study`` reply, not the
    top-level conclusion ``verdict``, and not a disk read of
    ``report_card_verdict.json`` (no such file exists anywhere in this tmp
    workspace)."""
    workspace, written = _write(
        tmp_path, configs=[{"name": "basal", "condition": "basal"},
                           {"name": "with_aa", "condition": "with_aa"}])
    assert set(written["study_paths"]) == {PARCA_STUDY_NAME, "basal", "with_aa"}

    class _FakePool:
        def __init__(self):
            self.calls = []

        def call(self, workspace, method, params):
            self.calls.append((workspace, method, params))
            return {"written": ["matrix.html"], "errors": []}

    fake_pool = _FakePool()
    monkeypatch.setattr(
        "vivarium_workbench.lib.env_worker_pool.get_pool", lambda: fake_pool)

    calls: list = []
    summary = run_investigation_composite(
        workspace, INVEST_SLUG,
        run_study_fn=_recorder_with_comparison_verdict(calls))

    assert set(calls) == {PARCA_STUDY_NAME, "basal", "with_aa"}
    assert summary["errors"] == []

    assert len(fake_pool.calls) == 1
    _, method, params = fake_pool.calls[0]
    assert method == "run_investigation_analysis"
    assert params["name"] == "comparison_matrix"

    # The matrix was dispatched with the PER-CONFIG COMPARISON verdicts
    # (analyses.comparison_cards.verdict), extracted from the wired study
    # result stores -- NOT the raw run_study reply, and NOT the top-level
    # conclusion verdict ({"overall": "cnc"}), which would prove the
    # extraction picked the wrong field. config_verdicts is keyed by every
    # composite member (InvestigationAnalysisStep wires ALL studies, incl.
    # parca, not just the config studies), so parca's extracted entry is
    # present too -- the assertion below checks each config study's entry
    # specifically (the thing the matrix actually renders, via
    # config_studies) while still proving the dict as a whole came from
    # per-slug store extraction, not a single copied value.
    config_verdicts = params["config"]["config_verdicts"]
    assert config_verdicts["basal"] == {"overall": "within_tol", "cfg": "basal"}
    assert config_verdicts["with_aa"] == {"overall": "within_tol", "cfg": "with_aa"}
    assert params["config"]["config_studies"] == ["basal", "with_aa"]

    # No report_card_verdict.json exists anywhere in the tmp workspace for
    # either config study -- the matrix nonetheless received real
    # (non-placeholder) verdicts above, proving they arrived via the
    # composite's wired stores, not a disk read.
    for slug in ("basal", "with_aa"):
        verdict_path = workspace / "studies" / slug / "report_card_verdict.json"
        assert not verdict_path.exists(), verdict_path


def test_parca_runs_before_each_config_study_and_matrix_runs_after(tmp_path, monkeypatch):
    """Full (a)-(d): ordering, dispatch, and matrix-analysis wiring, all
    driven through the real substrate (stubbed worker + stubbed analysis
    pool -- no live sims, no live env worker)."""
    workspace, written = _write(
        tmp_path, configs=[{"name": "basal", "condition": "basal"},
                           {"name": "with_aa", "condition": "with_aa"}])
    assert set(written["study_paths"]) == {PARCA_STUDY_NAME, "basal", "with_aa"}

    # Stub the investigation-level analysis dispatch (comparison_matrix runs
    # via the env worker pool, same as any other study/analysis run) -- same
    # pattern the substrate's own test_runner_unification.py uses. Only the
    # analysis path needs this; run_study_fn (below) covers study dispatch.
    class _FakePool:
        def __init__(self):
            self.calls = []

        def call(self, workspace, method, params):
            self.calls.append((workspace, method, params))
            return {"written": ["matrix.html"], "errors": []}

    fake_pool = _FakePool()
    monkeypatch.setattr(
        "vivarium_workbench.lib.env_worker_pool.get_pool", lambda: fake_pool)

    calls: list = []
    summary = run_investigation_composite(
        workspace, INVEST_SLUG, run_study_fn=_recorder(calls))

    # (a) the recorder saw run_study for parca AND each config study.
    assert set(calls) == {PARCA_STUDY_NAME, "basal", "with_aa"}
    assert set(summary["studies_ran"]) == {PARCA_STUDY_NAME, "basal", "with_aa"}

    # (b) parca ran before every config study.
    order = summary["studies_ran"]
    assert order.index(PARCA_STUDY_NAME) < order.index("basal"), order
    assert order.index(PARCA_STUDY_NAME) < order.index("with_aa"), order

    # (c) the comparison_matrix analysis is present -- it only runs once its
    # InvestigationAnalysisStep's inputs (every config study's result store)
    # are available, so the scheduler necessarily ran it after both configs.
    assert summary["analyses"] == ["comparison_matrix"]
    assert len(fake_pool.calls) == 1
    _, method, params = fake_pool.calls[0]
    assert method == "run_investigation_analysis"
    assert params["name"] == "comparison_matrix"
    # config_studies (the member slug list Task 2 wires the matrix to) is
    # exactly the two config slugs -- assembled from the written study.yaml,
    # not hardcoded here.
    assert params["config"]["config_studies"] == ["basal", "with_aa"]

    # (d) no errors from ordering/dispatch itself (the worker/analysis are
    # stubbed, so this is not asserting real verdicts -- only that the
    # RUN/order succeeded).
    assert summary["errors"] == []


def test_single_config_study_results_carry_the_stubbed_reply(tmp_path, monkeypatch):
    """1-config variant of the same wiring proof, plus a check that
    study_results (the composite's per-study reply map) actually carries the
    recorder's stub reply through to the summary -- proving the StudyStep's
    output store is the thing the InvestigationAnalysisStep would read a real
    verdict from, not a side channel the substrate ignores."""
    workspace, written = _write(tmp_path)
    assert set(written["study_paths"]) == {PARCA_STUDY_NAME, "basal"}

    monkeypatch.setattr(
        "vivarium_workbench.lib.env_worker_pool.get_pool",
        lambda: type("P", (), {"call": staticmethod(
            lambda workspace, method, params: {"written": [], "errors": []})})())

    calls: list = []
    summary = run_investigation_composite(
        workspace, INVEST_SLUG, run_study_fn=_recorder(calls))

    assert calls == [PARCA_STUDY_NAME, "basal"]
    assert summary["studies_ran"] == [PARCA_STUDY_NAME, "basal"]
    assert summary["analyses"] == ["comparison_matrix"]
    assert summary["errors"] == []
    assert summary["study_results"]["basal"]["verdict"] == {"overall": "within_tol"}
