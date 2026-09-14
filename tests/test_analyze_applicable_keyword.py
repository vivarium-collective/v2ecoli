"""``v2ecoli-analyze`` must honour the ``"applicable"`` keyword in its config.

Regression for a live failure on sim 1318 (2026-09-14). viva-api's chain-dispatch
campaign gather writes

    {"analysis_options": "applicable", "out_dir": "s3://...", "sim_data_path": "..."}

and ``main()`` passed that straight through, so ``run_analyses`` received a bare
``str`` and raised ``AttributeError: 'str' object has no attribute 'items'``
(rc=1) for every such campaign. The keyword was already honoured by
``scripts/run_standalone_analysis.py`` and ``scripts/run_multi_node_analysis.py``
-- just not by the entry point the gather actually invokes.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast

from v2ecoli.workflow.analysis_runner import resolve_analysis_options  # noqa: E402


def test_an_explicit_mapping_is_used_verbatim() -> None:
    options = {"single": {"mass_fraction_summary": {}}}
    assert resolve_analysis_options(options, "/nowhere") is options


def test_a_json_encoded_mapping_is_parsed() -> None:
    got = resolve_analysis_options('{"multiseed": {"doubling_time_distribution": {}}}', "/nowhere")
    assert got == {"multiseed": {"doubling_time_distribution": {}}}


def test_empty_and_none_resolve_to_no_analyses() -> None:
    for value in (None, "", {}, "none", "NONE", "  none  "):
        assert resolve_analysis_options(value, "/nowhere") == {}


def test_the_applicable_keyword_resolves_against_the_sweeps_own_shape(monkeypatch) -> None:
    """The exact payload that failed, and the shape comes from the DATA."""
    import v2ecoli.workflow.analysis_runner as ar

    # two seeds x three generations, as hive partition keys
    keys = [
        {"variant": 0, "lineage_seed": s, "generation": g, "agent_id": "0"}
        for s in (0, 1)
        for g in (0, 1, 2)
    ]
    monkeypatch.setattr(ar, "cell_keys", lambda sweep_dir: keys)

    seen: dict = {}

    def fake_build(analyses, *, n_seeds, n_generations, **kw):
        seen.update(analyses=analyses, n_seeds=n_seeds, n_generations=n_generations)
        return {"multigeneration": {"ptools_rna_multigeneration": {}}}

    import v2ecoli.steps.batch_baseline_runner as bbr

    monkeypatch.setattr(bbr, "build_analysis_options", fake_build)

    got = ar.resolve_analysis_options("applicable", "/sweep")
    assert got == {"multigeneration": {"ptools_rna_multigeneration": {}}}
    assert seen == {"analyses": "applicable", "n_seeds": 2, "n_generations": 3}


def test_applicable_is_case_and_whitespace_insensitive(monkeypatch) -> None:
    import v2ecoli.steps.batch_baseline_runner as bbr
    import v2ecoli.workflow.analysis_runner as ar

    monkeypatch.setattr(ar, "cell_keys", lambda sweep_dir: [{"lineage_seed": 0, "generation": 0}])
    monkeypatch.setattr(bbr, "build_analysis_options", lambda *a, **k: {"single": {}})
    assert ar.resolve_analysis_options("  Applicable  ", "/sweep") == {"single": {}}


def test_an_empty_sweep_still_resolves_to_a_sane_shape(monkeypatch) -> None:
    """A listing that returns nothing must not produce n_seeds=0."""
    import v2ecoli.steps.batch_baseline_runner as bbr
    import v2ecoli.workflow.analysis_runner as ar

    monkeypatch.setattr(ar, "cell_keys", lambda sweep_dir: [])
    seen: dict = {}
    monkeypatch.setattr(
        bbr,
        "build_analysis_options",
        lambda a, *, n_seeds, n_generations, **k: seen.update(n_seeds=n_seeds, n_generations=n_generations) or {},
    )
    ar.resolve_analysis_options("applicable", "/sweep")
    assert seen == {"n_seeds": 1, "n_generations": 1}


def test_run_analyses_resolves_options_at_the_library_boundary(monkeypatch) -> None:
    """The gap this closes: ``main()`` resolved the keyword, but ``run_analyses``
    trusted its arg -- so a direct caller (an old dispatcher, a script, viva-api's
    submit path) that reached ``run_analyses`` with ``"applicable"`` or a JSON
    string still hit the ``.items()`` loops as a bare ``str`` and died with the
    sim-1318 ``AttributeError``. ``run_analyses`` must now route its arg through
    ``resolve_analysis_options`` BEFORE any ``.items()``.

    Proven by making the resolver raise a unique sentinel: if ``run_analyses``
    calls it first, that sentinel surfaces; if it skipped straight to ``.items()``
    on the bare string, we'd get ``AttributeError`` instead.
    """
    import v2ecoli.workflow.analysis_runner as ar

    class _ResolverReached(Exception):
        pass

    def _fake_resolve(analyses, sweep_dir):
        raise _ResolverReached(repr(analyses))

    monkeypatch.setattr(ar, "resolve_analysis_options", _fake_resolve)
    with pytest.raises(_ResolverReached, match="applicable"):
        ar.run_analyses("/sweep", "applicable")
