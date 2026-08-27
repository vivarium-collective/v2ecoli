"""The exchange-flux ``basis`` is declared ONCE and honoured by BOTH arms.

``counts`` and ``gdcw`` are different measurements, not different units: the
first is a lineage-cumulative molecule total read from ``environment.exchange``,
the second a per-tick mmol/gDCW/h rate. On the reference (wrapped-vEcoli) arm the
rate is not derived — it is read from the wrapped metabolism's own
``listeners.fba_results.external_exchange_fluxes`` — the same leaf genuine
vEcoli's own analyses read, though they index it positionally while this reads it
by key, so it works only for a metabolism that writes that leaf as a mapping.

Each test below names the failure it exists to catch; several of them pass
trivially if the basis is dropped anywhere along the chain, so the chain is
tested at each hop rather than end-to-end only.
"""
from types import SimpleNamespace

import pytest

from v2ecoli.library.vivarium_ecoli_engine import _select_exchange_fluxes

FLUXES = {"glucose_exchange": "GLC[p]", "product_exchange": "SOME-PRODUCT[c]"}


class _StubLoader:
    """_get_special_step reads unique-molecule names off the loader before it
    reaches any step branch. This deriver needs none of them, so a stub keeps the
    test a unit test rather than requiring a 90MB ParCa load."""

    def __init__(self):
        self.sim_data = SimpleNamespace(
            internal_state=SimpleNamespace(
                unique_molecule=SimpleNamespace(
                    unique_molecule_definitions={})))

# A store shaped like the real one: the environment carries glucose as a
# cumulative count and has NO key for the product, while fba_results carries
# both as rates. That asymmetry is the whole defect this switch addresses.
ENVIRONMENT = {"exchange": {"GLC[p]": 0.0, "GLC": -2.8e7}}
LISTENERS = {"fba_results": {"external_exchange_fluxes": {
    "GLC[p]": -7.3, "SOME-PRODUCT[c]": 0.129}}}


def test_counts_basis_reads_the_environment_store():
    """Default behaviour is unchanged: counts comes from environment.exchange."""
    out = _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="counts",
                                  listeners=LISTENERS)
    assert out["glucose_exchange"] == pytest.approx(-2.8e7)


def test_counts_basis_cannot_see_a_molecule_absent_from_the_environment():
    """The defect, pinned: on counts the product reads 0.0 — indistinguishable
    from a true zero — because the environment store has no key for it."""
    out = _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="counts",
                                  listeners=LISTENERS)
    assert out["product_exchange"] == 0.0


def test_gdcw_basis_reads_fba_results_and_surfaces_that_molecule():
    """The fix. Catches: basis ignored, or gdcw still reading the environment —
    either of which leaves the product at 0.0 while the controls look healthy."""
    out = _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="gdcw",
                                  listeners=LISTENERS)
    assert out["product_exchange"] == pytest.approx(0.129)
    assert out["glucose_exchange"] == pytest.approx(-7.3)


def test_the_two_bases_disagree_by_orders_of_magnitude():
    """Guards against a future 'harmonisation' that quietly makes them the same
    number: they are different quantities and must not be interchangeable."""
    counts = _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="counts",
                                     listeners=LISTENERS)["glucose_exchange"]
    gdcw = _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="gdcw",
                                   listeners=LISTENERS)["glucose_exchange"]
    assert abs(counts) / abs(gdcw) > 1e5


def test_unknown_basis_raises_rather_than_defaulting():
    """A silently-defaulted basis emits a running total under a rate's name, so
    the refusal is the feature. Catches a `basis or "counts"` fallback."""
    with pytest.raises(ValueError, match="basis"):
        _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="per-cell",
                                listeners=LISTENERS)


def test_gdcw_refuses_a_positional_array_source():
    """Not every metabolism keys that leaf: some write a POSITIONAL ARRAY, whose
    id->index map lives in emit metadata and is not in the store. Treating that as
    empty would emit 0.0 on every leaf of every tick — a flat zero trace that reads
    exactly like a cell producing none of the molecule. Refused instead."""
    import numpy as np
    listeners = {"fba_results": {"external_exchange_fluxes": np.array([-7.3, 0.129])}}
    with pytest.raises(TypeError, match="metabolite id"):
        _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="gdcw",
                                listeners=listeners)
    # a plain list is the same hazard
    with pytest.raises(TypeError):
        _select_exchange_fluxes(
            ENVIRONMENT, FLUXES, basis="gdcw",
            listeners={"fba_results": {"external_exchange_fluxes": [-7.3, 0.129]}})


def test_unknown_basis_raises_even_with_an_empty_flux_map():
    """Validation must precede the empty-map short-circuit, so a bad basis is
    refused on every call — the deriver validates in initialize() regardless of
    its map, and the two surfaces must agree."""
    with pytest.raises(ValueError, match="basis"):
        _select_exchange_fluxes(ENVIRONMENT, {}, basis="per-cell")


def test_gdcw_with_no_fba_results_is_REFUSED_not_read_as_zero():
    """⛔ This test previously asserted the opposite, and the opposite was the bug.

    An absent `external_exchange_fluxes` fell through to `{}`, so every leaf read
    0.0 on every tick — a flat zero trace indistinguishable from a cell producing
    none of the molecule, and the card then graded it `ungraded`, which the shared
    severity model scores 0, i.e. no worse than a pass.

    ⚠ MEASURED, and it is why this is not hypothetical: the PUBLIC vEcoli's
    `metabolism_redux` does not write this leaf at all (it writes
    `estimated_exchange_dmdt`); only the fork does. So the silently-zero
    configuration was the public one, while the loud refusal for a positional
    array only ever fired for stock `metabolism.py`. Absent and wrong-shaped are
    now refused on the same footing."""
    with pytest.raises(TypeError, match="does not write it"):
        _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="gdcw", listeners={})


def test_gdcw_with_an_EMPTY_keyed_leaf_still_traces_rather_than_raising():
    """The other side of that line, and it is what keeps the fork working.

    A metabolism that DECLARES the leaf in its schema (default `{}`) but has not
    populated it on this tick has written it — the key exists, the mapping is
    simply empty. That is a real not-yet-solved tick, not an unusable metabolism,
    so it traces 0.0 rather than taking the run down."""
    out = _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="gdcw",
                                  listeners={"fba_results":
                                             {"external_exchange_fluxes": {}}})
    assert out == {"glucose_exchange": 0.0, "product_exchange": 0.0}


def test_empty_flux_map_is_a_no_op_on_either_VALID_basis():
    assert _select_exchange_fluxes(ENVIRONMENT, {}, basis="gdcw") == {}
    assert _select_exchange_fluxes(ENVIRONMENT, {}, basis="counts") == {}


# --- the chain: each hop that could silently drop the basis -----------------

def test_deriver_is_actually_built_with_the_declared_basis():
    """Builds the step through _get_special_step and reads the basis off the
    INSTANCE. The previous version of this test only asserted the module global
    it had just set, so deleting the `basis` key from the deriver's config in
    _get_special_step left it green — the exact failure it claimed to catch."""
    from v2ecoli.composites import _helpers
    from v2ecoli.core import build_core
    core = build_core()
    _helpers.set_exchange_fluxes_override({"glucose_exchange": "GLC[p]"})
    _helpers.set_exchange_flux_basis_override("gdcw")
    loader = _StubLoader()
    try:
        instance, _topo, _kind = _helpers._get_special_step(
            loader, "exchange_flux_listener", core)
    finally:
        _helpers.set_exchange_fluxes_override({})
        _helpers.set_exchange_flux_basis_override(None)
    assert getattr(instance, "basis", None) == "gdcw"


def test_deriver_defaults_to_counts_when_nothing_declared():
    """The other half: an undeclared basis must reach the step as counts, not as
    whatever the previous build left in the module global.

    ⚠ This test USED TO PASS BY ACCIDENT OF ORDERING. It read the module global
    without setting it, so it was really asserting whatever the previous test's
    teardown had left there — measured: flipping the declared default to "gdcw"
    made it fail when run ALONE and stay green when run in file order. It now
    reloads the module so it reads the DECLARED default rather than a residue.
    """
    import importlib

    from v2ecoli.composites import _helpers
    from v2ecoli.core import build_core
    importlib.reload(_helpers)          # ⚠ load-bearing: see the docstring
    core = build_core()
    assert _helpers._EXCHANGE_FLUX_BASIS_OVERRIDE == "counts", (
        "the module's DECLARED default is not counts, so an undeclared build "
        "would silently change quantity for every existing study")
    _helpers.set_exchange_fluxes_override({"glucose_exchange": "GLC[p]"})
    loader = _StubLoader()
    try:
        instance, _topo, _kind = _helpers._get_special_step(
            loader, "exchange_flux_listener", core)
    finally:
        _helpers.set_exchange_fluxes_override({})
    assert getattr(instance, "basis", None) == "counts"


def test_the_card_no_longer_re_derives_the_basis_from_the_study_config():
    """The second reader is DELETED, not repaired. Two readers of one setting
    disagreed once (engines took `comparison:` only, the card's helper preferred a
    top-level key) and graded a cumulative total as a rate. This pins the deletion:
    re-adding the key to the card's state re-creates the disagreement."""
    import inspect, re
    import scripts.comparison_report_card as crc
    src = inspect.getsource(crc)
    m = re.search(r'"config":\s*\{(.*?)\},\n', src, re.S)
    assert m, "could not locate the card state's config dict"
    assert "exchange_flux_basis" not in m.group(1), (
        "the card must read the basis off the RUN, not re-derive it from config")


def test_both_spec_routes_resolve_the_basis_by_ONE_rule(tmp_path):
    """The bug: the study route read `comparison:` while the investigation route's
    helper preferred a top-level key, so one file gave two answers and the run and
    the card could disagree. Both now go through this helper, and it is
    `comparison:`-only like every other per-study measurement key."""
    from scripts._compare.study_spec import exchange_flux_basis_from_study_yaml
    y = tmp_path / "study.yaml"
    y.write_text("exchange_flux_basis: gdcw\ncomparison:\n  seeds: 4\n",
                 encoding="utf-8")
    # a stray TOP-LEVEL key must NOT win — that asymmetry was the defect
    assert exchange_flux_basis_from_study_yaml(y, fallback="counts") == "counts"
    y.write_text("comparison:\n  exchange_flux_basis: gdcw\n", encoding="utf-8")
    assert exchange_flux_basis_from_study_yaml(y, fallback="counts") == "gdcw"


def _spec(**kw):
    """A minimal StudySpec for the runner tests below."""
    from scripts._compare.study_spec import StudySpec
    base = dict(name="s", condition="basal", seeds=1, gens=1, cards=[],
                invest_name="i", v2_cache="v2c", ve_cache="vec",
                study_path="workspace/studies/s/study.yaml", config="basal")
    base.update(kw)
    return StudySpec(**base)


def _argvs_from_run_engines(monkeypatch, spec):
    """Run the REAL _run_engines with subprocess.run captured. Returns the argv
    of every engine invocation it made."""
    from scripts._compare import runner
    calls = []
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda argv, **kw: calls.append(list(argv)))
    runner._run_engines(spec, out="out/x", mode="serial")
    return calls


def _flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_runner_puts_the_basis_on_BOTH_engine_command_lines(monkeypatch):
    """Replaces a source-substring test that only proved the STRING
    "--exchange-flux-basis" appeared somewhere in runner.py — which stays true
    if the flag is hardcoded to "counts", appended to only one of the two
    subprocess calls, or built from a field that no longer exists.

    This runs the real _run_engines with subprocess.run captured and reads the
    value off each argv. Two arms carrying different quantities under one leaf
    name is the failure the whole basis surface exists to prevent, so ONE arm is
    not enough."""
    calls = _argvs_from_run_engines(
        monkeypatch, _spec(exchange_fluxes={"product_exchange": "X[c]"},
                           exchange_flux_basis="gdcw"))
    assert len(calls) == 2, "expected one invocation per engine"
    for argv in calls:
        assert _flag_value(argv, "--exchange-flux-basis") == "gdcw", argv
        # and it must ride WITH the map, not instead of it
        assert "--exchange-flux" in argv, argv


def test_runner_carries_counts_too_so_the_flag_cannot_be_hardcoded(monkeypatch):
    """The other value, so the test above cannot pass against a literal."""
    calls = _argvs_from_run_engines(
        monkeypatch, _spec(exchange_fluxes={"product_exchange": "X[c]"},
                           exchange_flux_basis="counts"))
    assert [_flag_value(a, "--exchange-flux-basis") for a in calls] == \
        ["counts", "counts"]


def test_runner_emits_no_basis_flag_when_no_fluxes_are_declared(monkeypatch):
    """A study declaring no exchange fluxes must be unchanged: the basis is
    meaningless without the map, and an unconditional flag would make every
    baseline study's command line differ from what it ran before."""
    calls = _argvs_from_run_engines(monkeypatch, _spec())
    for argv in calls:
        assert "--exchange-flux-basis" not in argv
        assert "--exchange-flux" not in argv


def _fake_workspace(tmp_path, study_yaml_text, name="s"):
    """Lay out workspace/{investigations/i,studies/<name>} the way
    studies_root_for() resolves it (inv_dir.parent.parent / "studies")."""
    inv = tmp_path / "workspace" / "investigations" / "i"
    inv.mkdir(parents=True)
    study_dir = tmp_path / "workspace" / "studies" / name
    study_dir.mkdir(parents=True)
    (study_dir / "study.yaml").write_text(study_yaml_text, encoding="utf-8")
    return inv, study_dir / "study.yaml"


def _ctx(inv_dir, **kw):
    from scripts._compare.reference import ReferenceEngine
    ctx = {"invest_name": "i", "v2_cache": "a", "ve_cache": "b",
           "reference": ReferenceEngine.from_spec({}), "configs": [],
           "defaults": {}, "default_cards": [], "inv_dir": inv_dir}
    ctx.update(kw)
    return ctx


def test_investigation_route_lands_the_basis_ON_THE_SPEC(tmp_path):
    """Replaces a test that called the helper directly and therefore proved
    nothing about the route: deleting the `exchange_flux_basis=` argument from
    the StudySpec built in specs_from_configs left it green, and the spec is
    what the runner reads.

    This builds the spec through the real specs_from_configs and asserts on the
    FIELD. configs[] entries carry no study.yaml keys, so a declaration in the
    file the investigation NAMES reaches the spec only via the bridge."""
    from scripts._compare.study_spec import specs_from_configs
    inv, _ = _fake_workspace(
        tmp_path, "comparison:\n  exchange_flux_basis: gdcw\n")
    ctx = _ctx(inv, configs=[{"name": "s", "condition": "basal"}])
    spec, = specs_from_configs(ctx)
    assert spec.exchange_flux_basis == "gdcw"


def test_investigation_route_default_is_counts_not_a_stale_value(tmp_path):
    """The other side, so the test above cannot pass by the field defaulting to
    'gdcw' or by the bridge ignoring the file entirely."""
    from scripts._compare.study_spec import specs_from_configs
    inv, _ = _fake_workspace(tmp_path, "comparison:\n  seeds: 1\n")
    ctx = _ctx(inv, configs=[{"name": "s", "condition": "basal"}])
    spec, = specs_from_configs(ctx)
    assert spec.exchange_flux_basis == "counts"


def test_investigation_defaults_reach_the_spec_when_the_study_is_silent(tmp_path):
    """An investigation-level declaration must survive the bridge: the helper
    receives it as the fallback, and a bridge that passed its own literal
    default instead would silently drop it."""
    from scripts._compare.study_spec import specs_from_configs
    inv, _ = _fake_workspace(tmp_path, "comparison:\n  seeds: 1\n")
    ctx = _ctx(inv, configs=[{"name": "s", "condition": "basal"}],
               defaults={"exchange_flux_basis": "gdcw"})
    spec, = specs_from_configs(ctx)
    assert spec.exchange_flux_basis == "gdcw"


def test_study_route_lands_the_basis_ON_THE_SPEC(tmp_path):
    """The other first-class route. _spec_from_study builds the spec the runner
    reads; a missing `exchange_flux_basis=` there silently runs every study on
    counts while its YAML says otherwise."""
    from scripts._compare.study_spec import _spec_from_study
    inv, study = _fake_workspace(
        tmp_path,
        "name: s\ncondition: basal\ncomparison:\n  exchange_flux_basis: gdcw\n")
    spec = _spec_from_study(study, _ctx(inv))
    assert spec.exchange_flux_basis == "gdcw"


def test_study_route_default_is_counts(tmp_path):
    from scripts._compare.study_spec import _spec_from_study
    inv, study = _fake_workspace(
        tmp_path, "name: s\ncondition: basal\ncomparison:\n  seeds: 1\n")
    assert _spec_from_study(study, _ctx(inv)).exchange_flux_basis == "counts"


def test_the_two_spec_routes_agree_on_one_file(tmp_path):
    """The bug that shipped: one study.yaml, two routes, two answers. Asserted
    on the SPECS the two routes produce, not on the shared helper — a helper
    both routes agree about proves nothing if one route stops calling it."""
    from scripts._compare.study_spec import _spec_from_study, specs_from_configs
    text = ("name: s\ncondition: basal\n"
            "exchange_flux_basis: gdcw\n"          # stray TOP-LEVEL key
            "comparison:\n  exchange_flux_basis: counts\n")
    inv, study = _fake_workspace(tmp_path, text)
    ctx = _ctx(inv, configs=[{"name": "s", "condition": "basal"}])
    from_configs, = specs_from_configs(ctx)
    from_study = _spec_from_study(study, ctx)
    assert from_configs.exchange_flux_basis == from_study.exchange_flux_basis
    # and `comparison:` is the rule both follow — top level must NOT win
    assert from_study.exchange_flux_basis == "counts"


def test_investigation_fallback_applies_when_the_study_is_silent(tmp_path):
    """And an investigation-level declaration still wins where the study says
    nothing — the bridge must not clobber the fallback with its own default."""
    from scripts._compare.study_spec import exchange_flux_basis_from_study_yaml
    y = tmp_path / "study.yaml"
    y.write_text("comparison:\n  seeds: 4\n", encoding="utf-8")
    assert exchange_flux_basis_from_study_yaml(y, fallback="gdcw") == "gdcw"
    assert exchange_flux_basis_from_study_yaml(
        tmp_path / "missing.yaml", fallback="gdcw") == "gdcw"


# --- the gdcw path against REAL composite value types ------------------------

def test_gdcw_deriver_tolerates_pint_quantities_for_mass_and_timestep():
    """⚠ Regression for a crash that unit tests could not see. On the real
    composite ``listeners.mass.dry_mass`` is a pint Quantity in femtograms, and a
    bare ``float()`` on it raises
    ``DimensionalityError: Cannot convert from 'femtogram' to 'dimensionless'``,
    taking the whole run down on the first tick of the gdcw basis.

    It survived because the path was UNREACHABLE — no study could set a basis, so
    it had only ever run against tests passing plain floats. Unreachable and
    untested were the same fact. This asserts the real value type."""
    from v2ecoli.steps.derivers.exchange_flux_listener import _as_float_fg
    import pint
    ureg = pint.UnitRegistry()
    assert _as_float_fg(300.0 * ureg.femtogram) == 300.0
    assert _as_float_fg(2.0) == 2.0          # plain floats still work
    assert _as_float_fg(None) == 0.0         # absent -> no rate, not a crash
    assert _as_float_fg("not a number") == 0.0


# --- the on-disk sidecar contract: the REAL writer against the REAL reader ---
#
# The filename, the JSON key and the two arm prefixes are a CONTRACT between
# `run_comparison_ensemble._write_exchange_flux_sidecar` and the generic
# `exchange_flux_basis.basis_from_runs`. Re-typing either side by hand is
# how a rename stays green here and breaks every real run, so these tests call
# both production functions and never spell the filename or the key themselves.

def _write_real_sidecar(out_root, prefix, basis, leaves=None):
    """The PRODUCTION writer. Deliberately not a hand-rolled json.dump."""
    from scripts.run_comparison_ensemble import _write_exchange_flux_sidecar
    if leaves is None:
        leaves = {"product_exchange": "X[c]"}
    _write_exchange_flux_sidecar(str(out_root), prefix, dict(leaves), basis)


def test_the_sidecar_round_trips_from_the_real_writer_to_the_real_reader(tmp_path):
    """GAP: nothing executed writer and reader together. Renaming the file or
    the ``basis`` key in the writer left every card test green (they re-typed
    both) while every real run lost its basis and the card refused to grade.

    Both arms write into ONE out_root, exactly as a real run does, so the arm
    PREFIXES are part of what round-trips."""
    from scripts._compare.exchange_flux_basis import basis_from_runs
    _write_real_sidecar(tmp_path, "v2ecoli", "gdcw")
    _write_real_sidecar(tmp_path, "vecoli", "gdcw")
    basis, why = basis_from_runs({"v2_dir": str(tmp_path),
                                       "ve_dir": str(tmp_path)})
    assert (basis, why) == ("gdcw", "")


def test_the_round_trip_keeps_the_two_arms_apart_in_one_out_root(tmp_path):
    """The arm prefixes are load-bearing: both engines share one out_root, so a
    writer that dropped the prefix (or a reader that looked for the wrong one)
    would let ONE arm's basis answer for both — which is precisely the
    two-arms-disagree failure the sidecar exists to expose."""
    from scripts._compare.exchange_flux_basis import basis_from_runs
    _write_real_sidecar(tmp_path, "v2ecoli", "gdcw")
    _write_real_sidecar(tmp_path, "vecoli", "counts")
    basis, why = basis_from_runs({"v2_dir": str(tmp_path),
                                       "ve_dir": str(tmp_path)})
    assert basis is None
    assert "'gdcw'" in why and "'counts'" in why, why


def test_the_round_trip_carries_counts_too(tmp_path):
    """So the test above cannot pass by the reader hardcoding 'gdcw'."""
    from scripts._compare.exchange_flux_basis import basis_from_runs
    _write_real_sidecar(tmp_path, "v2ecoli", "counts")
    _write_real_sidecar(tmp_path, "vecoli", "counts")
    assert basis_from_runs({"v2_dir": str(tmp_path),
                                 "ve_dir": str(tmp_path)}) == ("counts", "")


def test_a_run_that_declared_no_fluxes_writes_no_sidecar_and_is_refused(tmp_path):
    """The writer short-circuits on an empty flux map. That is intended (nothing
    to describe), and the reader must treat the absent file as a REFUSAL rather
    than guessing — a run predating the sidecar looks identical."""
    from scripts._compare.exchange_flux_basis import basis_from_runs
    _write_real_sidecar(tmp_path, "v2ecoli", "gdcw", leaves={})
    _write_real_sidecar(tmp_path, "vecoli", "gdcw", leaves={})
    basis, why = basis_from_runs({"v2_dir": str(tmp_path),
                                       "ve_dir": str(tmp_path)})
    assert basis is None and "sidecar" in why


# --- the ensemble driver: CLI -> run_one -> composite build ------------------
#
# Every hop below is a place the basis is handed from one layer to the next.
# They are tested by EXECUTING the caller with the callee captured, so severing
# the hop (dropping the keyword, renaming it, hardcoding a literal) reds them —
# which asserting on the callee's signature or on module source would not.

class _Stop(Exception):
    """Sentinel: stop the caller the moment the hop under test has been made,
    so no test ever needs a ParCa cache or a simulation."""


def test_the_CLI_hands_the_parsed_basis_to_make_run_one(monkeypatch, tmp_path):
    """Hop: main() -> make_run_one(exchange_flux_basis=...). The runner's flag
    reaches the process, argparse parses it, and then it has to be forwarded —
    a dropped keyword here means every arm of every study runs on counts while
    both YAML and command line say gdcw."""
    import scripts.run_comparison_ensemble as rce
    import v2ecoli.library.parallel_seeds as ps
    seen = {}

    def _fake_make_run_one(**kw):
        seen.update(kw)
        return lambda seed: {"seed": seed}

    monkeypatch.setattr(rce, "make_run_one", _fake_make_run_one)
    monkeypatch.setattr(ps, "run_seeds_parallel",
                        lambda seeds, run_one, **kw: [])
    rce.main(["--composite", "v2ecoli", "--condition", "basal",
              "--cache-dir", str(tmp_path), "--n-seeds", "1",
              "--out-root", str(tmp_path), "--mode", "serial",
              "--exchange-flux", "product_exchange=X[c]",
              "--exchange-flux-basis", "gdcw"])
    assert seen["exchange_flux_basis"] == "gdcw"
    # the map has to arrive too — the basis is meaningless without it
    assert seen["exchange_fluxes"] == {"product_exchange": "X[c]"}


def test_the_CLI_defaults_the_basis_to_counts(monkeypatch, tmp_path):
    """So the test above cannot pass against a hardcoded 'gdcw'."""
    import scripts.run_comparison_ensemble as rce
    import v2ecoli.library.parallel_seeds as ps
    seen = {}
    monkeypatch.setattr(rce, "make_run_one",
                        lambda **kw: (seen.update(kw), (lambda s: {}))[1])
    monkeypatch.setattr(ps, "run_seeds_parallel",
                        lambda seeds, run_one, **kw: [])
    rce.main(["--composite", "v2ecoli", "--condition", "basal",
              "--cache-dir", str(tmp_path), "--n-seeds", "1",
              "--out-root", str(tmp_path), "--mode", "serial"])
    assert seen["exchange_flux_basis"] == "counts"


def _run_one(monkeypatch, out_root, **kw):
    import scripts.run_comparison_ensemble as rce
    base = dict(composite_kind="v2ecoli", condition="basal", cache_dir="c",
                max_generations=1, max_steps=1, chunk=1, out_root=str(out_root))
    base.update(kw)
    return rce.make_run_one(**base)


def test_run_one_builds_the_CANDIDATE_with_the_declared_basis(monkeypatch, tmp_path):
    """Hop: make_run_one's closure -> _build_v2ecoli(exchange_flux_basis=...).
    Captured at the call, not at the signature: a caller that stopped passing
    the keyword would still satisfy any signature check while every candidate
    run silently reverted to counts."""
    import scripts.run_comparison_ensemble as rce
    seen = {}

    def _fake_build(seed, condition, cache_dir, overrides=None,
                    exchange_fluxes=None, exchange_flux_basis=None):
        seen.update(basis=exchange_flux_basis, fluxes=exchange_fluxes)
        raise _Stop()

    monkeypatch.setattr(rce, "_build_v2ecoli", _fake_build)
    run_one = _run_one(monkeypatch, tmp_path,
                       exchange_fluxes={"product_exchange": "X[c]"},
                       exchange_flux_basis="gdcw")
    with pytest.raises(_Stop):
        run_one(0)
    assert seen == {"basis": "gdcw", "fluxes": {"product_exchange": "X[c]"}}


def test_run_one_runs_the_REFERENCE_engine_on_the_declared_basis_and_records_it(
        monkeypatch, tmp_path):
    """Two hops on the reference arm in one execution, because they must agree:
    the basis reaches the wrapped-vEcoli engine, AND the sidecar written beside
    that run records the same value. A sidecar written from a different source
    than the engine ran on is exactly the disagreement this design removed.

    The assertion goes through the card's REAL reader, so the file name and key
    are not re-typed here."""
    from v2ecoli.library import vivarium_ecoli_engine as vee
    from scripts._compare.exchange_flux_basis import basis_from_runs
    seen = {}

    def _fake_engine(**kw):
        seen.update(kw)
        return {"generations": 1, "build_config": None}

    monkeypatch.setattr(vee, "run_vivarium_ecoli_pbg_multigen", _fake_engine)
    run_one = _run_one(monkeypatch, tmp_path, composite_kind="vecoli",
                       exchange_fluxes={"product_exchange": "X[c]"},
                       exchange_flux_basis="gdcw")
    run_one(0)
    assert seen["exchange_flux_basis"] == "gdcw"
    # ...and the run said so on disk, in the file the card reads. The candidate
    # arm's sidecar stands in (written by the same production writer) so the
    # reader has the pair it requires; the reference half is the one under test.
    _write_real_sidecar(tmp_path, "v2ecoli", "gdcw")
    basis, why = basis_from_runs({"v2_dir": str(tmp_path),
                                       "ve_dir": str(tmp_path)})
    assert (basis, why) == ("gdcw", ""), why


def test_run_one_records_the_CANDIDATE_arm_basis_beside_its_stores(
        monkeypatch, tmp_path):
    """The same contract on the other arm: the v2ecoli sidecar is written from
    inside run_one, in the same best-effort try/except as the build-config
    sidecar — so a failure there is PRINTED AND SWALLOWED, and the only visible
    symptom is a card that refuses every axis on an otherwise healthy run.

    Nothing executed that block before, which is why a broken call there could
    survive review."""
    import scripts.run_comparison_ensemble as rce
    from v2ecoli.library import xarray_run
    from scripts._compare.exchange_flux_basis import basis_from_runs

    monkeypatch.setattr(rce, "_build_v2ecoli",
                        lambda *a, **k: SimpleNamespace(state={}))
    monkeypatch.setattr(rce, "extract_v2_build_config",
                        lambda *a, **k: {"n_processes": 0})
    monkeypatch.setattr(xarray_run, "run_multigen_xarray",
                        lambda *a, **kw: {"steps": 1, "generations": [1]})
    run_one = _run_one(monkeypatch, tmp_path,
                       exchange_fluxes={"product_exchange": "X[c]"},
                       exchange_flux_basis="gdcw")
    run_one(0)
    _write_real_sidecar(tmp_path, "vecoli", "gdcw")     # the other arm stands in
    basis, why = basis_from_runs({"v2_dir": str(tmp_path),
                                       "ve_dir": str(tmp_path)})
    assert (basis, why) == ("gdcw", ""), why


def test_build_v2ecoli_hands_the_basis_to_build_composite(monkeypatch):
    """Hop: _build_v2ecoli -> build_composite("ecoli_baseline",
    exchange_flux_basis=...). The last hop before the composite generator; a
    drop here is invisible because the leaves are still emitted, just carrying
    the other quantity."""
    import v2ecoli
    from scripts.run_comparison_ensemble import _build_v2ecoli
    seen = {}

    def _fake_build_composite(name, **kwargs):
        seen["name"] = name
        seen.update(kwargs)
        return object()          # no .state -> the media assertion is skipped

    monkeypatch.setattr(v2ecoli, "build_composite", _fake_build_composite)
    # condition="" short-circuits the per-condition ParCa regen, so this needs
    # no cache on disk.
    _build_v2ecoli(0, "", "out/cache",
                   exchange_fluxes={"product_exchange": "X[c]"},
                   exchange_flux_basis="gdcw")
    assert seen["name"] == "ecoli_baseline"
    assert seen["exchange_flux_basis"] == "gdcw"
    assert seen["exchange_fluxes"] == {"product_exchange": "X[c]"}


def test_build_v2ecoli_declares_no_basis_when_no_fluxes_are_declared(monkeypatch):
    """An undeclared study must build the composite exactly as before — no new
    keyword — so enabling this feature cannot change a baseline run."""
    import v2ecoli
    from scripts.run_comparison_ensemble import _build_v2ecoli
    seen = {}
    monkeypatch.setattr(v2ecoli, "build_composite",
                        lambda name, **kw: (seen.update(kw), object())[1])
    _build_v2ecoli(0, "", "out/cache")
    assert "exchange_flux_basis" not in seen and "exchange_fluxes" not in seen


# --- the composite: baseline() -> the module-level override -----------------

def _stub_bundle():
    """The smallest bundle baseline() will accept. Passing `bundle=` skips
    load_cache_bundle entirely, so this needs no ParCa cache (and cannot be
    invalidated by one going stale)."""
    return {"initial_state": {"environment": {"media_id": "minimal"}},
            "configs": {}, "unique_names": [], "dry_mass_inc_dict": {}}


def _baseline_basis_override(monkeypatch, **kw):
    """Run the REAL baseline() with its step-building loop stubbed out, and
    return every value it pushed through set_exchange_flux_basis_override."""
    from v2ecoli.composites import ecoli_baseline as eb
    seen = []
    monkeypatch.setattr(eb, "set_exchange_flux_basis_override", seen.append)
    monkeypatch.setattr(eb, "_get_step_config", lambda *a, **k: None)
    eb.baseline(bundle=_stub_bundle(), emitter="null", **kw)
    return seen


def test_baseline_pushes_the_declared_basis_onto_the_build_override(monkeypatch):
    """⚠ THE HIGHEST-VALUE HOP. baseline() threads the basis to the deriver
    through a module-level override; deleting that ONE line reverts the whole
    feature to counts with no error, no warning and no missing leaf — the run
    completes and every number is a different quantity than the card believes.

    Executes the real baseline() (with a synthetic bundle and the step loop
    stubbed, so it takes ~1s and needs no cache) and reads the value the
    override actually received."""
    seen = _baseline_basis_override(
        monkeypatch, exchange_fluxes={"product_exchange": "X[c]"},
        exchange_flux_basis="gdcw")
    assert seen and seen[0] == "gdcw", (
        "baseline() did not push the declared basis onto the exchange-flux "
        "override, so the deriver is built on the default")


def test_baseline_restores_the_override_so_it_cannot_leak_to_a_later_build(monkeypatch):
    """The other half of the same line's contract: the override is process-wide,
    so a build that left 'gdcw' set would silently re-base the NEXT composite
    built in the same process (a sweep, a daughter, a test)."""
    seen = _baseline_basis_override(
        monkeypatch, exchange_fluxes={"product_exchange": "X[c]"},
        exchange_flux_basis="gdcw")
    assert seen[-1] is None, f"override not restored: {seen}"


def test_baseline_resolves_an_undeclared_basis_to_counts(monkeypatch):
    """So the test above cannot pass against a hardcoded literal, and an
    undeclared build is pinned to the unchanged behaviour."""
    seen = _baseline_basis_override(monkeypatch)
    assert seen and seen[0] == "counts"


# --- surviving division: the composite -> Division -> the daughter ----------

class _DivisionLoader(_StubLoader):
    """The attributes _get_special_step's 'division' branch reads, on top of the
    unique-molecule names it reads for every step."""

    def __init__(self):
        super().__init__()
        self.sim_data.expectedDryMassIncreaseDict = {}
        self.unique_names = []
        self.cache_dir = "out/cache"

    def get_config_by_name(self, name):
        return {}


def _division_step(fluxes, basis):
    from v2ecoli.composites import _helpers
    from v2ecoli.core import build_core
    core = build_core()
    _helpers.set_exchange_fluxes_override(fluxes)
    _helpers.set_exchange_flux_basis_override(basis)
    try:
        instance, _topo, _kind = _helpers._get_special_step(
            _DivisionLoader(), "division", core)
    finally:
        _helpers.set_exchange_fluxes_override({})
        _helpers.set_exchange_flux_basis_override(None)
    return instance


def test_the_division_step_is_built_carrying_the_basis(monkeypatch):
    """Two hops, executed together because neither is worth anything alone:
    _helpers' division branch must put the basis in div_config, and
    Division.initialize must lift it off self.parameters. Read off the built
    INSTANCE, so setting the key under a different name, or holding it in
    parameters without lifting it, both red."""
    step = _division_step({"product_exchange": "X[c]"}, "gdcw")
    assert getattr(step, "_exchange_flux_basis", None) == "gdcw"


def test_the_division_step_carries_counts_too(monkeypatch):
    """Not a hardcoded 'gdcw' — and 'counts' is the value a daughter reverts to
    when the hop is severed, so it must be carried explicitly rather than
    arrived at by accident."""
    step = _division_step({"product_exchange": "X[c]"}, "counts")
    assert getattr(step, "_exchange_flux_basis", None) == "counts"


def test_no_declared_fluxes_leaves_the_division_step_untouched(monkeypatch):
    """A baseline study's division step must be built exactly as before."""
    step = _division_step({}, "gdcw")
    assert getattr(step, "_exchange_flux_basis", None) is None


def test_a_dividing_cell_rebuilds_BOTH_daughters_on_the_declared_basis(monkeypatch):
    """The regression that shipped for the sibling field, pinned by EXECUTION:
    drive the real Division.next_update through a real division event with
    baseline() captured, and read the basis off both daughter rebuild calls.

    A daughter rebuilt without it reverts to counts and emits a lineage-
    cumulative running total under a leaf the card reads as mmol/gDCW/h — while
    dry mass, growth and division all look normal. Generation 1 carries the
    declared quantity and every later generation carries the other one.

    The sibling test for `exchange_fluxes` checks this against module SOURCE
    TEXT; this runs the code instead, so a rebuild that passes the wrong
    variable, or passes it only to one of the two daughters, is caught too."""
    import numpy as np
    from v2ecoli.steps import division as division_mod
    from v2ecoli.composites import ecoli_baseline as eb
    from v2ecoli.library import division as division_lib

    step = object.__new__(division_mod.Division)
    step.core = None
    division_mod.Division.initialize(step, {
        "exchange_fluxes": {"product_exchange": "X[c]"},
        "exchange_flux_basis": "gdcw",
    })

    bulk = np.zeros(1, dtype=[("count", "i8")])
    daughter_state = {"bulk": bulk, "unique": {}, "environment": {},
                      "boundary": {}}
    monkeypatch.setattr(division_lib, "divide_cell",
                        lambda cell: (dict(daughter_state), dict(daughter_state)))

    rebuilds = []
    monkeypatch.setattr(eb, "baseline", lambda **kw: (
        rebuilds.append(kw),
        {"state": {"agents": {"0": {"listeners": {}}}}})[1])
    monkeypatch.setattr(eb, "seed_mass_listener", lambda *a, **k: None)

    chromosomes = np.zeros(2, dtype=[("_entryState", "i8")])
    chromosomes["_entryState"] = 1
    update = step.next_update(1.0, {
        "bulk": bulk, "unique": {"full_chromosome": chromosomes},
        "listeners": {"mass": {"dry_mass": 500.0}}, "environment": {},
        "boundary": {}, "global_time": 100.0, "divide": True})

    assert len(update["agents"]["_add"]) == 2, "no division happened"
    assert len(rebuilds) == 2, "both daughters must be rebuilt"
    assert [r.get("exchange_flux_basis") for r in rebuilds] == ["gdcw", "gdcw"]
    # the map has to travel with it, or the daughter declares no leaves at all
    assert [r.get("exchange_fluxes") for r in rebuilds] == \
        [{"product_exchange": "X[c]"}] * 2


# --- the gdcw CALL SITE against real composite value types -------------------
#
# ⚠ The helper test above (`..._tolerates_pint_quantities_...`) asserts on
# `_as_float_fg` in ISOLATION. That is the "covers the helper, not the call
# site" failure this lane has already catalogued: reverting `update()` to a bare
# `float(...)` — i.e. restoring the crash the coercion exists to fix — left the
# whole suite green (measured: 111 pass across five files). These execute the
# Step's real `update()` with the value TYPE the composite supplies.

def _gdcw_step(fluxes):
    """The real Step on the gdcw basis, built the way the composite builds it."""
    from v2ecoli.core import build_core
    from v2ecoli.steps.derivers.exchange_flux_listener import ExchangeFluxListener
    return ExchangeFluxListener({"fluxes": fluxes, "basis": "gdcw"},
                                core=build_core())


def _fg(x):
    """A dry mass as the real composite carries it: a pint Quantity in fg."""
    from v2ecoli.types.quantity import ureg
    return x * ureg.fg


def test_gdcw_update_survives_a_pint_dry_mass_at_the_CALL_SITE():
    """⚠ Regression for the crash that took down the first real gdcw run.

    `listeners.mass.dry_mass` is a `quantity[float,fg]` on the composite while
    this Step declares it a bare `float`, so `update()` receives a Quantity. A
    bare `float()` on it raises DimensionalityError on the FIRST tick.

    Asserts through `update()` rather than through the coercion helper, because
    the helper was already covered and the call site was not — reverting only
    the call site reproduced the shipped crash with every test still green."""
    step = _gdcw_step({"product_exchange": "X[c]"})
    common = {"global_time": 0.0, "timestep": 1.0, "mass": {"dry_mass": _fg(400.0)}}
    step.update({"exchange": {"X[c]": 1.0e6}, **common})       # priming tick
    out = step.update({"exchange": {"X[c]": 3.0e6}, **common})
    rate = out["listeners"]["exchange_flux"]["product_exchange"]
    assert rate > 0, "a secretion on the gdcw basis must report a positive rate"
    assert 1e-3 < rate < 1e3, f"rate {rate} is outside any physiological band"


def test_gdcw_reads_a_pint_dry_mass_IN_FEMTOGRAMS_not_by_bare_magnitude():
    """⚠ The same mass in a different unit must give the SAME rate.

    Taking `.magnitude` off a Quantity verbatim is only correct while the
    Quantity happens to be in fg — nothing in this Step's port contract says it
    is, since the port declares a bare `float`. A picogram-valued Quantity read
    by bare magnitude yields 0.4 where 400.0 is meant, so the rate comes out
    1000x too large with no error: a silently wrong QUANTITY under a correct
    name, which is the failure the basis exists to remove."""
    from v2ecoli.types.quantity import ureg

    def _rate(dry_mass):
        step = _gdcw_step({"product_exchange": "X[c]"})
        common = {"global_time": 0.0, "timestep": 1.0,
                  "mass": {"dry_mass": dry_mass}}
        step.update({"exchange": {"X[c]": 1.0e6}, **common})
        return step.update({"exchange": {"X[c]": 3.0e6},
                            **common})["listeners"]["exchange_flux"]["product_exchange"]

    in_fg = _rate(400.0 * ureg.fg)
    in_pg = _rate((400.0 * ureg.fg).to(ureg.pg))     # the SAME mass
    assert in_fg == pytest.approx(in_pg, rel=1e-9), (
        f"same dry mass, different unit, different rate: fg->{in_fg} pg->{in_pg}; "
        "the magnitude is being read without converting")


def test_an_uncoercible_dry_mass_is_no_rate_rather_than_a_crash():
    """The tolerance the conversion must not lose: a value that is neither a
    number nor a mass Quantity yields 0.0 ("no rate is defined"), not an
    exception that takes a completed run down at the last tick."""
    step = _gdcw_step({"product_exchange": "X[c]"})
    common = {"global_time": 0.0, "timestep": 1.0,
              "mass": {"dry_mass": "not a mass"}}
    step.update({"exchange": {"X[c]": 1.0e6}, **common})
    out = step.update({"exchange": {"X[c]": 3.0e6}, **common})
    assert out["listeners"]["exchange_flux"]["product_exchange"] == 0.0


# --- the sidecar's RUN SHAPE, through the production writer ------------------
#
# ⚠ GAP MEASURED: `_write_real_sidecar` above calls the writer without
# `seeds=`/`generations=`, so nothing exercised the run-shape fields at all.
# Deleting both from `_write_exchange_flux_sidecar` left 42 tests green — the
# staleness guard added alongside them could be removed silently, because the
# reader skips its check whenever either side is None.

def _write_real_sidecar_with_shape(out_root, prefix, basis, seeds, generations):
    """The PRODUCTION writer, exercising the run-shape keyword arguments."""
    from scripts.run_comparison_ensemble import _write_exchange_flux_sidecar
    _write_exchange_flux_sidecar(str(out_root), prefix, {"product_exchange": "X[c]"},
                                 basis, seeds=seeds, generations=generations)


def test_the_writer_records_the_run_shape_the_reader_compares(tmp_path):
    """Round trip: the real writer's run-shape fields reach the real reader.

    Spelled through both production functions rather than a hand-rolled dict,
    because re-typing the contract in the test is how a rename stays green here
    and refuses on every real run."""
    import json
    from scripts._compare.exchange_flux_basis import basis_from_runs
    _write_real_sidecar_with_shape(tmp_path, "v2ecoli", "gdcw", 4, 8)
    _write_real_sidecar_with_shape(tmp_path, "vecoli", "gdcw", 4, 8)

    doc = json.loads((tmp_path / "v2ecoli_exchange_flux.json").read_text())
    assert doc.get("seeds") == 4 and doc.get("generations") == 8, (
        f"the writer did not record the run shape: {doc}")
    basis, why = basis_from_runs({"v2_dir": str(tmp_path), "ve_dir": str(tmp_path)})
    assert (basis, why) == ("gdcw", ""), (basis, why)


def test_two_arms_with_different_run_shapes_are_refused_as_stale(tmp_path):
    """⚠ The guard the fields exist for. Both arms write into ONE out_root and
    nothing cleans it, so re-running one arm leaves the other's sidecar and
    stores in place. Two sidecars agreeing on 'gdcw' would otherwise pass the
    agreement check while describing different invocations."""
    from scripts._compare.exchange_flux_basis import basis_from_runs
    v2, ve = tmp_path / "v2", tmp_path / "ve"
    v2.mkdir(), ve.mkdir()
    _write_real_sidecar_with_shape(v2, "v2ecoli", "gdcw", seeds=4, generations=8)
    _write_real_sidecar_with_shape(ve, "vecoli", "gdcw", seeds=1, generations=1)

    basis, why = basis_from_runs({"v2_dir": str(v2), "ve_dir": str(ve)})
    assert basis is None, "a stale arm was accepted as a matching run"
    assert "stale" in why and "seeds=4" in why and "seeds=1" in why, why


def test_matching_run_shapes_are_not_refused(tmp_path):
    """So the test above cannot pass by refusing everything."""
    from scripts._compare.exchange_flux_basis import basis_from_runs
    v2, ve = tmp_path / "v2b", tmp_path / "veb"
    v2.mkdir(), ve.mkdir()
    _write_real_sidecar_with_shape(v2, "v2ecoli", "gdcw", seeds=4, generations=8)
    _write_real_sidecar_with_shape(ve, "vecoli", "gdcw", seeds=4, generations=8)
    assert basis_from_runs({"v2_dir": str(v2), "ve_dir": str(ve)}) == ("gdcw", "")


# --- the REFERENCE arm's in-engine chain ------------------------------------
#
# ⛔ MEASURED GAP: five consecutive hops here survived mutation, and no test
# anywhere in tests/ called `VivariumEcoliProcess.update()`. Coverage stopped at
# `make_run_one -> run_vivarium_ecoli_pbg_multigen(exchange_flux_basis=...)` and
# resumed only at the pure helper `_select_exchange_fluxes`. Everything between
# was untested, which is the half of this change that is actually new — the
# reference arm READING the wrapped metabolism's own rate instead of deriving.
#
# The sharpest of them: deleting `listeners=obs.get("listeners")` from the call
# site killed nothing. The helper cannot tell "listener not populated yet" from
# "listeners never wired at all", so that one keyword argument is the whole
# guard against a flat-zero gdcw trace on the reference arm.

_FBA_RATE = 0.129129     # a real measured mmol/gDCW/h product secretion


def _wrapped_process(**cfg):
    """A VivariumEcoliProcess with no real EcoliSim behind it: the pending-handle
    branch skips the sim_data build, exactly as test_exchange_flux_observables
    already does for outputs()."""
    from v2ecoli.core import build_core
    from v2ecoli.library.vivarium_ecoli_engine import VivariumEcoliProcess
    VivariumEcoliProcess._PENDING_HANDLE = object()
    try:
        return VivariumEcoliProcess(config=cfg, core=build_core())
    finally:
        VivariumEcoliProcess._PENDING_HANDLE = None


def _fake_observables(**listeners):
    """What cell_observables() returns, with the seven scalar axes stubbed."""
    from v2ecoli.library.vivarium_ecoli_engine import COUNT_OBS, MASS_OBS
    obs = {k: 1.0 for k in MASS_OBS}
    obs.update({k: 1.0 for k in COUNT_OBS})
    obs["environment"] = {"exchange": {"GLC[p]": -5.0e7, "X[c]": 9.9e9}}
    obs["listeners"] = listeners
    return obs


def _run_one_tick(monkeypatch, proc):
    """Drive the REAL update(), with the engine and observables stubbed."""
    import v2ecoli.library.vivarium_ecoli_engine as eng

    class _Eng:
        def run_for(self, _):
            return None

    monkeypatch.setattr(proc, "_handle", type("H", (), {"engine": _Eng()})())
    monkeypatch.setattr(eng, "cell_observables", lambda _e: _fake_observables(
        fba_results={"external_exchange_fluxes": {"X[c]": _FBA_RATE,
                                                  "GLC[p]": -7.43}}))
    return proc.update({}, 1.0)["listeners"]["exchange_flux"]


def test_the_wrapped_process_reads_the_FBA_RATE_on_gdcw(monkeypatch):
    """⚠ The hop nothing executed. On gdcw the reference arm must report the
    wrapped metabolism's own listener value — NOT the cumulative counts store,
    which the same stubbed observables also carry (9.9e9 vs 0.129).

    Deleting `basis=` or `listeners=` from the call site both silently produce
    the wrong one of those two numbers, and both were green."""
    proc = _wrapped_process(exchange_fluxes={"product_exchange": "X[c]"},
                            exchange_flux_basis="gdcw")
    leaves = _run_one_tick(monkeypatch, proc)
    assert leaves["product_exchange"] == pytest.approx(_FBA_RATE), (
        f"expected the fba_results rate {_FBA_RATE}, got "
        f"{leaves['product_exchange']} — if this is 9.9e9 the call site dropped "
        "basis=; if it is 0.0 it dropped listeners=")


def test_the_wrapped_process_reads_the_COUNTS_STORE_on_counts(monkeypatch):
    """The discriminating other half: the same stubbed tick on the default basis
    must give the cumulative store's number, so the test above cannot pass
    against a process that ignores the basis and always reads one source."""
    proc = _wrapped_process(exchange_fluxes={"product_exchange": "X[c]"},
                            exchange_flux_basis="counts")
    leaves = _run_one_tick(monkeypatch, proc)
    assert leaves["product_exchange"] == pytest.approx(9.9e9)


def test_the_wrapped_process_honours_an_UNDECLARED_basis_as_counts(monkeypatch):
    """So neither test above can pass against a hardcoded literal."""
    proc = _wrapped_process(exchange_fluxes={"product_exchange": "X[c]"})
    leaves = _run_one_tick(monkeypatch, proc)
    assert leaves["product_exchange"] == pytest.approx(9.9e9)


def test_the_composite_builder_lands_the_basis_on_the_process(monkeypatch):
    """Hop: build_vivarium_ecoli_composite(exchange_flux_basis=) -> the process
    config. Captured at construction, so a builder that stopped passing it —
    or hardcoded 'counts' into the config dict — is caught."""
    import v2ecoli.library.vivarium_ecoli_engine as eng
    seen = {}

    class _Stop(Exception):
        pass

    class _Spy(eng.VivariumEcoliProcess):
        def __init__(self, config=None, core=None):
            seen.update(config or {})
            raise _Stop()

    monkeypatch.setattr(eng, "build_vivarium_ecoli", lambda **kw: object())
    monkeypatch.setattr(eng, "VivariumEcoliProcess", _Spy)
    with pytest.raises(_Stop):
        eng.build_vivarium_ecoli_composite(
            sim_data_path="x", condition="basal", seed=0,
            exchange_fluxes={"product_exchange": "X[c]"},
            exchange_flux_basis="gdcw")
    assert seen.get("exchange_flux_basis") == "gdcw", seen
    assert seen.get("exchange_fluxes") == {"product_exchange": "X[c]"}


# --- B7b: the DECLARED defaults, pinned directly -----------------------------
#
# ⛔ MEASURED GAP: flipping `StudySpec.exchange_flux_basis` or
# `_helpers._EXCHANGE_FLUX_BASIS_OVERRIDE` from "counts" to "gdcw" left the whole
# suite green. Every "the default is counts" test reached its answer through an
# explicit `or "counts"` on some OTHER code path, so the declarations themselves
# were never under test. They are the PR's "every existing study is byte-unchanged"
# claim: flip either and every undeclared run silently changes quantity.

def test_the_study_spec_DECLARES_counts_as_its_default():
    from scripts._compare.study_spec import StudySpec
    import dataclasses
    field = {f.name: f for f in dataclasses.fields(StudySpec)}["exchange_flux_basis"]
    assert field.default == "counts", (
        f"StudySpec declares {field.default!r}; a study that says nothing would "
        "run on that quantity")


def test_the_composite_override_DECLARES_counts_as_its_default():
    """Read after a reload so a leaked value from another test cannot supply the
    answer — the failure mode the sibling default test above shipped with."""
    import importlib

    from v2ecoli.composites import _helpers
    importlib.reload(_helpers)
    assert _helpers._EXCHANGE_FLUX_BASIS_OVERRIDE == "counts"


def test_restoring_the_override_returns_it_to_counts_not_to_whatever_was_set():
    """`set_exchange_flux_basis_override(None)` is the restore path baseline() runs
    in its `finally`. If None resolved to anything but counts, a single gdcw build
    would re-base every later composite in the same process."""
    from v2ecoli.composites import _helpers
    _helpers.set_exchange_flux_basis_override("gdcw")
    _helpers.set_exchange_flux_basis_override(None)
    assert _helpers._EXCHANGE_FLUX_BASIS_OVERRIDE == "counts"


# --- B5: the staleness guard's OWN INPUT, through the real CLI ---------------
#
# ⛔ MEASURED GAP: nothing checked that the real `--n-seeds` reaches the sidecar.
# Dropping `n_seeds=args.n_seeds` from main()'s make_run_one call left the suite
# green. If it failed, every sidecar would record seeds=1, BOTH arms would agree,
# and the staleness guard would never fire on any run — silently inert, which is
# the exact shape that already bit this feature twice (`seeds=len(seeds)` raising
# NameError into a best-effort guard, and before that the sidecar never written).
#
# Driven end to end: the real argparse, the real make_run_one, the real writer,
# read back through the real reader. Only the simulation is stubbed.

def test_the_REAL_cli_n_seeds_reaches_the_sidecar(monkeypatch, tmp_path):
    """`--n-seeds 4` must arrive in the sidecar as seeds=4, not as the default 1."""
    import scripts.run_comparison_ensemble as rce
    from v2ecoli.library import parallel_seeds as ps

    seen = {}
    real_make_run_one = rce.make_run_one

    def _spy(**kw):
        seen.update(kw)
        # Build the real closure, then write the sidecar the way run_one does,
        # without running a simulation.
        real_make_run_one(**kw)
        rce._write_exchange_flux_sidecar(
            kw["out_root"], "v2ecoli", kw.get("exchange_fluxes") or {},
            kw.get("exchange_flux_basis") or "counts",
            seeds=kw.get("n_seeds"), generations=kw.get("max_generations"))
        rce._write_exchange_flux_sidecar(
            kw["out_root"], "vecoli", kw.get("exchange_fluxes") or {},
            kw.get("exchange_flux_basis") or "counts",
            seeds=kw.get("n_seeds"), generations=kw.get("max_generations"))
        return lambda s: {}

    monkeypatch.setattr(rce, "make_run_one", _spy)
    monkeypatch.setattr(ps, "run_seeds_parallel", lambda seeds, run_one, **kw: [])
    rce.main(["--composite", "v2ecoli", "--condition", "basal",
              "--cache-dir", str(tmp_path), "--n-seeds", "4",
              "--max-generations", "3", "--out-root", str(tmp_path),
              "--mode", "serial", "--exchange-flux", "product_exchange=X[c]",
              "--exchange-flux-basis", "gdcw"])

    assert seen.get("n_seeds") == 4, (
        f"main() forwarded n_seeds={seen.get('n_seeds')!r} — the staleness guard "
        "would compare a constant and never fire")

    from scripts._compare.exchange_flux_basis import basis_from_runs
    import json
    doc = json.loads((tmp_path / "v2ecoli_exchange_flux.json").read_text())
    assert (doc["seeds"], doc["generations"]) == (4, 3), doc
    assert basis_from_runs({"v2_dir": str(tmp_path),
                            "ve_dir": str(tmp_path)}) == ("gdcw", "")


# --------------------------------------------------------------------------- #
# THE VARIANT'S HOPS — the ones a declaration-and-argv test cannot see.
#
# ⛔ WHY THESE EXIST. `--variant` was added with coverage at YAML -> StudySpec and
# at StudySpec -> argv, and none between argv and the engine. Reverting the fix
# outright (`variant=variant` back to `variant=0` at the engine call) left the
# whole suite GREEN. The bug the flag was written to fix lived in exactly the
# untested hop: the value arrived at the engine and was then discarded, because
# `apply_variant` runs only on the whole-config route and the driving config did
# not auto-enable it.
# --------------------------------------------------------------------------- #
def test_the_CLI_hands_the_parsed_variant_to_make_run_one(monkeypatch, tmp_path):
    """Hop 1: main() -> make_run_one(variant=...)."""
    import scripts.run_comparison_ensemble as rce
    import v2ecoli.library.parallel_seeds as ps
    seen = {}
    monkeypatch.setattr(rce, "make_run_one",
                        lambda **kw: (seen.update(kw), (lambda s: {}))[1])
    monkeypatch.setattr(ps, "run_seeds_parallel", lambda seeds, run_one, **kw: [])
    rce.main(["--composite", "vecoli", "--condition", "basal", "--variant", "2",
              "--cache-dir", str(tmp_path), "--n-seeds", "1",
              "--out-root", str(tmp_path), "--mode", "serial"])
    assert seen["variant"] == 2


def test_run_one_hands_the_variant_to_the_REFERENCE_engine(monkeypatch, tmp_path):
    """⭐ Hop 2: make_run_one's closure -> run_vivarium_ecoli_pbg_multigen(variant=...).

    THE HOP THAT WAS MISSING. Captured at the call, because a caller that stops
    forwarding the keyword still satisfies every signature check while every
    reference run silently reverts to the unperturbed baseline.
    """
    import scripts.run_comparison_ensemble as rce
    from v2ecoli.library import vivarium_ecoli_engine as vee
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(vee, "run_vivarium_ecoli_pbg_multigen", _fake)
    run_one = _run_one(monkeypatch, tmp_path, composite_kind="vecoli", variant=3)
    with pytest.raises(_Stop):
        run_one(0)
    assert seen["variant"] == 3, (
        "the reference arm was built on a different variant than was requested")


def test_a_requested_variant_that_CANNOT_be_applied_is_REFUSED(monkeypatch, tmp_path):
    """⛔⛔ THE INVARIANT, at the point of discard.

    Every `apply_variant` gate is `_cfgfile and int(variant)`, so without a
    whole-config the variant is threaded to the engine and then ignored — the run
    completes as the unperturbed baseline while its metadata records the variant
    as applied. Measured on a real config before the fix: variant=1,
    whole_config=None, apply_variant never called.

    ⚠ The refusal and the provenance stamp in `metadata_base` are ONE invariant in
    two places. If this test is ever deleted, that stamp starts lying.
    """
    from v2ecoli.library import vivarium_ecoli_engine as vee
    with pytest.raises(ValueError, match="silently discarded|whole-config"):
        vee.build_vivarium_ecoli(
            sim_data_path=str(tmp_path / "nope.cPickle"), condition="basal",
            seed=0, variant=1)


def test_variant_zero_needs_no_whole_config(monkeypatch, tmp_path):
    """The refusal is about a variant that would be DROPPED. Baseline drops
    nothing, so it must not trip it — otherwise every unvaried study breaks."""
    from v2ecoli.library import vivarium_ecoli_engine as vee
    with pytest.raises(Exception) as ei:
        vee.build_vivarium_ecoli(
            sim_data_path=str(tmp_path / "nope.cPickle"), condition="basal",
            seed=0, variant=0)
    assert "silently discarded" not in str(ei.value), (
        "variant 0 tripped the missing-whole-config refusal")


def test_a_declared_variant_AUTO_ENABLES_the_whole_config_route(monkeypatch, tmp_path):
    """⭐ The other half of the invariant: the refusal must not fire on a study
    that legitimately declares a variant.

    A config declaring `variants` alongside `swap_processes` — and neither
    `add_processes` nor `spatial_environment_config` — took the swap route, where
    the variant cannot be applied. Requesting one now auto-enables the route that
    can apply it, so the refusal above stays a genuine error rather than a wall.
    """
    import scripts.run_comparison_ensemble as rce
    from v2ecoli.library import vivarium_ecoli_engine as vee
    seen = {}

    # ⚠ The stub MUST declare `variants` — that is what selects the route. An
    # earlier stub omitted it and the route was keyed on the variant INDEX
    # instead, which this test could not see. ⊕ Only the `config_adapter` patch
    # binds: `make_run_one` imports the resolver INSIDE the function, so a
    # module-level patch on `rce` never takes effect (it needed `raising=False`
    # to not error, which is the tell). Removed.
    import scripts._compare.config_adapter as ca
    monkeypatch.setattr(ca, "resolve_vecoli_config_local",
                        lambda cfg, fork: {"swap_processes": {"a": "b"},
                                           "variants": {"some_pathway_shift": {}}})

    def _fake(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(vee, "run_vivarium_ecoli_pbg_multigen", _fake)
    run_one = _run_one(monkeypatch, tmp_path, composite_kind="vecoli", variant=1,
                       from_vecoli_config="configs/some_config.json")
    with pytest.raises(_Stop):
        run_one(0)
    assert seen.get("whole_config"), (
        "a requested variant did not enable the whole-config route, so "
        "apply_variant could never run")


def test_variant_ZERO_takes_the_SAME_ROUTE_as_a_variant_arm(monkeypatch, tmp_path):
    """⛔⛔ THE BASELINE ARM AND THE VARIANT ARM MUST BE THE SAME MODEL.

    `--variant 0` exists so a study can declare a DELIBERATE BASELINE reference
    arm. That arm is the CONTROL for the variant arm, so the only thing allowed to
    differ between them is what `apply_variant` does.

    A previous fix keyed the whole-config route on `int(variant or 0)` — so
    `variant 0` took the swap/flow route while `variant 1` took the native one.
    Those are NOT the same model: the native path carries
    `exclude_processes: ['exchange_data']`, which `build_vivarium_ecoli` MERGES
    with the caller's list, so `ExchangeData` — the Step that writes metabolism's
    uptake bounds — runs on one arm and not the other. Nothing in the zarr
    metadata, the sidecar or the card records which route ran, so a
    baseline-vs-variant comparison would silently confound the perturbation with a
    model change.

    ⇒ The route is selected by the CONFIG declaring `variants`, never by the index.
    """
    import scripts.run_comparison_ensemble as rce
    from v2ecoli.library import vivarium_ecoli_engine as vee
    import scripts._compare.config_adapter as ca
    monkeypatch.setattr(ca, "resolve_vecoli_config_local",
                        lambda cfg, fork: {"swap_processes": {"a": "b"},
                                           "variants": {"some_pathway_shift": {}}})
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(vee, "run_vivarium_ecoli_pbg_multigen", _fake)
    routes = {}
    for v in (0, 1):
        seen.clear()
        run_one = _run_one(monkeypatch, tmp_path, composite_kind="vecoli", variant=v,
                           from_vecoli_config="configs/some_config.json")
        with pytest.raises(_Stop):
            run_one(0)
        routes[v] = bool(seen.get("whole_config"))
    assert routes[0] == routes[1] is True, (
        f"baseline and variant arms took DIFFERENT routes: variant 0 native="
        f"{routes[0]}, variant 1 native={routes[1]} — the control is not the "
        f"same model as the treatment")


def test_a_config_that_DECLARES_variants_REFUSES_an_omitted_variant(monkeypatch, tmp_path):
    """⛔⛔ THE REFUSAL ITSELF, which had NO coverage in any form.

    Deleting `_declared_variants`' body, reverting its resolver, or removing the
    `p.error` branch entirely all passed the full suite. The guard is the only
    thing standing between "the study declared variants and the operator said
    nothing" and a reference arm that silently runs the unvaried strain.

    ⚠ It reads the config through the SAME resolver as the route decision. When
    it used a stricter loader instead, the two disagreed on 10 of 86 real fork
    configs — the route switching on variants the guard could not see — and the
    guard failed OPEN on exactly those.
    """
    import scripts.run_comparison_ensemble as rce
    import scripts._compare.config_adapter as ca
    monkeypatch.setattr(ca, "resolve_vecoli_config_local",
                        lambda cfg, fork: {"swap_processes": {"a": "b"},
                                           "variants": {"some_pathway_shift": {}}})
    with pytest.raises(SystemExit):
        rce.main(["--composite", "vecoli", "--condition", "basal",
                  "--cache-dir", str(tmp_path), "--n-seeds", "1",
                  "--from-vecoli-config", "configs/some_config.json",
                  "--out-root", str(tmp_path), "--mode", "serial"])


def test_a_config_with_NO_variants_needs_no_choice_and_stays_on_the_swap_route(
        monkeypatch, tmp_path):
    """⭐ THE NEGATIVE CASE, in both directions — the route rule's other corner.

    Making the route unconditional (`if True`), or dropping the
    `add_processes`/`spatial` trigger, or letting `--vecoli-whole-config off` be
    ignored, all passed the suite: only the positive corner was pinned. A config
    declaring NO variants must neither be refused nor switched to the native
    route, or every unvaried study on this harness silently changes model.
    """
    import scripts.run_comparison_ensemble as rce
    import scripts._compare.config_adapter as ca
    from v2ecoli.library import vivarium_ecoli_engine as vee
    monkeypatch.setattr(ca, "resolve_vecoli_config_local",
                        lambda cfg, fork: {"swap_processes": {"a": "b"}})
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(vee, "run_vivarium_ecoli_pbg_multigen", _fake)
    run_one = _run_one(monkeypatch, tmp_path, composite_kind="vecoli",
                       from_vecoli_config="configs/some_config.json")
    with pytest.raises(_Stop):
        run_one(0)
    assert not seen.get("whole_config"), (
        "a config declaring no variants was switched to the native route, which "
        "changes the process set")


def test_vecoli_whole_config_OFF_still_overrides_a_declared_variant(
        monkeypatch, tmp_path):
    """⛔ `off` is the operator's only escape back to the pre-branch swap route
    for a variants-declaring config. Letting `_needs_native` bypass the mode
    check passed the suite; nothing referenced `vecoli_whole_config` in tests."""
    import scripts._compare.config_adapter as ca
    from v2ecoli.library import vivarium_ecoli_engine as vee
    monkeypatch.setattr(ca, "resolve_vecoli_config_local",
                        lambda cfg, fork: {"swap_processes": {"a": "b"},
                                           "variants": {"some_pathway_shift": {}}})
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(vee, "run_vivarium_ecoli_pbg_multigen", _fake)
    run_one = _run_one(monkeypatch, tmp_path, composite_kind="vecoli", variant=0,
                       from_vecoli_config="configs/some_config.json",
                       vecoli_whole_config="off")
    with pytest.raises(_Stop):
        run_one(0)
    assert not seen.get("whole_config"), "--vecoli-whole-config off was ignored"


def test_a_NEGATIVE_variant_is_refused_at_the_CLI(monkeypatch, tmp_path):
    """⛔ TWO ENTRY POINTS MUST NOT DISAGREE ABOUT THE SAME VALUE.

    `variant_from_study_yaml` rejects a negative index. argparse did not — and
    `_select_variant_params` treats ANY index <= 0 as "baseline", so a typo'd
    `--variant -1` skipped the missing-variant guard (it is not None) and then
    silently ran the unperturbed model, on the one arm this whole flag exists to
    protect. A study declaring `-1` errors; a command line passing it did not.
    """
    import scripts.run_comparison_ensemble as rce
    with pytest.raises(SystemExit):
        rce.main(["--composite", "vecoli", "--condition", "basal",
                  "--variant", "-1", "--cache-dir", str(tmp_path),
                  "--n-seeds", "1", "--out-root", str(tmp_path),
                  "--mode", "serial"])
