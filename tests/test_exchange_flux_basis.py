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


def test_gdcw_with_no_fba_results_yields_zero_not_a_crash():
    """A reference arm whose wrapped process has not populated the listener yet
    must emit a continuous trace rather than raise mid-run."""
    out = _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="gdcw",
                                  listeners={})
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
    whatever the previous build left in the module global."""
    from v2ecoli.composites import _helpers
    from v2ecoli.core import build_core
    core = build_core()
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


def test_runner_emits_the_basis_flag_alongside_the_flux_map():
    """Catches the study-level declaration never reaching either arm's CLI."""
    import inspect
    from scripts._compare import runner
    src = inspect.getsource(runner)
    assert "--exchange-flux-basis" in src, (
        "the runner must pass the basis to run_comparison_ensemble; without it "
        "a study declaring gdcw silently gets counts on both arms")


def test_study_yaml_declaration_survives_the_investigation_route(tmp_path):
    """A study.yaml declaring the basis must win on BOTH spec routes. The
    investigation route builds from configs[] entries, which carry no study.yaml
    keys — so a declaration in the file the investigation NAMES would otherwise
    be silently ignored. The previous version of this test only inspected the
    dataclass field, so deleting both parse hunks left it green."""
    from scripts._compare.study_spec import exchange_flux_basis_from_study_yaml
    y = tmp_path / "study.yaml"
    y.write_text("comparison:\n  exchange_flux_basis: gdcw\n", encoding="utf-8")
    assert exchange_flux_basis_from_study_yaml(y, fallback="counts") == "gdcw"


def test_investigation_fallback_applies_when_the_study_is_silent(tmp_path):
    """And an investigation-level declaration still wins where the study says
    nothing — the bridge must not clobber the fallback with its own default."""
    from scripts._compare.study_spec import exchange_flux_basis_from_study_yaml
    y = tmp_path / "study.yaml"
    y.write_text("comparison:\n  seeds: 4\n", encoding="utf-8")
    assert exchange_flux_basis_from_study_yaml(y, fallback="gdcw") == "gdcw"
    assert exchange_flux_basis_from_study_yaml(
        tmp_path / "missing.yaml", fallback="gdcw") == "gdcw"
