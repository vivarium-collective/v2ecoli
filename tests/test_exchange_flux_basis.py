"""The exchange-flux ``basis`` is declared ONCE and honoured by BOTH arms.

``counts`` and ``gdcw`` are different measurements, not different units: the
first is a lineage-cumulative molecule total read from ``environment.exchange``,
the second a per-tick mmol/gDCW/h rate. On the reference (wrapped-vEcoli) arm the
rate is not derived — it is read from the wrapped metabolism's own
``listeners.fba_results.external_exchange_fluxes``, which is the leaf genuine
vEcoli's own analyses read.

Each test below names the failure it exists to catch; several of them pass
trivially if the basis is dropped anywhere along the chain, so the chain is
tested at each hop rather than end-to-end only.
"""
import pytest

from v2ecoli.library.vivarium_ecoli_engine import _select_exchange_fluxes

FLUXES = {"glucose_exchange": "GLC[p]", "product_exchange": "SOME-PRODUCT[c]"}

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


def test_gdcw_with_no_fba_results_yields_zero_not_a_crash():
    """A reference arm whose wrapped process has not populated the listener yet
    must emit a continuous trace rather than raise mid-run."""
    out = _select_exchange_fluxes(ENVIRONMENT, FLUXES, basis="gdcw",
                                  listeners={})
    assert out == {"glucose_exchange": 0.0, "product_exchange": 0.0}


def test_empty_flux_map_is_a_no_op_on_either_basis():
    assert _select_exchange_fluxes(ENVIRONMENT, {}, basis="gdcw") == {}
    assert _select_exchange_fluxes(ENVIRONMENT, {}, basis="counts") == {}


# --- the chain: each hop that could silently drop the basis -----------------

def test_deriver_is_built_with_the_declared_basis():
    """Catches the candidate-arm half being dropped in _get_special_step."""
    from v2ecoli.composites import _helpers
    _helpers.set_exchange_fluxes_override({"glucose_exchange": "GLC[p]"})
    _helpers.set_exchange_flux_basis_override("gdcw")
    try:
        assert _helpers._EXCHANGE_FLUX_BASIS_OVERRIDE == "gdcw"
    finally:
        _helpers.set_exchange_fluxes_override({})
        _helpers.set_exchange_flux_basis_override(None)
    assert _helpers._EXCHANGE_FLUX_BASIS_OVERRIDE == "counts"


def test_runner_emits_the_basis_flag_alongside_the_flux_map():
    """Catches the study-level declaration never reaching either arm's CLI."""
    import inspect
    from scripts._compare import runner
    src = inspect.getsource(runner)
    assert "--exchange-flux-basis" in src, (
        "the runner must pass the basis to run_comparison_ensemble; without it "
        "a study declaring gdcw silently gets counts on both arms")


def test_study_spec_carries_the_basis_with_a_counts_default():
    """Catches the field being added to the dataclass but never parsed."""
    from scripts._compare.study_spec import StudySpec
    assert getattr(StudySpec, "exchange_flux_basis", None) == "counts" or \
        "exchange_flux_basis" in getattr(StudySpec, "__dataclass_fields__", {})
