"""Candidate-arm exchange-flux listener: re-homes named environment.exchange
fluxes onto listeners.exchange_flux.<name> leaves so the listeners-only compact
view carries them. Generic — the flux map is config; no pathway is special-cased.
"""
import pytest

from v2ecoli.steps.derivers.exchange_flux_listener import (
    ExchangeFluxListener, derive_fluxes, resolve_exchange_key)

FLUXES = {"violacein_exchange": "VIOLACEIN[c]", "glucose_exchange": "GLC[p]"}


def test_resolve_key_is_compartment_tolerant():
    # v2ecoli strips compartments (GLC); fork ids carry them (GLC[p]). One config
    # value must match either store convention.
    stripped_store = {"GLC": -8.5, "VIOLACEIN": 0.042}
    assert resolve_exchange_key(stripped_store, "GLC[p]") == -8.5
    assert resolve_exchange_key(stripped_store, "VIOLACEIN[c]") == 0.042
    full_store = {"GLC[p]": -8.5}
    assert resolve_exchange_key(full_store, "GLC") == -8.5      # reverse direction
    assert resolve_exchange_key(full_store, "GLC[p]") == -8.5   # exact
    assert resolve_exchange_key({"GLC": -8.5}, "MISSING[c]") is None


def test_derive_matches_fork_ids_against_stripped_store():
    # study config uses fork ids; candidate store is stripped -> still matches
    out = derive_fluxes({"GLC": -8.5, "VIOLACEIN": 0.042}, FLUXES)
    assert out == {"violacein_exchange": 0.042, "glucose_exchange": -8.5}


def test_derive_selects_named_fluxes_preserving_sign():
    exch = {"VIOLACEIN[c]": 0.042, "GLC[p]": -8.5, "CO2[p]": 12.0}
    assert derive_fluxes(exch, FLUXES) == {
        "violacein_exchange": 0.042, "glucose_exchange": -8.5}


def test_derive_missing_key_is_zero_not_gap():
    assert derive_fluxes({"GLC[p]": -8.5}, FLUXES) == {
        "violacein_exchange": 0.0, "glucose_exchange": -8.5}


def test_derive_empty():
    assert derive_fluxes({}, {}) == {}
    assert derive_fluxes(None, {"x": "Y"}) == {"x": 0.0}


@pytest.mark.fast
def test_outputs_declared_from_config_flux_map():
    from v2ecoli.core import build_core
    core = build_core()
    step = ExchangeFluxListener({"fluxes": FLUXES}, core=core)
    out = step.outputs()
    assert set(out["listeners"]["exchange_flux"]) == set(FLUXES)
    # no fluxes configured -> no leaves declared (feature effectively off)
    step0 = ExchangeFluxListener({"fluxes": {}}, core=core)
    assert step0.outputs()["listeners"]["exchange_flux"] == {}


@pytest.mark.fast
def test_update_writes_listener_leaves():
    from v2ecoli.core import build_core
    core = build_core()
    step = ExchangeFluxListener({"fluxes": FLUXES}, core=core)
    upd = step.update({"exchange": {"VIOLACEIN[c]": 0.042, "GLC[p]": -8.5},
                       "global_time": 0.0, "timestep": 1.0})
    assert upd["listeners"]["exchange_flux"]["violacein_exchange"] == 0.042
    assert upd["listeners"]["exchange_flux"]["glucose_exchange"] == -8.5


@pytest.mark.fast
def test_feature_inserts_step_after_mass_listener():
    from v2ecoli.composites.ecoli_baseline import build_execution_layers
    flat = [s for L in build_execution_layers(["exchange_flux"]) for s in L]
    assert "exchange_flux_listener" in flat
    assert flat.index("exchange_flux_listener") > flat.index("ecoli-mass-listener")
