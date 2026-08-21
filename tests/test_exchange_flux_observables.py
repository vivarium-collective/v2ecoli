"""Reference-arm (genuine vEcoli pbg node) generic exchange-flux emit.

VivariumEcoliProcess can lift named metabolic exchange fluxes out of the cell's
environment.exchange store onto listeners.exchange_flux.<leaf> leaves, the same
way observable_bulk_ids surfaces bulk counts. Deliberately generic: the flux map
is config, so the shared model stays agnostic to any particular pathway.
"""
from v2ecoli.library.vivarium_ecoli_engine import _select_exchange_fluxes


FLUXES = {"acetate_exchange": "AC[p]", "glucose_exchange": "GLC[p]"}


def test_selects_named_fluxes_preserving_sign():
    env = {"exchange": {"AC[p]": 0.042, "GLC[p]": -8.5, "CO2[p]": 12.0}}
    out = _select_exchange_fluxes(env, FLUXES)
    assert out == {"acetate_exchange": 0.042, "glucose_exchange": -8.5}


def test_missing_key_defaults_to_zero_not_crash():
    env = {"exchange": {"GLC[p]": -8.5}}
    out = _select_exchange_fluxes(env, FLUXES)
    assert out == {"acetate_exchange": 0.0, "glucose_exchange": -8.5}


def test_empty_fluxes_returns_empty():
    assert _select_exchange_fluxes({"exchange": {"GLC[p]": -8.5}}, {}) == {}


def test_no_exchange_substore_yields_zeros_no_crash():
    # environment present but no exchange key yet (e.g. before metabolism's first
    # write) -> zeros, so the leaf stays a continuous trace
    assert _select_exchange_fluxes({}, FLUXES) == {
        "acetate_exchange": 0.0, "glucose_exchange": 0.0}
    assert _select_exchange_fluxes(None, FLUXES) == {
        "acetate_exchange": 0.0, "glucose_exchange": 0.0}


def test_process_declares_exchange_flux_output_only_when_configured():
    from v2ecoli.core import build_core
    from v2ecoli.library.vivarium_ecoli_engine import VivariumEcoliProcess
    core = build_core()

    def _make(cfg):
        # Avoid a real EcoliSim build: inject a dummy handle so __init__ takes
        # the pending-handle branch (no sim_data needed).
        VivariumEcoliProcess._PENDING_HANDLE = object()
        try:
            return VivariumEcoliProcess(config=cfg, core=core)
        finally:
            VivariumEcoliProcess._PENDING_HANDLE = None

    out = _make({"exchange_fluxes": FLUXES}).outputs()
    assert set(out["listeners"]["exchange_flux"]) == set(FLUXES)
    assert out["listeners"]["exchange_flux"]["acetate_exchange"] == "overwrite[float]"

    assert "exchange_flux" not in _make({}).outputs()["listeners"]
