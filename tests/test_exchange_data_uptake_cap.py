"""The configurable aerobic carbon-uptake cap must not perturb existing configs.

`exchange_data_from_concentrations` gained an `aerobic_cap` argument so a study
can titrate glucose uptake. The safety property this file pins is that the knob
is inert unless set: absent/None must reproduce the stock 20.0 mmol/gDCW/h cap
exactly, and the anaerobic branch must not be reachable by it at all.

The method reads only four attributes off ExternalState, so these tests build it
with `object.__new__` and set those four — no ParCa cache, no sim.
"""
from v2ecoli.processes.parca.reconstruction.ecoli.dataclasses.state.external_state import (
    ExternalState,
)
from v2ecoli.processes.parca.wholecell.utils import units
from v2ecoli.steps.exchange_data import ExchangeData

GLC = "GLC[p]"
OXY = "OXYGEN-MOLECULE[p]"
FLUX = units.mmol / units.g / units.h


def _external_state():
    """Minimal ExternalState carrying only what the method under test reads."""
    es = object.__new__(ExternalState)
    es.env_to_exchange_map = {"GLC": GLC, "OXYGEN-MOLECULE": OXY}
    es.import_constraint_threshold = 1e-5
    es.carbon_sources = [GLC]
    es.secretion_exchange_molecules = {"ACET[p]"}
    return es


def _glc_cap(aerobic_cap=None, oxygen=1.0):
    """Aerobic glucose cap, as a plain float in mmol/gDCW/h."""
    res = _external_state().exchange_data_from_concentrations(
        {"GLC": 10.0, "OXYGEN-MOLECULE": oxygen}, aerobic_cap=aerobic_cap
    )
    return res["importConstrainedExchangeMolecules"][GLC].asNumber(FLUX)


def test_absent_cap_is_the_stock_default():
    # The whole safety argument for this PR: not passing the argument at all, and
    # passing None explicitly, both land on the pre-existing hardcoded 20.0.
    res = _external_state().exchange_data_from_concentrations(
        {"GLC": 10.0, "OXYGEN-MOLECULE": 1.0}
    )
    assert res["importConstrainedExchangeMolecules"][GLC].asNumber(FLUX) == 20.0
    assert _glc_cap(aerobic_cap=None) == 20.0


def test_cap_is_applied_when_set():
    assert _glc_cap(aerobic_cap=6.0) == 6.0
    assert _glc_cap(aerobic_cap=2.5) == 2.5


def test_cap_of_zero_is_honored_not_treated_as_absent():
    # 0.0 is falsy; a truthiness check instead of an `is None` check would
    # silently restore 20.0 and make a full-starvation variant a no-op.
    assert _glc_cap(aerobic_cap=0.0) == 0.0


def test_anaerobic_branch_is_untouched():
    # The cap is aerobic-only. Without oxygen the stock 100.0 must stand even
    # when a cap is set, so the knob cannot leak into anaerobic configs.
    assert _glc_cap(aerobic_cap=None, oxygen=0.0) == 100.0
    assert _glc_cap(aerobic_cap=6.0, oxygen=0.0) == 100.0


def test_rest_of_the_payload_is_unchanged_by_the_cap():
    # Only the constrained-flux value may move; membership sets must not.
    mols = {"GLC": 10.0, "OXYGEN-MOLECULE": 1.0}
    stock = _external_state().exchange_data_from_concentrations(mols)
    capped = _external_state().exchange_data_from_concentrations(mols, aerobic_cap=6.0)
    for key in ("externalExchangeMolecules", "importExchangeMolecules",
                "importUnconstrainedExchangeMolecules", "secretionExchangeMolecules"):
        assert stock[key] == capped[key]
    assert stock["importConstrainedExchangeMolecules"].keys() == \
        capped["importConstrainedExchangeMolecules"].keys()


def test_step_config_default_is_none():
    # An existing config that never mentions the field must resolve to None, which
    # is what makes the fallback above the default path.
    assert ExchangeData.config_schema["glc_uptake_cap_aerobic"]["_default"] is None


def test_step_threads_config_value_through():
    step = object.__new__(ExchangeData)
    step.initialize({"glc_uptake_cap_aerobic": 6.0})
    assert step.glc_uptake_cap_aerobic == 6.0

    absent = object.__new__(ExchangeData)
    absent.initialize({})
    assert absent.glc_uptake_cap_aerobic is None
