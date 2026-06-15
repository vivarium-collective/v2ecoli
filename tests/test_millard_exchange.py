"""Millard medium-exchange accounting (mbp-09 req-1).

The Millard kinetic-metabolism cell must populate ``environment.exchange`` with
its MEDIUM exchanges (O2 / external glucose / acetate) so the reactor coupler can
drive O2 consumption and the mass-conservation deriver can evaluate on the
Millard cell.

Convention matched (v2ecoli/processes/metabolism.py ``_fba_output_to_deltas`` +
the ``environment.exchange`` write): keys are BARE molecule names (no [c]/[p]
suffix); value = signed per-tick molecule-COUNT delta; POSITIVE = secretion
(added to medium), NEGATIVE = uptake (removed). Count from mM via
``count = mM * 1e-3 * V_live_L * N_A``.

Why FLUX-based (not concentration-delta): in the Millard SBML, O2 is a ``fixed``
species (its mM never changes -> a conc-delta would be 0) and external glucose
``GLCx`` is held up by an artificial ``_GLC_FEED`` (its mM RISES even while the
cell consumes it -> a conc-delta would have the WRONG sign). The physically
correct medium exchange is the boundary-reaction flux integrated over the tick.
"""
import warnings; warnings.filterwarnings("ignore")
import pytest

from v2ecoli.core import build_core
from v2ecoli.steps.millard_pdmp_metabolism import (
    MillardPDMPMetabolism,
    AVOGADRO,
    FG_PER_G,
    DEFAULT_CELL_DENSITY_G_PER_L,
)


@pytest.fixture(scope="module")
def core():
    return build_core()


@pytest.mark.sim
def test_outputs_schema_has_environment_exchange(core):
    s = MillardPDMPMetabolism(config={}, core=core)
    outs = s.outputs()
    assert "environment" in outs
    assert "exchange" in outs["environment"]


@pytest.mark.sim
def test_millard_emits_signed_medium_exchange(core):
    """One tick: O2 and glucose are taken up (NEGATIVE exchange counts), and the
    counts equal the boundary-flux-derived expectation with the WCM sign."""
    cell_mass_fg = 1000.0
    s = MillardPDMPMetabolism(config={}, core=core)
    out = s.update(
        {
            "lqr_control": {},
            "bulk": None,
            "listeners_mass": {"cell_mass": cell_mass_fg, "dry_mass": 300.0},
            "external_concentrations": {},
        },
        1.0,
    )

    assert "environment" in out, "no environment output emitted"
    exch = out["environment"]["exchange"]
    fluxes = out["central_fluxes"]

    # Reproduce the live-volume conc->count factor independently.
    live_volume_L = cell_mass_fg * FG_PER_G / DEFAULT_CELL_DENSITY_G_PER_L
    conc_to_count = 1e-3 * live_volume_L * AVOGADRO
    tick_s = 1.0

    # --- O2 (respiration consumes O2 via CYTBO -> uptake -> NEGATIVE) ---
    assert "OXYGEN-MOLECULE" in exch, "O2 medium exchange missing"
    assert exch["OXYGEN-MOLECULE"] < 0.0, "O2 exchange must be negative (uptake)"
    expected_o2 = round(-1.0 * fluxes["CYTBO"] * tick_s * conc_to_count)
    assert exch["OXYGEN-MOLECULE"] == expected_o2, (
        f"O2 exchange {exch['OXYGEN-MOLECULE']} != expected {expected_o2}"
    )

    # --- glucose (XCH_GLC: GLCx -> GLCp transport -> uptake -> NEGATIVE) ---
    assert "GLC" in exch, "glucose medium exchange missing"
    assert exch["GLC"] < 0.0, "glucose exchange must be negative (uptake)"
    expected_glc = round(-1.0 * fluxes["XCH_GLC"] * tick_s * conc_to_count)
    assert exch["GLC"] == expected_glc, (
        f"glucose exchange {exch['GLC']} != expected {expected_glc}"
    )

    # --- CO2 efflux (decarboxylation) + H2O efflux (CYTBO respiration) ---
    # The Millard SBML holds HCO3/H2O as FIXED buffer species; their efflux is
    # reconstructed from reaction stoichiometry (BYPRODUCT_EFFLUX_REACTIONS) and
    # emitted as medium SECRETION (POSITIVE count) so the boundary mass
    # accounting closes (substrate-C leaves as CO2, consumed O2 leaves as H2O).
    assert "CARBON-DIOXIDE" in exch, "CO2 efflux missing"
    assert exch["CARBON-DIOXIDE"] > 0.0, "CO2 exchange must be positive (secretion)"
    assert "WATER" in exch, "respiratory H2O efflux missing"
    assert exch["WATER"] > 0.0, "H2O exchange must be positive (secretion)"
    # CO2 net efflux = (GND+PDH+ICD+LPD+MAE+PCK) − PPC, integrated over the tick.
    co2_rxns = {"GND": 1.0, "PDH": 1.0, "ICD": 1.0, "LPD": 1.0,
                "MAE": 1.0, "PCK": 1.0, "PPC": -1.0}
    expected_co2 = round(
        sum(coeff * fluxes.get(r, 0.0) for r, coeff in co2_rxns.items())
        * tick_s * conc_to_count)
    assert exch["CARBON-DIOXIDE"] == expected_co2, (
        f"CO2 exchange {exch['CARBON-DIOXIDE']} != expected {expected_co2}"
    )
    expected_h2o = round(2.0 * fluxes.get("CYTBO", 0.0) * tick_s * conc_to_count)
    assert exch["WATER"] == expected_h2o, (
        f"H2O exchange {exch['WATER']} != expected {expected_h2o}"
    )


@pytest.mark.sim
def test_baseline_millard_mass_conservation_evaluable():
    """Integration: with the mass_conservation feature, the Millard cell now
    populates environment.exchange (O2 finite) AND the conservation listener
    produces a finite conservation_relative_cumulative (was None before — the
    Millard-cell mass gate is now evaluable)."""
    from v2ecoli import build_composite
    import math

    c = build_composite(
        "baseline_millard", seed=0, cache_dir="out/cache",
        features=["mass_conservation"],
    )
    c.run(14)
    ag = (c.state.get("agents") or {}).get("0") or {}

    exch = (ag.get("environment") or {}).get("exchange") or {}
    assert "OXYGEN-MOLECULE" in exch, "Millard cell did not populate O2 exchange"
    assert math.isfinite(float(exch["OXYGEN-MOLECULE"]))

    cum = ag.get("listeners", {}).get("mass", {}).get(
        "conservation_relative_cumulative")
    assert cum is not None, "conservation_relative_cumulative still None"
    assert math.isfinite(float(cum)), f"cum not finite: {cum!r}"
