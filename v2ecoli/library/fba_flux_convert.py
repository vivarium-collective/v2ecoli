"""Convert a Millard reaction flux (mM/s) to the v2ecoli FBA bound basis.

The WCM sets reaction flux bounds in CONC_UNITS magnitude using a `coefficient`
(mass*time/volume) — the same factor `Metabolism.set_reaction_bounds` uses to
map mmol/gDCW/hr to a mM basis. The Millard ODE reports reaction fluxes in mM/s
(``basico.get_reactions()['flux']``), so the bridge mirrors that coefficient to
keep pinned bounds dimensionally consistent with the FBA's other bounds. `scale`
carries the reaction-map stoichiometry/sign for a Millard reaction that maps to a
v2ecoli reaction with a different convention.
"""
from __future__ import annotations


def millard_flux_to_fba_bound(flux_mM_per_s: float, coefficient: float,
                              scale: float = 1.0) -> float:
    """Convert a Millard reaction flux (mM/s) to a v2ecoli FBA flux bound.

    bound = flux_mM_per_s * coefficient * scale  (pure, deterministic).
    """
    return float(flux_mM_per_s) * float(coefficient) * float(scale)
