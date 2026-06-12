"""FBAFluxCoupler — pin v2ecoli FBA reactions to Millard central-carbon fluxes.

Each tick this Step reads a ``central_fluxes`` dict (Millard reaction id ->
flux in mM/s, e.g. from ``basico.get_reactions()['flux']`` on the Millard 2017
ODE), maps each Millard reaction onto its v2ecoli FBA *base* reaction id(s) via
the reaction map (``v2ecoli/data/millard_v2ecoli_reaction_map.yaml``), and
converts the flux into the FBA bound basis with the shared coefficient. It
emits ``pinned_flux_targets`` (FBA reaction id -> flux bound) for the
Metabolism process to consume as pinned reaction bounds.

Millard reactions with no clean v2ecoli counterpart (``millard_only``) are
excluded by the reaction-map loader and so are never pinned.
"""
from __future__ import annotations

from process_bigraph import Process

from v2ecoli.types.stores import InPlaceDict
from v2ecoli.library.fba_reaction_map import load_reaction_map
from v2ecoli.library.fba_flux_convert import millard_flux_to_fba_bound


class FBAFluxCoupler(Process):
    """Map+convert Millard central fluxes to FBA reaction-flux pins.

    Topology:
      inputs:
        central_fluxes: Millard reaction id -> flux (mM/s)
      outputs:
        pinned_flux_targets: FBA reaction id -> flux bound
        bridge_diagnostics: per-tick coupler diagnostics
    """

    name = "fba-flux-coupler"
    topology = {
        "central_fluxes": ("central_fluxes",),
        "pinned_flux_targets": ("pinned_flux_targets",),
        "bridge_diagnostics": ("bridge_diagnostics",),
    }
    config_schema = {
        "reaction_map_file": {
            "_type": "string",
            "_default": "v2ecoli/data/millard_v2ecoli_reaction_map.yaml"},
        "coefficient": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.parameters = config or {}
        map_file = self.parameters.get(
            "reaction_map_file",
            self.config_schema["reaction_map_file"]["_default"])
        self.reaction_map = load_reaction_map(map_file)
        self.coefficient = float(self.parameters.get("coefficient", 1.0))

    def inputs(self):
        return {"central_fluxes": InPlaceDict()}

    def outputs(self):
        return {
            "pinned_flux_targets": InPlaceDict(),
            "bridge_diagnostics": InPlaceDict(),
        }

    def update(self, state, interval=None):
        fluxes = state.get("central_fluxes") or {}
        pins: dict[str, float] = {}
        for millard_rxn, v in fluxes.items():
            for fba_id, scale in self.reaction_map.get(millard_rxn, []):
                pins[fba_id] = millard_flux_to_fba_bound(
                    float(v), self.coefficient, scale)
        return {
            "pinned_flux_targets": pins,
            "bridge_diagnostics": {"n_pinned": len(pins)},
        }


def register(core):
    core.register_link("FBAFluxCoupler", FBAFluxCoupler)
