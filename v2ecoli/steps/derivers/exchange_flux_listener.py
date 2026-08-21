"""
=======================
Exchange-Flux Listener
=======================

Generic derived-emit Step: each tick it reads named keys out of the cell's
``environment.exchange`` store and writes them as plain scalar leaves under
``listeners.exchange_flux.<name>``.

WHY. The compact comparison view (and the XArray emitter view generally) only
carries ``listeners.*`` leaves, so a metabolic exchange flux — which lives in
``environment.exchange`` — never reaches the emitted store otherwise. This
listener bridges that gap by re-homing named fluxes onto listener leaves. It
also sidesteps v2ecoli#547 (dict-valued listener leaves drop on the injection
path): it READS the exchange dict (which populates) and WRITES plain scalars.

DELIBERATELY GENERIC. It knows nothing about any particular pathway: the caller
supplies a ``fluxes`` map ``{leaf_name: exchange_key}`` (e.g.
``{"violacein_exchange": "VIOLACEIN[c]", "glucose_exchange": "GLC[p]"}``). Enable
it via the ``ecoli_baseline`` generator's ``exchange_fluxes`` param. Sign is
preserved verbatim (uptake negative, secretion positive).
"""
from v2ecoli.library.ecoli_step import EcoliStep as Step

NAME = "exchange_flux_listener"
TOPOLOGY = {
    "exchange": ("environment", "exchange"),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


def derive_fluxes(exchange: dict, fluxes: dict) -> dict:
    """Pure core: pull each configured exchange key into its leaf name. A key
    absent this tick yields 0.0 so the leaf stays a continuous trace."""
    exchange = exchange or {}
    return {leaf: (float(exchange[key]) if exchange.get(key) is not None else 0.0)
            for leaf, key in (fluxes or {}).items()}


class ExchangeFluxListener(Step):
    """Re-home named exchange fluxes onto ``listeners.exchange_flux.<name>``."""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # {leaf_name: exchange_key} — the fluxes to surface.
        "fluxes": "map[string]",
        "time_step": "float{1.0}",
    }

    def initialize(self, config):
        self.fluxes = dict(self.parameters.get("fluxes") or {})

    def inputs(self):
        return {
            "exchange": {"_type": "map[float]", "_default": {}},
            "global_time": {"_type": "float", "_default": 0.0},
            "timestep": {"_type": "float", "_default": 1.0},
        }

    def outputs(self):
        # Recomputed-absolute each tick → 'set'/overwrite semantics. Leaves are
        # declared from the configured flux map (no fluxes → nothing declared).
        return {"listeners": {"exchange_flux": {
            leaf: {"_type": "overwrite[float]", "_default": 0.0}
            for leaf in self.fluxes}}}

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        if not self.fluxes:
            return {}
        return {"listeners": {"exchange_flux":
                              derive_fluxes(states.get("exchange"), self.fluxes)}}
