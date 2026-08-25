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
``{"acetate_exchange": "AC[p]", "glucose_exchange": "GLC[p]"}``). Enable
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


def _strip_compartment(mid: str) -> str:
    """``AC[p]`` -> ``AC``; unchanged if no ``[..]`` suffix."""
    return mid[:-3] if len(mid) > 3 and mid.endswith("]") and mid[-3] == "[" else mid


def resolve_exchange_key(exchange: dict, key: str):
    """Compartment-tolerant lookup: exchange stores may key by full metabolite id
    (``AC[p]``, the fork convention) or compartment-stripped (``AC``,
    v2ecoli's convention), and a genuine-vEcoli store can carry BOTH forms for the
    same molecule — a zero-valued compartment-tagged placeholder alongside the
    real flux on the stripped key (e.g. ``{"GLC[p]": 0, "GLC": -1.28e7}``). So an
    exact-first match would return the 0 placeholder and miss the real value.
    Instead gather every entry whose compartment-stripped form matches the
    request and return the one with the largest magnitude (placeholders are 0;
    the real flux is not). Returns None if no key matches."""
    exchange = exchange or {}
    stripped = _strip_compartment(key)
    matches = [v for k, v in exchange.items()
               if k == key or _strip_compartment(k) == stripped]
    if not matches:
        return None
    return max(matches, key=lambda v: abs(float(v)) if v is not None else 0.0)


def derive_fluxes(exchange: dict, fluxes: dict) -> dict:
    """Pure core: pull each configured exchange key into its leaf name. A key
    absent this tick yields 0.0 so the leaf stays a continuous trace."""
    exchange = exchange or {}
    out = {}
    for leaf, key in (fluxes or {}).items():
        v = resolve_exchange_key(exchange, key)
        out[leaf] = float(v) if v is not None else 0.0
    return out


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
