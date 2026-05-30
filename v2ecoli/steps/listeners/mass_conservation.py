"""
==========================
Mass Conservation Listener
==========================

Observe-only runtime check that cellular mass is conserved.

Over one timestep, the cell's **dry-mass change** should equal the **net mass
imported across the environment boundary** by metabolism. Every other process
(transcription, translation, complexation, …) only repackages atoms the cell
already holds — it consumes monomers from the bulk pools and produces polymers,
conserving mass internally. So the only legitimate source/sink of cell mass is
the metabolic exchange with the environment.

This Step reads the dry mass and the per-tick environment exchange counts,
converts the exchange to a mass via per-molecule masses, and emits the residual

    residual = Δdry_mass − net_mass_imported

each tick, plus its magnitude relative to |Δdry_mass|. When the relative
residual exceeds ``tolerance`` it ``warnings.warn``\\s — it never raises, so a
long simulation is never halted by the check (the residual is observable in the
``listeners.mass`` history instead).

A healthy run sits at rounding noise; a large residual flags a process that is
creating or destroying mass without an explicit, justified source/sink (the
conservation expectation in AGENTS.md check #4).

Sign convention: ``environment.exchange[name]`` is the count added to the
*environment* this tick (FBA convention: secretion positive, uptake negative),
so the mass entering the *cell* is ``-Σ count·mass``.
"""

import warnings

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import as_quantity

NAME = "ecoli-mass-conservation"
TOPOLOGY = {
    "listeners": ("listeners",),
    "environment": ("environment",),
}


class MassConservationListener(Step):
    """Emit the per-tick mass-conservation residual; warn (never raise) on drift."""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # environment molecule name (compartment-stripped, matching
        # environment.exchange keys) -> mass per molecule (fg per count).
        "exchange_masses": {"_type": "map[quantity[float,fg]]", "_default": {}},
        # relative-residual threshold above which a warning is emitted.
        "tolerance": {"_type": "float", "_default": 1.0e-2},
    }

    def inputs(self):
        return {
            "listeners": {
                "mass": {
                    "dry_mass": {"_type": "quantity[float,fg]", "_default": 0.0},
                },
            },
            "environment": {"exchange": "map[float]"},
        }

    def outputs(self):
        return {
            "listeners": {
                "mass": {
                    "conservation_residual": {
                        "_type": "overwrite[quantity[float,fg]]",
                        "_default": 0.0,
                    },
                    "conservation_residual_relative": {
                        "_type": "overwrite[float]",
                        "_default": 0.0,
                    },
                    "exchange_mass_in": {
                        "_type": "overwrite[quantity[float,fg]]",
                        "_default": 0.0,
                    },
                },
            },
        }

    def initialize(self, config):
        self.exchange_masses = self.parameters["exchange_masses"]
        self.tolerance = self.parameters.get("tolerance", 1.0e-2)
        # Previous dry mass (None until the first tick supplies a baseline).
        self._prev_dry_mass = None

    def net_mass_imported(self, exchange):
        """Net mass (Quantity[fg]) entering the cell from the environment.

        ``exchange[name]`` is the count added to the environment this tick;
        mass entering the cell is the negative of the secreted mass.
        """
        total = 0.0 * units.fg
        for name, count in exchange.items():
            mass_per = self.exchange_masses.get(name)
            if mass_per is None or count == 0:
                continue
            total = total - count * mass_per
        return total

    def update(self, states, interval=None):
        dry_mass = as_quantity(states["listeners"]["mass"]["dry_mass"], units.fg)
        exchange = states.get("environment", {}).get("exchange", {}) or {}
        mass_in = self.net_mass_imported(exchange)

        if self._prev_dry_mass is None:
            # First observation: establish the baseline, nothing to balance yet.
            self._prev_dry_mass = dry_mass
            return {
                "listeners": {
                    "mass": {
                        "conservation_residual": 0.0 * units.fg,
                        "conservation_residual_relative": 0.0,
                        "exchange_mass_in": mass_in,
                    }
                }
            }

        delta_dry = dry_mass - self._prev_dry_mass
        self._prev_dry_mass = dry_mass
        residual = delta_dry - mass_in

        denom = abs(delta_dry.to(units.fg).magnitude)
        rel = abs(residual.to(units.fg).magnitude) / denom if denom else 0.0
        if rel > self.tolerance:
            warnings.warn(
                f"{self.name}: mass-conservation residual "
                f"{residual.to(units.fg).magnitude:.4g} fg "
                f"(relative {rel:.3g}) exceeds tolerance {self.tolerance}: "
                f"Δdry_mass={delta_dry.to(units.fg).magnitude:.4g} fg vs "
                f"net exchange={mass_in.to(units.fg).magnitude:.4g} fg.",
                stacklevel=2,
            )

        return {
            "listeners": {
                "mass": {
                    "conservation_residual": residual,
                    "conservation_residual_relative": rel,
                    "exchange_mass_in": mass_in,
                }
            }
        }
