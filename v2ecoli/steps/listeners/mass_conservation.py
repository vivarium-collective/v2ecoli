"""
==========================
Mass Conservation Listener
==========================

Observe-only runtime check that cellular mass is conserved.

Over one timestep, the change in **total cell mass** (including water) should
equal the **net mass exchanged across the environment boundary** by metabolism.
Every other process (transcription, translation, complexation, …) only
repackages atoms the cell already holds — it consumes monomers from the bulk
pools and produces polymers, conserving mass internally. So the only legitimate
source/sink of cell mass is the metabolic exchange with the environment:

    residual = Δcell_mass − net_mass_imported

emitted each tick (plus its magnitude relative to |Δcell_mass|). When the
relative residual exceeds ``tolerance`` it ``warnings.warn``\\s — it never
raises, so a long run is never halted; the residual is observable in the
``listeners.mass`` history.

Two subtleties that this Step handles (and that an earlier version got wrong):

1. **``environment.exchange`` is CUMULATIVE**, not per-tick — metabolism *adds*
   ``delta_nutrients`` to it every tick (it tracks media depletion), so
   ``exchange[name]`` grows monotonically. The per-tick amount is the diff vs
   the previous tick, which this Step computes (``_prev_exchange``). Reading the
   store directly as if per-tick over-counts by ~the tick number.
2. **Total cell mass, not dry mass.** The exchange includes water and ions, so
   it balances against ``cell_mass`` (incl water), not ``dry_mass``.

Sign convention: ``environment.exchange[name]`` is the cumulative count added to
the *environment* (FBA convention: secretion positive, uptake negative), so the
per-tick mass entering the *cell* is ``−Σ Δcount·mass``.

STATUS — opt-in (``v2ecoli.composites.baseline.enable_features('mass_conservation')``;
OFF by default). MEASURED (baseline seed 0, 60 ticks, 2026-05-30): with the
per-tick diff + full per-molecule masses, the **net boundary exchange
(0.22 fg/tick) matches metabolism's own bulk addition (0.20 fg/tick)** and
tracks cell growth — i.e. **metabolism's exchange is mass-balanced** (the FBA's
S·v=0 enforces it). The residual that remains is small and is the real
conservation signal: it should sit near rounding noise on a healthy run, and a
spike flags a process creating/destroying mass without a justified source/sink
(AGENTS.md check #4). Still opt-in pending a clean multi-seed baseline.
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
        # relative-CUMULATIVE-residual threshold for the warning. Per-tick
        # ratios are noisy (Δcell≈0 on some ticks); the cumulative drift is the
        # physically meaningful conservation signal. A healthy baseline sits
        # near ~1%, so the default leaves headroom before warning.
        "tolerance": {"_type": "float", "_default": 5.0e-2},
        # Ticks to skip before accumulating: cell initialization carries a fixed
        # one-time mass offset (first metabolic solves / listener spin-up) that
        # is not a leak. Accumulate from steady state so the cumulative drift
        # and its warning reflect ongoing conservation, not the transient.
        "warmup_ticks": {"_type": "integer", "_default": 10},
    }

    def inputs(self):
        return {
            "listeners": {
                "mass": {
                    "cell_mass": {"_type": "quantity[float,fg]", "_default": 0.0},
                },
            },
            "environment": {"exchange": "map[float]"},
        }

    def outputs(self):
        return {
            "listeners": {
                "mass": {
                    "conservation_residual": {
                        "_type": "overwrite[quantity[float,fg]]", "_default": 0.0},
                    "conservation_residual_relative": {
                        "_type": "overwrite[float]", "_default": 0.0},
                    "exchange_mass_in": {
                        "_type": "overwrite[quantity[float,fg]]", "_default": 0.0},
                    # Cumulative since the run start — the meaningful signal.
                    "conservation_residual_cumulative": {
                        "_type": "overwrite[quantity[float,fg]]", "_default": 0.0},
                    "conservation_relative_cumulative": {
                        "_type": "overwrite[float]", "_default": 0.0},
                },
            },
        }

    def initialize(self, config):
        self.exchange_masses = self.parameters["exchange_masses"]
        self.tolerance = self.parameters.get("tolerance", 5.0e-2)
        self.warmup_ticks = self.parameters.get("warmup_ticks", 10)
        self._tick = 0                    # post-baseline tick counter
        self._prev_cell_mass = None       # None until the first tick
        self._prev_exchange = {}          # cumulative environment.exchange last tick
        self._cum_dcell = 0.0 * units.fg  # Σ Δcell_mass since warmup
        self._cum_exchange = 0.0 * units.fg  # Σ net exchange since warmup

    def per_tick_exchange(self, cumulative):
        """Per-tick exchange counts = diff of the cumulative environment.exchange
        against the previous tick. Advances ``_prev_exchange``."""
        delta = {name: total - self._prev_exchange.get(name, 0.0)
                 for name, total in cumulative.items()}
        self._prev_exchange = dict(cumulative)
        return delta

    def net_mass_imported(self, per_tick):
        """Net mass (Quantity[fg]) entering the cell this tick = −Σ Δcount·mass."""
        total = 0.0 * units.fg
        for name, count in per_tick.items():
            mass_per = self.exchange_masses.get(name)
            if mass_per is None or count == 0:
                continue
            total = total - count * mass_per
        return total

    def update(self, states, interval=None):
        cell_mass = as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)
        cumulative = states.get("environment", {}).get("exchange", {}) or {}
        mass_in = self.net_mass_imported(self.per_tick_exchange(cumulative))

        if self._prev_cell_mass is None:
            self._prev_cell_mass = cell_mass
            return {"listeners": {"mass": {
                "conservation_residual": 0.0 * units.fg,
                "conservation_residual_relative": 0.0,
                "exchange_mass_in": mass_in,
                "conservation_residual_cumulative": 0.0 * units.fg,
                "conservation_relative_cumulative": 0.0,
            }}}

        delta_cell = cell_mass - self._prev_cell_mass
        self._prev_cell_mass = cell_mass
        residual = delta_cell - mass_in
        denom = abs(delta_cell.to(units.fg).magnitude)
        rel = abs(residual.to(units.fg).magnitude) / denom if denom else 0.0

        # Skip the cell-initialization transient; accumulate from steady state.
        self._tick += 1
        if self._tick <= self.warmup_ticks:
            return {"listeners": {"mass": {
                "conservation_residual": residual,
                "conservation_residual_relative": rel,
                "exchange_mass_in": mass_in,
                "conservation_residual_cumulative": 0.0 * units.fg,
                "conservation_relative_cumulative": 0.0,
            }}}

        # Cumulative drift (since warmup) — the meaningful, low-noise signal.
        self._cum_dcell = self._cum_dcell + delta_cell
        self._cum_exchange = self._cum_exchange + mass_in
        cum_residual = self._cum_dcell - self._cum_exchange
        cum_denom = abs(self._cum_dcell.to(units.fg).magnitude)
        cum_rel = abs(cum_residual.to(units.fg).magnitude) / cum_denom if cum_denom else 0.0

        # Only warn once enough cell-mass has accumulated for the ratio to be
        # meaningful (avoids a small-denominator transient right after warmup).
        if cum_denom > 5.0 and cum_rel > self.tolerance:
            warnings.warn(
                f"{self.name}: cumulative mass-conservation drift "
                f"{cum_residual.to(units.fg).magnitude:.4g} fg "
                f"(relative {cum_rel:.3g}) exceeds tolerance {self.tolerance}: "
                f"Σ Δcell_mass={self._cum_dcell.to(units.fg).magnitude:.4g} fg vs "
                f"Σ net exchange={self._cum_exchange.to(units.fg).magnitude:.4g} fg.",
                stacklevel=2,
            )

        return {"listeners": {"mass": {
            "conservation_residual": residual,
            "conservation_residual_relative": rel,
            "exchange_mass_in": mass_in,
            "conservation_residual_cumulative": cum_residual,
            "conservation_relative_cumulative": cum_rel,
        }}}
