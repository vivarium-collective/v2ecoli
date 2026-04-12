"""
Carbon-budget listener.

Reads the per-step count deltas metabolism writes to ``environment.exchange``
(negative = imported by cell, positive = secreted), weights them by the
carbon content of each molecule, and emits an aggregate carbon budget to
``listeners.carbon_budget``. Purpose: make "free carbon" — the LP
satisfying biomass without a boundary carbon source — visible as a
diagnostic.

Fields written (counts are mmol-of-carbon per simulation step,
converted via Avogadro):

  c_in_mmol         Sum over imported carbon (only molecules with
                    C > 0 contribute); positive by construction.
  c_out_mmol        Sum over secreted carbon; positive by construction.
  c_net_mmol        c_in - c_out; when negative, cell is releasing
                    more carbon than it takes in — an obvious symptom
                    of draining internal pools.
  cumulative_c_in   Running total since t=0.
  cumulative_c_out  Running total since t=0.
  biomass_c_est_mmol_per_step
                    Rough dry-mass delta × 0.48 (g C / g DCW) / 12
                    (g C / mmol C). Back-of-the-envelope "how much C
                    went into biomass this step".

Runs after ``ecoli-metabolism`` + ``environment_update`` in the
layer order.
"""

from __future__ import annotations

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.carbon_counts import carbon_of


AVOGADRO = 6.02214076e23
BIOMASS_C_FRACTION = 0.48          # g C / g DCW (Neidhardt textbook value)
C_GRAMS_PER_MMOL = 12.011e-3       # 12 g/mol → g/mmol
# dry_mass is stored in femtograms — convert to grams for the carbon calc.
DRY_MASS_UNIT_TO_G = 1e-15


class CarbonBudget(Step):
    """Aggregate per-step carbon-exchange into a running budget."""

    name = "carbon_budget_listener"

    config_schema = {
        "time_step": {"_default": 1},
    }

    topology = {
        "environment": ("environment",),
        "listeners": ("listeners",),
    }

    def initialize(self, config):
        self.parameters = config or {}
        # Running totals — instance state preserved across composite ticks.
        self._cum_c_in_mmol = 0.0
        self._cum_c_out_mmol = 0.0
        self._last_dry_mass_fg = None

    def inputs(self):
        return {
            "environment": {
                "exchange": {
                    "_type": "map[float]", "_default": {}},
            },
            "listeners": {
                "mass": {
                    "dry_mass": {"_type": "float[fg]", "_default": 0.0},
                },
            },
        }

    def outputs(self):
        return {
            "listeners": {
                "carbon_budget": {
                    "c_in_mmol": {
                        "_type": "overwrite[float]", "_default": 0.0},
                    "c_out_mmol": {
                        "_type": "overwrite[float]", "_default": 0.0},
                    "c_net_mmol": {
                        "_type": "overwrite[float]", "_default": 0.0},
                    "cumulative_c_in_mmol": {
                        "_type": "overwrite[float]", "_default": 0.0},
                    "cumulative_c_out_mmol": {
                        "_type": "overwrite[float]", "_default": 0.0},
                    "biomass_c_est_mmol_per_step": {
                        "_type": "overwrite[float]", "_default": 0.0},
                },
            },
        }

    def update(self, states, interval=None):
        exchange = states.get("environment", {}).get("exchange", {}) or {}

        c_in = 0.0   # mmol C imported this step
        c_out = 0.0  # mmol C secreted this step
        for mol, count in exchange.items():
            c = carbon_of(mol)
            if c == 0:
                continue
            try:
                n = float(count)
            except (TypeError, ValueError):
                continue
            mmol_flux = n / AVOGADRO * 1e3   # count → mmol
            mmol_c = mmol_flux * c           # mmol molecule → mmol C
            if mmol_c < 0:
                c_in += -mmol_c             # negative exchange = import
            elif mmol_c > 0:
                c_out += mmol_c

        self._cum_c_in_mmol += c_in
        self._cum_c_out_mmol += c_out

        # Biomass C estimate from dry-mass delta this step.
        dm_fg = float(states.get("listeners", {}).get("mass", {})
                             .get("dry_mass", 0.0))
        if self._last_dry_mass_fg is None:
            biomass_c = 0.0
        else:
            delta_g = (dm_fg - self._last_dry_mass_fg) * DRY_MASS_UNIT_TO_G
            biomass_c = (delta_g * BIOMASS_C_FRACTION) / C_GRAMS_PER_MMOL
        self._last_dry_mass_fg = dm_fg

        return {
            "listeners": {
                "carbon_budget": {
                    "c_in_mmol": c_in,
                    "c_out_mmol": c_out,
                    "c_net_mmol": c_in - c_out,
                    "cumulative_c_in_mmol": self._cum_c_in_mmol,
                    "cumulative_c_out_mmol": self._cum_c_out_mmol,
                    "biomass_c_est_mmol_per_step": biomass_c,
                },
            },
        }
