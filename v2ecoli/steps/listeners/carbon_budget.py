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
from v2ecoli.library.carbon_counts import carbon_of, mw_of


AVOGADRO = 6.02214076e23
BIOMASS_C_FRACTION = 0.48          # g C / g DCW (Neidhardt textbook value)
C_GRAMS_PER_MMOL = 12.011e-3       # 12 g/mol → g/mmol
# dry_mass / cell_mass are stored in femtograms.
DRY_MASS_UNIT_TO_G = 1e-15
G_TO_FG = 1e15


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
        self._cum_mass_in_fg = 0.0         # non-water imports, in fg
        self._cum_mass_out_fg = 0.0        # secretions, in fg
        self._cum_water_in_fg = 0.0        # tracked separately
        self._initial_dry_mass_fg = None
        self._initial_cell_mass_fg = None
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
                    "cell_mass": {"_type": "float[fg]", "_default": 0.0},
                },
            },
        }

    def outputs(self):
        # All values are per-step except the cumulative_* fields.
        f = lambda: {"_type": "overwrite[float]", "_default": 0.0}
        return {
            "listeners": {
                "carbon_budget": {
                    # Per-step carbon flux (mmol C)
                    "c_in_mmol": f(), "c_out_mmol": f(), "c_net_mmol": f(),
                    "cumulative_c_in_mmol": f(),
                    "cumulative_c_out_mmol": f(),
                    "biomass_c_est_mmol_per_step": f(),
                    # Per-step mass flux (fg) — non-water imports vs
                    # secretions vs water, tracked separately.
                    "mass_in_fg": f(), "mass_out_fg": f(),
                    "water_in_fg": f(),
                    "cumulative_mass_in_fg": f(),
                    "cumulative_mass_out_fg": f(),
                    "cumulative_water_in_fg": f(),
                    # Mass-balance delta: what imports should explain
                    # vs what actually grew.
                    "dry_mass_delta_fg": f(),
                    "cumulative_dry_mass_gained_fg": f(),
                    # (cumulative_mass_in - cumulative_mass_out) - dry_gained.
                    # Negative = imports can't explain growth (carbon
                    # appearing from nowhere). Near zero = balanced.
                    "mass_balance_deficit_fg": f(),
                },
            },
        }

    def update(self, states, interval=None):
        exchange = states.get("environment", {}).get("exchange", {}) or {}

        c_in = 0.0   # mmol C imported this step
        c_out = 0.0  # mmol C secreted this step
        mass_in_fg = 0.0   # fg of non-water mass imported this step
        mass_out_fg = 0.0  # fg of mass secreted this step
        water_in_fg = 0.0  # fg of water imported (tracked separately —
                           # water contributes to cell_mass, not dry_mass)

        for mol, count in exchange.items():
            try:
                n = float(count)
            except (TypeError, ValueError):
                continue
            if n == 0:
                continue

            # Carbon budget
            c = carbon_of(mol)
            if c:
                mmol_c = (n / AVOGADRO * 1e3) * c
                if mmol_c < 0:
                    c_in += -mmol_c
                else:
                    c_out += mmol_c

            # Mass budget — same flux, weighted by MW instead of C count.
            mw = mw_of(mol)
            if not mw:
                continue
            # count × g_per_mol / avogadro → grams; ×1e15 → fg.
            fg_flux = n / AVOGADRO * mw * G_TO_FG
            mol_bare = mol.split("[", 1)[0] if "[" in mol else mol
            is_water = mol_bare == "WATER"
            if fg_flux < 0:  # import
                if is_water:
                    water_in_fg += -fg_flux
                else:
                    mass_in_fg += -fg_flux
            else:            # secretion
                mass_out_fg += fg_flux

        self._cum_c_in_mmol += c_in
        self._cum_c_out_mmol += c_out
        self._cum_mass_in_fg += mass_in_fg
        self._cum_mass_out_fg += mass_out_fg
        self._cum_water_in_fg += water_in_fg

        # Dry-mass accounting
        dm_fg = float(states.get("listeners", {}).get("mass", {})
                             .get("dry_mass", 0.0))
        if self._initial_dry_mass_fg is None:
            self._initial_dry_mass_fg = dm_fg
        dm_gained = dm_fg - self._initial_dry_mass_fg

        if self._last_dry_mass_fg is None:
            biomass_c = 0.0
            dm_delta = 0.0
        else:
            dm_delta = dm_fg - self._last_dry_mass_fg
            delta_g = dm_delta * DRY_MASS_UNIT_TO_G
            biomass_c = (delta_g * BIOMASS_C_FRACTION) / C_GRAMS_PER_MMOL
        self._last_dry_mass_fg = dm_fg

        # Net non-water imports vs dry-mass gained (both in fg).
        # In a carbon-balanced cell: cumulative_mass_in ≈ cumulative_mass_out
        # + cumulative_dry_mass_gained. The deficit below highlights the
        # gap — when negative, imports/exports don't account for growth.
        deficit = (self._cum_mass_in_fg - self._cum_mass_out_fg) - dm_gained

        return {
            "listeners": {
                "carbon_budget": {
                    "c_in_mmol": c_in,
                    "c_out_mmol": c_out,
                    "c_net_mmol": c_in - c_out,
                    "cumulative_c_in_mmol": self._cum_c_in_mmol,
                    "cumulative_c_out_mmol": self._cum_c_out_mmol,
                    "biomass_c_est_mmol_per_step": biomass_c,
                    "mass_in_fg": mass_in_fg,
                    "mass_out_fg": mass_out_fg,
                    "water_in_fg": water_in_fg,
                    "cumulative_mass_in_fg": self._cum_mass_in_fg,
                    "cumulative_mass_out_fg": self._cum_mass_out_fg,
                    "cumulative_water_in_fg": self._cum_water_in_fg,
                    "dry_mass_delta_fg": dm_delta,
                    "cumulative_dry_mass_gained_fg": dm_gained,
                    "mass_balance_deficit_fg": deficit,
                },
            },
        }
