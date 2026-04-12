"""
Dark-matter accountant — diagnostic listener.

Tracks the cell-scale mass-conservation identity

    Δ(cell_mass)  =  (imports − secretions)

by differencing the mass-listener's ``cell_mass`` against boundary
exchange flux each step. ``cell_mass`` already aggregates every
compartment's mass including unique molecules (ribosomes, RNAPs,
chromosomes, and RNA in transcription), so assembly events that move
subunits from bulk into unique are captured correctly and do not
appear as "mass loss". This version supersedes an earlier bulk-only
implementation that gave misleadingly-negative pool values when mass
moved from bulk into the unique pool.

    dark_matter(t+dt) = dark_matter(t) + (Δ cell_mass − Δ boundary_mass)

Sign:
  * Positive dark matter = cell grew MORE than boundary imports → LP
    produced mass without a source. *This is the violation we care
    about.*
  * Near zero = balanced — what a physical cell must hit.
  * Negative dark matter = cell grew LESS than boundary imports. Possible
    causes: (a) secretions the ``mw_of`` table doesn't recognise;
    (b) per-step accounting lag (imports this step, biomass next);
    (c) unique-pool assembly releasing byproducts back to bulk. Not a
    mass-creation violation — tracked separately as "unaccounted".
"""

from __future__ import annotations

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.carbon_counts import mw_of as _fallback_mw_of


AVOGADRO = 6.02214076e23
G_TO_FG = 1e15


class DarkMatterAccountant(Step):
    """Track the mass imbalance between LP bulk updates and exchange flow."""

    name = "dark_matter_listener"

    config_schema = {
        "time_step": {"_default": 1},
        # Full bulk-molecule MW table from sim_data (see
        # LoadSimData.get_dark_matter_config). Falls back to the
        # small carbon_counts table when a molecule is missing.
        "mw_table": {"_type": "map[float]", "_default": {}},
    }

    topology = {
        "environment": ("environment",),
        "listeners": ("listeners",),
    }

    def initialize(self, config):
        self.parameters = config or {}
        self._mw_table = dict(self.parameters.get("mw_table", {}) or {})
        # Persistent pool state.
        self._dark_matter_fg = 0.0
        self._last_cell_mass_fg = None
        self._cumulative_cell_mass_in_fg = 0.0
        self._cumulative_exchange_mass_in_fg = 0.0
        # Two separate counters to make the story honest:
        self._cumulative_violations_fg = 0.0     # sustained Δ > 0: mass created
        self._cumulative_unaccounted_fg = 0.0    # Δ < 0: mass missing (secretions, lag)

    def _mw(self, mol_id: str) -> float:
        """Lookup MW with compartment-agnostic fallback."""
        if mol_id in self._mw_table:
            return self._mw_table[mol_id]
        # Strip compartment suffix (e.g. "GLC[c]" → "GLC")
        bare = mol_id.split("[", 1)[0] if "[" in mol_id else mol_id
        if bare in self._mw_table:
            return self._mw_table[bare]
        return _fallback_mw_of(mol_id)

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "environment": {
                "exchange": {
                    "_type": "map[float]", "_default": {}},
            },
        }

    def outputs(self):
        f = lambda: {"_type": "overwrite[float]", "_default": 0.0}
        return {
            "listeners": {
                "dark_matter": {
                    # Per-step
                    "cell_mass_change_fg": f(),
                    "exchange_mass_change_fg": f(),
                    "dark_matter_delta_fg": f(),
                    # Persistent state.
                    "dark_matter_fg": f(),
                    "cumulative_cell_mass_in_fg": f(),
                    "cumulative_exchange_mass_in_fg": f(),
                    # Distinguish the two directions.
                    "cumulative_violations_fg": f(),     # mass created (Δ > 0)
                    "cumulative_unaccounted_fg": f(),    # mass missing (Δ < 0)
                },
            },
        }

    def update(self, states, interval=None):
        mass = states.get("listeners", {}).get("mass", {}) or {}
        cell_mass_fg = float(mass.get("cell_mass", 0.0))

        if self._last_cell_mass_fg is None:
            cell_mass_change_fg = 0.0
        else:
            cell_mass_change_fg = cell_mass_fg - self._last_cell_mass_fg
        self._last_cell_mass_fg = cell_mass_fg

        # Boundary flux this step. Negative exchange count = import
        # (cell gains mass); positive = secretion (loses). Net into
        # cell = -Σ(count × MW).
        exchange = states.get("environment", {}).get("exchange", {}) or {}
        exchange_mass_change_fg = 0.0
        for mol, count in exchange.items():
            try:
                n = float(count)
            except (TypeError, ValueError):
                continue
            mw = self._mw(mol)
            if mw <= 0 or n == 0:
                continue
            exchange_mass_change_fg += -n * mw / AVOGADRO * G_TO_FG

        # Mass-balance identity: cell grew iff imports supplied it.
        dark_delta = cell_mass_change_fg - exchange_mass_change_fg
        self._dark_matter_fg += dark_delta
        self._cumulative_cell_mass_in_fg += cell_mass_change_fg
        self._cumulative_exchange_mass_in_fg += exchange_mass_change_fg
        if dark_delta > 0:
            self._cumulative_violations_fg += dark_delta
        elif dark_delta < 0:
            self._cumulative_unaccounted_fg += -dark_delta

        return {
            "listeners": {
                "dark_matter": {
                    "cell_mass_change_fg": cell_mass_change_fg,
                    "exchange_mass_change_fg": exchange_mass_change_fg,
                    "dark_matter_delta_fg": dark_delta,
                    "dark_matter_fg": self._dark_matter_fg,
                    "cumulative_cell_mass_in_fg":
                        self._cumulative_cell_mass_in_fg,
                    "cumulative_exchange_mass_in_fg":
                        self._cumulative_exchange_mass_in_fg,
                    "cumulative_violations_fg":
                        self._cumulative_violations_fg,
                    "cumulative_unaccounted_fg":
                        self._cumulative_unaccounted_fg,
                },
            },
        }
