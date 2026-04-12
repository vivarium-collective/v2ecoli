"""
Dark-matter accountant — phase 1 (diagnostic).

Implements the "biomass dark matter" concept as a *post-hoc* accountant
that tracks the mass the LP would need to create/destroy each step to
produce its bulk update. The pool models the cell's mass imbalance:

    dark_matter(t+dt) = dark_matter(t) + (bulk_mass_change − exchange_mass_change)

Sign convention:
  * bulk_mass_change = Σ_i (Δcount_i × MW_i) over all bulk species the
    listener has molecular weights for (from carbon_counts.MOLECULAR_
    WEIGHTS — currently ~60 species, covers imports/secretions + the
    main biomass components). Captures what the LP is writing to the
    cell pool.
  * exchange_mass_change = Σ_m ((−count_imported_m) × MW_m) over
    boundary exchange counts. Captures what crossed the boundary.
  * Difference = "mass the LP synthesized without a matching import" =
    deposit into dark matter (positive) or withdrawal (negative).

Invariants (user specification):
  * Mass can NOT be created → ``dark_matter`` must stay at 0 on
    average. Sustained ``dark_matter > 0`` means the LP is producing
    bulk mass that doesn't have a boundary source — a violation.
  * Mass can be destroyed (secreted or decomposed), and temporary
    ``dark_matter < 0`` is fine (imports that haven't been
    incorporated into biomass yet, or mass that's been lost to
    secretions the listener's MW table doesn't recognise).
  * If the homeostatic biomass targets can't be met because carbon
    is limiting, that's ALLOWED — the cell simply fails to grow.

This listener does NOT enforce the invariant — it only measures and
reports. Phase 2 adds an LP-level dark-matter flux whose bounds turn
this into a hard constraint: draws from a pool bounded at the current
``dark_matter`` value, deposits unbounded, both strongly penalised in
the objective so the LP tries to run at zero pool usage.
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
        "bulk": ("bulk",),
        "environment": ("environment",),
        "listeners": ("listeners",),
    }

    def initialize(self, config):
        self.parameters = config or {}
        self._mw_table = dict(self.parameters.get("mw_table", {}) or {})
        # Persistent pool state — single scalar tracking cumulative
        # bulk-mass minus exchange-mass delta. Starts at 0 and must
        # stay ≥ 0 if mass is actually conserved.
        self._dark_matter_fg = 0.0
        self._last_bulk_total_mass_fg = None
        self._cumulative_bulk_mass_in_fg = 0.0
        self._cumulative_exchange_mass_in_fg = 0.0
        self._cumulative_violations_fg = 0.0

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
                    "bulk_mass_change_fg": f(),
                    "exchange_mass_change_fg": f(),
                    "dark_matter_delta_fg": f(),
                    # Persistent pool state (fg). Negative = mass created
                    # from nothing.
                    "dark_matter_fg": f(),
                    "cumulative_bulk_mass_in_fg": f(),
                    "cumulative_exchange_mass_in_fg": f(),
                    "cumulative_violations_fg": f(),
                    # Fraction of bulk species we have MW coverage for —
                    # lets the reader trust or discount the numbers.
                    "mw_coverage_fraction": f(),
                },
            },
        }

    def update(self, states, interval=None):
        bulk = states.get("bulk")
        if bulk is None:
            ids, counts = [], []
        else:
            # `bulk` is a structured numpy array ('id', 'count'); neither
            # a dict nor guaranteed non-None.
            try:
                ids = list(bulk["id"])
                counts = list(bulk["count"])
            except (KeyError, TypeError, ValueError, IndexError):
                ids, counts = [], []

        # Current total bulk mass (fg) across molecules we have MW for.
        covered = 0
        total_counts = 0
        total_mass_g = 0.0
        for mid, ct in zip(ids, counts):
            mw = self._mw(mid)
            total_counts += 1
            if mw > 0:
                covered += 1
                total_mass_g += ct * mw / AVOGADRO
        total_bulk_mass_fg = total_mass_g * G_TO_FG
        coverage = covered / max(total_counts, 1)

        # Δ bulk mass this step (what the upstream metabolism step
        # actually produced, integrating across all processes that
        # write bulk).
        if self._last_bulk_total_mass_fg is None:
            bulk_mass_change_fg = 0.0
        else:
            bulk_mass_change_fg = total_bulk_mass_fg - self._last_bulk_total_mass_fg
        self._last_bulk_total_mass_fg = total_bulk_mass_fg

        # Mass actually crossing the boundary this step. Negative
        # exchange count = cell imported (cell gains mass); positive
        # = cell secreted (cell loses mass). Net import = -Σ (count × MW).
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
            # count / N_A × MW gives grams; ×1e15 → fg. Flipped sign so
            # positive = net import = mass into cell.
            exchange_mass_change_fg += -n * mw / AVOGADRO * G_TO_FG

        # Dark-matter accounting: "how much mass did the LP produce
        # beyond what the boundary actually supplied?" Positive dark
        # matter means the cell has extra cushion; negative means the
        # LP manufactured mass without a source.
        dark_delta = bulk_mass_change_fg - exchange_mass_change_fg
        self._dark_matter_fg += dark_delta
        self._cumulative_bulk_mass_in_fg += bulk_mass_change_fg
        self._cumulative_exchange_mass_in_fg += exchange_mass_change_fg
        # Violation accounting: positive dark matter = LP produced bulk
        # mass without a matching boundary import, which violates mass
        # conservation. Negative dark matter is fine (imports pending
        # incorporation, or uncovered secretions). Track only the bad
        # direction.
        if dark_delta > 0:
            self._cumulative_violations_fg += dark_delta

        return {
            "listeners": {
                "dark_matter": {
                    "bulk_mass_change_fg": bulk_mass_change_fg,
                    "exchange_mass_change_fg": exchange_mass_change_fg,
                    "dark_matter_delta_fg": dark_delta,
                    "dark_matter_fg": self._dark_matter_fg,
                    "cumulative_bulk_mass_in_fg":
                        self._cumulative_bulk_mass_in_fg,
                    "cumulative_exchange_mass_in_fg":
                        self._cumulative_exchange_mass_in_fg,
                    "cumulative_violations_fg":
                        self._cumulative_violations_fg,
                    "mw_coverage_fraction": coverage,
                },
            },
        }
