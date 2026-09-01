"""ReactorCellCoupler — bridge v2ecoli's cell population to the BiRD reactor.

Build-phase scaffold for mbp-03 (req-2). PBG Step that runs each emit cycle,
implementing **Option B** of the v2ecoli <-> pbg-bioreactordesign coupling:

  * BiRD owns transport physics + reads biomass as an input.
  * v2ecoli owns biomass + the metabolic exchange demand.

This Step is the thin translator between the two halves. Each cycle it:

1. **Biomass passthrough.** Read ``population.biomass_concentration_gL`` (g/L,
   produced by :class:`PopulationAggregator`) and write it to
   ``reactor.biomass`` (g/L) for ``BiRDTransportProcess`` to consume. This is a
   passthrough/overwrite of the absolute value (the ``reactor.biomass`` port
   must be wired as an overwrite type in the composite).

2. **Dissolved-gas -> environment concentration.** Read ``reactor.dissolved_o2``
   and ``reactor.dissolved_co2`` (mg/L) and write the cell environment's
   ``external_concentrations`` (mM) for the two gases.

   Conversion (the load-bearing one):

       mM = (mg/L) / MW[g/mol]

   because  (mg/L) / (g/mol) = (1e-3 g/L) / (g/mol) = 1e-3 mol/L = 1 mM.
   e.g. 8 mg/L O2 / 31.999 ~= 0.25 mM. These are absolute concentrations
   (overwrite the environment port, not additive).

3. **Metabolic demand -> dissolved-gas delta.** Aggregate each agent's
   metabolic exchange flux into an *additive* dissolved-gas delta (mg/L) the
   reactor accumulates this interval:

       delta[mg/L] = counts * cells_per_agent / N_A[1/mol]
                     * MW[g/mol] * 1000[mg/g] / volume_L

   MW in g/mol numerically equals mg/mmol, so the same constant serves both
   the mM conversion and this mg/L delta.

   NOTE this is COUNT-based and touches no mass listener. An earlier revision
   of this docstring described a flux x biomass form with the biomass taken as
   ``cell_mass_fg * cells_per_agent * 1e-15`` — stale twice over: the code has
   not worked that way for some time, and ``cell_mass`` is TOTAL (wet) mass,
   ~3.33x dry, so using it as a gDW basis is the same error since corrected in
   PopulationAggregator. Do not reintroduce it.

Sign convention (verified against ``v2ecoli/processes/metabolism.py``
:func:`_fba_output_to_deltas`, lines 469-492): ``external_exchange_fluxes`` is
reported on a gDCW basis where the FBA exchange flux feeds
``delta_nutrients = (1/counts_to_molar) * exchange_fluxes`` — described there as
"the cumulative count added to the environment store". So a POSITIVE flux means
the molecule is *secreted* (added to the environment) and a NEGATIVE flux means
*uptake* (removed). Therefore O2 (consumed) carries a negative flux and yields a
NEGATIVE ``reactor.dissolved_o2`` delta (uptake depletes dissolved O2), while
CO2 (produced) carries a positive flux and yields a POSITIVE
``reactor.dissolved_co2`` delta. The Step passes the signed flux straight
through the conversion chain, so the sign is preserved without special-casing.

Time base: ``BiRDTransportProcess`` works in HOURS, while v2ecoli's cell sim
steps in SECONDS. The metabolic-exchange path needs NO interval scaling: it
differences ``environment.exchange``, a cumulative COUNT, and one difference
already covers exactly one WCM step. ``SECONDS_PER_HOUR`` and ``FG_PER_GRAM``
are retained as published constants but are not used by this Step; an earlier
version of this docstring described a per-hour flux conversion that the code
has never performed.

⚠ STANDING HAZARD for anything cumulative added to this Step
------------------------------------------------------------
The Step flow runs ONCE AT CYCLE START, before the processes and before the
PopulationAggregator have written anything (measured: ``calls_per_tick =
[2,1,1,...]``), so the first invocation sees an unpopulated state -- notably
``population.biomass_concentration_gL == 0.0``.

Any quantity that latches "the initial value" on its first invocation latches a
pre-population zero. Three distinct defects have come from this:

* mbp-03's O2 mass balance anchored its numerator after tick 0 while its
  denominator was anchored before it -- a 5.3% phantom deficit read as a leak;
* the per-agent first-observation rule exists precisely because tick 0's
  exchange store is not yet meaningful;
* the elemental ledger below latched ``b0 = 0`` and charged the INOCULUM against
  consumed glucose (measured carbon_residual -8.48, i.e. more carbon in cells
  than was ever consumed).

=> Defer the baseline until the store you depend on is live, and latch every
related baseline together so they refer to the same instant.
"""

from __future__ import annotations

import math
from typing import Any

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.stores import InPlaceDict


# --- Constants --------------------------------------------------------------

# Molecular weights (g/mol == mg/mmol).
MW_O2: float = 31.999
MW_CO2: float = 44.010

# Metabolism exchange-flux map keys (periplasmic compartment). Used for the
# environment.external_concentrations write (the cell-side env port IS keyed with
# the [p] compartment suffix).
O2_ID: str = "OXYGEN-MOLECULE[p]"
CO2_ID: str = "CARBON-DIOXIDE[p]"

# v2ecoli reports environmental exchange as CUMULATIVE molecule COUNTS at
# agents.*.environment.exchange, keyed by the BARE molecule name (no compartment
# suffix). Negative == uptake (removed from the environment), positive ==
# secretion. This is the real consumption source the coupler reads (the molar
# external_exchange_fluxes path the original draft assumed does not exist in
# v2ecoli's live state — see metabolism._fba_output_to_deltas: delta_nutrients is
# "the cumulative count added to the environment store"). ⚠ CUMULATIVE is literal:
# the store is a lineage running total that does not reset at division, so this
# Step DIFFERENCES it per agent to recover this tick's exchange.
O2_EXCHANGE_KEY: str = "OXYGEN-MOLECULE"
CO2_EXCHANGE_KEY: str = "CARBON-DIOXIDE"

SECONDS_PER_HOUR: float = 3600.0
FG_PER_GRAM: float = 1.0e-15
AVOGADRO: float = 6.02214076e23  # 1/mol

DEFAULT_CELLS_PER_AGENT: float = 1.0
DEFAULT_REACTOR_VOLUME_L: float = 1.0

# Medium concentration accumulator (#225 req-3 — close the ungraded
# substrate/glucose-conc + byproducts comparison axes). Maps a reactor store
# leaf (mmol/L, ADDITIVE float) to the bare environment.exchange molecule key the
# cell reports per-step COUNTS for (negative == uptake, positive == secretion).
# Each tick the coupler integrates those counts into a mmol/L delta on the leaf,
# so the reactor accumulates a medium CONCENTRATION the Beulig batch CSVs (also
# mmol/L) can be graded against — the same count->concentration conversion the
# dissolved-gas delta uses, minus the MW factor (mmol/L instead of mg/L).
#
#   delta[mmol/L] = counts * cells_per_agent / N_A[1/mol] * 1000 / volume_L
#
# Glucose (GLC) is UPTAKE (negative counts): seeded at the medium recipe glucose
# (reactor.glucose_medium_mM initial) and DRAWN DOWN. The byproducts are SECRETED
# (positive counts) and seeded at 0 so the leaf == cumulative secreted
# concentration. acetate/pyruvate are tracked too (sim observable) even though
# the WCM does not secrete them in this aerobic state (leaf stays ~0) — that is a
# genuine sim prediction, not a gap.
GLUCOSE_MEDIUM_LEAF: str = "glucose_medium_mM"
# Environment-store id for medium glucose, in the compartment-tagged form the
# rest of this Step's env_concs writes use (EnvironmentMirror resolves it onto
# the bare `GLC` boundary key).
GLUCOSE_ID: str = "GLC[p]"
GLUCOSE_EXCHANGE_KEY: str = "GLC"
# Medium AMMONIUM — the nitrogen source, tracked on exactly the same footing as
# glucose so the nitrogen ledger can close (mbp-04 declares a nitrogen_residual
# criterion). Seeded from the medium recipe and DRAWN DOWN by uptake.
# ⚠ The compartment tag is [c], NOT [p] as for glucose: measured, the importable
# id in exchange_data is `AMMONIUM[c]`. EnvironmentMirror strips the tag either
# way onto the bare `AMMONIUM` boundary key, but the id is not guessable from
# the glucose case.
AMMONIUM_MEDIUM_LEAF: str = "ammonium_medium_mM"
AMMONIUM_ID: str = "AMMONIUM[c]"
AMMONIUM_EXCHANGE_KEY: str = "AMMONIUM"

# --- Elemental ledger (reactor.diagnostics.{carbon,nitrogen}_residual) --------
# Carbon/nitrogen atoms per molecule, for the closure check. Byproduct entries
# key off the same reactor leaves the coupler already accumulates.
CARBON_ATOMS: dict[str, int] = {
    "acetate_mM": 2, "lactate_mM": 3, "formate_mM": 1,
    "ethanol_mM": 2, "pyruvate_mM": 3, "succinate_mM": 4,
}
GLUCOSE_CARBON_ATOMS: int = 6
# Elemental composition of E. coli dry mass. These are LITERATURE CONSTANTS, not
# fitted: a residual computed against them tests the simulation, not the fit.
# ⚠ BOTH are overridable via config (`biomass_c_fraction` / `biomass_n_fraction`)
# so a composite that holds sim_data can pass a MODEL-DERIVED value instead of a
# literature one. The coupler is reactor-side and has no sim_data of its own.
BIOMASS_C_FRACTION: float = 0.46    # gC / gDW
# ⭐⭐ 0.126, and THE BASIS IS THE WHOLE POINT. sim_data yields TWO internally
# consistent nitrogen fractions that differ by which dry mass you mean:
#   route B  0.1356  ParCa's DECLARED composition (`mass.mass_fractions`, 9
#                    classes, basal, tau=44 min)
#   route A  0.1260  the model's ACTUAL t=0 cell (initial_state bulk + unique)
# They differ because the initial state carries 38.6 fg -- 10.1% of dry mass --
# of small metabolites (the 125 targets in `mass._metTargetIds`) that the 9-class
# accounting omits, diluting protein (0.554 -> 0.481) and RNA (0.173 -> 0.126)
# by more N than the extra pool supplies.
# ⇒ THIS CONSTANT MULTIPLIES ROUTE A'S DENOMINATOR, so it must be route A's
# value. `biomass_gL` <- population.biomass_concentration_gL <- total_biomass_gDW
# <- `listeners.mass.dry_mass`. Route A is independently validated: dry/(dry+
# water) = 380.62/(380.62+888.11) = 0.3000, exactly sim_data's
# `cell_dry_mass_fraction`.
# ⚠ HISTORY, because both errors are instructive. This shipped as 0.12 -- the
# Kjeldahl 16%-N / factor-6.25 convention, and ~5% below even the lowest
# defensible model route. It was then "fixed" to 0.135, which is the RIGHT
# derivation on the WRONG basis: route B's value against route A's multiplicand,
# ~7% high. Literature (0.11-0.14) brackets all three, which is exactly why none
# of it was visible. The model's own proteome gives 17.38% N -- a protein factor
# of 5.75, the accepted bacterial value, not a model artifact.
# ⚠ CONSEQUENCE for the OD10 nitrogen claim, RE-DERIVED directly rather than
# scaled: at OD10 (3.4 gDW/L via DEFAULT_OD_TO_GDW=0.34) the requirement is
# 30.58 mM N = 101.0% of the 30.272 mM M9 recipe pool -- MARGINAL, essentially
# exactly at the line. NOT the "already insufficient" that 0.135 implied (32.77
# mM, 108%), and not the comfortable 96% that 0.12 implied.
# ⛔ BIOMASS_C_FRACTION below has the IDENTICAL basis question and has NOT been
# re-derived on either route. It is now the least-supported number in this
# ledger. Do not assume 0.46 is basis-consistent just because 0.126 now is.
BIOMASS_N_FRACTION: float = 0.126   # gN / gDW, route A (listeners.mass.dry_mass)
MW_C: float = 12.011
MW_N: float = 14.007
DIAGNOSTICS_LEAF: str = "diagnostics"
# Enumerated once here so the store seed, the coupler's output schema and the
# writer below cannot drift apart. ⚠ A leaf absent from the output schema is
# SILENTLY DROPPED by the InPlaceDict port (the same trap documented for the
# *_mM medium leaves), so adding a diagnostic means adding it HERE.
DIAGNOSTIC_LEAVES: tuple[str, ...] = (
    # ⭐ 1.0 == this tick's ledger is meaningful; 0.0 == it is NOT, and every
    # residual/fraction leaf below is NaN. FILTER ON THIS. It exists because the
    # residual previously emitted 0.0 for BOTH "balanced" and "no data" -- the
    # same value meaning two opposite things, in the instrument built to catch
    # exactly that class of defect.
    "ledger_valid",
    "carbon_residual", "nitrogen_residual",
    "carbon_in_mM", "carbon_biomass_mM", "carbon_co2_mM", "carbon_byproducts_mM",
    "nitrogen_in_mM", "nitrogen_biomass_mM",
    # ⭐ The PARTITION, reported beside the residual because closure alone does
    # not validate it. `[m@31Aug]` this model closes carbon to within a few
    # percent while routing ~98% of glucose carbon into BIOMASS and ~1% to CO2;
    # real aerobic growth sends 40-50% to CO2. A residual near zero on a
    # physically impossible split is exactly the "passing gate that cannot see
    # the failure" this diagnostic exists to avoid becoming.
    "carbon_to_biomass_frac", "carbon_to_co2_frac", "carbon_to_byproducts_frac",
)
# reactor leaf (mmol/L) -> bare environment.exchange byproduct key.
BYPRODUCT_LEAVES: dict[str, str] = {
    "acetate_mM":   "ACET",
    "lactate_mM":   "D-LACTATE",
    "formate_mM":   "FORMATE",
    "ethanol_mM":   "ETOH",
    "pyruvate_mM":  "PYRUVATE",
    "succinate_mM": "SUC",
}


class ReactorCellCoupler(Step):
    """Translate between v2ecoli's cell population and the BiRD reactor stores.

    See module docstring; spec in
    ``studies/mbp-03-*/study.yaml`` req-2 (reactor<->cell coupler).
    """

    name = "reactor_cell_coupler"
    # Bare type names (not {"_default": ...}) so config overrides propagate —
    # see PopulationAggregator's schema comment for the bigraph-schema rationale.
    config_schema = {
        "cells_per_agent":  "float",
        "reactor_volume_L": "float",
        "time_step":        "float",
        "track_medium":     "boolean",
        # Elemental composition used by the closure ledger. Optional: a caller
        # holding sim_data can pass a model-derived value rather than the
        # literature default (see BIOMASS_*_FRACTION).
        "biomass_c_fraction": "float",
        "biomass_n_fraction": "float",
    }
    topology = {
        "population":  ("population",),
        "reactor":     ("reactor",),
        "environment": ("environment",),
        "agents":      ("agents",),
    }

    def initialize(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.cells_per_agent = float(
            cfg.get("cells_per_agent") or DEFAULT_CELLS_PER_AGENT)
        self.reactor_volume_L = float(
            cfg.get("reactor_volume_L") or DEFAULT_REACTOR_VOLUME_L)
        # Accumulate medium glucose/byproduct concentrations into the reactor
        # store so the substrate/byproducts axes grade (#225 req-3), and publish
        # glucose back to the cell's environment (see next_update).
        #
        # ⚠ The `is None` branch below reads as "default ON", and is not: the
        # config_schema declares `track_medium: boolean`, so the framework fills
        # it as False whenever a caller omits it, and the branch never fires.
        # Direct construction (`ReactorCellCoupler(config={})`) therefore gets
        # medium tracking OFF. Composites switch it on explicitly
        # (`reactor_bird_coupled` passes track_medium=True), which is why this
        # has never shown up in a run. Left as-is rather than changed here:
        # flipping it would alter behaviour for any direct caller relying on
        # today's effective default.
        # ⚠⚠ FALSY-CHECK ON PURPOSE. `is None` looks more correct and IS WRONG
        # here: config_schema declares these as `float`, and the framework FILLS
        # a declared float key with **0.0** when the caller omits it (verified,
        # not assumed -- the same fill documented for `track_medium` above).
        # So `cfg.get(...)` never returns None for a declared key, `is None`
        # never fires, and an omitted fraction would be taken as 0.0 --
        # ZEROING the entire biomass side of the ledger silently in every
        # default-constructed coupler.
        # ⇒ 0.0 must mean "unset" here. The cost is that a caller cannot ask for
        # a genuine 0.0, which is the right trade: a zero elemental mass
        # fraction is unphysical, and silently honouring one would produce a
        # ledger that reports perfect closure by construction.
        # ⊕ `test_an_omitted_biomass_fraction_falls_back_to_the_constant` pins
        # this, because the falsy check reads like a bug and invites "fixing".
        self.biomass_c_fraction = float(
            cfg.get("biomass_c_fraction") or BIOMASS_C_FRACTION)
        self.biomass_n_fraction = float(
            cfg.get("biomass_n_fraction") or BIOMASS_N_FRACTION)
        track = cfg.get("track_medium")
        self.track_medium = True if track is None else bool(track)
        # How many invocations fell back to the configured `cells_per_agent`
        # because no population cell_count was readable. Expected to be exactly
        # 1 in a composite that wires the aggregator (the first pass, before
        # its write commits); a growing count means the population scale is not
        # reaching this Step at all.
        self.scale_fallbacks: int = 0
        # agents.*.environment.exchange is a LINEAGE-CUMULATIVE running total
        # (ecoli_baseline's `exchange_flux_basis` doc: "a LINEAGE-CUMULATIVE
        # molecule total that does not reset at division, so its time-average is
        # not a rate"). This Step needs THIS TICK's exchange, so it differences
        # the store per agent. Keyed by agent_id: at division each daughter
        # inherits the parent's total under a NEW id, so differencing the summed
        # total across agents would DOUBLE-COUNT it: with the store's
        # negative-uptake convention, summing two daughters that each carry the
        # parent's total doubles it in the NEGATIVE direction, i.e. a spurious
        # extra UPTAKE of one full lineage total on the division tick -- not a
        # sign flip, and not secretion. (An earlier version of this comment said
        # "reads as secretion"; that was backwards, and the division test's
        # assertion direction was derived from it.)
        self._prev_exchange: dict[str, dict[str, float]] = {}
        # A leaf's first observation for an agent yields 0.0 rather than a first
        # difference -- the same choice the gdcw deriver makes and for the same
        # reason: the store carries across division, so differencing against an
        # assumed zero would dump a whole generation's accumulation into one
        # tick. Costs one tick of exchange per new agent; a spike would be worse.
        self.first_observation_ticks: int = 0
        # --- elemental ledger state (reactor.diagnostics.*) ------------------
        # The medium pools are read on the FIRST tick and held: the coupler is
        # not told the recipe, and re-reading them each tick would compare the
        # pool against itself and the residual could never fail.
        self._c_ledger_glc0: float | None = None
        self._c_ledger_nh40: float | None = None
        # Cell-produced CO2 (mmol/L), accumulated here rather than read back off
        # `dissolved_co2`: that store is ALSO written by transport (stripping),
        # so the dissolved value is production MINUS what has been stripped and
        # is not the amount produced.
        self._cum_co2_mM: float = 0.0
        # Standing biomass at ledger start. The inoculum was NOT built from this
        # reactor's carbon, so charging it against consumed glucose makes the
        # residual scale with inoculum size (measured: a -15.2 residual at
        # OD~1.1, entirely from counting the seed cells as product).
        self._c_ledger_b0: float | None = None

    def inputs(self) -> dict[str, Any]:
        return {
            "population": InPlaceDict(),
            "reactor":    InPlaceDict(),
            "agents":     InPlaceDict(),
        }

    def outputs(self) -> dict[str, Any]:
        return {
            "reactor":     InPlaceDict(),
            "environment": InPlaceDict(),
        }

    # --- main update -------------------------------------------------------

    def next_update(self, timestep, states):
        update: dict[str, Any] = {}

        reactor_out: dict[str, float] = {}
        env_concs: dict[str, float] = {}

        reactor = states.get("reactor") or {}
        # Reactor volume: prefer the live store, fall back to config default.
        volume_L = _as_float(reactor.get("volume_L"))
        if not volume_L or volume_L <= 0.0:
            volume_L = self.reactor_volume_L

        # 1. Biomass passthrough (g/L) — overwrite the reactor biomass input.
        population = states.get("population") or {}
        biomass_gL = population.get("biomass_concentration_gL")
        if biomass_gL is not None:
            reactor_out["biomass"] = _as_float(biomass_gL)

        # 2. Dissolved gas (mg/L) -> environment concentration (mM).
        do2 = reactor.get("dissolved_o2")
        dco2 = reactor.get("dissolved_co2")
        if do2 is not None:
            env_concs[O2_ID] = _as_float(do2) / MW_O2
        if dco2 is not None:
            env_concs[CO2_ID] = _as_float(dco2) / MW_CO2
        # Medium glucose (mM) -> the cell's environment, on the same footing as
        # the dissolved gases. Before this, `glucose_medium_mM` was written by
        # this Step and read by NOTHING: draining the pool changed nothing the
        # cells experienced, so glucose-limited growth was unreachable however
        # the reactor was configured. The leaf is already seeded at the medium
        # recipe and drawn down below, and is already in mM, so no MW
        # conversion applies. Clamp at zero: an exhausted pool is zero
        # available, never negative.
        if self.track_medium:
            glc_medium = reactor.get(GLUCOSE_MEDIUM_LEAF)
            if glc_medium is not None:
                env_concs[GLUCOSE_ID] = max(_as_float(glc_medium), 0.0)
            nh4_medium = reactor.get(AMMONIUM_MEDIUM_LEAF)
            if nh4_medium is not None:
                env_concs[AMMONIUM_ID] = max(_as_float(nh4_medium), 0.0)

        # 3. Metabolic exchange demand -> additive dissolved-gas delta (mg/L).
        #
        # v2ecoli emits environmental exchange as CUMULATIVE molecule COUNTS at
        # agents.*.environment.exchange (negative == uptake), DIFFERENCED per
        # agent above to recover this tick's exchange. Convert counts/step
        # for one agent, scaled by cells_per_agent (representative-sampling
        # population scale), to a mg/L delta on the shared dissolved store:
        #
        #   delta[mg/L] = counts * cells_per_agent / N_A[1/mol]
        #                 * MW[g/mol] * 1000[mg/g] / volume_L
        #
        # The DIFFERENCED counts cover one WCM step, so NO interval
        # (timestep/3600) scaling is applied here. (Before 2026-08-29 this Step
        # consumed the store's value directly, on the reading that it was already
        # per-step. It is not -- it is a running total -- so the reactor was
        # drained by ~N/2 for an N-tick run: measured 22.2 mM of a 22.2 mM pool
        # gone while the population formed 2.3 mg of biomass, an apparent yield
        # of 0.0006 g/g against a ~0.54 ceiling.) Sign flows straight through: negative O2
        # counts (uptake) -> negative dissolved_o2 delta. This is the per-step
        # mass exchanged between the cell population and the reactor liquid.
        agents = states.get("agents") or {}
        o2_counts = 0.0   # molecules/step, signed (negative == uptake)
        co2_counts = 0.0  # molecules/step, signed (positive == secretion)
        # Medium glucose + byproduct counts (signed; negative == uptake).
        glc_counts = 0.0
        nh4_counts = 0.0
        byproduct_counts = {leaf: 0.0 for leaf in BYPRODUCT_LEAVES}
        seen_agents: set[str] = set()
        for _agent_id, agent_state in agents.items():
            exch = _extract_environment_exchange(agent_state)
            seen_agents.add(_agent_id)
            # F4: decide NEW-ness from whether this agent has been SEEN, before
            # setdefault creates its entry. `not prev` was equivalent only while
            # every tick wrote a baseline for every key; with the absent-key
            # guard below, an agent whose snapshot carries none of our keys
            # leaves `prev` empty forever and would count as new EVERY tick,
            # conflating "many divisions" with "one agent with no exchange data".
            new_agent = _agent_id not in self._prev_exchange
            prev = self._prev_exchange.setdefault(_agent_id, {})

            def _tick_delta(key: str) -> float:
                """This tick's exchange for one molecule: the store is a running
                total, so difference it against this agent's previous reading.

                Two cases yield 0.0 rather than a difference:

                * FIRST OBSERVATION of a (agent, molecule) pair. The store
                  carries across division, so differencing against an assumed
                  zero would dump a whole generation's accumulation into one
                  tick. The gdcw deriver makes the same choice for the same
                  reason (steps/derivers/exchange_flux_listener.py).
                * KEY MISSING OR ``None`` in this tick's snapshot. The emit
                  cadence can snapshot mid-init, so ``exch`` may be missing a
                  key it had last tick, or carry it as ``None`` (see
                  _extract_environment_exchange). Treating either as 0.0 would
                  overwrite the baseline with 0.0 and then emit the whole
                  running total as a spike on the next tick -- the very spike
                  the first-observation rule exists to prevent. Leave the
                  baseline untouched and skip the tick.
                  ⚠ ``None`` must be tested SEPARATELY from absence: a bare
                  ``key not in exch`` passes a present ``None`` through to
                  ``_as_float``, which maps it to 0.0 -- measured at +0.332 mM
                  of spurious secretion followed by -0.498 mM of spurious
                  uptake, against a -0.166 mM normal tick.
                """
                value = exch.get(key)
                if value is None:
                    return 0.0
                total = _as_float(value)
                previous = prev.get(key)
                prev[key] = total
                return 0.0 if previous is None else total - previous

            o2_counts += _tick_delta(O2_EXCHANGE_KEY)
            co2_counts += _tick_delta(CO2_EXCHANGE_KEY)
            if self.track_medium:
                glc_counts += _tick_delta(GLUCOSE_EXCHANGE_KEY)
                nh4_counts += _tick_delta(AMMONIUM_EXCHANGE_KEY)
                for leaf, key in BYPRODUCT_LEAVES.items():
                    byproduct_counts[leaf] += _tick_delta(key)
            if new_agent:
                self.first_observation_ticks += 1
        # Drop agents that are gone (division retires the parent id) so the
        # bookkeeping cannot grow without bound over a long lineage.
        for stale in set(self._prev_exchange) - seen_agents:
            del self._prev_exchange[stale]

        if agents:
            # Population scale. `PopulationAggregator` multiplies biomass and
            # cell_count by 2**doublings under `representative_doubling`, so the
            # followed lineage stands in for a growing population; this Step
            # scaled exchange by `cells_per_agent` ALONE. The reactor therefore
            # saw an accumulating biomass consuming a constant amount of
            # substrate -- by generation 4, 8x the biomass eating 1x the
            # glucose and oxygen. That is a mass-conservation violation, not an
            # accuracy problem, and it makes any carbon/oxygen closure
            # criterion fail by construction.
            #
            # `population.cell_count` already carries cells_per_agent *
            # growth_factor, so deriving the per-agent scale from it keeps ONE
            # authoritative number rather than a second copy that can drift.
            # The aggregator precedes this Step in flow_order, so from the
            # second invocation onward the value is current for this tick.
            #
            # ⚠ NOT on the first. Measured under `Composite.run()`: the
            # aggregator's write is not committed to the store within the first
            # pass, so the coupler's FIRST invocation reads 0.0 and takes the
            # fallback below -- exactly once per composite. (An earlier revision
            # of this comment claimed tick-0 propagation was verified. It is
            # path-dependent, and on the `run()` path used by the harness and
            # the figures script it is false. Corrected after measurement.)
            #
            # Consequence: in `fixed` mode the fallback is harmless, because
            # cell_count = n_agents * cells_per_agent * 1.0 makes the two
            # expressions identical. Under `representative_doubling` at
            # generation g, that one tick of exchange is under-scaled by
            # 2**(g-1). Small, but it is the very error this Step is being
            # fixed for, so the fallback now RECORDS that it fired rather than
            # switching scales silently.
            n_agents = len(agents)
            cell_count = _as_float(population.get("cell_count"))
            if cell_count > 0.0 and n_agents:
                cells_per_agent_effective = cell_count / n_agents
            else:
                cells_per_agent_effective = self.cells_per_agent
                self.scale_fallbacks += 1
            counts_to_mgL = cells_per_agent_effective / AVOGADRO * 1000.0 / volume_L
            o2_delta = o2_counts * counts_to_mgL * MW_O2
            co2_delta = co2_counts * counts_to_mgL * MW_CO2
            # Safety clamp: a single step's uptake must not drive the dissolved
            # store negative (transport replenishes next step). The coupler runs
            # after transport in the flow, so reactor.dissolved_o2 already holds
            # this step's transport delta; cap the consumption at what's present.
            do2_now = _as_float(reactor.get("dissolved_o2"))
            if o2_delta < 0.0 and (do2_now + o2_delta) < 0.0:
                o2_delta = -do2_now
            dco2_now = _as_float(reactor.get("dissolved_co2"))
            if co2_delta < 0.0 and (dco2_now + co2_delta) < 0.0:
                co2_delta = -dco2_now
            reactor_out["dissolved_o2"] = o2_delta
            reactor_out["dissolved_co2"] = co2_delta

            # Medium concentration accumulator (mmol/L) — same count->conc
            # conversion as the dissolved-gas delta minus the MW factor
            # (counts_to_mgL is already counts->mmol/L; *MW gives mg/L).
            if self.track_medium:
                counts_to_mM = counts_to_mgL  # = cells/N_A * 1000 / volume_L
                # Glucose: uptake (negative) -> draw down the seeded medium pool.
                # Clamp the draw at what remains; unlike the dissolved gases
                # (which are replenished by transport) an exhausted medium pool
                # has nothing to restore it, so an unclamped delta drives the
                # concentration negative and the cell would be told it has less
                # than nothing.
                glc_delta = glc_counts * counts_to_mM
                glc_now = _as_float(reactor.get(GLUCOSE_MEDIUM_LEAF))
                if glc_delta < 0.0 and (glc_now + glc_delta) < 0.0:
                    glc_delta = -glc_now
                reactor_out[GLUCOSE_MEDIUM_LEAF] = glc_delta
                # Ammonium: same shape and the same clamp reasoning as glucose —
                # an exhausted medium pool has nothing to restore it, so an
                # unclamped draw would drive the concentration negative and the
                # cell would be told it has less than nothing.
                nh4_delta = nh4_counts * counts_to_mM
                nh4_now = _as_float(reactor.get(AMMONIUM_MEDIUM_LEAF))
                if nh4_delta < 0.0 and (nh4_now + nh4_delta) < 0.0:
                    nh4_delta = -nh4_now
                reactor_out[AMMONIUM_MEDIUM_LEAF] = nh4_delta
                # Byproducts: secretion (positive) -> accumulate from 0.
                for leaf in BYPRODUCT_LEAVES:
                    reactor_out[leaf] = byproduct_counts[leaf] * counts_to_mM

                # --- elemental closure ------------------------------------------
                # ⚖ CONVENTION, stated because it changes the number: both residuals
                # are a RATIO OF CUMULATIVE TOTALS (sum of in, sum of out), not a
                # mean of per-tick ratios. On comparable data those differ by ~25%,
                # which is wider than the bands a card would grade against, so the
                # convention is part of the result and is recorded with it.
                self._cum_co2_mM += co2_counts * counts_to_mM
                biomass_now = _as_float(population.get("biomass_concentration_gL"))
                # ⚠ Defer the ENTIRE baseline until the population store is actually
                # populated. The Step flow runs once at CYCLE START -- before the
                # PopulationAggregator has written anything -- so the first
                # invocation sees biomass_concentration_gL == 0.0. Latching the
                # baseline there sets b0 = 0 and every subsequent tick charges the
                # INOCULUM against consumed glucose: measured C_res -8.48 at t=500,
                # with carbon_biomass_mM (16.3) an order of magnitude above
                # carbon_in_mM (1.7). All three baselines are latched together so
                # they refer to the same instant.
                latched_this_tick = False
                if self._c_ledger_glc0 is None and biomass_now > 0.0:
                    self._c_ledger_glc0 = _as_float(reactor.get(GLUCOSE_MEDIUM_LEAF))
                    self._c_ledger_nh40 = _as_float(reactor.get(AMMONIUM_MEDIUM_LEAF))
                    self._c_ledger_b0 = biomass_now
                    latched_this_tick = True
                # Biomass FORMED in this reactor, read from the population store
                # (reactor.biomass is the coupler's own passthrough and reads 0.0
                # until the first write, so it cannot supply the baseline).
                biomass_gL = biomass_now - (self._c_ledger_b0 or 0.0)

                glc_now_after = _as_float(reactor.get(GLUCOSE_MEDIUM_LEAF)) + glc_delta
                nh4_now_after = _as_float(reactor.get(AMMONIUM_MEDIUM_LEAF)) + nh4_delta

                if self._c_ledger_glc0 is None:
                    # Baseline not latched yet (pre-population tick): emit nothing
                    # rather than a residual computed against a zero baseline.
                    self._cum_co2_mM = 0.0
                    glc_now_after = nh4_now_after = 0.0
                c_in = ((self._c_ledger_glc0 or 0.0) - glc_now_after) * GLUCOSE_CARBON_ATOMS
                c_biomass = biomass_gL * self.biomass_c_fraction / MW_C * 1000.0
                c_byproducts = sum(
                    (_as_float(reactor.get(leaf)) + byproduct_counts[leaf] * counts_to_mM)
                    * atoms
                    for leaf, atoms in CARBON_ATOMS.items()
                )
                c_out = c_biomass + self._cum_co2_mM + c_byproducts

                n_in = (self._c_ledger_nh40 or 0.0) - nh4_now_after
                n_out = biomass_gL * self.biomass_n_fraction / MW_N * 1000.0

                # ⭐⭐ VALIDITY. Three ways a tick carries no meaningful ledger,
                # and ALL of them previously emitted 0.0 -- indistinguishable
                # from "perfectly balanced".
                #   (a) baseline not latched (pre-population tick);
                #   (a2) ⭐ THE LATCH TICK ITSELF. `b0 = biomass_now` and then
                #       `biomass_gL = biomass_now - b0` == EXACTLY 0.0, every
                #       run, by construction -- so the whole "out" side is zero
                #       while `c_in` is a real uptake. `[m@01Sep]` t=0 emitted
                #       carbon_residual = -1.929 (293% of carbon from nowhere)
                #       and was marked VALID, because 0.0 >= 0.0 passes.
                #       ⛔ It is a GUARANTEED-EVERY-RUN artifact that no run
                #       length dilutes, so max_abs over the valid series
                #       inherited it however long you ran. A/B on one window:
                #       max_abs 1.929 admitting it vs 0.739 excluding it,
                #       against mbp-04's 0.02 band.
                #       ⚠ Fixed by naming the ACTUAL defect -- no interval has
                #       elapsed since the baseline -- NOT by tightening the
                #       biomass test to `> 0.0`, which would also discard
                #       legitimate zero-growth steady-state ticks later in a
                #       run, where no net biomass change alongside real glucose
                #       consumption is meaningful data.;
                #   (b) no glucose consumed yet since the baseline, so the
                #       residual's DENOMINATOR is zero or negative;
                #   (c) ⚠ biomass BELOW its own latched baseline. `[m@31Aug]` the
                #       population store dips early -- 0.38062 -> 0.37981, ~11
                #       ticks to recover -- so `biomass_gL` goes NEGATIVE and
                #       carbon_biomass_mM with it (11 of 120 ticks; at t=12
                #       bio=-0.175, byp=+1.175). The three fractions still summed
                #       to 1.0 throughout, which is why the unit test passed: a
                #       partition summing to 1 is not a partition being sane.
                valid = (
                    self._c_ledger_glc0 is not None
                    and not latched_this_tick
                    and c_in > 0.0
                    and biomass_gL >= 0.0
                )
                # ⚖ CONVENTION, declared here because it changes the number a
                # criterion computes: `c_in` is glucose consumed SINCE THE
                # BASELINE LATCHED, not this tick's flux -- so each emitted
                # residual is already a RATIO OF CUMULATIVE AGGREGATES. Reducing
                # the series with `mean` therefore gives a mean-of-ratios over
                # ratios-of-aggregates, which is NOT the endpoint residual and is
                # not what "the balance closes" means. Grade the ENDPOINT (last
                # valid tick), or max_abs over VALID ticks only.
                _nan = math.nan
                diagnostics = {
                    "ledger_valid": 1.0 if valid else 0.0,
                    # Fractional closure error; 0.0 == the ledger balances, NaN ==
                    # this tick carries no ledger. Sign is (in - out)/in, so
                    # POSITIVE means carbon went missing.
                    "carbon_residual":   ((c_in - c_out) / c_in) if valid else _nan,
                    "nitrogen_residual": ((n_in - n_out) / n_in) if (valid and n_in > 0) else _nan,
                    # Components, so a failing residual says WHICH term is wrong
                    # rather than only that something is. Emitted even when the
                    # tick is invalid -- they are what diagnose WHY it is.
                    "carbon_in_mM":        c_in,
                    "carbon_biomass_mM":   c_biomass,
                    "carbon_co2_mM":       self._cum_co2_mM,
                    "carbon_byproducts_mM": c_byproducts,
                    "nitrogen_in_mM":      n_in,
                    "nitrogen_biomass_mM": n_out,
                    # Fractions of carbon OUT. These are where a physiological
                    # violation shows up; the residual above cannot see it.
                    "carbon_to_biomass_frac":    (c_biomass / c_out) if (valid and c_out > 0) else _nan,
                    "carbon_to_co2_frac":        (self._cum_co2_mM / c_out) if (valid and c_out > 0) else _nan,
                    "carbon_to_byproducts_frac": (c_byproducts / c_out) if (valid and c_out > 0) else _nan,
                }
                # ⚠ The writer must emit EXACTLY the enumerated leaves. A leaf in
                # the dict but not in DIAGNOSTIC_LEAVES is SILENTLY DROPPED by the
                # InPlaceDict port; a leaf enumerated but not written keeps its
                # stale value forever. The constant was introduced to stop that
                # drift and then not actually used -- so assert it here rather
                # than trusting two hand-maintained lists to agree.
                # ⚠ SET equality, not tuple: only membership matters to the
                # port, and an order-sensitive check made a semantically
                # harmless reorder of two writer keys red 14 tests. It also
                # raises inside next_update, so keep it cheap and correct --
                # aborting a long simulation over key ORDER would be worse than
                # the drift it guards against.
                # HONEST NOTE: deleting this assertion leaves the suite green
                # (measured) -- test_the_writer_emits_exactly_the_enumerated_
                # leaves already covers the invariant, and the dict below is a
                # literal with no conditional branch, so today this CANNOT fire.
                # Kept as cheap insurance for the day that dict becomes
                # conditional, NOT because it is currently load-bearing. Do not
                # cite it as coverage.
                assert set(diagnostics) == set(DIAGNOSTIC_LEAVES), (
                    "ledger writer and DIAGNOSTIC_LEAVES disagree: "
                    f"written={sorted(diagnostics)} "
                    f"enumerated={sorted(DIAGNOSTIC_LEAVES)}")
                reactor_out[DIAGNOSTICS_LEAF] = diagnostics

        if reactor_out:
            update["reactor"] = reactor_out
        if env_concs:
            update["environment"] = {"external_concentrations": env_concs}
        return update

    def update(self, state, interval=None):
        return self.next_update(state.get("timestep", 1.0), state)


# --- helpers ----------------------------------------------------------------

def _as_float(value: Any) -> float:
    """Coerce a possibly pint-Quantity scalar to a bare float."""
    if value is None:
        return 0.0
    if hasattr(value, "magnitude"):
        return float(value.magnitude)
    return float(value)



def _extract_exchange_fluxes(agent_state: dict | Any) -> dict[str, Any]:
    """Return ``metabolism.external_exchange_fluxes`` map, or {} if absent.

    Retained for reference/back-compat; v2ecoli's live state does NOT populate
    this path (see ``_extract_environment_exchange``, the real source).
    """
    try:
        metabolism = agent_state.get("metabolism", {}) if hasattr(agent_state, "get") else {}
        fluxes = metabolism.get("external_exchange_fluxes")
        if isinstance(fluxes, dict):
            return fluxes
        return {}
    except (AttributeError, KeyError, TypeError):
        return {}


def _extract_environment_exchange(agent_state: dict | Any) -> dict[str, Any]:
    """Return ``environment.exchange`` (CUMULATIVE molecule COUNTS), or {} if absent.

    A dict keyed by the bare molecule name (e.g. ``OXYGEN-MOLECULE``), value =
    the signed molecule count added to the environment **since the start of the
    lineage** (negative == uptake).

    ⚠ This is a RUNNING TOTAL, not a per-step delta, and it does NOT reset at
    division — a daughter inherits the parent's total. Callers must difference
    it per agent to recover one tick's exchange; see ``ReactorCellCoupler`` and
    ``ecoli_baseline``'s ``exchange_flux_basis`` documentation. Reading it as a
    per-step value was the #632 defect: it drained the reactor by elapsed time
    rather than by cell demand, ~N/2 too much over N ticks.

    Defensive against missing intermediate keys (emit cadence can snapshot
    mid-init) — returns {} rather than raising.
    """
    try:
        env = agent_state.get("environment", {}) if hasattr(agent_state, "get") else {}
        exch = env.get("exchange")
        if isinstance(exch, dict):
            return exch
        return {}
    except (AttributeError, KeyError, TypeError):
        return {}
