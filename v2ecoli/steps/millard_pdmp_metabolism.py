"""MillardPDMPMetabolism — Millard 2017 ODE + LQR + bulk delta emission.

Composite-side seam for the v2ecoli-pdmp Phase 1 milestone: replaces
v2ecoli's tFBA Metabolism process with a single Process that
(a) advances the Millard 2017 kinetic ODE one WCM tick via basico/COPASI,
(b) accepts an LQR control signal on its `lqr_control` input port,
(c) translates the resulting mM concentrations into v2ecoli bulk-count
deltas using the millard_v2ecoli_species_map, and
(d) emits those deltas to the structured bulk store the WCM's downstream
processes (Equilibrium, TfBinding, transcription, etc.) actually read.

Why this is one Process rather than a Millard+Bridge+Indexer chain:
process-bigraph silently drops same-tick writes to a store when another
edge in the same composite declares that store as an input. The
intermediate `central_metabolites` store the staged chain wanted to share
hit this issue and lost Millard's updates. Internalising the chain in
one Process eliminates the shared-store wiring and is the right
architectural shape anyway — Millard's mM concentrations are
implementation state, not biology the WCM bulk store needs to mirror.

`central_metabolites` is still exposed as an OUTPUT for observability;
no other Step or Process reads it (otherwise the wiring quirk above
would kick in again).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from process_bigraph import Process

from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.quantity_helpers import fg_magnitude
from v2ecoli.types.stores import InPlaceDict


DEFAULT_MAPPING = "v2ecoli/data/millard_v2ecoli_species_map.yaml"
DEFAULT_CELL_VOLUME_L = 1.0e-15   # fallback when listeners.mass is unset
DEFAULT_CELL_DENSITY_G_PER_L = 1100.0
AVOGADRO = 6.02214076e23
FG_PER_G = 1.0e-15

# --- Medium-exchange (boundary) accounting ---------------------------------
# Maps a Millard boundary REACTION -> (bare v2ecoli env name, stoichiometric
# coefficient of the exchanged species in that reaction, sign). Used to emit
# ``environment.exchange`` — the per-cell, per-tick molecule-count exchanged
# with the medium — matching the WCM convention in
# ``v2ecoli/processes/metabolism.py`` (BARE names, NEGATIVE = uptake / removed
# from medium, POSITIVE = secretion / added to medium).
#
# Why flux-based (NOT concentration-delta): in the Millard SBML, O2 is a
# ``fixed`` species (its mM is held constant -> a per-tick conc-delta is always
# 0), and external glucose ``GLCx`` is replenished by an artificial chemostat
# feed ``_GLC_FEED`` (flux ~10x uptake) so its mM RISES even as the cell
# consumes it -> a conc-delta would report the WRONG sign (apparent secretion).
# The physically correct medium exchange is the boundary-reaction flux (mM/s)
# integrated over the tick. The flux is read from the same per-tick
# ``central_fluxes`` map this Process already computes.
#
# CO2 / H2O: the Millard SBML has no free-CO2 species and holds HCO3 + H2O as
# FIXED buffer species, so their efflux is accounted separately from the
# reaction stoichiometry — see BYPRODUCT_EFFLUX_REACTIONS below.
# --- Energy/redox currency homeostasis ------------------------------------
# Millard-governed metabolites that whole-cell processes OUTSIDE the Millard
# reaction network (translation, transcription, charging, ...) consume
# cell-wide. The FBA Metabolism this Process replaces regenerates them every
# tick to balance that demand, so in the FBA arm ATP[c] stays ~steady. Millard
# 2017's reduced central-carbon network does NOT carry the whole-cell ATP/redox
# turnover, and in delta_mode it only nudges the bulk by its own tiny per-tick
# ΔmM. With nothing replenishing them, these pools drain monotonically to zero
# (ATP[c] ~tick 130) and then negative, which makes the ATP-dependent
# TF-phosphorylation reactions in the Equilibrium process unsolvable
# ("Negative values at equilibrium steady state"). The homeostatic floor in
# update() tops these — and only these — pools back up to the concentration
# Millard's kinetics sustain (current_mM x V), so Millard acts as the cell's
# metabolic engine for the energy currency exactly as the FBA arm does.
HOMEOSTATIC_COFACTOR_MILLARD_IDS = frozenset({
    "ATP", "ADP", "AMP", "NAD", "NADH", "NADP", "NADPH",
})


MEDIUM_EXCHANGE_REACTIONS: dict[str, tuple[str, float, float]] = {
    # reaction:   (bare_name,          species_coeff, sign)
    "CYTBO":      ("OXYGEN-MOLECULE",  1.0,           -1.0),  # O2 consumed (uptake)
    "XCH_GLC":    ("GLC",              1.0,           -1.0),  # GLCx -> GLCp (uptake)
    "_ACE_OUT":   ("ACET",            1.0,           +1.0),  # ACEx -> medium (secretion)
}


# --- Byproduct efflux (CO2 / H2O) -----------------------------------------
# Decarboxylation CO2 and respiratory H2O cross the cell boundary as efflux,
# but in the Millard SBML their carriers (HCO3, H2O) are FIXED species — the
# kinetic network produces them into an infinite buffer, so they never appear
# as a tracked pool and (before this) their mass left the boundary accounting
# entirely. That made O2 + glucose UPTAKE look like phantom mass IMPORT (the
# substrate carbon actually leaves as CO2, the consumed O2 leaves as H2O),
# inflating the Millard-cell mass-conservation residual ~40x (measured: net
# boundary mass-in +2.46 fg / 40 ticks vs an actual Δcell_mass of ~+0.06 fg).
# Accounting the NET byproduct production as medium SECRETION (positive count,
# WCM convention) closes the boundary: for a respiring cell, glucose-C in =
# CO2-C out and O2 in = H2O out, so net boundary mass ≈ retained biomass.
#
# CO2 is modeled in Millard as bicarbonate (HCO3) produced by the
# decarboxylating reactions (GND/PDH/ICD/LPD/MAE/PCK); PPC (anaplerotic) re-
# fixes it, so the NET CO2 efflux is producers − PPC. Mass is accounted as CO2
# (44 g/mol — the carbon physically leaving the organic pool, the respiratory-
# quotient convention), NOT bicarbonate: the extra O/H of HCO3 come from the
# fixed H2O buffer, not the dynamic cell pools. H2O efflux is the 2 H2O per
# CYTBO turnover. Maps a BARE v2ecoli env name -> {reaction: stoich_coeff of
# the byproduct in that reaction} (positive coeff = produced/secreted).
BYPRODUCT_EFFLUX_REACTIONS: dict[str, dict[str, float]] = {
    "CARBON-DIOXIDE": {
        "GND": 1.0, "PDH": 1.0, "ICD": 1.0, "LPD": 1.0,
        "MAE": 1.0, "PCK": 1.0, "PPC": -1.0,
    },
    "WATER": {"CYTBO": 2.0},
}


# --- Reactor->Millard O2 feedback (#225 item #4) ---------------------------
# The reactor's dissolved O2 (and external glucose) reach this Process through
# the ``external_concentrations`` input, but the bioreactor coupler / environment
# mirror write them under v2ecoli molecule names (bare or [p]/[c]-suffixed),
# whereas COPASI species are keyed by their SBML id. This alias table maps the
# v2ecoli names onto the Millard SBML ids so the overwritten boundary value
# actually drives the rate law. Raw SBML ids (e.g. "O2") still pass straight
# through via ``self.sbml_to_name``; this only rescues the aliased names.
#
# O2 is a ``fixed`` species in the SBML, but its CYTBO rate law reads [O2]
# linearly (Vmax/(...)*(QH2^2*O2 - Q^2/Keq)); overwriting the fixed value each
# tick therefore THROTTLES respiration as the reactor's dissolved O2 falls — the
# reverse leg of the reactor<->cell O2 loop. Driving the fixed-species value
# (rather than un-fixing O2) keeps the ODE stable: O2 is never integrated, so it
# can never be drawn negative / stiff, and the standalone cell (no external
# drive) keeps the model's calibrated air-saturated 0.21 mM.
EXTERNAL_NAME_TO_SBML: dict[str, str] = {
    "OXYGEN-MOLECULE":    "O2",
    "OXYGEN-MOLECULE[p]": "O2",
    "OXYGEN-MOLECULE[c]": "O2",
    "GLC":                "GLCx",
    "GLC[p]":             "GLCx",
    "GLC[c]":             "GLCx",
}


def _set_initial_concentrations(changes, dm) -> None:
    """Overwrite the initial concentration of named species on a COPASI model.

    Mirrors pbg_copasi.processes._set_initial_concentrations: set each
    species' InitialConcentration then commit via updateInitialValues so the
    next time course starts the overwritten species at the new value while
    every other (internal) species carries over from the model's current
    state.

    changes: iterable of (copasi_species_name, value) pairs
    dm: COPASI DataModel as returned by basico.load_model
    """
    import COPASI

    model = dm.getModel()
    references = COPASI.ObjectStdVector()
    for name, value in changes:
        species = model.getMetabolite(name)
        if species is None:
            continue
        species.setInitialConcentration(float(value))
        references.append(species.getInitialConcentrationReference())
    if len(references) > 0:
        model.updateInitialValues(references)


def _load_millard_to_v2ecoli(mapping_file: str) -> dict[str, str]:
    path = Path(mapping_file)
    if not path.is_absolute():
        path = Path.cwd() / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    m2v: dict[str, str] = {}
    for section in ("adenylates", "redox", "glycolysis", "tca", "ppp", "common"):
        for entry in raw.get(section, []) or []:
            if entry.get("role") != "shared":
                continue
            mid = entry.get("millard_id") or entry.get("millard")
            vid = entry.get("v2ecoli_id") or entry.get("v2ecoli")
            if mid and vid:
                m2v[mid] = vid
    return m2v


class MillardPDMPMetabolism(Process):
    """Millard 2017 ODE + LQR control + bulk delta emission, in one Process."""

    name = "millard-pdmp-metabolism"
    config_schema = {
        "model_source": {"_default":
            "v2ecoli/models/sbml/millard2017_central_metabolism.xml"},
        "tick_s": {"_default": 1.0},
        "intervals": {"_default": 10},
        "control_reaction": {"_default": "PTS_4"},
        "control_parameter": {"_default": "kF"},
        "u_clip": {"_default": 0.5},
        "mapping_file": {"_default": DEFAULT_MAPPING},
        "cell_volume_L": {"_default": DEFAULT_CELL_VOLUME_L},
        # When True, ignore the static cell_volume_L config and compute V
        # dynamically from listeners.mass.cell_mass / cell_density on every
        # tick. This is the biology-accurate path — v2ecoli's cell volume
        # grows during the cycle, and a static V causes Millard counts to
        # drift from the bulk's growing reference frame.
        "use_live_volume": {"_default": True},
        "cell_density_g_per_L": {"_default": DEFAULT_CELL_DENSITY_G_PER_L},
        "min_count": {"_default": 0.0},
        # When True (default), bulk deltas are computed from the CHANGE in
        # Millard mM since last tick × V — preserves v2ecoli's initial bulk
        # state and adds only the kinetic perturbation. When False, deltas
        # drive bulk to the absolute Millard SS target (legacy behavior
        # that collapsed initial amino-acid pools at t=0).
        "delta_mode": {"_default": True},
    }

    def initialize(self, config):
        self.parameters = config or {}
        import basico
        self._basico = basico
        # Keep an explicit handle to THIS process's COPASI model so per-tick
        # flux reads (basico.get_reactions(model=self._model)) are guaranteed
        # to read the same model this process advances, regardless of any
        # other basico model that may have become the global "current" model.
        self._model = basico.load_model(self.parameters["model_source"])
        # SBML-species-id -> COPASI display name, so external_concentrations
        # (keyed by SBML id) can be applied via getMetabolite(name). Mirrors
        # pbg_copasi.processes.BaseCopasi.sbml_to_name.
        spec_df = basico.get_species(model=self._model)
        self.sbml_to_name = {
            spec_df.loc[name, "sbml_id"]: name for name in spec_df.index
        }
        self.tick_s = float(self.parameters.get("tick_s", 1.0))
        self.intervals = int(self.parameters.get("intervals", 10))
        self.control_reaction = self.parameters.get("control_reaction", "PTS_4")
        self.control_parameter = self.parameters.get("control_parameter", "kF")
        self.u_clip = float(self.parameters.get("u_clip", 0.5))
        params = basico.get_reaction_parameters(reaction_name=self.control_reaction)
        param_row_name = f"({self.control_reaction}).{self.control_parameter}"
        if params is not None and param_row_name in params.index:
            self.baseline_value = float(params.loc[param_row_name]["value"])
        else:
            self.baseline_value = 1.0
        # Bulk-indexing setup
        self.mapping_file = self.parameters.get("mapping_file", DEFAULT_MAPPING)
        self.cell_volume_L_static = float(self.parameters.get(
            "cell_volume_L", DEFAULT_CELL_VOLUME_L))
        self.use_live_volume = bool(self.parameters.get("use_live_volume", True))
        self.cell_density_g_per_L = float(self.parameters.get(
            "cell_density_g_per_L", DEFAULT_CELL_DENSITY_G_PER_L))
        self.min_count = float(self.parameters.get("min_count", 0.0))
        self.delta_mode = bool(self.parameters.get("delta_mode", True))
        self._prev_mM: dict[str, float] | None = None
        self._m2v = _load_millard_to_v2ecoli(self.mapping_file)
        # mM × 1e-3 × V_L × Avogadro = count. When use_live_volume=True
        # this is recomputed each tick from listeners.mass; the static
        # value is the fallback when listener data isn't available yet.
        self._conc_to_count_static = 1e-3 * self.cell_volume_L_static * AVOGADRO
        # Resolved lazily on first update (need bulk['id'] from state).
        self._mids: list[str] | None = None
        self._bulk_idx: np.ndarray | None = None
        self._cofactor_mask: np.ndarray | None = None
        self._tick = 0

    def __init__(self, config=None, core=None):
        super().__init__(config or {}, core)
        # Re-run initialize against the schema-filled config (self.config) so
        # config_schema defaults (e.g. model_source) are present even when the
        # caller passes an empty/partial config. The raw `config or {}` used
        # previously raised KeyError('model_source') on config={}.
        self.initialize(self.config)

    def inputs(self):
        return {
            "lqr_control": InPlaceDict(),
            "bulk": "bulk_array",
            "listeners_mass": {
                "_type": "node",
                "_default": {"cell_mass": 0.0, "dry_mass": 0.0},
            },
            # Optional bioreactor-environment drive: {sbml_species_id: conc_mM}.
            # When non-empty, these species' initial concentrations are
            # overwritten on the COPASI model before integrating each tick so
            # the Millard kinetics respond to external nutrient levels (e.g.
            # GLCx glucose, O2). Internal metabolites are untouched and carry
            # over from the previous tick.
            "external_concentrations": {"_type": "node", "_default": {}},
        }

    def outputs(self):
        return {
            # mM concentrations / mM/s fluxes (see module docstring + the medium-
            # exchange notes above). inplace_dict[<unit>] keeps the in-place merge
            # apply these shared stores require while declaring the value unit so
            # units_resolver labels them like the tFBA Metabolism's fba_results.
            "species_concentrations": "inplace_dict[float[mM]]",
            "central_fluxes": "inplace_dict[float[mM/s]]",
            "control_applied": InPlaceDict(),
            "bulk": "bulk_array",
            # Per-tick signed molecule-count exchange with the medium, keyed by
            # BARE v2ecoli name (WCM convention: NEGATIVE = uptake, POSITIVE =
            # secretion). map[float] accumulates per-tick deltas in the store,
            # exactly like the WCM metabolism's environment.exchange write, so
            # the mass-conservation deriver's per-tick diff is valid.
            "environment": {"exchange": "map[float]"},
        }

    def _conc_to_count(self, state) -> float:
        """mM -> per-cell molecule count factor for this tick.

        Uses the live cell volume (listeners.mass.cell_mass / density) when
        ``use_live_volume`` is set, else the static config volume. Identical to
        the factor the bulk-delta path uses, so exchange counts share the bulk's
        reference frame.
        """
        conc_to_count = self._conc_to_count_static
        if self.use_live_volume:
            mass_in = state.get("listeners_mass") or {}
            cell_mass_fg = fg_magnitude(mass_in.get("cell_mass", 0.0))
            if cell_mass_fg > 0.0:
                live_volume_L = (cell_mass_fg * FG_PER_G
                                 / self.cell_density_g_per_L)
                conc_to_count = 1e-3 * live_volume_L * AVOGADRO
        return conc_to_count

    def _apply_control(self, ctrl: dict) -> tuple[float, dict]:
        """Read lqr_control, set basico parameters, return (tick_value, applied).

        Skip set_reaction_parameters when the new value equals the previously
        applied one — every call dirties COPASI's model, triggering a
        full recompile on the next run_time_course (measured at ~184 ms/tick,
        71% of the WCM tick). Cache last-applied values in self._last_applied
        and short-circuit unchanged values.
        """
        basico = self._basico
        applied: dict[str, float] = {}
        if not hasattr(self, "_last_applied"):
            self._last_applied: dict[str, float] = {}

        # Multi-input path: u_dict maps full param names to deltas.
        if isinstance(ctrl.get("u_dict"), dict) and ctrl["u_dict"]:
            for param_full, u_raw in ctrl["u_dict"].items():
                u_clipped = max(-self.u_clip, min(self.u_clip, float(u_raw)))
                if "." in param_full and param_full.startswith("("):
                    reaction = param_full.split(")", 1)[0][1:]
                    param = param_full.split(".", 1)[-1]
                else:
                    continue
                ps = basico.get_reaction_parameters(reaction_name=reaction)
                if ps is None or param_full not in ps.index:
                    continue
                base = float(ps.loc[param_full]["value"])
                target = base * (1.0 + u_clipped)
                if abs(target - self._last_applied.get(param_full, float("nan"))) > 1e-12:
                    basico.set_reaction_parameters(name=param_full, value=target)
                    self._last_applied[param_full] = target
                applied[param_full] = target
            tick_value = self.baseline_value  # observability only
            return tick_value, applied

        # Single-input back-compat path.
        u_raw = float(ctrl.get("u", 0.0))
        u_clipped = max(-self.u_clip, min(self.u_clip, u_raw))
        tick_value = self.baseline_value * (1.0 + u_clipped)
        param_full = f"({self.control_reaction}).{self.control_parameter}"
        if abs(tick_value - self._last_applied.get(param_full, float("nan"))) > 1e-12:
            basico.set_reaction_parameters(name=param_full, value=tick_value)
            self._last_applied[param_full] = tick_value
        applied[param_full] = tick_value
        return tick_value, applied

    def update(self, state, interval):
        basico = self._basico
        ctrl = state.get("lqr_control") or {}
        tick_value, applied = self._apply_control(ctrl)

        # Drive the kinetics from the bioreactor environment: overwrite ONLY
        # the named external species' initial concentrations before integrating
        # (internal metabolites carry over). Mirrors
        # CopasiUTCProcess._set_initial_concentrations.
        external = state.get("external_concentrations") or {}
        if external:
            # Skip unmapped species and non-finite values: a diverging reactor can
            # feed NaN/inf into external_concentrations, and float()/COPASI must not
            # crash the WCM update on it (overwrite only clean, mapped boundaries).
            changes = []
            for raw_id, conc_mM in external.items():
                # Accept raw SBML ids directly; otherwise map a v2ecoli molecule
                # name (e.g. the coupler's "OXYGEN-MOLECULE[p]") to its SBML id.
                sbml_id = (raw_id if raw_id in self.sbml_to_name
                           else EXTERNAL_NAME_TO_SBML.get(raw_id))
                if sbml_id is None or sbml_id not in self.sbml_to_name:
                    continue
                try:
                    val = float(conc_mM)
                except (TypeError, ValueError):
                    continue
                # Clamp negatives to 0 (a diverging reactor must not feed a
                # negative boundary concentration into the rate law) and skip
                # non-finite values entirely.
                if not math.isfinite(val):
                    continue
                if val < 0.0:
                    val = 0.0
                changes.append((self.sbml_to_name[sbml_id], val))
            if changes:
                _set_initial_concentrations(changes, self._model)

        # Advance the Millard ODE by one WCM tick. Pass model=self._model so the
        # integration acts on the same model the external overwrite (and flux
        # read below) target, regardless of basico's global "current" model.
        try:
            ts = basico.run_time_course(
                duration=self.tick_s,
                intervals=self.intervals,
                update_model=True,
                use_sbml_id=True,
                model=self._model,
            )
        except Exception as e:
            self._tick += 1
            return {
                "control_applied": {
                    "error": str(e)[:120],
                    "tick_value": tick_value,
                    "applied_per_param": applied,
                }
            }

        species = {sid: float(ts[sid].iloc[-1]) for sid in ts.columns}
        self._tick += 1

        # Read per-reaction fluxes (mM/s) from the SAME model the ODE just
        # advanced. basico.get_reactions() returns a DataFrame indexed by
        # reaction name with a `flux` column; pass model=self._model so we
        # never read a different model that happens to be basico-current.
        central_fluxes: dict[str, float] = {}
        try:
            fl = basico.get_reactions(model=self._model)
            if fl is not None and "flux" in getattr(fl, "columns", []):
                for rxn in fl.index:
                    val = fl.loc[rxn, "flux"]
                    if val is not None and not (isinstance(val, float)
                                                and math.isnan(val)):
                        central_fluxes[str(rxn)] = float(val)
        except Exception:
            central_fluxes = {}

        update: dict[str, Any] = {
            "species_concentrations": species,
            "central_fluxes": central_fluxes,
            "control_applied": {
                "tick": self._tick,
                "tick_value": tick_value,
                "applied_per_param": applied,
                "baseline_value": self.baseline_value,
            },
        }

        # mM -> per-cell count factor (shared by exchange + bulk paths below).
        conc_to_count = self._conc_to_count(state)

        # Medium-exchange accounting: emit the per-tick signed molecule-count
        # exchanged with the medium for each boundary species (O2, glucose,
        # acetate). Flux-based, NOT concentration-delta — see
        # MEDIUM_EXCHANGE_REACTIONS for why. This is the MEDIUM side; the bulk
        # path below carries the intracellular metabolite deltas, so there is no
        # double-counting (different stores, disjoint molecule sets).
        exchange: dict[str, float] = {}
        for rxn, (bare, coeff, sign) in MEDIUM_EXCHANGE_REACTIONS.items():
            flux = central_fluxes.get(rxn)
            if flux is None or not math.isfinite(flux):
                continue
            # mM exchanged this tick = flux[mM/s] * tick_s * stoich.
            mM = flux * self.tick_s * coeff
            count = round(sign * mM * conc_to_count)
            if count != 0.0:
                exchange[bare] = exchange.get(bare, 0.0) + count

        # Byproduct efflux (CO2 from decarboxylation, H2O from respiration): the
        # NET production summed over its source reactions -> medium SECRETION
        # (positive count). Closes the boundary mass accounting the O2/glucose
        # uptake otherwise leaves open. See BYPRODUCT_EFFLUX_REACTIONS.
        for bare, rxn_coeffs in BYPRODUCT_EFFLUX_REACTIONS.items():
            net_mM = 0.0
            seen = False
            for rxn, coeff in rxn_coeffs.items():
                flux = central_fluxes.get(rxn)
                if flux is None or not math.isfinite(flux):
                    continue
                net_mM += coeff * flux * self.tick_s
                seen = True
            if not seen:
                continue
            count = round(net_mM * conc_to_count)  # +secretion (WCM convention)
            if count != 0.0:
                exchange[bare] = exchange.get(bare, 0.0) + count
        if exchange:
            update["environment"] = {"exchange": exchange}

        # Translate mM → count deltas and emit to bulk.
        bulk = state.get("bulk")
        if bulk is not None and hasattr(bulk, "dtype") and bulk.dtype.names:
            if self._mids is None:
                bulk_ids = bulk["id"]
                resolved_mids: list[str] = []
                resolved_idx: list[int] = []
                for mid, vid in self._m2v.items():
                    if mid not in species:
                        continue
                    idx = bulk_name_to_idx(vid, bulk_ids, strict=False)
                    if idx is None or (isinstance(idx, np.ndarray) and idx.size == 0):
                        continue
                    resolved_mids.append(mid)
                    resolved_idx.append(int(idx))
                self._mids = resolved_mids
                self._bulk_idx = np.asarray(resolved_idx, dtype=np.int64)
                # Mask (aligned with _mids / _bulk_idx) marking the energy/redox
                # currency cofactors that get the homeostatic floor.
                self._cofactor_mask = np.fromiter(
                    (mid in HOMEOSTATIC_COFACTOR_MILLARD_IDS
                     for mid in resolved_mids),
                    dtype=bool, count=len(resolved_mids))

            if self._bulk_idx is not None and self._bulk_idx.size > 0:
                # Same live-volume mM→count factor used for medium exchange.
                current_mM = np.fromiter(
                    (species.get(mid, 0.0) for mid in self._mids),
                    dtype=np.float64, count=len(self._mids))
                current = counts(bulk, self._bulk_idx).astype(
                    np.float64, copy=False)
                if self.delta_mode:
                    # Apply the per-tick ΔmM × V × N_A as a bulk delta.
                    # Preserves v2ecoli's initial bulk state (so other
                    # processes aren't starved at t=0) and only adds
                    # Millard's kinetic perturbation. On the first tick
                    # prev_mM is None ⇒ delta = 0 (no jump).
                    if self._prev_mM is None:
                        delta = np.zeros_like(current, dtype=np.int64)
                    else:
                        delta_mM = current_mM - np.fromiter(
                            (self._prev_mM.get(mid, 0.0) for mid in self._mids),
                            dtype=np.float64, count=len(self._mids))
                        delta = np.rint(delta_mM * conc_to_count).astype(np.int64)
                    self._prev_mM = {mid: current_mM[i]
                                     for i, mid in enumerate(self._mids)}
                else:
                    # Legacy absolute-target mode (keeps the test-equivalence
                    # path open; see config "delta_mode": False).
                    targets = current_mM * conc_to_count
                    delta = np.rint(targets - current).astype(np.int64)
                # Per-metabolite lower bound for the resulting bulk count.
                # Default = min_count (≥0). For the energy/redox currency
                # cofactors, raise the bound to the homeostatic level Millard's
                # kinetics sustain (current_mM × V): these pools are drained
                # cell-wide by processes outside the Millard network (which the
                # FBA Metabolism this Process replaces used to balance), so
                # without this floor they drain to zero and crash the
                # ATP-dependent Equilibrium reactions. The floor only ever tops
                # a pool back UP when a consumer drew it below the sustained
                # concentration; a pool already above it (e.g. v2ecoli's larger
                # initial inventory) is left untouched — no t=0 collapse, no
                # pinning-down — exactly mirroring how the FBA arm holds ATP
                # steady. See HOMEOSTATIC_COFACTOR_MILLARD_IDS.
                floor = np.full_like(current, float(self.min_count),
                                     dtype=np.float64)
                if self._cofactor_mask.any():
                    homeostatic = np.rint(current_mM * conc_to_count)
                    floor = np.where(
                        self._cofactor_mask,
                        np.maximum(homeostatic, float(self.min_count)),
                        floor,
                    )
                new_counts = current + delta
                below = new_counts < floor
                if below.any():
                    delta = np.where(
                        below,
                        (floor - current).astype(np.int64),
                        delta,
                    )
                if delta.any():
                    update["bulk"] = [(self._bulk_idx, delta)]

        return update
