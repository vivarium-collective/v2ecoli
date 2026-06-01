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
        basico.load_model(self.parameters["model_source"])
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
        self._tick = 0

    def __init__(self, config=None, core=None):
        super().__init__(config or {}, core)
        self.initialize(config or {})

    def inputs(self):
        return {
            "lqr_control": InPlaceDict(),
            "bulk": "bulk_array",
            "listeners_mass": {
                "_type": "node",
                "_default": {"cell_mass": 0.0, "dry_mass": 0.0},
            },
        }

    def outputs(self):
        return {
            "species_concentrations": InPlaceDict(),
            "control_applied": InPlaceDict(),
            "bulk": "bulk_array",
        }

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

        # Advance the Millard ODE by one WCM tick.
        try:
            ts = basico.run_time_course(
                duration=self.tick_s,
                intervals=self.intervals,
                update_model=True,
                use_sbml_id=True,
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

        update: dict[str, Any] = {
            "species_concentrations": species,
            "control_applied": {
                "tick": self._tick,
                "tick_value": tick_value,
                "applied_per_param": applied,
                "baseline_value": self.baseline_value,
            },
        }

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

            if self._bulk_idx is not None and self._bulk_idx.size > 0:
                # Compute the mM→count conversion factor from the live cell
                # volume when use_live_volume is set.
                conc_to_count = self._conc_to_count_static
                if self.use_live_volume:
                    mass_in = state.get("listeners_mass") or {}
                    cell_mass_fg = fg_magnitude(mass_in.get("cell_mass", 0.0))
                    if cell_mass_fg > 0.0:
                        live_volume_L = (cell_mass_fg * FG_PER_G
                                         / self.cell_density_g_per_L)
                        conc_to_count = 1e-3 * live_volume_L * AVOGADRO
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
                # Floor at min_count: don't push counts below zero.
                new_counts = current + delta
                below = new_counts < self.min_count
                if below.any():
                    delta = np.where(
                        below,
                        np.int64(self.min_count) - current.astype(np.int64),
                        delta,
                    )
                if delta.any():
                    update["bulk"] = [(self._bulk_idx, delta)]

        return update
