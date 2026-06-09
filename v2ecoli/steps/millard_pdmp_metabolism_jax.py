"""MillardPDMPMetabolismJAX — JAX/Diffrax backend for the PDMP metabolism slot.

Drop-in replacement for v2ecoli.steps.millard_pdmp_metabolism.MillardPDMPMetabolism
that uses the JAX-compiled Millard rhs from v2ecoli.library.millard_jax_full
in place of the basico/COPASI integrator. Lands the M4 (compiled-backend)
lever from this PR's architectural roadmap.

Performance — Phase 4 standalone benchmark (5000 s of Millard):
    basico (LSODA, tight tol)       ~51 ms warm
    JAX Kvaerno3 (rtol=1e-6)        ~98 ms warm   (slower at tight tol)
    JAX Kvaerno3 (rtol=1e-3)        ~29 ms warm   (1.8x faster than basico)

This Process uses the LOOSE tolerance setting by default to actually beat
basico. L_inf vs basico at loose tol stays ~1e-7 across all 64 state
species (the slack is well below the count-rounding step in the bulk
indexer, so it does not propagate to v2ecoli's discrete-count world).

LQR control: omitted for now. The JAX rhs source bakes parameter values at
build time; modulating PTS_4.kF / PFK.Vmax / PYK.Vmax would require
rebuilding the rhs each tick (defeats the JIT amortization) or refactoring
the SBML->JAX translator to expose those as runtime args. The current
LQRControllerMultiState falls back to zero gain anyway (Riccati solve
fails on the Millard linearization), so dropping control here changes no
observable behavior. Re-enabling control is task #19 (post-merge).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from process_bigraph import Process

from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.types.stores import InPlaceDict


DEFAULT_MAPPING = "v2ecoli/data/millard_v2ecoli_species_map.yaml"
DEFAULT_CELL_VOLUME_L = 1.0e-15
AVOGADRO = 6.02214076e23


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


class MillardPDMPMetabolismJAX(Process):
    """Millard ODE via JAX/Diffrax + bulk delta emission."""

    name = "millard-pdmp-metabolism-jax"
    config_schema = {
        "model_source": {"_default":
            "v2ecoli/models/sbml/millard2017_central_metabolism.xml"},
        "tick_s": {"_default": 1.0},
        # Diffrax integration controls. Default tolerances match the
        # Phase-4 "loose" config that's 1.8x faster than basico and stays
        # within ~1e-7 of the tight-tol reference.
        "rtol": {"_default": 1e-3},
        "atol": {"_default": 1e-6},
        # 500_000 matches the Phase-4 standalone benchmark setting; the
        # full Millard chemistry is stiff and routinely needs ~10⁴-10⁵
        # implicit steps for a 1 s integration at tight tolerance.
        "max_steps": {"_default": 500_000},
        "mapping_file": {"_default": DEFAULT_MAPPING},
        "cell_volume_L": {"_default": DEFAULT_CELL_VOLUME_L},
        "min_count": {"_default": 0.0},
    }

    def initialize(self, config):
        self.parameters = config or {}
        self.tick_s = float(self.parameters.get("tick_s", 1.0))
        self.rtol = float(self.parameters.get("rtol", 1e-3))
        self.atol = float(self.parameters.get("atol", 1e-6))
        self.max_steps = int(self.parameters.get("max_steps", 4096))
        self.cell_volume_L = float(self.parameters.get(
            "cell_volume_L", DEFAULT_CELL_VOLUME_L))
        self.min_count = float(self.parameters.get("min_count", 0.0))

        # Build the JAX-compiled Millard rhs once at init. cold compile ~14 s
        # for the full chemistry; amortized across the run lifetime.
        import jax
        import jax.numpy as jnp
        import diffrax
        from v2ecoli.library.millard_jax_full import build_jax_model
        self._jax = jax
        self._jnp = jnp
        self._diffrax = diffrax
        self._model = build_jax_model(self.parameters["model_source"])
        # State vector and species ID list (only dynamic state species).
        self._sids = list(self._model.state_species_ids)
        self._sid_idx = {sid: i for i, sid in enumerate(self._sids)}

        # Keep y as a JAX array between ticks. The original implementation
        # rebound from numpy each tick, which forced diffrax to re-trace
        # (Python-float t0/t1 + numpy y0 = static args + non-JAX state) and
        # blew per-tick wall up to ~16 s/sim-s. With the integration step
        # jit-compiled once over JAX-array arguments, per-tick cost drops
        # back to the Phase-4 amortized number.
        solver = diffrax.Kvaerno3()
        term = diffrax.ODETerm(self._model.rhs)
        stepsize = diffrax.PIDController(rtol=self.rtol, atol=self.atol)
        max_steps = self.max_steps

        # Critical: t0/t1 must be traced (JAX scalars), NOT Python floats —
        # otherwise jit specializes on each (t0, t1) value pair and we pay
        # a 14-s recompile on every tick.
        # dt0=0.1 matches the standalone JAX pilot config that produced
        # ATP=2.5819 (bit-identical to basico). Earlier ratio-based dt0
        # ((t1-t0)*0.1) was also 0.1 here, so that's not the bug — but
        # leaving this explicit avoids accidental rebinding to ratio form.
        dt0 = 0.1

        @jax.jit
        def _step(y, t0, t1):
            sol = diffrax.diffeqsolve(
                term, solver,
                t0=t0, t1=t1,
                dt0=dt0,
                y0=y,
                stepsize_controller=stepsize,
                max_steps=max_steps,
            )
            return sol.ys[-1]

        self._step_jit = _step
        self._y = jnp.asarray(self._model.y0)
        self._t = jnp.asarray(0.0)
        # Pre-cast the tick interval to a JAX scalar once — used to advance
        # _t each call (kept on-device so the addition stays traced).
        self._dt = jnp.asarray(self.tick_s)

        # Bulk-indexing setup
        self.mapping_file = self.parameters.get("mapping_file", DEFAULT_MAPPING)
        self._m2v = _load_millard_to_v2ecoli(self.mapping_file)
        self._conc_to_count = 1e-3 * self.cell_volume_L * AVOGADRO
        self._mids: list[str] | None = None
        self._bulk_idx: np.ndarray | None = None

        # Pre-warm the JIT — pays the XLA compile up front so the first WCM
        # tick doesn't carry a ~14 s spike. Use a real 1-tick interval so
        # the trace matches the steady-state call shape.
        _ = self._step_jit(self._y, self._t, self._t + self._dt)
        _.block_until_ready()
        self._tick = 0

    def __init__(self, config=None, core=None):
        super().__init__(config or {}, core)
        self.initialize(config or {})

    def inputs(self):
        return {
            "lqr_control": InPlaceDict(),   # accepted but currently ignored
            "bulk": "bulk_array",
        }

    def outputs(self):
        return {
            "species_concentrations": InPlaceDict(),
            "control_applied": InPlaceDict(),
            "bulk": "bulk_array",
        }

    def update(self, state, interval):
        # Run the ODE for tick_s seconds. self._y and self._t are JAX
        # scalars/arrays kept between ticks; with t0/t1 traced (not
        # specialized) the jit cache hits and each tick runs in ~ms.
        t0 = self._t
        t1 = self._t + self._dt
        try:
            self._y = self._step_jit(self._y, t0, t1)
        except Exception as e:
            self._tick += 1
            return {
                "control_applied": {
                    "error": str(e)[:120], "tick": self._tick,
                    "backend": "jax",
                }
            }
        self._t = t1

        # Materialise to numpy once for the bulk delta + species dict path.
        y_np = np.asarray(self._y, dtype=np.float64)
        species = {sid: float(y_np[i]) for i, sid in enumerate(self._sids)}
        self._tick += 1

        update: dict[str, Any] = {
            "species_concentrations": species,
            "control_applied": {
                "tick": self._tick,
                "backend": "jax",
                "rtol": self.rtol,
                "atol": self.atol,
            },
        }

        # Translate mM -> count deltas and emit to bulk (same logic as
        # MillardPDMPMetabolism — kept in-line so this Process is fully
        # self-contained as the JAX-backed slot).
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
                targets = np.fromiter(
                    (species.get(mid, 0.0) * self._conc_to_count
                     for mid in self._mids),
                    dtype=np.float64,
                    count=len(self._mids),
                )
                current = counts(bulk, self._bulk_idx).astype(np.float64, copy=False)
                delta = np.rint(targets - current).astype(np.int64)
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
