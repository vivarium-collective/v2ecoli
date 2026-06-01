"""LineageProcess — one (variant, seed) lineage as an embeddable Process.

Wraps a baseline cell composite (the EcoliWCM embedding pattern) and runs it
generation-by-generation, carrying a single daughter forward (vEcoli's
``single_daughters=true`` default). Variant overrides are applied at build
time. Each generation emits either partitioned parquet (default) or — when
``emitter == "xarray"`` — a hive-partitioned zarr store via an external
XArrayEmitter (the validated v2ecoli/library/xarray_run.py pattern), with
its own metadata. The meta-composite ticks this process via update(); it
reports ``complete`` when ``generations`` cells have been run.
"""

from __future__ import annotations

import warnings
from typing import Any

from process_bigraph import Process


def select_carry_daughter(agents_before, agents_now, mother_snapshot):
    """State to seed the next generation (single-daughter lineage), or None.

    The inner baseline composite's Division step already splits the mother
    into ``…0`` / ``…1`` daughters and adds them to its agents map. Carry the
    ``…0`` daughter's biological state DIRECTLY — re-dividing it would halve an
    already-halved cell, producing quarter-mass, slow-growing daughters (the
    multigeneration bug this guards against). Only when no structural daughter
    surfaced (a divide-flag / exception signal with no agents-map change) fall
    back to dividing the pre-run mother snapshot exactly once.
    """
    keys = ("bulk", "unique", "environment", "boundary")
    new_ids = set(agents_now) - set(agents_before)
    d0_id = next((i for i in sorted(new_ids) if i.endswith("0")), None)
    if d0_id is not None:
        dcell = agents_now.get(d0_id, {}) or {}
        return {k: dcell.get(k) for k in keys}
    if mother_snapshot and mother_snapshot.get("bulk") is not None:
        from v2ecoli.library.division import divide_cell
        d1, _d2 = divide_cell(mother_snapshot)
        return d1
    return None


# Default xarray view: scalar mass gauges (no vector coord arrays needed).
# Override via emitter_arg["view"] (JSON list roots are accepted). Leaves the
# composite doesn't emit are filtered out at open time (xarray is strict).
DEFAULT_XARRAY_VIEW = [{
    "root": ("listeners", "mass"),
    "variables": {
        name: [{"path": name, "dtype": "<f4"}]
        for name in ("dry_mass", "cell_mass", "protein_mass", "rna_mass", "dna_mass")
    },
}]


class LineageProcess(Process):
    config_schema = {
        "cache_dir": {"_type": "string", "_default": "out/cache"},
        "seed": {"_type": "integer", "_default": 0},
        "lineage_seed": {"_type": "integer", "_default": 0},
        "variant_index": {"_type": "integer", "_default": 0},
        "variant_name": {"_type": "string", "_default": "baseline"},
        "config_overrides": {"_default": {}},
        "generations": {"_type": "integer", "_default": 1},
        "single_daughters": {"_type": "boolean", "_default": True},
        "experiment_id": {"_type": "string", "_default": "default"},
        "out_dir": {"_type": "string", "_default": "out/workflow"},
        "max_duration_per_gen": {"_type": "float", "_default": 3600.0},
        "time_step": {"_type": "float", "_default": 1.0},
        # "parquet" (default) or "xarray". xarray drives an external
        # XArrayEmitter per lineage (validated multigen pattern); the internal
        # baseline emitter step then falls back to RAM (not read).
        "emitter": {"_type": "string", "_default": "parquet"},
        "emitter_arg": {"_default": {}},
        # Optional: opt-in baseline feature modules (replaces DEFAULT_FEATURES
        # when non-empty). dnaa-2 passes
        # ["ppgpp_regulation", "dnaa_nucleotide"] to add the DnaA intrinsic
        # hydrolysis step + nucleotide-state listener. Empty → baseline default.
        "features": {"_default": []},
        # Optional: seed generation 0 from a saved cell dill (divided once)
        # instead of cold-starting from the cache. Lets every seed in a
        # multiseed sweep start from the SAME steady-state init (e.g. dnaa-0's
        # saved succinate gen-3 state), then diverge stochastically by seed.
        "resume_dill": {"_type": "string", "_default": ""},
    }

    def initialize(self, config):
        self._composite = None
        self._generation = 0          # 0-based current generation
        self._agent_id = "0"
        self._gen_elapsed = 0.0
        self._carry_state: dict | None = None
        self._complete = False
        self._summaries: list[dict] = []
        self._needs_build = True      # True → call _build_generation on next tick
        # Optional resume-from-dill: seed gen 0's carry-state from a saved
        # mother cell (divided once), so every seed starts from the same
        # steady-state init. Carries the same biological keys the daughter walk
        # uses; the inner build re-seeds the mass listener (see _build_generation).
        resume_dill = (config.get("resume_dill") or "").strip()
        if resume_dill:
            import dill
            from v2ecoli.library.division import divide_cell
            with open(resume_dill, "rb") as f:
                mother = dill.load(f)
            d1, _d2 = divide_cell(mother)
            self._carry_state = {k: d1.get(k) for k in
                                 ("bulk", "unique", "environment", "boundary")}
        # xarray emitter state (only used when config["emitter"] == "xarray")
        self._xarray_em = None        # live XArrayEmitter for the current gen
        self._xarray_pending = False  # True → open on first populated emit tick
        self._xarray_view = None      # filtered view in use for this lineage
        self._xarray_store = None     # zarr store path (stable across gens)

    def _is_xarray(self) -> bool:
        return self.config.get("emitter", "parquet") == "xarray"

    def inputs(self):
        return {}

    def outputs(self):
        return {"summary": "map", "complete": "boolean"}

    # --- build / run helpers (stubbed in unit tests) ---------------------

    def _build_generation(self):
        from process_bigraph import Composite
        from v2ecoli.core import build_core
        from v2ecoli.composites.baseline import baseline, seed_mass_listener

        core = build_core()
        gen_seed = (int(self.config["seed"]) + self._generation) % (2 ** 31)
        overrides = dict(self.config.get("config_overrides") or {})
        features = list(self.config.get("features") or []) or None

        if self._is_xarray():
            # External XArrayEmitter path. The internal baseline emitter step is
            # minimised to global_time only (set_null_emitter_override) so it
            # doesn't waste memory — we emit out of band. The XArrayEmitter is
            # opened lazily on the first populated emit tick (see _emit_xarray),
            # so the view can be filtered against real state — xarray is strict
            # about missing emit paths.
            from v2ecoli.composites._helpers import set_null_emitter_override
            set_null_emitter_override(True)
            try:
                doc = baseline(core=core, seed=gen_seed,
                               cache_dir=self.config["cache_dir"],
                               config_overrides=overrides, features=features)
            finally:
                set_null_emitter_override(False)
            self._xarray_pending = True
        else:
            from v2ecoli.composites._helpers import set_parquet_emitter_override
            from v2ecoli.library.emitter_presets import parquet_vecoli
            emitter_cfg = parquet_vecoli(
                out_dir=self.config["out_dir"],
                experiment_id=self.config["experiment_id"],
                variant=int(self.config["variant_index"]),
                lineage_seed=int(self.config["lineage_seed"]),
                agent_id=self._agent_id,
                generation=self._generation,
            )
            set_parquet_emitter_override(emitter_cfg)
            try:
                doc = baseline(core=core, seed=gen_seed,
                               cache_dir=self.config["cache_dir"],
                               config_overrides=overrides, features=features)
            finally:
                set_parquet_emitter_override(None)

        if self._carry_state is not None:
            agent = doc["state"]["agents"]["0"]
            for key in ("bulk", "unique", "environment", "boundary"):
                if key in self._carry_state:
                    agent[key] = self._carry_state[key]
            agent["listeners"]["mass"] = {"dry_mass": 0.0, "cell_mass": 0.0}
            seed_mass_listener(agent, core)

        self._composite = Composite(doc, core=core)
        self._core = core
        self._gen_elapsed = 0.0

    def _open_xarray_emitter(self, emit_cell):
        """Open an XArrayEmitter for the current generation, filtering the view
        against ``emit_cell`` (populated state) and discovering vector coords.
        Mirrors the validated v2ecoli/library/xarray_run.py pattern."""
        import os
        import shutil
        from v2ecoli.library.xarray_run import (
            _build_emitter, filter_view_to_existing_leaves,
            extract_output_metadata_from_state)

        arg = dict(self.config.get("emitter_arg") or {})
        raw_view = arg.get("view") or DEFAULT_XARRAY_VIEW
        raw_view = [dict(e, root=tuple(e["root"])) for e in raw_view]
        transducer = arg.get("transducer") or {}
        buf = ((transducer.get("buffer") or {}).get("size"))
        buf = max(3, int(buf or 4))  # transducer requires buffer.size > 2
        predicate = transducer.get("predicate")
        writer = arg.get("writer")
        out_dir = arg.get("out_dir") or self.config["out_dir"]

        wrapped = {"agents": {"0": emit_cell}}
        view = filter_view_to_existing_leaves(wrapped, raw_view)
        if not view:
            warnings.warn("LineageProcess: xarray view has no leaves present in "
                          "composite state; skipping xarray emission.")
            self._xarray_pending = False
            return
        output_metadata = extract_output_metadata_from_state(wrapped, view)

        if self._xarray_store is None:
            self._xarray_store = os.path.join(
                out_dir,
                f"{self.config['experiment_id']}_v{int(self.config['variant_index'])}"
                f"_s{int(self.config['lineage_seed'])}.zarr")
        if self._generation == 0 and os.path.exists(self._xarray_store):
            shutil.rmtree(self._xarray_store)  # fresh store for a new lineage
        os.makedirs(out_dir, exist_ok=True)

        metadata_base = {
            "experiment_id": self.config["experiment_id"],
            "variant": int(self.config["variant_index"]),
            "lineage_seed": int(self.config["lineage_seed"]),
            "time_step": float(self.config.get("time_step", 1.0)),
            "max_duration": float(self.config["max_duration_per_gen"]),
        }
        self._xarray_view = view
        self._xarray_em = _build_emitter(
            core=self._core, store_path=self._xarray_store, view=view,
            metadata_base=metadata_base, generation=self._generation,
            agent_id=self._agent_id, buffer_size=buf,
            output_metadata=output_metadata, writer=writer, predicate=predicate)
        self._xarray_pending = False

    def _emit_xarray(self, agents_now):
        """Emit the inner cell's filtered state to the xarray emitter (opening
        it lazily on the first populated tick)."""
        emit_cell = agents_now.get("0")  # inner composite always names the cell "0"
        if not isinstance(emit_cell, dict):
            return
        if self._xarray_pending and self._xarray_em is None:
            self._open_xarray_emitter(emit_cell)
        if self._xarray_em is None:
            return
        from v2ecoli.library.xarray_run import _filter_agent_state
        payload = _filter_agent_state(emit_cell, self._xarray_view)
        try:
            self._xarray_em.update({
                "time": float(self._gen_elapsed),
                "global_time": float(self._gen_elapsed),
                "agents": {self._agent_id: payload},
            })
        except Exception as e:
            warnings.warn(f"LineageProcess: xarray emit failed at generation "
                          f"{self._generation} t={self._gen_elapsed}: {e}")

    def _run_until_division(self, interval):
        """Run the internal composite for ``interval`` seconds. Returns
        ``(divided, daughter_cell_data_or_None, final_dry_mass)``."""
        agents = self._composite.state.get("agents") or {}
        agents_before = set(agents.keys())
        # Snapshot the mother's divisible state BEFORE running: the inner
        # Division step removes the mother mid-run (and adds daughters), so
        # reading after the run samples an already-divided daughter. Only the
        # snapshot is used for the exception/divide-flag fallback path.
        mother = agents.get(self._agent_id) or next(iter(agents.values()), {})
        mother_snapshot = (
            {k: mother.get(k) for k in ("bulk", "unique", "environment", "boundary")}
            if isinstance(mother, dict) else None)

        divided = False
        try:
            self._composite.run(interval)
        except Exception as e:
            msg = str(e).lower()
            # Division surfaces as a structural update; the message mentions it.
            # Non-division exceptions must propagate.
            if "divide" in msg or "division" in msg:
                divided = True
            else:
                raise
        self._gen_elapsed += interval

        agents_now = self._composite.state.get("agents") or {}
        agents_after = set(agents_now.keys())
        if agents_before and agents_after != agents_before:
            divided = True
        # MarkDPeriod sets a divide flag without changing the agents map; honor it
        # too (mirrors the three-signal detection in v2ecoli/bridge.py).
        survivor = agents_now.get(self._agent_id, {})
        if isinstance(survivor, dict) and survivor.get("divide"):
            divided = True

        cell = agents_now.get(self._agent_id) or next(iter(agents_now.values()), {})
        dry_mass = float(cell.get("listeners", {}).get("mass", {}).get("dry_mass", 0.0))

        if self._is_xarray():
            self._emit_xarray(agents_now)

        daughter = None
        if divided:
            daughter = select_carry_daughter(agents_before, agents_now, mother_snapshot)
        return divided, daughter, dry_mass

    # --- main tick -------------------------------------------------------

    def update(self, state, interval):
        if not self.config.get("single_daughters", True):
            raise NotImplementedError(
                "single_daughters=False (binary-tree lineage) is deferred; "
                "MVP supports the single-lineage walk only.")
        if self._complete:
            return {"complete": True}
        if self._needs_build:
            self._build_generation()
            self._needs_build = False

        divided, daughter, dry_mass = self._run_until_division(interval)
        timed_out = self._gen_elapsed >= float(self.config["max_duration_per_gen"])
        if not (divided or timed_out):
            return {}

        # End of this generation: flush/close emitter, record summary.
        if self._is_xarray():
            if self._xarray_em is not None:
                try:
                    self._xarray_em.close(success=True)
                except Exception as e:
                    warnings.warn(f"LineageProcess: xarray close failed for "
                                  f"generation {self._generation}: {e}")
                self._xarray_em = None
            self._xarray_pending = False
        else:
            from v2ecoli.composites._helpers import flush_parquet
            try:
                flush_parquet(self._composite, success=True)
            except Exception as e:
                warnings.warn(f"LineageProcess: parquet flush failed for "
                              f"generation {self._generation} ({self._agent_id}): {e}")
        self._summaries.append({
            "generation": self._generation,
            "agent_id": self._agent_id,
            "duration": self._gen_elapsed,
            "dry_mass": dry_mass,
            "divided": bool(divided),
        })

        self._generation += 1
        if self._generation >= int(self.config["generations"]):
            self._complete = True
            self._composite = None
            return {"complete": True, "summary": {"generations": self._summaries}}

        # Carry daughter 0 forward; rebuild a fresh composite next tick.
        from v2ecoli.steps.division import daughter_phylogeny_id
        self._carry_state = daughter
        self._agent_id = daughter_phylogeny_id(self._agent_id)[0]
        self._composite = None
        self._needs_build = True
        return {"summary": {"generations": self._summaries}}
