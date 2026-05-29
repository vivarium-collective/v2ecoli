"""LineageProcess — one (variant, seed) lineage as an embeddable Process.

Wraps a baseline cell composite (the EcoliWCM embedding pattern) and runs it
generation-by-generation, carrying a single daughter forward (vEcoli's
``single_daughters=true`` default). Variant overrides are applied at build
time; each generation emits partitioned parquet with its own metadata.
The meta-composite ticks this process via update(); it reports ``complete``
when ``generations`` cells have been run.
"""

from __future__ import annotations

import warnings
from typing import Any

from process_bigraph import Process


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

    def inputs(self):
        return {}

    def outputs(self):
        return {"summary": "map", "complete": "boolean"}

    # --- build / run helpers (stubbed in unit tests) ---------------------

    def _build_generation(self):
        from process_bigraph import Composite
        from v2ecoli.core import build_core
        from v2ecoli.composites.baseline import baseline, seed_mass_listener
        from v2ecoli.composites._helpers import set_parquet_emitter_override
        from v2ecoli.library.emitter_presets import parquet_vecoli

        core = build_core()
        gen_seed = (int(self.config["seed"]) + self._generation) % (2 ** 31)
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
            doc = baseline(
                core=core, seed=gen_seed, cache_dir=self.config["cache_dir"],
                config_overrides=dict(self.config.get("config_overrides") or {}))
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
        self._gen_elapsed = 0.0

    def _run_until_division(self, interval):
        """Run the internal composite for ``interval`` seconds. Returns
        ``(divided, daughter_cell_data_or_None, final_dry_mass)``."""
        agents_before = set((self._composite.state.get("agents") or {}).keys())
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
        agents_after = set((self._composite.state.get("agents") or {}).keys())
        if agents_before and agents_after != agents_before:
            divided = True
        # MarkDPeriod sets a divide flag without changing the agents map; honor it
        # too (mirrors the three-signal detection in v2ecoli/bridge.py).
        cur_cell = (self._composite.state.get("agents", {}) or {}).get(self._agent_id, {})
        if isinstance(cur_cell, dict) and cur_cell.get("divide"):
            divided = True

        cell = (self._composite.state.get("agents", {}).get(self._agent_id)
                or next(iter(self._composite.state.get("agents", {}).values()), {}))
        dry_mass = float(cell.get("listeners", {}).get("mass", {}).get("dry_mass", 0.0))

        daughter = None
        if divided:
            from v2ecoli.library.division import divide_cell
            cell_data = {
                "bulk": cell.get("bulk"),
                "unique": cell.get("unique", {}),
                "environment": cell.get("environment", {}),
                "boundary": cell.get("boundary", {}),
            }
            if cell_data["bulk"] is not None:
                d1, _d2 = divide_cell(cell_data)
                daughter = d1
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

        # End of this generation: flush emitter, record summary.
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
