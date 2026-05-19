"""
=====================================
DnaA Initiation Mechanism (shadow obs)
=====================================

A non-destructive observer that computes when a DnaA-driven mechanistic
initiation trigger WOULD fire, with a SeqA-v0 fixed-timer refractory.
Does NOT actually replace v2ecoli's existing initiation Process — that
would require editing the chromosome-replication subsystem directly,
which is out of scope for the dnaa-04 first-pass implementation.

Instead, this Step emits to ``listeners.dnaA_initiation.*`` a parallel
"would-fire" signal that can be compared against the existing heuristic's
actual initiation events. That lets the dnaa-04 behavior_tests
(``initiation-mass-mean-matches-heuristic``, ``one-initiation-per-
generation``, etc.) be evaluated WITHOUT swapping out the live initiation
logic.

Trigger condition (replaceable later with literature-anchored kinetics):

  would_fire == TRUE iff
      oric_high_occupied      ≥ oric_high_threshold (default 0.8)
  AND atp_fraction            ≥ atp_fraction_threshold (default 0.3)
  AND (global_time - last_fire_time) ≥ refractory_seconds (default 600 = 10 min)

The SeqA-v0 refractory is a fixed timer modeled after the in-vivo
post-initiation SeqA sequestration of newly-replicated oriC. Production
SeqA (dnaa-06) will replace this with explicit methylation-state
sequestration.

Emits per tick:
  listeners.dnaA_initiation.would_fire             : bool (1 / 0)
  listeners.dnaA_initiation.oric_high_obs          : float (read)
  listeners.dnaA_initiation.atp_fraction_obs       : float (read)
  listeners.dnaA_initiation.in_refractory          : bool (1 / 0)
  listeners.dnaA_initiation.t_since_last_fire_s    : float
  listeners.dnaA_initiation.cumulative_fires       : int (since sim start)
"""

from __future__ import annotations

from v2ecoli.library.ecoli_step import EcoliStep as Step


NAME = "dnaa-initiation-mechanism"
TOPOLOGY = {
    "listeners":   ("listeners",),
    "unique":      ("unique",),
    "global_time": ("global_time",),
}


class DnaaInitiationMechanism(Step):
    """Shadow observer for DnaA-driven mechanistic initiation + SeqA-v0.

    Reads from ``listeners.dnaA_binding.oric_high_occupied`` and
    ``listeners.dnaA_cycle.atp_fraction`` (both produced by upstream
    dnaa-03 + dnaa-02 Steps). Emits a per-tick ``would_fire`` boolean
    plus auxiliary state under ``listeners.dnaA_initiation.*``.

    Does NOT modify the chromosome / replication state. Production
    integration (replacing the heuristic trigger) belongs to a later
    pass after biology validation.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "oric_high_threshold":          {"_type": "float",   "_default": 0.8},
        "atp_fraction_threshold":       {"_type": "float",   "_default": 0.3},
        "refractory_seconds":           {"_type": "float",   "_default": 600.0},
        # Hard biology constraint: don't fire while existing replication is
        # active. Models SeqA's role at the v0 abstraction level. -1 disables
        # the check (for tests where you want to fire regardless).
        "n_replisomes_max_for_fire":    {"_type": "integer", "_default": 0},
        "time_step":                    {"_type": "float",   "_default": 1.0},
    }

    def initialize(self, config):
        self.oric_high_threshold       = float(self.parameters["oric_high_threshold"])
        self.atp_fraction_threshold    = float(self.parameters["atp_fraction_threshold"])
        self.refractory_seconds        = float(self.parameters["refractory_seconds"])
        self.n_replisomes_max_for_fire = int(self.parameters["n_replisomes_max_for_fire"])
        # State maintained across ticks.
        self._last_fire_time: float = -1e18  # Far in the past => not in refractory
        self._cumulative_fires: int = 0

    def inputs(self):
        return {
            "listeners":   "node",
            "unique":      "node",
            "global_time": {"_type": "float", "_default": 0.0},
        }

    def outputs(self):
        return {
            "listeners": {
                "dnaA_initiation": {
                    "would_fire":            {"_type": "overwrite[integer]", "_default": 0},
                    "oric_high_obs":         {"_type": "overwrite[float]",   "_default": 0.0},
                    "atp_fraction_obs":      {"_type": "overwrite[float]",   "_default": 0.0},
                    "in_refractory":         {"_type": "overwrite[integer]", "_default": 0},
                    "replisomes_active":     {"_type": "overwrite[integer]", "_default": 0},
                    "t_since_last_fire_s":   {"_type": "overwrite[float]",   "_default": 0.0},
                    "cumulative_fires":      {"_type": "overwrite[integer]", "_default": 0},
                },
            },
        }

    def update(self, states, interval=None):
        listeners = states.get("listeners") or {}
        t = float(states.get("global_time", 0.0))

        # Walk into the upstream listener emits. dnaA_binding emits with
        # nested oric.* / chromosome.* / dnaap.* paths (see DnaaBoxBinding
        # process). dnaA_cycle is flat.
        bind = listeners.get("dnaA_binding") or {}
        cyc  = listeners.get("dnaA_cycle") or {}

        oric_node = bind.get("oric") if isinstance(bind, dict) else None
        if isinstance(oric_node, dict):
            oric_high = float(oric_node.get("high_affinity_occupied", 0.0) or 0.0)
        else:
            # Backward-compat: flat field path if the listener convention changes.
            oric_high = float(bind.get("oric_high_occupied", 0.0) or 0.0)
        atp_frac   = float(cyc.get("atp_fraction", 0.0) or 0.0)

        t_since_last = t - self._last_fire_time
        in_refractory = t_since_last < self.refractory_seconds

        # SeqA-v0 biology gate: don't fire while existing replication is in
        # progress. v2ecoli's chromosome state has active_replisome instances
        # whose _entryState flags indicate "this slot holds a real replisome".
        unique = states.get("unique") or {}
        rep = unique.get("active_replisome") if isinstance(unique, dict) else None
        n_replisomes = 0
        if rep is not None and hasattr(rep, "dtype"):
            names = (rep.dtype.names or ())
            if "_entryState" in names:
                try:
                    n_replisomes = int(rep["_entryState"].sum())
                except Exception:
                    n_replisomes = 0
        replisome_gate_blocks = (
            self.n_replisomes_max_for_fire >= 0 and
            n_replisomes > self.n_replisomes_max_for_fire
        )

        condition_met = (
            oric_high >= self.oric_high_threshold and
            atp_frac  >= self.atp_fraction_threshold and
            not in_refractory and
            not replisome_gate_blocks
        )

        if condition_met:
            self._last_fire_time = t
            self._cumulative_fires += 1

        return {
            "listeners": {
                "dnaA_initiation": {
                    "would_fire":          1 if condition_met else 0,
                    "oric_high_obs":       oric_high,
                    "atp_fraction_obs":    atp_frac,
                    "in_refractory":       1 if in_refractory else 0,
                    "replisomes_active":   n_replisomes,
                    "t_since_last_fire_s": t_since_last,
                    "cumulative_fires":    self._cumulative_fires,
                },
            },
        }
