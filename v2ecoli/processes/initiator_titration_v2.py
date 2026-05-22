"""
=============================
Initiator-titration model v2
=============================

Standalone implementation of the Fu / Xiao / Jun (2023) initiator-titration
model v2 ("ITv2") for *E. coli* replication initiation. See
``references/notes/FuXiaoJun2023.md`` for the paper digest.

The model is mean-field — it tracks DnaA copy numbers (DnaA-ATP + DnaA-ADP),
cell volume, and replication-fork progress without resolving individual
DnaA boxes, RNA polymerases, or any other biomolecule. The purpose of
this Process is to provide a clean comparison target for v2ecoli's much
richer DnaA mechanism: same observables (DnaA-ATP fraction, initiation
mass, cell-cycle timing), drastically simpler underlying dynamics.

Equations (paper Appendix D, with explicit RIDA gating):

    dV/dt   = λ · V                                    (exponential growth)
    dI_T/dt = λ · (I_T + I_D) − ν_eff(d) · I_T         (balanced synthesis as DnaA-ATP minus hydrolysis)
    dI_D/dt =                    ν_eff(d) · I_T         (ADP fed by hydrolysis flux)
    dρ_i/dt = 1/C        for each ρ_i < 1               (forks travel ori → ter in time C)

Where:
    λ = ln(2) / τ                                       (growth rate, from doubling time)
    ν_eff(d) = ν_intrinsic + ν_RIDA · 1[d > 0]         (RIDA hydrolysis is replication-dependent)

Binding-site count (Eq. 6, with d generations active):

    B(ρ) = N_B · (1 + Σ_{i=1..d} ρ_i · 2^(d−i)) + 2^d · n_B

Initiation rule:
    When I_T ≥ B AND no replication initiation in the previous
    ``eclipse_s`` seconds → fire a new round:
      ρ ← (0, ρ_1, ..., ρ_d)
      d ← d + 1

Division rule:
    When a generation's ρ reaches 1.0 (termination), schedule a division
    ``D`` seconds later. At division:
      V ← V / 2 ; I_T ← I_T / 2 ; I_D ← I_D / 2
    and drop the terminated generation from ρ.

All time inputs are in seconds inside the Process. The ``time_step`` is
the v2ecoli-style outer tick; events are caught at sub-tick resolution
via a small internal Euler step (``substep_s``, default 1 s).

Outputs (under ``listeners.itv2``):
  volume                 cell volume (µm³)
  dnaa_atp_count         DnaA-ATP molecule count (I_T)
  dnaa_adp_count         DnaA-ADP molecule count (I_D)
  dnaa_total_count       I_T + I_D
  dnaa_atp_fraction      I_T / (I_T + I_D)
  binding_sites          B(t)
  n_generations          d (number of active replication generations)
  fork_progress          ρ array (variable length; emitted as a list)
  initiation_event       1 if an initiation fired during this tick, else 0
  initiation_mass        V at the most recent initiation (µm³); 0 before any
  initiations_so_far     cumulative initiation count
  divisions_so_far       cumulative division count
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from v2ecoli.library.ecoli_step import EcoliProcess as Process


NAME = "initiator-titration-v2"
TOPOLOGY = {"listeners": ("listeners",)}


class InitiatorTitrationV2(Process):
    """Mean-field initiator-titration model v2 (FXJ 2023) as a Process.

    Distinct from v2ecoli's whole-cell DnaA mechanism — this Process
    owns ALL its state internally (no bulk-array reads). Wire it into
    a standalone composite for side-by-side comparison runs.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # ── Cell-physics parameters ──
        # Mass-doubling time τ in minutes. λ = ln(2)/τ.
        "tau_min":      {"_type": "float",   "_default": 60.0},
        # Replication-period C in minutes (ori → ter traversal time).
        "C_min":        {"_type": "float",   "_default": 40.0},
        # Termination → division delay D in minutes.
        "D_min":        {"_type": "float",   "_default": 20.0},

        # ── DnaA-box / oriC binding sites ──
        "N_B":          {"_type": "integer", "_default": 300},   # chromosomal
        "n_B":          {"_type": "integer", "_default": 10},    # per oriC

        # ── DnaA pool parameters ──
        # Target initiator concentration in copies / µm³. Default ≈ 600
        # copies per 1 µm³ cell, in the same order of magnitude as
        # Schmidt 2016 / Mori 2021 mass-spec values.
        "c_I_per_um3":  {"_type": "float",   "_default": 600.0},
        # Intrinsic DnaA-ATP → DnaA-ADP hydrolysis rate (per minute).
        # 1/(15 min) ≈ 0.046 from Boesen 2024 / Katayama 2017.
        "nu_intrinsic_per_min": {"_type": "float", "_default": 0.046},
        # Additional RIDA hydrolysis rate when at least one replication
        # fork is active. Set to 0.0 to recover the Δ4-mutant behaviour
        # (no extrinsic conversion). Default ≈ 4× intrinsic.
        "nu_rida_per_min":      {"_type": "float", "_default": 0.20},

        # ── Eclipse period (post-initiation refractory) ──
        # No initiation can fire within ``eclipse_s`` of the prior one.
        # Paper §IIE: ~10 min in E. coli (SeqA-mediated sequestration).
        "eclipse_s":    {"_type": "float",   "_default": 600.0},

        # ── Initial conditions ──
        "V0_um3":       {"_type": "float",   "_default": 1.0},
        "I_T0":         {"_type": "float",   "_default": 500.0},
        "I_D0":         {"_type": "float",   "_default": 100.0},

        # ── Division behaviour ──
        # If true, halve V / I_T / I_D when a generation terminates +
        # D_min later. If false, run as the paper's "protocell" (no
        # division; chromosome accumulates indefinitely).
        "divide_on_terminate": {"_type": "boolean", "_default": True},

        # ── Numerics ──
        # Internal Euler substep in seconds. Smaller = more accurate
        # event timing. Default 1.0 — fine enough for τ ≈ 60 min runs.
        "substep_s":    {"_type": "float",   "_default": 1.0},
        # v2ecoli-style outer tick in seconds.
        "time_step":    {"_type": "float",   "_default": 60.0},
        "seed":         {"_type": "integer", "_default": 0},
    }

    def initialize(self, config):
        p = self.parameters
        # Convert all rates to per-second.
        self.tau_s = float(p["tau_min"]) * 60.0
        self.C_s = float(p["C_min"]) * 60.0
        self.D_s = float(p["D_min"]) * 60.0
        self.lam = math.log(2.0) / self.tau_s

        self.N_B = int(p["N_B"])
        self.n_B = int(p["n_B"])
        self.c_I = float(p["c_I_per_um3"])
        self.nu_intrinsic = float(p["nu_intrinsic_per_min"]) / 60.0
        self.nu_rida = float(p["nu_rida_per_min"]) / 60.0
        self.eclipse_s = float(p["eclipse_s"])
        self.substep_s = max(0.05, float(p["substep_s"]))
        self.divide_on_terminate = bool(p["divide_on_terminate"])

        # Model state (owned by the Process; nothing leaks to the
        # composite state tree except via the listener emit).
        self._t = 0.0
        self._V = float(p["V0_um3"])
        self._I_T = float(p["I_T0"])
        self._I_D = float(p["I_D0"])
        # Fork progress, one entry per active generation. Empty list
        # = no replication ongoing (d = 0, the initial chromosome).
        self._rho: list[float] = []
        # Time of the most recent initiation (−inf so the first one is
        # always allowed).
        self._t_last_init = -math.inf
        self._initiations = 0
        self._divisions = 0
        # Volume at the most recent initiation (the "initiation mass").
        self._initiation_volume = 0.0
        # Pending divisions: list of (t_division_due,) markers. When a
        # generation terminates we append now + D_s; the main loop
        # checks the head of this queue each substep.
        self._pending_divisions: list[float] = []
        # Did an initiation fire during the current outer tick?
        self._init_this_tick = 0

    # ── No external inputs: this is a standalone model ──
    def inputs(self):
        return {}

    def outputs(self):
        return {
            "listeners": {
                "itv2": {
                    "volume":              {"_type": "overwrite[float]",   "_default": 0.0},
                    "dnaa_atp_count":      {"_type": "overwrite[float]",   "_default": 0.0},
                    "dnaa_adp_count":      {"_type": "overwrite[float]",   "_default": 0.0},
                    "dnaa_total_count":    {"_type": "overwrite[float]",   "_default": 0.0},
                    "dnaa_atp_fraction":   {"_type": "overwrite[float]",   "_default": 0.0},
                    "binding_sites":       {"_type": "overwrite[float]",   "_default": 0.0},
                    "n_generations":       {"_type": "overwrite[integer]", "_default": 0},
                    "fork_progress":       {"_type": "overwrite[list[float]]", "_default": []},
                    "initiation_event":    {"_type": "overwrite[integer]", "_default": 0},
                    "initiation_mass":     {"_type": "overwrite[float]",   "_default": 0.0},
                    "initiations_so_far":  {"_type": "overwrite[integer]", "_default": 0},
                    "divisions_so_far":    {"_type": "overwrite[integer]", "_default": 0},
                },
            },
        }

    # ── Core model maths ──
    def _binding_sites(self) -> float:
        """B(t) per Eq. 6, with d active generations.

        With d = 0 (no replication): B = N_B + n_B.
        With d ≥ 1: B = N_B · (1 + Σ ρ_i · 2^(d−i)) + 2^d · n_B.
        """
        d = len(self._rho)
        if d == 0:
            return float(self.N_B + self.n_B)
        chrom_sum = 0.0
        for i, rho in enumerate(self._rho, start=1):
            # i is 1-indexed; exponent is (d − i)
            chrom_sum += rho * (2 ** (d - i))
        return self.N_B * (1.0 + chrom_sum) + (2 ** d) * self.n_B

    def _step(self, dt: float) -> None:
        """One internal Euler substep of length dt seconds."""
        d = len(self._rho)

        # ── Continuous dynamics ──
        I_total = self._I_T + self._I_D
        nu_eff = self.nu_intrinsic + (self.nu_rida if d > 0 else 0.0)
        dI_T = self.lam * I_total - nu_eff * self._I_T
        dI_D = nu_eff * self._I_T
        dV = self.lam * self._V

        self._I_T += dI_T * dt
        self._I_D += dI_D * dt
        self._V += dV * dt
        self._t += dt

        # ── Fork progression ──
        terminated_indices: list[int] = []
        for i in range(d):
            if self._rho[i] < 1.0:
                self._rho[i] += dt / self.C_s
                if self._rho[i] >= 1.0:
                    self._rho[i] = 1.0
                    terminated_indices.append(i)

        # ── Termination events: schedule a division D seconds later
        for _ in terminated_indices:
            if self.divide_on_terminate:
                self._pending_divisions.append(self._t + self.D_s)

        # ── Division event (drain the queue head) ──
        while self._pending_divisions and self._pending_divisions[0] <= self._t:
            self._pending_divisions.pop(0)
            self._V *= 0.5
            self._I_T *= 0.5
            self._I_D *= 0.5
            # Drop the oldest terminated generation if any are at ρ=1.
            for j in range(len(self._rho)):
                if self._rho[j] >= 1.0:
                    self._rho.pop(j)
                    break
            self._divisions += 1

        # ── Initiation check ──
        B = self._binding_sites()
        in_eclipse = (self._t - self._t_last_init) < self.eclipse_s
        if self._I_T >= B and not in_eclipse:
            # Fire initiation: prepend a fresh generation.
            self._rho.insert(0, 0.0)
            self._t_last_init = self._t
            self._initiations += 1
            self._initiation_volume = self._V
            self._init_this_tick = 1

    def update(self, state: dict, interval: float) -> dict:
        # Reset the "initiation this tick" flag at the start of each
        # outer tick. _step sets it to 1 if any sub-step fires an init.
        self._init_this_tick = 0
        # Sub-step until we cover the tick interval.
        remaining = float(interval)
        while remaining > 1e-9:
            dt = min(self.substep_s, remaining)
            self._step(dt)
            remaining -= dt

        d = len(self._rho)
        total = self._I_T + self._I_D
        atp_fraction = (self._I_T / total) if total > 0 else 0.0
        return {
            "listeners": {
                "itv2": {
                    "volume":              float(self._V),
                    "dnaa_atp_count":      float(self._I_T),
                    "dnaa_adp_count":      float(self._I_D),
                    "dnaa_total_count":    float(total),
                    "dnaa_atp_fraction":   float(atp_fraction),
                    "binding_sites":       float(self._binding_sites()),
                    "n_generations":       int(d),
                    "fork_progress":       list(self._rho),
                    "initiation_event":    int(self._init_this_tick),
                    "initiation_mass":     float(self._initiation_volume),
                    "initiations_so_far":  int(self._initiations),
                    "divisions_so_far":    int(self._divisions),
                },
            },
        }
