"""Simplified scalar LQR controller for Millard 2017 + Bridge composite.

v2ecoli-pdmp Phase 1 deliverable: outer LQR control layer that drives a
biomass-proxy growth rate toward the Phase 0 M9-glucose reference μ(t)
ensemble mean.

Scope (intentionally minimal — proof of concept, not production):
  - **1-state LQR**. State x = biomass-proxy estimate. Control u = scalar
    gain on glucose-uptake (intended to feed back into Millard's GLT rate
    constant, currently logged as a diagnostic — closed-loop wiring is the
    follow-up task).
  - **Continuous-time analytical Riccati for 1D**: with state dynamics
    ẋ = a x + b u and cost J = ∫ Q(x-x_ref)² + R u² dt, the optimal gain
    is K = (1/R) × b × P, where P solves the algebraic Riccati equation
    P b² / R = 2 a P + Q. For a 1D system: P = (a + √(a² + Q b² / R)) / (b²/R).
  - **Reference μ(t)** loaded from v2ecoli/data/phase0_glucose_mu_ref.npy
    (real Phase 0 M9-glucose ensemble mean, N=64).
  - **Biomass-proxy** = weighted sum of (ATP + NADH + R5P); the growth-rate
    estimate is d(proxy)/dt from finite difference across recent ticks.

What this delivers (for the lqr-growth-rate-tracking primary test):
  - millard_lqr composite that runs without error
  - Per-tick log of (μ_estimated, μ_reference, u_control, tracking_error)
  - RMS tracking error metric

What it does NOT do (honest gaps):
  - Doesn't actually feed u back into Millard's reactions. The CopasiUTCProcess
    wrapper doesn't expose parameter mutation; closing the loop needs either
    a new CopasiUTCProcess feature or a wrapper that rebuilds the model per
    tick with adjusted parameters. Tracked as follow-up.
  - Single-state, not multi-state. Sobol-scannable LQR weights (the
    millard-with-lqr-tuning planned run target) need a multi-state controller.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any

import numpy as np

from process_bigraph import Process
from v2ecoli.types.stores import InPlaceDict


# Biomass-proxy weights (per mM concentration). Crude proxy: ATP for energy,
# NADH for redox state, R5P for nucleotide-synthesis precursor pool.
PROXY_WEIGHTS = {
    "ATP": 1.0,
    "NADH": 0.5,
    "R5P": 2.0,
}


def proxy_from_state(state: dict) -> float:
    val = 0.0
    for sp, w in PROXY_WEIGHTS.items():
        v = state.get(sp)
        if v is None: continue
        try: val += w * float(v)
        except (TypeError, ValueError): continue
    return val


def lqr_1d_gain(a: float, b: float, Q: float, R: float) -> float:
    """Closed-form LQR gain for the scalar system ẋ = a x + b u under
    cost J = ∫ Q(x-x_ref)² + R u² dt. Returns K such that u = -K(x-x_ref).
    Uses the positive root of the algebraic Riccati equation."""
    if b == 0 or R <= 0:
        return 0.0
    disc = a * a + Q * b * b / R
    P = (a + math.sqrt(disc)) * R / (b * b)
    return P * b / R


class LQRController(Process):
    """Scalar LQR controller. See module docstring for scope + honest gaps.

    Topology (composite-side):
      inputs:
        central_metabolites_millard: ["shared", "central_metabolites"]
      outputs:
        lqr_control: ["shared", "lqr_control"]            (scalar u, latest)
        lqr_diagnostics: ["shared", "lqr_diagnostics"]    (per-tick log)
    """

    name = "lqr_controller"
    config_schema = {
        "reference_npy": {"_default": "v2ecoli/data/phase0_glucose_mu_ref.npy"},
        "Q": {"_default": 1.0},   # state-tracking weight
        "R": {"_default": 0.1},   # control-effort weight
        "a": {"_default": -0.001}, # assumed linearization: state decays slowly
        "b": {"_default": 1.0},    # control gain
        "tick_s": {"_default": 100.0},
    }

    def initialize(self, config):
        self.parameters = config or {}
        ref_path = Path(self.parameters.get(
            "reference_npy",
            self.config_schema["reference_npy"]["_default"]))
        if ref_path.exists():
            ref = np.load(ref_path)
            self.ref_t = ref[0]   # time array
            self.ref_mu = ref[1]  # μ(t) array
            self._have_ref = True
        else:
            self.ref_t = None
            self.ref_mu = None
            self._have_ref = False

        self.Q = float(self.parameters.get("Q", 1.0))
        self.R = float(self.parameters.get("R", 0.1))
        self.a = float(self.parameters.get("a", -0.001))
        self.b = float(self.parameters.get("b", 1.0))
        self.tick_s = float(self.parameters.get("tick_s", 100.0))
        self.K = lqr_1d_gain(self.a, self.b, self.Q, self.R)

        # Per-tick state
        self._prev_proxy: float | None = None
        self._t: float = 0.0
        self._log: list[dict] = []

    def __init__(self, config=None, core=None):
        super().__init__(config or {}, core)
        self.initialize(config or {})

    def inputs(self):
        return {"central_metabolites_millard": InPlaceDict()}

    def outputs(self):
        return {
            "lqr_control": InPlaceDict(),
            "lqr_diagnostics": InPlaceDict(),
        }

    def update(self, state, interval):
        return self.next_update(interval, state)

    def next_update(self, timestep, states):
        m = states.get("central_metabolites_millard", {}) or {}
        proxy = proxy_from_state(m)
        # Estimate μ via finite difference of proxy across ticks.
        # Skip on first tick (no prev) AND on the SECOND tick if prev_proxy
        # was effectively zero (composite-store initialised empty; first
        # tick reads {} and computes proxy=0 → next tick's finite diff
        # would explode by dividing by ~0).
        MIN_PROXY = 1e-3  # mM-scale floor; below this we consider proxy uninitialised
        if (self._prev_proxy is None or proxy <= 0
                or self._prev_proxy < MIN_PROXY):
            mu_est = 0.0
        else:
            # Relative growth rate (1/s): d(log proxy)/dt = (Δproxy/proxy)/Δt
            mu_est = (proxy - self._prev_proxy) / self._prev_proxy / timestep
        # Reference μ at current time (last value if past end).
        if self._have_ref and self.ref_t is not None:
            idx = min(int(self._t / max(self.ref_t[1] - self.ref_t[0], 1.0)),
                      len(self.ref_mu) - 1)
            mu_ref = float(self.ref_mu[idx])
        else:
            mu_ref = 0.00025  # fallback: Phase 0 steady-state ~2.5e-4 /s
        # LQR control: u = -K (x - x_ref). Use μ as the state proxy for the
        # control law since we want growth-rate tracking.
        tracking_err = mu_est - mu_ref
        u = -self.K * tracking_err

        self._log.append({
            "t": self._t,
            "proxy": proxy,
            "mu_est": mu_est,
            "mu_ref": mu_ref,
            "tracking_err": tracking_err,
            "u_control": u,
        })
        self._prev_proxy = proxy
        self._t += timestep

        return {
            "lqr_control": {"u": u, "K": self.K, "Q": self.Q, "R": self.R},
            "lqr_diagnostics": {
                "last_tick": self._log[-1],
                "n_ticks": len(self._log),
                "rms_tracking_err": float(np.sqrt(np.mean([
                    e["tracking_err"]**2 for e in self._log
                ]))) if self._log else 0.0,
            },
        }


def register(core):
    core.register_link("LQRController", LQRController)


__all__ = ["LQRController", "register", "lqr_1d_gain", "proxy_from_state"]
