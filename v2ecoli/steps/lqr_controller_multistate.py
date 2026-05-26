"""Multi-state LQR for Millard 2017 — closes the bandwidth limit of the scalar variant.

Uses linearization saved at v2ecoli/data/millard_linearization.npz (15-state
Jacobian + B vector around Millard's published steady state). Solves the
continuous-time Riccati via scipy.linalg.solve_continuous_are for the
optimal gain K (1×15 row), then applies u = -K(x - x_ss) per tick.

The scalar LQR's bandwidth limit (best Sobol RMS ~8.6e-4) came from
estimating growth-rate from a 3-species biomass proxy via finite difference;
this controller uses 15-state deviation directly, no derivative noise.

For the lqr-growth-rate-tracking primary test: the multi-state K can drive
multiple state errors simultaneously, which is what scalar-on-proxy can't do.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np

from process_bigraph import Process
from v2ecoli.types.stores import InPlaceDict


class LQRControllerMultiState(Process):
    """Multi-state continuous-time LQR controller for Millard.

    Topology:
      inputs:
        central_metabolites_millard: ["shared", "central_metabolites"]
      outputs:
        lqr_control: ["shared", "lqr_control"]
        lqr_diagnostics: ["shared", "lqr_diagnostics"]
    """

    name = "lqr_controller_multistate"
    config_schema = {
        "linearization_npz": {"_default": "v2ecoli/data/millard_linearization.npz"},
        "Q_diag_weight": {"_default": 1.0},   # weight on diag(Q)
        "R": {"_default": 0.1},               # scalar weight on control effort
        "tick_s": {"_default": 100.0},
    }

    def initialize(self, config):
        self.parameters = config or {}
        lin_path = Path(self.parameters.get(
            "linearization_npz",
            self.config_schema["linearization_npz"]["_default"]))
        data = np.load(lin_path, allow_pickle=True)
        self.A = np.asarray(data["A"], dtype=float)          # (n, n)
        self.B = np.asarray(data["B"], dtype=float).reshape(-1, 1)  # (n, 1)
        self.x_ss = np.asarray(data["x_ss"], dtype=float)    # (n,)
        self.species = [str(s) for s in data["species"]]
        self.baseline_kF = float(data["baseline_kF"])

        n = self.A.shape[0]
        q_w = float(self.parameters.get("Q_diag_weight", 1.0))
        r_w = float(self.parameters.get("R", 0.1))
        Q = q_w * np.eye(n)
        R = np.array([[r_w]])

        # Solve continuous-time Riccati: A^T P + P A - P B R^{-1} B^T P + Q = 0
        try:
            from scipy.linalg import solve_continuous_are
            P = solve_continuous_are(self.A, self.B, Q, R)
            self.K = (np.linalg.inv(R) @ self.B.T @ P).flatten()  # (n,)
        except Exception as e:
            print(f"[LQRMultiState] Riccati solve failed ({e!r}); using zero gain")
            self.K = np.zeros(n)

        self.tick_s = float(self.parameters.get("tick_s", 100.0))
        self._tick = 0
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
        m = state.get("central_metabolites_millard", {}) or {}
        # Build state vector x from species observations (default to x_ss for missing)
        x = np.array([
            float(m[sp]) if sp in m and m[sp] is not None else self.x_ss[i]
            for i, sp in enumerate(self.species)
        ])
        deviation = x - self.x_ss
        u = float(-self.K @ deviation)  # scalar

        # Tracking error metric = max abs deviation (signed)
        # Use Q-weighted norm for a "single-number" tracking
        track_err = float(np.sqrt(np.sum(deviation ** 2)))

        self._log.append({
            "tick": self._tick,
            "u": u,
            "max_abs_deviation": float(np.max(np.abs(deviation))),
            "deviation_norm": track_err,
            "max_deviation_species": self.species[int(np.argmax(np.abs(deviation)))],
        })
        self._tick += 1

        return {
            "lqr_control": {"u": u, "K_norm": float(np.linalg.norm(self.K)),
                            "n_state": len(self.species)},
            "lqr_diagnostics": {
                "last_tick": self._log[-1],
                "n_ticks": len(self._log),
                "rms_deviation_norm": float(np.sqrt(np.mean([
                    e["deviation_norm"] ** 2 for e in self._log
                ]))) if self._log else 0.0,
            },
        }


def register(core):
    core.register_link("LQRControllerMultiState", LQRControllerMultiState)


__all__ = ["LQRControllerMultiState", "register"]
