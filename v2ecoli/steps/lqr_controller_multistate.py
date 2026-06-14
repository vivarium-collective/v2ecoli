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
        # Stabilizing shift for the CARE: the Millard Jacobian has two genuine
        # conserved-moiety modes at lambda=0 that are uncontrollable; solving the
        # CARE on (A - care_shift*I) makes those modes strictly stable for scipy
        # without materially changing the controllable-subspace gain.
        "care_shift": {"_default": 1e-3},
    }

    def initialize(self, config):
        self.parameters = config or {}
        lin_path = Path(self.parameters.get(
            "linearization_npz",
            self.config_schema["linearization_npz"]["_default"]))
        data = np.load(lin_path, allow_pickle=True)
        self.A = np.asarray(data["A"], dtype=float)               # (n, n)
        # B is (n, n_controls). Earlier single-input npz stored B as (n,);
        # reshape to (n, 1) for back-compat.
        B_raw = np.asarray(data["B"], dtype=float)
        self.B = B_raw if B_raw.ndim == 2 else B_raw.reshape(-1, 1)
        self.x_ss = np.asarray(data["x_ss"], dtype=float)         # (n,)
        self.species = [str(s) for s in data["species"]]
        # control_params + baseline_values are new (multi-input npz); single-
        # input back-compat: baseline_kF + assume PTS_4.kF.
        if "control_params" in data.files:
            self.control_params = [str(p) for p in data["control_params"]]
            self.baseline_values = [float(v) for v in data["baseline_values"]]
        else:
            self.control_params = ["(PTS_4).kF"]
            self.baseline_values = [float(data["baseline_kF"])]
        self.n_controls = self.B.shape[1]

        n = self.A.shape[0]
        q_w = float(self.parameters.get("Q_diag_weight", 1.0))
        r_w = float(self.parameters.get("R", 0.1))
        Q = q_w * np.eye(n)
        R = r_w * np.eye(self.n_controls)

        # Solve continuous-time Riccati: A^T P + P A - P B R^{-1} B^T P + Q = 0.
        # ``self.A`` is the true Jacobian J (near-Hurwitz). Its two conserved-
        # moiety modes sit at lambda=0 and are uncontrollable, which scipy's
        # solver rejects (imaginary-axis uncontrollable modes). Shifting to
        # (A - care_shift*I) makes them strictly stable so the CARE is
        # well-posed; the resulting gain still stabilises the controllable
        # subspace and leaves the (genuinely uncontrollable) conservation modes
        # untouched.
        shift = float(self.parameters.get("care_shift", 1e-3))
        self.gain_degenerate = False
        try:
            from scipy.linalg import solve_continuous_are
            P = solve_continuous_are(self.A - shift * np.eye(n), self.B, Q, R)
            self.K = np.linalg.inv(R) @ self.B.T @ P  # (n_controls, n)
        except Exception as e:
            # Loud, NOT silent: a zero-gain controller is an UNCONTROLLED run.
            # Surface it via gain_degenerate so lqr_diagnostics records it
            # rather than the run masquerading as "controlled".
            self.gain_degenerate = True
            print(
                "[LQRMultiState] WARNING: Riccati solve failed even after "
                f"care_shift={shift:g} ({e!r}); falling back to ZERO GAIN — "
                "the metabolism is effectively UNCONTROLLED this run."
            )
            self.K = np.zeros((self.n_controls, n))

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
        u_vec = -self.K @ deviation  # (n_controls,) — multi-input

        # Tracking error metric = L2 norm of state deviation
        track_err = float(np.sqrt(np.sum(deviation ** 2)))

        # Per-control breakdown: dict of param-name → control value
        u_dict = {p: float(u_vec[k]) for k, p in enumerate(self.control_params)}

        self._log.append({
            "tick": self._tick,
            "u": u_dict,
            "u_norm": float(np.linalg.norm(u_vec)),
            "max_abs_deviation": float(np.max(np.abs(deviation))),
            "deviation_norm": track_err,
            "max_deviation_species": self.species[int(np.argmax(np.abs(deviation)))],
        })
        self._tick += 1

        return {
            # NEW multi-input output: per-parameter control AND legacy scalar 'u' for back-compat
            "lqr_control": {
                "u_dict": u_dict,
                "u": float(u_vec[0]) if self.n_controls == 1 else float(np.linalg.norm(u_vec)),
                "K_norm": float(np.linalg.norm(self.K)),
                "n_state": len(self.species),
                "n_controls": self.n_controls,
                "control_params": self.control_params,
            },
            "lqr_diagnostics": {
                "last_tick": self._log[-1],
                "n_ticks": len(self._log),
                "rms_deviation_norm": float(np.sqrt(np.mean([
                    e["deviation_norm"] ** 2 for e in self._log
                ]))) if self._log else 0.0,
                # True when the Riccati solve failed and the controller is
                # running at zero gain (i.e. the metabolism is uncontrolled).
                "gain_degenerate": bool(self.gain_degenerate),
            },
        }


def register(core):
    core.register_link("LQRControllerMultiState", LQRControllerMultiState)


__all__ = ["LQRControllerMultiState", "register"]
