"""Linearize Millard 2017 ODE around its published steady state.

Numerically computes:
  x_ss: steady-state concentrations of N key species
  A:    N×N Jacobian via 1% finite-difference perturbation
  B:    N×1 control vector for PTS_4.kF (glucose-uptake forward rate)

Saves two .npz files:
  v2ecoli/data/millard_linearization.npz       — full 15-state (true Jacobian J)
  v2ecoli/data/millard_linearization_sub.npz   — 4-state subspace (G6P, F6P,
                                                 PEP, PYR), retained for
                                                 reference only.

NOTE (2026-06-11): an earlier version stored ``I + J`` instead of J (see the
Jacobian step below), which made the full (A, B) pair look like it had
"unstable uncontrollable modes (Riccati infeasible)" — the original reason the
4-state subspace + multi-input workarounds were introduced. That was a
misdiagnosis of the +I bug: with the true Jacobian J, the full 15-state pair is
stable except for two genuine conservation-law (lambda=0) modes, and the LQR
solves fine on the full system (the controller applies a small ``-eps*I`` shift
so those two uncontrollable conservation modes are strictly stable for scipy).

Run from worktree root:
    python scripts/linearize_millard.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import basico

FULL_SPECIES = ["ATP", "ADP", "AMP", "NAD", "NADH", "NADP", "NADPH",
                "G6P", "F6P", "PEP", "PYR", "AKG", "MAL", "OAA", "CIT"]
SUB_SPECIES = ["G6P", "F6P", "PEP", "PYR"]
# Multi-input control: 3 reaction parameters spanning glucose uptake /
# glycolysis-commit / glycolysis-exit. Multi-input improves controllability of
# the glycolytic states; the two remaining uncontrollable modes are genuine
# conservation laws (lambda=0), handled by the controller's -eps*I shift rather
# than by dropping to a subspace.
CONTROL_PARAMS = [
    ("PTS_4", "(PTS_4).kF"),   # glucose uptake (kF)
    ("PFK",   "(PFK).Vmax"),   # F6P → FDP (glycolysis commitment)
    ("PYK",   "(PYK).Vmax"),   # PEP → PYR (glycolysis exit)
]
MODEL_PATH = "v2ecoli/models/sbml/millard2017_central_metabolism.xml"
OUT_DIR = Path("v2ecoli/data")


def main():
    print("1. Running Millard to steady state (5000 s)...")
    basico.load_model(MODEL_PATH)
    ts = basico.run_time_course(start_time=0, duration=5000.0, intervals=100,
                                use_sbml_id=True, update_model=True)
    last = ts.iloc[-1]
    x_ss = {sp: float(last.get(sp, 0.0)) for sp in FULL_SPECIES}
    print(f"   ATP_ss = {x_ss['ATP']:.4f} mM (expected ~2.57)")

    def reset_and_run(perturb_sp: str | None, delta: float,
                      param_delta: tuple[str, float] | None = None,
                      dt: float = 1.0) -> np.ndarray:
        """param_delta = (param_full_name, delta_value) — adds delta to the
        baseline value for that parameter; no perturbation if None."""
        basico.load_model(MODEL_PATH)
        for sp, v in x_ss.items():
            try: basico.set_species(sp, initial_concentration=v)
            except Exception: pass
        if perturb_sp is not None:
            cur = basico.get_species(perturb_sp).iloc[0]["initial_concentration"]
            basico.set_species(perturb_sp, initial_concentration=float(cur) + delta)
        if param_delta is not None:
            param_name, delta_val = param_delta
            rxn_name = param_name.split(".")[0].lstrip("(").rstrip(")")
            base_val = float(basico.get_reaction_parameters(reaction_name=rxn_name)
                            .loc[param_name]["value"])
            basico.set_reaction_parameters(name=param_name, value=base_val + delta_val)
        out = basico.run_time_course(start_time=0, duration=dt, intervals=1,
                                     use_sbml_id=True, update_model=False)
        l = out.iloc[-1]
        return np.array([float(l.get(s, 0.0)) for s in FULL_SPECIES])

    print(f"\n2. Computing {len(FULL_SPECIES)}×{len(FULL_SPECIES)} Jacobian (1% finite diff)...")
    dt = 1.0
    n_sp = len(FULL_SPECIES)
    # Perturbing a STATE by delta and integrating for dt gives the integrated-
    # state sensitivity Phi(dt) = d x(dt)/d x0 = exp(J*dt) ~ I + J*dt, NOT the
    # derivative. The continuous-time Jacobian is therefore J = (Phi - I)/dt.
    # The previous version stored Phi/dt = I/dt + J (a spurious +I/dt), which
    # shifted every eigenvalue up by 1/dt: a stable Millard Jacobian looked
    # anti-stable and its conserved-moiety lambda=0 modes appeared at lambda=+1
    # (unstable + uncontrollable), making the LQR Riccati solve infeasible.
    Phi = np.zeros((n_sp, n_sp))
    base = reset_and_run(None, 0.0, None, dt)
    for j, sp in enumerate(FULL_SPECIES):
        delta = max(abs(x_ss[sp]) * 0.01, 1e-4)
        pert = reset_and_run(sp, delta, None, dt)
        Phi[:, j] = (pert - base) / delta
    A_full = (Phi - np.eye(n_sp)) / dt

    print(f"3. Computing B matrix ({len(CONTROL_PARAMS)} control inputs)...")
    B_matrix = np.zeros((len(FULL_SPECIES), len(CONTROL_PARAMS)))
    baseline_values = {}
    b_base = reset_and_run(None, 0.0, None, dt)
    for k, (rxn, param) in enumerate(CONTROL_PARAMS):
        rxn_name = param.split(".")[0].lstrip("(").rstrip(")")
        base_val = float(basico.get_reaction_parameters(reaction_name=rxn_name)
                         .loc[param]["value"])
        baseline_values[param] = base_val
        delta_val = base_val * 0.01
        b_pert = reset_and_run(None, 0.0, (param, delta_val), dt)
        B_matrix[:, k] = (b_pert - b_base) / delta_val / dt
        print(f"   {param} baseline={base_val:.3e}, |B[:, {k}]|_inf = {np.max(np.abs(B_matrix[:, k])):.3e}")

    x_ss_vec = np.array([x_ss[sp] for sp in FULL_SPECIES])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "millard_linearization.npz",
             x_ss=x_ss_vec, A=A_full, B=B_matrix,
             species=np.array(FULL_SPECIES),
             control_params=np.array([p for _, p in CONTROL_PARAMS]),
             baseline_values=np.array([baseline_values[p] for _, p in CONTROL_PARAMS]))
    print(f"   Saved {OUT_DIR/'millard_linearization.npz'}")
    print(f"   spectral radius: {np.max(np.abs(np.linalg.eigvals(A_full))):.4e}")

    # Controllability check on full system
    n_full = len(FULL_SPECIES)
    C_full = np.column_stack([np.linalg.matrix_power(A_full, k) @ B_matrix
                              for k in range(n_full)])
    print(f"   Controllability rank (full multi-input): {np.linalg.matrix_rank(C_full)} / {n_full}")

    idx = [FULL_SPECIES.index(s) for s in SUB_SPECIES]
    A_sub = A_full[np.ix_(idx, idx)]
    B_sub = B_matrix[idx, :]   # subspace x all controls
    x_ss_sub = x_ss_vec[idx]
    n_sub = len(SUB_SPECIES)
    C = np.column_stack([np.linalg.matrix_power(A_sub, k) @ B_sub
                         for k in range(n_sub)])
    print(f"\n4. Subspace {SUB_SPECIES}:")
    print(f"   Controllability rank: {np.linalg.matrix_rank(C)} / {n_sub}")
    np.savez(OUT_DIR / "millard_linearization_sub.npz",
             x_ss=x_ss_sub, A=A_sub, B=B_sub,
             species=np.array(SUB_SPECIES),
             control_params=np.array([p for _, p in CONTROL_PARAMS]),
             baseline_values=np.array([baseline_values[p] for _, p in CONTROL_PARAMS]))
    print(f"   Saved {OUT_DIR/'millard_linearization_sub.npz'}")


if __name__ == "__main__":
    main()
