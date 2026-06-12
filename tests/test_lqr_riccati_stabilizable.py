"""Regression test: the multi-state Millard LQR must produce a real
stabilizing gain, not silently fall back to zero gain.

Root cause this guards against: scripts/linearize_millard.py computed the
linearization as ``(x(dt) - x_ss)/delta/dt`` = the integrated-state
sensitivity Phi(dt) = exp(J*dt) ~ I + J*dt, divided by dt. With dt=1 that
stored ``A = I + J`` instead of the true Jacobian J. The spurious +I shift
turned the (stable) Millard Jacobian into an anti-stable matrix whose
conserved-moiety lambda=0 modes appeared at lambda=+1 (unstable AND
uncontrollable), so scipy.solve_continuous_are failed and the controller
silently used K = 0 -- i.e. NO control. Every PDMP-Phase-1 run was
uncontrolled, which is the root cause of the cell_mass ~3sigma gate miss.

The fix: the npz now stores the true Jacobian J (near-Hurwitz, with two
exact conservation-law lambda=0 modes), and the controller solves the CARE
on ``J - eps*I`` so those uncontrollable conservation modes are strictly
stable for the solver. The result is a non-zero stabilizing gain.
"""
import numpy as np
import pytest

from v2ecoli.steps.lqr_controller_multistate import LQRControllerMultiState


def test_multistate_lqr_gain_is_nonzero_and_stabilizing():
    from v2ecoli.core import build_core
    core = build_core()
    ctrl = LQRControllerMultiState(config={}, core=core)

    # 1) The gain must be real and non-zero (zero gain == silent no-control).
    K = np.asarray(ctrl.K, dtype=float)
    assert np.isfinite(K).all(), "LQR gain has non-finite entries"
    assert np.linalg.norm(K) > 1e-6, (
        "LQR gain is ~0 -> Riccati solve fell back to zero gain (no control). "
        f"||K|| = {np.linalg.norm(K):.3e}"
    )

    # 2) The closed loop on the true Jacobian must be stabilizing: every
    #    CONTROLLABLE mode is driven strictly stable, and only the genuine
    #    conserved-moiety modes are allowed to remain marginal (Re ~ 0).
    J = np.asarray(ctrl.A, dtype=float)        # true Jacobian (post-fix)
    B = np.asarray(ctrl.B, dtype=float)
    cl = np.linalg.eigvals(J - B @ K).real
    n = J.shape[0]
    n_marginal = int(np.sum(cl > 1e-6))
    # At most the two conservation laws may sit at the imaginary axis; nothing
    # should be genuinely unstable (Re > +1e-6).
    assert n_marginal == 0, (
        f"closed loop has {n_marginal} unstable modes (Re>0): {np.round(np.sort(cl),3)}"
    )
    n_stable = int(np.sum(cl < -1e-3))
    assert n_stable >= n - 2, (
        f"expected >= {n-2} strictly-stable closed-loop modes, got {n_stable}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
