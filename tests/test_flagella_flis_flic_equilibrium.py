"""Unit tests for FlagellaFliSFliCEquilibrium's exact closed-form solve.

No dedicated test previously existed for this Step's own quadratic math --
tests/test_flagella_filament_elongation.py only exercises the downstream
elongation Step's read of FLIS-FLIC-CPLX as an input, never this Step's own
update(). Added 2026-09-14 after a hand-retyped copy of this file (written
by Maya to work through the math directly) surfaced two real bugs a plain
import/smoke check wouldn't catch: a class-name mismatch that broke
ecoli_baseline.py's import, and three "default" (missing the leading
underscore this codebase's schema convention requires) keys that silently
dropped the timestep/global_time config defaults instead of erroring.

These tests independently re-derive the expected equilibrium point with
plain Python arithmetic (math.sqrt, not numpy) rather than re-calling the
Step's own code, so a real regression in the algebra itself -- not just the
scaffolding -- would actually be caught.
"""
import math

import numpy as np

from v2ecoli.processes.flagella_flis_flic_equilibrium import FlagellaFliSFliCEquilibrium
from v2ecoli.types.quantity import ureg as units


def _bulk(triples):
    """triples: list of (id, count)."""
    return np.array(triples, dtype=[("id", "U40"), ("count", "i8")])


def _proc():
    return FlagellaFliSFliCEquilibrium(parameters={})


def _states(free_fliS, free_fliC, cplx, cell_mass_fg):
    return {
        "bulk": _bulk([
            ("EG11388-MONOMER[c]", free_fliS),   # free FliS monomer
            ("EG10321-MONOMER[e]", free_fliC),   # free FliC
            ("FLIS-FLIC-CPLX[e]", cplx),         # existing complex
        ]),
        "listeners": {"mass": {"cell_mass": cell_mass_fg * units.fg}},
        "timestep": 2.0,
        "global_time": 0.0,
        "next_update_time": 0.0,
    }


def _expected_c_star(free_fliS, free_fliC, cplx, cell_mass_fg,
                      kd_molar=5.26e-8, cell_density=1100.0,
                      n_avogadro=6.02214076e23):
    """Independent re-derivation of the true equilibrium point (plain
    math, no numpy/sqrt call shared with the Step under test)."""
    d_tot = free_fliS // 2 + cplx
    b_tot = free_fliC + cplx
    cell_mass_g = cell_mass_fg * 1e-15
    cell_volume_L = cell_mass_g / cell_density
    kd_counts = kd_molar * n_avogadro * cell_volume_L
    s = d_tot + b_tot + kd_counts
    discriminant = max(s * s - 4.0 * d_tot * b_tot, 0.0)
    c_star = (s - math.sqrt(discriminant)) / 2.0
    return min(max(c_star, 0.0), d_tot, b_tot)


def test_equilibrium_matches_independent_quadratic_solve():
    """Core correctness check: the Step's computed complex delta matches
    an independently re-derived equilibrium point, not just internal
    self-consistency."""
    proc = _proc()
    free_fliS, free_fliC, cplx, cell_mass_fg = 200, 150, 0, 1000.0
    expected_c_star = _expected_c_star(free_fliS, free_fliC, cplx, cell_mass_fg)
    expected_delta = int(round(expected_c_star)) - cplx

    out = proc.update(_states(free_fliS, free_fliC, cplx, cell_mass_fg))
    updates = dict(out["bulk"])
    assert updates[proc.flis_fliC_complex_id] == expected_delta


def test_stoichiometry_two_fliS_per_complex_one_fliC_per_complex():
    """Auvray et al. 2001 homodimer correction: each unit of complex
    formed/dissociated must move 2 raw FliS monomers, not 1."""
    proc = _proc()
    out = proc.update(_states(free_fliS=400, free_fliC=300, cplx=0,
                               cell_mass_fg=1000.0))
    updates = dict(out["bulk"])
    delta = updates[proc.flis_fliC_complex_id]
    assert delta > 0, "expected net complex formation from an all-free start"
    assert updates[proc.fliS_id] == -2 * delta
    assert updates[proc.fliC_id] == -delta


def test_dissociation_when_complex_above_equilibrium():
    """If cplx starts artificially above the true equilibrium point, the
    Step must dissociate it (delta < 0), releasing FliS/FliC back."""
    proc = _proc()
    # Tiny free pools, huge pre-existing complex -- guaranteed above
    # equilibrium for any real Kd.
    out = proc.update(_states(free_fliS=10, free_fliC=10, cplx=1000,
                               cell_mass_fg=1000.0))
    updates = dict(out["bulk"])
    delta = updates[proc.flis_fliC_complex_id]
    assert delta < 0
    assert updates[proc.fliS_id] == -2 * delta  # positive: FliS released
    assert updates[proc.fliC_id] == -delta        # positive: FliC released


def test_no_op_when_no_free_fliS_or_complex():
    """d_tot == 0: nothing to bind, must be a clean no-op (no bulk key at
    all), not a crash or a spurious zero-delta update."""
    proc = _proc()
    out = proc.update(_states(free_fliS=0, free_fliC=500, cplx=0,
                               cell_mass_fg=1000.0))
    assert "bulk" not in out


def test_no_op_when_no_free_fliC_or_complex():
    """b_tot == 0: symmetric no-op case for the other reactant."""
    proc = _proc()
    out = proc.update(_states(free_fliS=500, free_fliC=0, cplx=0,
                               cell_mass_fg=1000.0))
    assert "bulk" not in out


def test_small_post_division_counts_do_not_crash():
    """Regression guard for the original motivating bug: the shared ODE
    solver crashed at exactly this scale (free FliS=278, free FliC=626, a
    ~1fL cell -- see module docstring). This Step must handle it as
    cleanly as any other input, with no tolerance to trip."""
    proc = _proc()
    out = proc.update(_states(free_fliS=278, free_fliC=626, cplx=0,
                               cell_mass_fg=1000.0))
    updates = dict(out["bulk"])
    # Must not overdraw free FliS: 2*delta <= free_fliS whenever forming.
    assert -updates[proc.fliS_id] <= 278
    assert -updates[proc.fliC_id] <= 626
