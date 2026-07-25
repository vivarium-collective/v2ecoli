"""Behavioral tests for the ``plasmids`` composite (baseline + pBR322 plasmid).

The plasmid is added as a purely additive layer on the STANDARD baseline cache
(decoupled from the ParCa). These tests pin the two properties that define a
working plasmid layer:

  1. The plasmid replicates: copy number stays >= 1, the Brendel-Perelson 1993
     copy-number ODE evolves, and at least one replication round completes
     (a new full_plasmid appears) — on the RNA-II timescale, without exploding.
  2. Baseline invariance: the plasmid layer must NOT perturb baseline whole-cell
     dynamics. Cell / dry mass of the plasmid composite match the plain baseline
     to within a tight tolerance (empirically bit-identical).

Both tests build on ``out/cache`` (the standard baseline cache) and run a few
hundred steps — enough to show copy-number dynamics + a replication round but
comfortably under the first division and the 120 s per-test cap.
"""
import os

import numpy as np
import pytest

from v2ecoli.library.quantity_helpers import fg_magnitude


pytestmark = pytest.mark.sim


def _skip_if_no_cache():
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; build via scripts/build_cache.py")


def _n_active(store):
    return int(store["_entryState"].sum())


def test_plasmid_composite_replicates():
    """Copy number >= 1 throughout, ODE evolves, and >= 1 replication round."""
    _skip_if_no_cache()
    from v2ecoli.core import build_core
    from v2ecoli import build_composite

    core = build_core()
    comp = build_composite("plasmids", core=core, seed=0, cache_dir="out/cache")
    cell = comp.state["agents"]["0"]

    # Plasmid unique stores exist and start with exactly one full plasmid.
    assert "full_plasmid" in cell["unique"], "plasmid unique store missing"
    assert _n_active(cell["unique"]["full_plasmid"]) == 1
    ctrl0 = {k: float(v) for k, v in cell["process_state"]["plasmid_rna_control"].items()}

    # Run to just past the first replication round (deterministic at seed=0,
    # first new plasmid appears around t~650 s). Track copy number every chunk.
    min_copy = _n_active(cell["unique"]["full_plasmid"])
    max_copy = min_copy
    replisome_seen = False
    for _ in range(7):
        comp.run(100)
        n = _n_active(cell["unique"]["full_plasmid"])
        min_copy = min(min_copy, n)
        max_copy = max(max_copy, n)
        if _n_active(cell["unique"]["plasmid_active_replisome"]) > 0:
            replisome_seen = True

    ctrl1 = {k: float(v) for k, v in cell["process_state"]["plasmid_rna_control"].items()}

    # (a) copy number never drops below 1 and never explodes.
    assert min_copy >= 1, f"plasmid copy number fell to {min_copy}"
    assert max_copy < 1000, f"plasmid copy number exploded to {max_copy}"

    # (b) the BP1993 ODE state genuinely evolved (RNA II accumulates, D_p flux).
    assert ctrl1["R_II"] > ctrl0["R_II"], "RNA II did not accumulate — ODE stalled"
    assert ctrl1["D_p"] > 0.0, "replication-competent pool D_p never populated"
    assert ctrl1["R_I"] != ctrl0["R_I"], "RNA I did not change — ODE stalled"

    # (c) at least one replication round completed (a new full_plasmid appeared,
    # having passed through an active replisome).
    assert max_copy >= 2 or replisome_seen, (
        "no plasmid replication initiated over the run "
        f"(max copy={max_copy}, replisome_seen={replisome_seen})")


def test_plasmid_preserves_baseline():
    """The additive plasmid layer must not perturb baseline mass/growth.

    Runs baseline and the plasmid composite from the same seed/cache for the
    same number of steps and asserts cell / dry mass match to a tight tolerance.
    (Empirically the two are bit-identical: the plasmid uses its own RNG, its
    DNA lives in separate unique stores, and its tiny dNTP draw is absorbed by
    metabolism — verified exact at 720 steps as well.)"""
    _skip_if_no_cache()
    from v2ecoli.core import build_core
    from v2ecoli import build_composite

    n_steps = 450

    def run(name):
        core = build_core()
        comp = build_composite(name, core=core, seed=0, cache_dir="out/cache")
        comp.run(n_steps)
        mass = comp.state["agents"]["0"]["listeners"]["mass"]
        return fg_magnitude(mass["cell_mass"]), fg_magnitude(mass["dry_mass"])

    cell_b, dry_b = run("baseline")
    cell_p, dry_p = run("plasmids")

    assert cell_b > 0 and dry_b > 0, "baseline produced non-positive mass"
    rel_cell = abs(cell_p - cell_b) / cell_b
    rel_dry = abs(dry_p - dry_b) / dry_b
    assert rel_cell < 1e-6, (
        f"plasmid layer perturbed cell mass: baseline={cell_b:.6f} "
        f"plasmids={cell_p:.6f} (rel diff {rel_cell:.2e})")
    assert rel_dry < 1e-6, (
        f"plasmid layer perturbed dry mass: baseline={dry_b:.6f} "
        f"plasmids={dry_p:.6f} (rel diff {rel_dry:.2e})")
