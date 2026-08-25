"""The declared-design landing check — the half that is engine-agnostic.

★ The property under test throughout: a target that CANNOT be judged must never
read as landed. A screen's ranking is only about its designs if the designs were
built, and "we could not tell" silently counted as "fine" is how a check stops
being able to fail.
"""
from __future__ import annotations

import pytest

from v2ecoli.library.design_landing import (
    ZERO_FLOOR, arm_targets, landing_violations, observed_fold_change,
    target_landed)


# --------------------------------------------------------------------------- #
# fold change
# --------------------------------------------------------------------------- #
def test_fold_change_is_relative_to_the_reference_arm():
    # 10x overexpression against a reference that expresses 120 units.
    assert observed_fold_change(1200.0, 120.0) == pytest.approx(10.0)
    assert observed_fold_change(60.0, 120.0) == pytest.approx(0.5)


@pytest.mark.parametrize("arm,ref", [
    (None, 120.0),          # target not observed in this arm
    (1200.0, None),         # target not observed in the reference
    (1200.0, 0.0),          # reference does not express it — nothing to be a factor OF
    ("n/a", 120.0),         # non-numeric, e.g. a missing-value sentinel
])
def test_undefined_comparisons_return_none_rather_than_a_number(arm, ref):
    """★ Each of these would otherwise produce a plausible-looking fold change —
    0, inf, or a crash. None is the only honest answer, and it is what makes the
    downstream 'unjudgeable is not a pass' rule expressible."""
    assert observed_fold_change(arm, ref) is None


# --------------------------------------------------------------------------- #
# landing
# --------------------------------------------------------------------------- #
def test_relative_tolerance_brackets_the_declared_multiplier():
    assert target_landed(2.0, 1.86, 0.3) is True       # 7% off
    assert target_landed(2.0, 2.59, 0.3) is True       # inside the 30% band
    assert target_landed(2.0, 2.61, 0.3) is False      # outside it
    assert target_landed(10.0, 6.0, 0.3) is False      # a 10x that only reached 6x
    # ⚠ Deliberately not asserted AT the boundary. `abs(2.60 - 2.0)` is
    # 0.6000000000000001 in binary floating point, so exactly-30%-off lands on
    # whichever side the representation falls — and no epsilon we invented here
    # would make that meaningful for a biological tolerance. Pick tolerances with
    # room in them; do not read the boundary as exact.


def test_a_declared_knockout_is_judged_on_an_absolute_floor():
    """★ Discriminating, and the reason the special case exists: a RELATIVE
    tolerance around a declared 0 is satisfied by nothing but exactly 0, so a
    knockout that left 1% of the protein behind would read as landed under the
    general rule only if the rule were written carelessly."""
    assert target_landed(0.0, 0.0, 0.3) is True
    assert target_landed(0.0, ZERO_FLOOR / 2, 0.3) is True
    assert target_landed(0.0, 0.01, 0.3) is False      # 1% left ≠ knocked out
    assert target_landed(0.0, 0.5, 0.99) is False      # tolerance cannot rescue it


def test_unjudgeable_is_none_not_a_pass():
    assert target_landed(2.0, None, 0.3) is None
    assert target_landed(None, 1.0, 0.3) is None


# --------------------------------------------------------------------------- #
# the arm block
# --------------------------------------------------------------------------- #
def test_every_declared_target_appears_even_when_unobserved():
    """★ The failure this check exists for: a target that quietly vanishes from
    the report. Absence must be recorded, not omitted."""
    declared = {"gene_a": 10.0, "gene_b": 0.0, "gene_c": 1.0}
    observed = {"gene_a": 1000.0, "gene_b": 0.0}          # gene_c never observed
    reference = {"gene_a": 100.0, "gene_b": 40.0, "gene_c": 55.0}
    out = arm_targets(declared, observed, reference, tolerance=0.3)

    assert sorted(out) == ["gene_a", "gene_b", "gene_c"]
    assert out["gene_a"] == {"expected": 10.0, "observed": 10.0, "within_tolerance": True}
    assert out["gene_b"]["within_tolerance"] is True       # knocked out, as declared
    assert out["gene_c"] == {"expected": 1.0, "observed": None, "within_tolerance": None}


def test_an_unchanged_target_still_has_to_be_checked():
    """A declared 1.0 is a claim — "this gene is not perturbed" — and a build that
    moved it anyway has not produced the declared design."""
    out = arm_targets({"gene_a": 1.0}, {"gene_a": 300.0}, {"gene_a": 100.0}, 0.3)
    assert out["gene_a"]["observed"] == pytest.approx(3.0)
    assert out["gene_a"]["within_tolerance"] is False


# --------------------------------------------------------------------------- #
# violations
# --------------------------------------------------------------------------- #
def test_violations_include_the_unjudgeable_not_only_the_failed():
    targets = {
        "arm_ok":      {"g": {"expected": 2.0, "observed": 2.0, "within_tolerance": True}},
        "arm_missed":  {"g": {"expected": 2.0, "observed": 0.9, "within_tolerance": False}},
        "arm_unknown": {"g": {"expected": 2.0, "observed": None, "within_tolerance": None}},
    }
    got = [(arm, t) for arm, t, _ in landing_violations(targets)]
    assert got == [("arm_missed", "g"), ("arm_unknown", "g")]


def test_a_panel_where_every_design_landed_has_no_violations():
    declared = {"g1": 10.0, "g2": 0.0}
    ref = {"g1": 100.0, "g2": 80.0}
    targets = {f"arm_{i}": arm_targets(declared, {"g1": 1000.0, "g2": 0.0}, ref, 0.3)
               for i in range(3)}
    assert landing_violations(targets) == []
