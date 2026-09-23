"""Did the declared design actually land? — the engine-agnostic half.

A design screen ranks engineered variants. The ranking is only about the designs
if the designs were actually built: a perturbation that silently failed to apply
still produces a cell, still produces a number, and still takes a place in the
ranking — ranked on something other than the design it is labelled with. So a
screen has to check, per arm, that each declared target moved the way it was
declared to.

WHAT THIS MODULE IS, AND IS NOT. It is the comparison and its accounting: given
each arm's declared targets, the observed value for each target, and the
reference arm they are relative to, it computes the observed fold-change and
whether it landed within tolerance. It is a pure function of numbers.

It is NOT the extraction. Turning a run into "the observed value for gene G in
arm A" requires that engine's stored output and its own id resolution (gene ->
cistron -> monomer), and differs per engine. That half is engine-specific by
nature; this half is written once and serves any of them.

⚠ FOLD-CHANGE IS RELATIVE TO THE REFERENCE ARM, NOT TO ONE. A perturbation of
1.0 means "unchanged from the reference", and the reference's own absolute level
is whatever the build produced. Comparing an absolute count to a declared
multiplier would grade the build rather than the design.
"""
from __future__ import annotations

#: A declared multiplier of 0 means "knocked out" — the observed value should be
#: at or near zero, and a RELATIVE tolerance around 0 can never be satisfied by
#: anything except exactly 0. So a zero target is judged on an absolute floor.
ZERO_FLOOR = 1e-9


def observed_fold_change(arm_value, reference_value):
    """The arm's value as a factor of the reference arm's, or ``None`` when the
    comparison is not defined.

    Returns None rather than raising or defaulting: a target with no reference
    is UNCHECKABLE, and reporting it as landed (or as failed) would both be
    claims the data does not support.
    """
    if arm_value is None or reference_value is None:
        return None
    try:
        arm_value = float(arm_value)
        reference_value = float(reference_value)
    except (TypeError, ValueError):
        return None
    if reference_value == 0.0:
        # Nothing to be a fold-change OF. Not an error — a gene the reference
        # does not express is a real situation, and it simply cannot be graded
        # this way.
        return None
    return arm_value / reference_value


def target_landed(expected, observed_fc, tolerance: float) -> bool | None:
    """Did one declared target land? ``None`` when it cannot be judged.

    ``tolerance`` is RELATIVE (0.3 = within 30% of the declared multiplier),
    except against a declared zero, which is judged on ``ZERO_FLOOR``.
    """
    if observed_fc is None or expected is None:
        return None
    expected = float(expected)
    if expected == 0.0:
        return abs(observed_fc) <= ZERO_FLOOR
    return abs(observed_fc - expected) <= abs(expected) * float(tolerance)


def arm_targets(declared: dict, observed: dict, reference_observed: dict,
                tolerance: float) -> dict:
    """The ``targets`` block for one arm: ``{target: {expected, observed,
    within_tolerance}}``.

    ``declared`` maps target id -> the multiplier the design asked for.
    ``observed`` and ``reference_observed`` map target id -> the measured level
    for this arm and for its reference arm.

    ⚠ Every DECLARED target appears in the output, including ones with no
    observation. A target that silently vanished from the report is the failure
    mode this check exists to catch, so absence is recorded as an explicit
    unjudgeable entry rather than an omission.
    """
    out = {}
    for target in sorted(declared):
        fc = observed_fold_change(observed.get(target),
                                  reference_observed.get(target))
        out[target] = {
            "expected": float(declared[target]),
            "observed": None if fc is None else round(fc, 6),
            "within_tolerance": target_landed(declared[target], fc, tolerance),
        }
    return out


def landing_violations(targets_by_arm: dict) -> list:
    """``[(arm, target, entry)]`` for every target that did not land or could not
    be judged.

    ⛔ ``within_tolerance is None`` counts as a violation. An unjudgeable target
    is not a pass: the screen cannot claim the design landed, and treating "we
    could not tell" as "fine" is exactly how a check stops being able to fail.
    """
    return [(arm, target, entry)
            for arm in sorted(targets_by_arm)
            for target, entry in sorted(targets_by_arm[arm].items())
            if entry.get("within_tolerance") is not True]
