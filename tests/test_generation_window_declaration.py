"""A study must be able to DECLARE a generation window, on both spec routes.

`aggregation.py` can window, but until this landed nothing could ask it to:
`generation_lower_bound` existed nowhere in `scripts/_compare/`, so the
mechanism was unreachable — the same shape as `exchange_flux_basis` being
declarable but inert.

⛔ THE TWO-ROUTE RULE THIS FILE EXISTS TO PIN. `study_spec` parses specs by two
paths — `comparison.configs[]` entries and a study.yaml — and they ALREADY
disagree on one key's spelling (`gens` vs `generations`), with BOTH silently
defaulting. A third key with its own split would be that defect again, so the
tests below assert both routes read the SAME key from the SAME place.
"""
from __future__ import annotations

import textwrap

import pytest

from scripts._compare.study_spec import (
    StudySpec,
    generation_lower_bound_from_study_yaml,
)


def _study(tmp_path, body: str):
    p = tmp_path / "study.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _spec(**kw):
    base = dict(name="s", condition="basal", seeds=4, gens=8, cards=[],
                invest_name="i", v2_cache="a", ve_cache="b", study_path="p")
    base.update(kw)
    return StudySpec(**base)


# --------------------------------------------------------------------------- #
# the reader
# --------------------------------------------------------------------------- #
def test_reads_the_bound_from_the_comparison_block(tmp_path):
    p = _study(tmp_path, """
        comparison:
          generation_lower_bound: 5
    """)
    assert generation_lower_bound_from_study_yaml(p) == 5


def test_a_top_level_key_is_ignored_like_every_other_measurement_key(tmp_path):
    """⭐ DISCRIMINATING. `comparison:`-only is the rule that stopped two readers
    of the basis disagreeing; a reader that also honoured top-level would
    reintroduce it for this key."""
    p = _study(tmp_path, """
        generation_lower_bound: 5
        comparison:
          seeds: 4
    """)
    assert generation_lower_bound_from_study_yaml(p, fallback=0) == 0


def test_silent_study_keeps_the_investigation_fallback(tmp_path):
    p = _study(tmp_path, """
        comparison:
          seeds: 4
    """)
    assert generation_lower_bound_from_study_yaml(p, fallback=3) == 3


def test_declared_bound_beats_the_fallback(tmp_path):
    p = _study(tmp_path, """
        comparison:
          generation_lower_bound: 5
    """)
    assert generation_lower_bound_from_study_yaml(p, fallback=3) == 5


def test_zero_is_a_real_declaration_not_a_missing_one(tmp_path):
    """0 is falsy — a `or fallback` implementation would silently substitute the
    investigation default and grade a window the study explicitly opted out of."""
    p = _study(tmp_path, """
        comparison:
          generation_lower_bound: 0
    """)
    assert generation_lower_bound_from_study_yaml(p, fallback=5) == 0


def test_unreadable_or_malformed_study_keeps_the_fallback(tmp_path):
    assert generation_lower_bound_from_study_yaml(tmp_path / "nope.yaml", 2) == 2
    p = _study(tmp_path, "comparison:\n  generation_lower_bound: not-a-number\n")
    assert generation_lower_bound_from_study_yaml(p, fallback=2) == 2


# --------------------------------------------------------------------------- #
# the guard — a window that grades nothing RELAXES the gate
# --------------------------------------------------------------------------- #
def test_bound_at_or_above_generations_is_refused_at_declaration_time():
    """⭐ THE GUARD THAT MATTERS.

    A bound excluding every generation yields no gradable cell -> the axis goes
    `ungraded` -> the severity model scores that 0, i.e. no worse than a pass.
    So the study would silently RELAX the gate it exists to enforce. Refuse
    where the author can see the number.
    """
    with pytest.raises(ValueError, match="excludes every generation"):
        _spec(gens=8, generation_lower_bound=8)
    with pytest.raises(ValueError, match="excludes every generation"):
        _spec(gens=8, generation_lower_bound=99)


def test_a_bound_that_admits_the_last_generation_only_is_allowed():
    """Narrow is legitimate; empty is not. The boundary must land between them."""
    assert _spec(gens=8, generation_lower_bound=7).generation_lower_bound == 7


def test_negative_bound_is_refused():
    with pytest.raises(ValueError, match="must be >= 0"):
        _spec(generation_lower_bound=-1)


def test_default_is_no_window_and_stays_valid():
    assert _spec().generation_lower_bound == 0


def test_an_explicit_zero_entry_is_not_overridden_by_an_investigation_default():
    """⭐ DISCRIMINATING, and a defect I shipped once already three lines away.

    `entry.get(k) or defaults.get(k) or 0` discards an explicitly-declared 0 —
    "grade every generation, deliberately" — in favour of the investigation
    default, because 0 is falsy. The reader guards this; the precedence chain
    feeding it must too, or the reader's care is undone one call up.
    """
    from scripts._compare.study_spec import _first_declared
    assert _first_declared(0, 5) == 0            # explicit opt-out survives
    assert _first_declared(None, 5) == 5         # silent entry inherits
    assert _first_declared(None, None) == 0      # nothing declared -> no window
    assert _first_declared(None, 0) == 0
