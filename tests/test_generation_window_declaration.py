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
    _context,
    _spec_from_study,
    generation_lower_bound_from_study_yaml,
    specs_from_configs,
)


def _study(tmp_path, body: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def test_an_unreadable_study_keeps_the_fallback_but_a_MALFORMED_bound_refuses(tmp_path):
    """⭐ The asymmetry is the point, and the old test had it wrong.

    A study we cannot OPEN declared nothing -> fallback is right. A study that
    DECLARED `generation_lower_bound: post-burn-in` did declare, and silently
    grading every generation is the same gate-relaxing failure the validator
    refuses at the other end of the range. The previous version asserted the
    silent fallback and so PINNED the bug as intended behaviour.
    """
    assert generation_lower_bound_from_study_yaml(tmp_path / "nope.yaml", 2) == 2

    for bad in ("post-burn-in", "true", "[3]", "{a: 1}"):
        p = _study(tmp_path / bad.strip("[]{} "),
                   f"comparison:\n  generation_lower_bound: {bad}\n")
        with pytest.raises(ValueError, match="integer generation index"):
            generation_lower_bound_from_study_yaml(p, fallback=2)


def test_a_quoted_integer_is_accepted(tmp_path):
    p = _study(tmp_path, "comparison:\n  generation_lower_bound: '5'\n")
    assert generation_lower_bound_from_study_yaml(p) == 5


# --------------------------------------------------------------------------- #
# END-TO-END THROUGH BOTH RESOLVERS
#
# ⛔ WHY THESE EXIST. An earlier version of this file tested `_first_declared`
# and the reader in isolation and asserted in its own docstring that "both
# routes read the SAME key from the SAME place" — while exercising NEITHER
# route. A review mutation-tested it: deleting the `generation_lower_bound=`
# kwarg from either resolver, restoring the `or`-chain the unit test was written
# against, or INVERTING the precedence all left the suite green. A unit test
# proving a helper is correct proves nothing about the helper being USED.
# --------------------------------------------------------------------------- #
def _workspace(tmp_path, *, inv: str, studies: dict):
    inv_dir = tmp_path / "investigations" / "inv"
    inv_dir.mkdir(parents=True)
    (inv_dir / "investigation.yaml").write_text(textwrap.dedent(inv), "utf-8")
    for nm, body in studies.items():
        d = tmp_path / "studies" / nm
        d.mkdir(parents=True)
        (d / "study.yaml").write_text(textwrap.dedent(body), "utf-8")
    return inv_dir


_INV = """
    name: inv
    members: [s1]
    comparison:
      candidate: v2ecoli
      reference: {repo: /x, kind: vecoli}
      v2_cache: a
      ve_cache: b
      defaults: {seeds: 1, gens: 8%(defaults)s}
      configs:
      - {name: s1, condition: basal%(entry)s}
"""


def _configs_route(tmp_path, *, defaults="", entry="", study=""):
    inv_dir = _workspace(
        tmp_path,
        inv=_INV % {"defaults": defaults, "entry": entry},
        studies={"s1": "condition: basal\ncomparison:\n  seeds: 1\n"
                       "  generations: 8\n" + study})
    return specs_from_configs(_context(inv_dir))[0]


def _study_route(tmp_path, *, defaults="", study=""):
    inv_dir = _workspace(
        tmp_path,
        inv=_INV % {"defaults": defaults, "entry": ""},
        studies={"s1": "condition: basal\ncomparison:\n  seeds: 1\n"
                       "  generations: 8\n" + study})
    return _spec_from_study(tmp_path / "studies" / "s1" / "study.yaml",
                            _context(inv_dir))


def test_configs_route_threads_a_study_declared_bound(tmp_path):
    """⭐ Kills 'delete the kwarg from specs_from_configs'."""
    assert _configs_route(tmp_path, study="  generation_lower_bound: 5\n"
                          ).generation_lower_bound == 5


def test_study_route_threads_a_study_declared_bound(tmp_path):
    """⭐ Kills 'delete the kwarg from _spec_from_study'."""
    assert _study_route(tmp_path, study="  generation_lower_bound: 5\n"
                        ).generation_lower_bound == 5


def test_both_routes_inherit_an_investigation_default(tmp_path):
    """⭐ Kills 'fallback -> 0', i.e. defaults silently ignored."""
    d = ", generation_lower_bound: 2"
    assert _configs_route(tmp_path / "a", defaults=d).generation_lower_bound == 2
    assert _study_route(tmp_path / "b", defaults=d).generation_lower_bound == 2


def test_study_declaration_beats_entry_and_defaults(tmp_path):
    """⭐ Kills INVERTED precedence, which the old suite could not see."""
    spec = _configs_route(tmp_path,
                          defaults=", generation_lower_bound: 2",
                          entry=", generation_lower_bound: 3",
                          study="  generation_lower_bound: 5\n")
    assert spec.generation_lower_bound == 5


def test_an_explicit_zero_in_a_study_is_not_overridden_by_a_default(tmp_path):
    """⭐ Kills the `or`-chain AT THE CALL SITE, not just in the helper."""
    for route in (_configs_route, _study_route):
        spec = route(tmp_path / route.__name__,
                     defaults=", generation_lower_bound: 5",
                     study="  generation_lower_bound: 0\n")
        assert spec.generation_lower_bound == 0


def test_a_negative_bound_in_a_study_yaml_reaches_the_guard(tmp_path):
    """⭐ The old test built a StudySpec directly, so nothing proved a negative
    declared in YAML ever reached validation — a reader sanitising with abs()
    survived."""
    with pytest.raises(ValueError, match="must be >= 0"):
        _study_route(tmp_path, study="  generation_lower_bound: -1\n")


# --------------------------------------------------------------------------- #
# the guard — a window that grades nothing RELAXES the gate
# --------------------------------------------------------------------------- #
def test_bound_at_or_above_generations_is_refused_at_declaration_time(tmp_path):
    """⭐ THE GUARD THAT MATTERS.

    A bound excluding every generation yields no gradable cell -> the axis goes
    `ungraded` -> the severity model scores that 0, i.e. no worse than a pass.
    So the study would silently RELAX the gate it exists to enforce. Refuse
    where the author can see the number.
    """
    with pytest.raises(ValueError, match="excludes every generation"):
        _study_route(tmp_path, study="  generation_lower_bound: 8\n")
    with pytest.raises(ValueError, match="excludes every generation"):
        _study_route(tmp_path / "b", study="  generation_lower_bound: 99\n")


def test_a_bound_that_admits_the_last_generation_only_is_allowed(tmp_path):
    """Narrow is legitimate; empty is not. The boundary must land between them."""
    assert _study_route(tmp_path, study="  generation_lower_bound: 7\n"
                        ).generation_lower_bound == 7


def test_default_is_no_window(tmp_path):
    assert _study_route(tmp_path).generation_lower_bound == 0


def test_one_narrow_member_does_not_kill_its_whole_investigation(tmp_path):
    """⭐ THE REGRESSION THE GUARD'S OLD PLACEMENT CAUSED.

    With the check in `StudySpec.__post_init__`, an investigation declaring a
    default of 5 with ONE member legitimately running `generations: 1` failed
    the ENTIRE investigation load — every unrelated member included — because
    `load_investigation` builds every spec. The refusal must be scoped to the
    study whose numbers actually conflict.
    """
    inv_dir = _workspace(
        tmp_path,
        inv=_INV % {"defaults": ", generation_lower_bound: 2", "entry": ""},
        studies={"s1": "condition: basal\ncomparison:\n  seeds: 1\n"
                       "  generations: 8\n"})
    # The healthy member resolves; it is not collateral damage.
    assert specs_from_configs(_context(inv_dir))[0].generation_lower_bound == 2


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


# --------------------------------------------------------------------------- #
# THE VARIANT DECLARATION — same two routes, same precedence trap, and a
# consequence that is worse when it is dropped.
#
# ⛔ A dropped `generation_lower_bound` widens a window. A dropped `variant`
# swaps the MODEL: for a config whose variant carries the strain plus its
# induction schedule, the unvaried reference arm has neither and emits a
# complete, healthy-looking result for a different organism. Nothing errors.
# --------------------------------------------------------------------------- #
def test_both_routes_thread_a_study_declared_variant(tmp_path):
    """⭐ Kills 'delete the kwarg' from EITHER resolver — the failure a unit test
    of the reader alone cannot see."""
    from scripts._compare.study_spec import variant_from_study_yaml
    assert _configs_route(tmp_path / "a", study="  variant: 1\n").variant == 1
    assert _study_route(tmp_path / "b", study="  variant: 1\n").variant == 1
    assert variant_from_study_yaml(
        tmp_path / "a" / "studies" / "s1" / "study.yaml") == 1


def test_an_UNDECLARED_variant_is_None_not_zero(tmp_path):
    """⛔ None and 0 are DIFFERENT ANSWERS and must not collapse.

    `None` means "the study did not say", which the runner is entitled to refuse.
    `0` means "baseline, deliberately". Defaulting an absent key to 0 would turn
    every silent study into an explicit request for the unvaried model — the
    exact substitution this key exists to make impossible.
    """
    assert _study_route(tmp_path / "a").variant is None
    assert _configs_route(tmp_path / "b").variant is None


def test_a_declared_ZERO_survives_an_investigation_default(tmp_path):
    """⛔ THE FALSY TRAP, and it is why `_first_declared` is used here.

    `variant: 0` is falsy, so an `or` chain hands back the investigation-level
    default instead — silently upgrading a study's deliberate baseline into
    whatever the investigation happened to declare.
    """
    d = ", variant: 2"
    assert _study_route(tmp_path / "a", defaults=d, study="  variant: 0\n").variant == 0
    assert _configs_route(tmp_path / "b", defaults=d, study="  variant: 0\n").variant == 0
    # ...and with the study silent, the investigation default still applies.
    assert _study_route(tmp_path / "c", defaults=d).variant == 2

    # ⛔⛔ AND THE FALSY TRAP AT THE *FALLBACK* LEVEL, which the assertions above
    # cannot reach. When the study.yaml DECLARES a value, `variant_from_study_yaml`
    # returns it directly and the `fallback=` expression is never evaluated — so
    # every test that puts the `0` in the study.yaml leaves `_first_declared`
    # itself unexercised, and replacing it with an `or` chain passed the whole
    # suite. The cell that matters is: study SILENT, `0` declared one level up.
    # Failure it admits: an investigation declares `defaults: {variant: 2}`, one
    # entry says `variant: 0` as its deliberate baseline arm, that study's YAML is
    # silent — and the baseline arm silently runs variant 2.
    assert _configs_route(tmp_path / "d", defaults=", variant: 2",
                          entry=", variant: 0").variant == 0, (
        "an entry-level `variant: 0` was discarded in favour of the "
        "investigation default")
    assert _study_route(tmp_path / "e", defaults=", variant: 0").variant == 0, (
        "an investigation-level `variant: 0` was discarded as falsy")


def test_the_study_gets_the_LAST_word_over_an_investigation_default(tmp_path):
    assert _configs_route(tmp_path / "a", defaults=", variant: 2",
                          entry=", variant: 3",
                          study="  variant: 1\n").variant == 1


def test_a_nonsense_variant_is_refused_rather_than_coerced(tmp_path):
    from scripts._compare.study_spec import variant_from_study_yaml
    p = tmp_path / "s.yaml"
    p.write_text("condition: basal\ncomparison:\n  variant: first\n", "utf-8")
    with pytest.raises(ValueError, match="variant"):
        variant_from_study_yaml(p)
    p.write_text("condition: basal\ncomparison:\n  variant: -1\n", "utf-8")
    with pytest.raises(ValueError, match=">= 0"):
        variant_from_study_yaml(p)
