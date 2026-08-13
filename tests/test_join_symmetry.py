"""The join treats its two operands as PEERS.

Layer 1 made both sides the same in-memory type. This is Layer 2: the grading
semantics. Each operand is filtered by ITS OWN rule, both contribute coverage
counts, and both are checked for a usable value — so a comparison whose sides are
both measurements (MS#8 #2) or both simulations (MS#8 #8) needs no new code path.

Two tests here are worth more than the rest:

* ``test_swapping_the_operands_mirrors_every_count`` — the property the whole
  layer exists to establish. If it fails, some behaviour has reacquired a
  preferred side.
* ``test_a_non_positive_value_on_side_b_is_excluded_and_counted`` — the defect
  this layer actually fixed. ``_loglog_r2`` filters non-positive pairs INTERNALLY
  and AFTER ppm renormalisation, so before this change an unusable B value was
  still counted in ``n_shared`` and then silently dropped from the fit, leaving
  the rendered shared-set size disagreeing with the R²'s own pair count.

Synthetic payloads only — no private data — so these run in public CI rather than
skipping behind the `private-data` extra.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# Layer 1's payload builder, reused rather than reinvented. `tests/` is not a
# package, so the directory goes on the path explicitly instead of relying on the
# runner's import mode.
sys.path.insert(0, str(Path(__file__).parent))

from test_operands import _payload            # noqa: E402
from v2ecoli.library import operands as ops   # noqa: E402

#: Every count the join is contracted to return. Pinned by name because these
#: keys land in the COMMITTED verdict JSON and are read by `_findings` and the
#: renderer by string key — a missing key does not raise, it renders as an absent
#: row. A silent rename is exactly the failure this set exists to make loud.
_COUNT_KEYS = {
    "n_shared", "n_measured", "n_detected",
    "n_a_outside_b_idspace", "n_b_outside_a_idspace",
    "n_declared_absent_by_a", "n_declared_absent_by_b",
    "n_nonpositive_a", "n_nonpositive_b",
    "n_provisional",
    "kind_a", "kind_b",
    "detection_informative_a", "detection_informative_b",
}


def _operands(root: Path, a_rows, b_rows, *, a_kind="measured",
              b_kind="model_predicted"):
    """Two promoted operands over synthetic payloads.

    Both sides resolve through the SAME call — which is the Layer 1 property
    these tests depend on, and the reason a sim<->sim case needs no fixture."""
    bundle = _payload(root,
                      ("a", a_kind, "unitsA", a_rows),
                      ("b", b_kind, "unitsB", b_rows))
    return (ops.promoted_operand("a", bundle, "proteome", "unitsA"),
            ops.promoted_operand("b", bundle, "proteome", "unitsB"))


class JoinContract(unittest.TestCase):

    def test_count_key_names_are_pinned(self):
        """★ Guards the rename. The renderer reads these by string key and fails
        SILENTLY on a miss, so the names are part of the contract."""
        with TemporaryDirectory() as td:
            a, b = _operands(Path(td),
                             [("E1", 5.0, "detected")],
                             [("E1", 7.0, "detected")])
            j = ops._join_vectors(a, b)
        self.assertTrue(_COUNT_KEYS <= set(j),
                        f"missing from the join: {sorted(_COUNT_KEYS - set(j))}")

    def test_the_detail_block_carries_both_directions_of_both_facts(self):
        """A reader must be able to see coverage and declared-absence for EACH
        side; one direction is not a symmetric report."""
        with TemporaryDirectory() as td:
            a, b = _operands(Path(td),
                             [("E1", 5.0, "detected"), ("only_a", 1.0, "detected")],
                             [("E1", 7.0, "detected"), ("only_b", 2.0, "detected")])
            j = ops._join_vectors(a, b)
        self.assertEqual(j["n_a_outside_b_idspace"], 1)
        self.assertEqual(j["n_b_outside_a_idspace"], 1)


class Symmetry(unittest.TestCase):

    def test_swapping_the_operands_mirrors_every_count(self):
        """★ The property the layer exists to establish.

        Join (a, b), then join (b, a): every per-side count must swap and the
        shared set must be identical. Nothing may prefer a side."""
        a_rows = [("shared1", 5.0, "detected"), ("shared2", 6.0, "detected"),
                  ("only_a", 1.0, "detected"), ("zero_a", 0.0, "detected"),
                  ("absent_a", 3.0, "below_limit")]
        b_rows = [("shared1", 7.0, "detected"), ("shared2", 8.0, "detected"),
                  ("only_b", 2.0, "detected"), ("absent_a", 4.0, "detected")]
        with TemporaryDirectory() as td:
            # BOTH measured, so `detection` is informative on both sides and the
            # mirror is exercised rather than short-circuited by a kind.
            a, b = _operands(Path(td), a_rows, b_rows,
                             a_kind="measured", b_kind="measured")
            fwd = ops._join_vectors(a, b)
            rev = ops._join_vectors(b, a)

        self.assertEqual(fwd["n_shared"], rev["n_shared"])
        self.assertEqual(set(fwd["ids"]), set(rev["ids"]))
        for x, y in (("n_a_outside_b_idspace", "n_b_outside_a_idspace"),
                     ("n_declared_absent_by_a", "n_declared_absent_by_b"),
                     ("n_nonpositive_a", "n_nonpositive_b")):
            self.assertEqual(fwd[x], rev[y], f"{x} did not mirror to {y}")
            self.assertEqual(fwd[y], rev[x], f"{y} did not mirror to {x}")

    def test_a_declared_absence_only_counts_where_the_other_side_covers_it(self):
        """An absence claim is evidence only where the other side has something
        to disagree with; outside its id-space it is just two panels differing."""
        with TemporaryDirectory() as td:
            a, b = _operands(Path(td),
                             [("shared", 5.0, "detected"),
                              ("covered", 1.0, "below_limit"),     # b covers it
                              ("uncovered", 1.0, "below_limit")],  # b does not
                             [("shared", 7.0, "detected"),
                              ("covered", 2.0, "detected")],
                             b_kind="measured")
            j = ops._join_vectors(a, b)
        self.assertEqual(j["n_declared_absent_by_a"], 1)


class KindDependentDetection(unittest.TestCase):
    """D5, mechanised: a measured zero may sit below a limit of detection; a
    simulated zero is a real zero."""

    def test_detection_is_informative_only_for_measured_kinds(self):
        with TemporaryDirectory() as td:
            a, b = _operands(Path(td),
                             [("E1", 5.0, "detected")],
                             [("E1", 7.0, "detected")])
        self.assertTrue(ops._detection_is_informative(a))
        self.assertFalse(ops._detection_is_informative(b),
                         "a simulation has no limit of detection, so its "
                         "`detection` column cannot carry information")

    def test_a_simulated_operand_declares_no_absences(self):
        """Even when the payload carries a non-`detected` status, a simulated
        operand makes no absence claim — filtering it would apply a test that
        cannot fail, which reads as a check and is not one."""
        with TemporaryDirectory() as td:
            a, b = _operands(Path(td),
                             [("E1", 5.0, "detected"), ("E2", 6.0, "detected")],
                             [("E1", 7.0, "detected"), ("E2", 8.0, "below_limit")])
            j = ops._join_vectors(a, b)
        self.assertEqual(j["n_declared_absent_by_b"], 0)
        self.assertFalse(j["detection_informative_b"])
        # ...and the row is USED rather than filtered out, because that status
        # carries no information on this side.
        self.assertEqual(j["n_shared"], 2)

    def test_the_structural_zero_is_reported_not_suppressed(self):
        """A reader who cannot see the count cannot tell "no disagreement" from
        "this comparison cannot express disagreement"."""
        with TemporaryDirectory() as td:
            a, b = _operands(Path(td),
                             [("E1", 5.0, "detected")],
                             [("E1", 7.0, "detected")])
            j = ops._join_vectors(a, b)
        self.assertIn("n_declared_absent_by_b", j)
        self.assertEqual(j["n_declared_absent_by_b"], 0)
        self.assertFalse(j["detection_informative_b"])


class NonPositiveAccounting(unittest.TestCase):

    def test_a_non_positive_value_on_side_b_is_excluded_and_counted(self):
        """★ The defect this layer fixed.

        B's zero cannot enter a log-log fit, and `_loglog_r2` already dropped it
        — but it used to be counted in `n_shared` first, so the reported
        shared-set size exceeded the number of points actually fitted."""
        with TemporaryDirectory() as td:
            a, b = _operands(Path(td),
                             [("ok", 5.0, "detected"), ("b_zero", 6.0, "detected")],
                             [("ok", 7.0, "detected"), ("b_zero", 0.0, "detected")])
            j = ops._join_vectors(a, b)
        self.assertEqual(j["n_shared"], 1)
        self.assertEqual(j["n_nonpositive_b"], 1)
        self.assertNotIn("b_zero", j["ids"])

    def test_shared_equals_the_number_of_points_actually_fitted(self):
        """★ The invariant that was silently violated before this change: the
        shared-set size the card RENDERS and the pair count the R² is computed
        over are the same number."""
        with TemporaryDirectory() as td:
            a, b = _operands(Path(td),
                             [("p1", 5.0, "detected"), ("p2", 6.0, "detected"),
                              ("p3", 7.0, "detected"), ("a_zero", 0.0, "detected")],
                             [("p1", 1.0, "detected"), ("p2", 2.0, "detected"),
                              ("p3", 3.0, "detected"), ("a_zero", 9.0, "detected")])
            j = ops._join_vectors(a, b)
            _, n = ops._loglog_r2(j["sim"], j["exp"])
        self.assertEqual(j["n_shared"], n)

    def test_the_id_space_is_the_panel_not_the_usable_rows(self):
        """An entity the other side covers but cannot quantify is a different
        fact from one it never covered, and the two must not collapse."""
        with TemporaryDirectory() as td:
            a, b = _operands(Path(td),
                             [("known", 5.0, "detected"), ("unknown", 6.0, "detected")],
                             [("known", 0.0, "detected")])
            j = ops._join_vectors(a, b)
        # 'known' is covered by b but unusable; 'unknown' is not covered at all.
        self.assertEqual(j["n_nonpositive_b"], 1)
        self.assertEqual(j["n_a_outside_b_idspace"], 1)


if __name__ == "__main__":
    unittest.main()
