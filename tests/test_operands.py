"""Both operands of a comparison resolve to one shape.

The card used to reach its two sides through two mechanisms that produced two
in-memory types — a pandas frame for the measurement, a bare ``{id: value}`` dict
for the model — so the join could only be written one-way-round. That is
invisible while the model is always one side and the reference always the other.
It stops being invisible when a comparison is exp<->exp or sim<->sim, which is
two of the six the CD1 evaluation asks for.

The load-bearing test here is
``test_a_simulated_group_resolves_exactly_like_a_measured_one``: a promoted
SIMULATION goes through the same call, returns the same shape, and differs only
in ``kind``. If that ever stops being true, the operand model has quietly
reacquired the kind-split it exists to remove.

Synthetic payloads only — no private data — so these run in public CI rather than
skipping behind the `private-data` extra.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from v2ecoli.library import operands as ops

_VECTOR_COLS = ["cultivation_group_id", "observable", "entity_id", "symbol",
                "units", "kind", "detection", "mean_arithmetic",
                "mean_geometric", "sd_log10", "n", "n_pos", "n_total"]
_BUNDLE_COLS = ["canonical_key", "cultivation_group_id", "observable",
                "source_path", "schema_name", "description", "units",
                "phase", "window"]


def _payload(root: Path, *groups) -> Path:
    """A minimal validation payload: one vector table per (group, kind, units)."""
    (root / "vectors").mkdir(parents=True, exist_ok=True)
    bundle = []
    for group, kind, units, rows in groups:
        rel = f"vectors/proteome__{group}.tsv"
        pd.DataFrame([{
            "cultivation_group_id": group, "observable": "proteome",
            "entity_id": eid, "symbol": "", "units": units, "kind": kind,
            "detection": det, "mean_arithmetic": val, "mean_geometric": val,
            "sd_log10": 0.1, "n": 3, "n_pos": 3, "n_total": 3,
        } for eid, val, det in rows], columns=_VECTOR_COLS).to_csv(
            root / rel, sep="\t", index=False)
        bundle.append({
            "canonical_key": f"{group}__proteome__{units}__exponential_batch",
            "cultivation_group_id": group, "observable": "proteome",
            "source_path": rel, "schema_name": "VectorObservationSchema",
            "description": f"{group} proteome", "units": units,
            "phase": "exponential_batch", "window": "3-4h",
        })
    p = root / "validation_bundle.tsv"
    pd.DataFrame(bundle, columns=_BUNDLE_COLS).to_csv(p, sep="\t", index=False)
    return p


def _fixture(root: Path, name: str, map_key: str, values: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(
        {"id_key": "EcoCyc monomer id", "n_cells": 20, "gen_lb": 3,
         map_key: values}), encoding="utf-8")
    return root / name


class OperandShape(unittest.TestCase):

    def test_both_resolvers_return_the_same_shape(self):
        """A caller cannot tell the two paths apart except by reading `path`.
        That is the property the whole re-cut rests on."""
        with TemporaryDirectory() as td:
            root = Path(td)
            bundle = _payload(root, ("g1", "measured", "TPM",
                                     [("EG10001", 5.0, "detected")]))
            a = ops.promoted_operand("g1", bundle, "proteome", "TPM")
            _fixture(root, "f.json", "by_id", {"EG10001": 7.0})
            b = ops.fixture_operand(root, "f.json", "by_id")
        for op in (a, b):
            self.assertIsInstance(op, ops.Operand)
            for col in ops.OPERAND_COLUMNS:
                self.assertIn(col, op.frame.columns, f"{op.path}: missing {col}")
        self.assertEqual({a.path, b.path}, {"promoted", "fixture"})

    def test_promoted_operand_reads_kind_from_the_data(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            bundle = _payload(root, ("sim1", "model_predicted", "counts_per_cell",
                                     [("EG10001", 5.0, "detected")]))
            op = ops.promoted_operand("sim1", bundle, "proteome", "counts_per_cell")
        self.assertEqual(op.kind, "model_predicted")

    def test_a_simulated_group_resolves_exactly_like_a_measured_one(self):
        """★ The one that matters. A promoted SIMULATION and a promoted
        MEASUREMENT go through the same call and come back the same shape;
        `kind` is the only difference. Comparison #8 (sim<->sim) and #2
        (exp<->exp) are both this, which is why neither needs a code path."""
        with TemporaryDirectory() as td:
            root = Path(td)
            rows = [("EG10001", 5.0, "detected"), ("EG10002", 9.0, "detected")]
            bundle = _payload(root,
                              ("exp1", "measured", "iBAQ", rows),
                              ("sim1", "model_predicted", "counts_per_cell", rows))
            a = ops.promoted_operand("exp1", bundle, "proteome", "iBAQ")
            b = ops.promoted_operand("sim1", bundle, "proteome", "counts_per_cell")
        self.assertEqual(a.path, b.path, "the PATH must not vary with kind")
        self.assertEqual(list(a.frame.columns), list(b.frame.columns))
        self.assertEqual(a.values, b.values)          # same numbers, different provenance
        self.assertNotEqual(a.kind, b.kind)           # kind still distinguishes them
        self.assertEqual({a.kind, b.kind}, {"measured", "model_predicted"})

    def test_missing_group_resolves_to_none_rather_than_raising(self):
        """A card degrades, it does not fail."""
        with TemporaryDirectory() as td:
            root = Path(td)
            bundle = _payload(root, ("g1", "measured", "TPM",
                                     [("EG10001", 5.0, "detected")]))
            self.assertIsNone(ops.promoted_operand("absent", bundle, "proteome", "TPM"))
            self.assertIsNone(ops.promoted_operand("g1", bundle, "proteome", "WRONG_UNITS"))


class OperandValues(unittest.TestCase):
    """`values` is the id-space view a join consumes. What it does and does NOT
    filter is load-bearing: filtering here would silently change what
    `n_shared` means on the other side of the join."""

    def _op(self, rows):
        with TemporaryDirectory() as td:
            root = Path(td)
            bundle = _payload(root, ("g", "measured", "TPM", rows))
            return ops.promoted_operand("g", bundle, "proteome", "TPM")

    def test_zero_and_negative_centres_are_KEPT(self):
        """Whether a non-positive value excludes an entity depends on which side
        of the join it is on, so that decision belongs to the join."""
        op = self._op([("A", 5.0, "detected"), ("B", 0.0, "detected")])
        self.assertEqual(set(op.values), {"A", "B"})

    def test_null_centres_are_dropped(self):
        """A null centre is not a number — it means no replicate was positive."""
        op = self._op([("A", 5.0, "detected"), ("B", None, "detected")])
        self.assertEqual(set(op.values), {"A"})

    def test_values_does_not_filter_on_detection(self):
        """Detection filtering is the join's job and differs by operand kind;
        doing it here would bake a measurement assumption into the resolver."""
        op = self._op([("A", 5.0, "detected"), ("B", 2.0, "below_limit")])
        self.assertEqual(set(op.values), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
