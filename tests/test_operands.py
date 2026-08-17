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


def _run_cache(sweep: Path, vector, *, observable="transcriptome",
               n_cells=20, gen_lb=0, run_commit=None) -> Path:
    """Write a synthetic run-vector cache so no parquet sweep is needed.

    ``load_or_extract`` returns a cached envelope verbatim when the schema
    matches, so the resolver can be exercised in public CI with no run data.
    """
    from v2ecoli.library import sim_vector_cache as svc

    path = svc.cache_path(str(sweep), gen_lb)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": svc.CACHE_SCHEMA,
        "run": {"experiment_id": "exp_live_1", "sweep_dir": str(sweep),
                "generation_lower_bound": gen_lb},
        "extractor": {"version": svc.EXTRACTOR_VERSION},
        "provenance": {"run_commit": run_commit,
                       "extracted_at_commit": "cafe1234"},
        "vectors": {"omics": {observable: {"vector": list(vector),
                                           "n_cells": n_cells}}},
    }), encoding="utf-8")
    return path


class RunOperand(unittest.TestCase):
    """The third path — a LIVE run inside the investigation.

    ``test_a_live_run_resolves_exactly_like_a_promoted_one`` is the load-bearing
    one: it is the direct test of `comparison-operands-plan` §10.1, which asks
    whether a live run collapses into the operand contract or needs its own
    resolution semantics. Until this path existed the question could not be
    asked, because every operand in play was read from a committed artifact.
    """

    def test_a_live_run_resolves_exactly_like_a_promoted_one(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            sweep = root / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [1.0, 2.0, 3.0], observable="proteome")
            bundle = _payload(root, ("g1", "measured", "TPM",
                                     [("A", 5.0, "detected"),
                                      ("B", 7.0, "detected")]))

            live = ops.run_operand(sweep, ["A", "B", "C"], observable="proteome")
            promoted = ops.promoted_operand("g1", bundle, "proteome", "TPM")

            # Same contract, same in-memory type, same accessor — the claim.
            # `at least OPERAND_COLUMNS`, exactly as the other two paths: a
            # promoted operand keeps the payload's full schema, a synthesised
            # one carries the guaranteed subset, and no caller may depend on
            # more than the subset.
            for op in (live, promoted):
                self.assertIsInstance(op, ops.Operand)
                for col in ops.OPERAND_COLUMNS:
                    self.assertIn(col, op.frame.columns, f"{op.path}: missing {col}")
                self.assertIsInstance(op.values, dict)
            # ...and they differ ONLY in how they were resolved.
            self.assertEqual({live.path, promoted.path},
                             {"in-investigation", "promoted"})

    def test_the_positional_vector_is_keyed_by_the_ids_it_is_given(self):
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [10.0, 20.0, 30.0])
            op = ops.run_operand(sweep, ["EG10001", "EG10002", "EG10003"])
            self.assertEqual(op.values,
                             {"EG10001": 10.0, "EG10002": 20.0, "EG10003": 30.0})

    def test_labels_from_the_wrong_sim_data_RAISE_rather_than_truncate(self):
        """The knockout trap, and the reason ``entity_ids`` is a parameter.

        A ParCa-level KO splices the genome, so the KO arm's cistron set is not
        the wild type's. Keying a KO sweep with WT labels would attribute every
        value past the deleted locus to the wrong gene — silently, and most
        damagingly on exactly the genes the knockout study is about.
        """
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [1.0, 2.0, 3.0])
            with self.assertRaises(ValueError) as caught:
                ops.run_operand(sweep, ["A", "B"])          # WT labels, KO sweep
            self.assertIn("width mismatch", str(caught.exception))

    def test_an_observable_the_sweep_did_not_record_is_None_not_a_raise(self):
        """Absent is a legitimate outcome; mis-keyed is not. Keep them distinct."""
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [1.0, 2.0], observable="transcriptome")
            self.assertIsNone(ops.run_operand(sweep, ["A", "B"],
                                              observable="proteome"))

    def test_collisions_are_summed_exactly_as_the_bake_path_sums_them(self):
        """Several mRNA cistrons legitimately map to one EcoCyc gene id.

        The live path and ``scripts/bake_model_omics.py`` must agree, or
        "we re-ran it live" quietly also means "we changed the aggregation".
        """
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [1.0, 2.0, 4.0])
            op = ops.run_operand(sweep, ["EG1", "EG1", "EG2"])
            self.assertEqual(op.values, {"EG1": 3.0, "EG2": 4.0})
            self.assertEqual(op.meta["n_collisions"], 1)

    def test_unmapped_ids_are_dropped_and_counted(self):
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [1.0, 2.0, 3.0])
            op = ops.run_operand(sweep, ["EG1", "", None])
            self.assertEqual(op.values, {"EG1": 1.0})
            self.assertEqual(op.meta["n_unmapped"], 2)

    def test_run_commit_stays_None_when_the_sweep_recorded_nothing(self):
        """Never substituted with the extracting tree's HEAD — that would look
        authoritative and mean something else (`comparison-operands-plan` §5)."""
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [1.0], run_commit=None)
            op = ops.run_operand(sweep, ["EG1"])
            self.assertIsNone(op.meta["run_commit"])
            self.assertEqual(op.meta["experiment_id"], "exp_live_1")

    def test_provenance_records_what_identifies_the_run(self):
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [1.0, 2.0], gen_lb=3, run_commit="abc123")
            op = ops.run_operand(sweep, ["EG1", "EG2"],
                                 generation_lower_bound=3)
            self.assertEqual(op.meta["run_commit"], "abc123")
            self.assertEqual(op.meta["gen_lb"], 3)
            self.assertEqual(op.meta["n_cells"], 20)
            self.assertEqual(op.kind, "model_predicted")

    def test_an_unknown_observable_names_the_ones_that_exist(self):
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [1.0])
            with self.assertRaises(KeyError) as caught:
                ops.run_operand(sweep, ["EG1"], observable="metabolome")
            self.assertIn("transcriptome", str(caught.exception))


class DeclaredZeros(unittest.TestCase):
    """A measured TRUE ZERO is recorded by the payload and, until now, reached
    no grader.

    It has no geometric mean (undefined for all-zero replicates), so it is
    stored as a NULL centre with the fact in ``mean_arithmetic``/``n_pos``.
    ``values`` drops nulls -- correctly -- and drops the recorded zero with
    them. `comparison-operands-plan` D5 is therefore honoured by the payload and
    invisible to the consumer, and the loss looks like a null rather than like a
    deletion, which is why nobody saw it.

    The invariant these tests protect: ``declared_zeros`` is **additive**.
    ``values`` is untouched, so ``n_shared`` cannot move under any card already
    rendered.
    """

    def _measured(self, rows):
        """rows: (entity_id, mean_arithmetic, mean_geometric, n_pos)."""
        frame = pd.DataFrame([{
            "cultivation_group_id": "g", "observable": "transcriptome",
            "entity_id": eid, "symbol": "", "units": "TPM", "kind": "measured",
            "detection": "detected", "mean_arithmetic": arith,
            "mean_geometric": geom, "sd_log10": None, "n": 6, "n_pos": npos,
            "n_total": 6,
        } for eid, arith, geom, npos in rows], columns=_VECTOR_COLS)
        return ops.Operand(frame=frame, path="promoted", kind="measured",
                           label="g transcriptome (TPM)")

    def test_a_measured_true_zero_is_visible_where_values_cannot_see_it(self):
        """The trpR case: 0.0 TPM on every replicate, in a dKO cultivation.

        The single most informative row in the comparison -- the knockout,
        visible in the data -- and the one row that reached no grader.
        """
        op = self._measured([("EG10001", 71.7, 71.4, 6),
                             ("EG11029", 0.0, None, 0)])      # trpR, knocked out
        self.assertEqual(set(op.values), {"EG10001"})
        self.assertEqual(op.declared_zeros, {"EG11029"})

    def test_values_and_declared_zeros_are_disjoint_by_construction(self):
        op = self._measured([("A", 5.0, 4.8, 6), ("B", 0.0, None, 0),
                             ("C", 1.0, 0.9, 3)])
        self.assertEqual(set(op.values) & op.declared_zeros, set())
        self.assertEqual(set(op.values), {"A", "C"})
        self.assertEqual(op.declared_zeros, {"B"})

    def test_a_null_centre_with_positive_replicates_is_NOT_a_declared_zero(self):
        """Absence of information is not a measurement of absence.

        A null centre with n_pos > 0 is malformed or censored -- either way the
        payload is not asserting a zero, so neither do we.
        """
        op = self._measured([("A", 5.0, None, 4)])
        self.assertEqual(op.declared_zeros, set())

    def test_adding_the_view_did_not_change_what_values_returns(self):
        """The regression guard. If this ever fails, `n_shared` has moved under
        every card already rendered -- including ones out for external review."""
        op = self._measured([("A", 5.0, 4.8, 6), ("B", 0.0, 0.0, 6),
                             ("C", -1.0, -1.0, 6), ("D", 0.0, None, 0)])
        # zeros and negatives KEPT, nulls dropped -- exactly as before.
        self.assertEqual(set(op.values), {"A", "B", "C"})

    def test_a_baked_fixture_declares_no_zeros(self):
        """A model has no limit of detection, so its zeros arrive as a real 0.0
        centre and `values` already keeps them. The asymmetry is real and
        belongs to the measured tier alone."""
        with TemporaryDirectory() as td:
            root = Path(td)
            _fixture(root, "f.json", "by_id", {"EG10001": 0.0, "EG10002": 7.0})
            op = ops.fixture_operand(root, "f.json", "by_id")
        self.assertEqual(op.declared_zeros, set())
        self.assertEqual(set(op.values), {"EG10001", "EG10002"})

    def test_a_live_run_declares_no_zeros(self):
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [0.0, 3.0])
            op = ops.run_operand(sweep, ["EG1", "EG2"])
        self.assertEqual(op.declared_zeros, set())
        self.assertEqual(set(op.values), {"EG1", "EG2"})


if __name__ == "__main__":
    unittest.main()
