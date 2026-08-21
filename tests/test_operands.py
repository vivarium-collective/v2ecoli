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

import dataclasses
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
               n_cells=20, gen_lb=0, run_commit=None, per_cell=None,
               group="omics", node_name=None) -> Path:
    """Write a synthetic run-vector cache so no parquet sweep is needed.

    ``load_or_extract`` returns a cached envelope verbatim when the schema
    matches, so the resolver can be exercised in public CI with no run data.

    ``per_cell`` writes the n_cells x n_features matrix the real extractor emits;
    ``group``/``node_name`` place the node somewhere other than
    ``omics.<observable>`` (the exchange observable lives under ``fluxes``).
    Deliberately does NOT write ``units``: the resolver must take units from its
    own declaration, so a fixture that supplied them would hide a regression.
    """
    from v2ecoli.library import sim_vector_cache as svc

    path = svc.cache_path(str(sweep), gen_lb)
    path.parent.mkdir(parents=True, exist_ok=True)
    node = {"vector": list(vector), "n_cells": n_cells}
    if per_cell is not None:
        node["per_cell"] = [list(r) for r in per_cell]
    path.write_text(json.dumps({
        "schema": svc.CACHE_SCHEMA,
        "run": {"experiment_id": "exp_live_1", "sweep_dir": str(sweep),
                "generation_lower_bound": gen_lb},
        "extractor": {"version": svc.EXTRACTOR_VERSION},
        "provenance": {"run_commit": run_commit,
                       "extracted_at_commit": "cafe1234"},
        "vectors": {group: {node_name or observable: node}},
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

    @staticmethod
    def _bake_keyed():
        """The REAL ``scripts/bake_model_omics.py::_keyed``, imported by path.

        Imported rather than restated: a copy asserts what we believe bake does,
        which is precisely the thing that can silently stop being true.
        """
        import importlib.util
        p = Path(__file__).resolve().parents[1] / "scripts" / "bake_model_omics.py"
        spec = importlib.util.spec_from_file_location("_bake_for_parity", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._keyed

    def test_collisions_are_summed_exactly_as_the_bake_path_sums_them(self):
        """Several mRNA cistrons legitimately map to one EcoCyc gene id.

        The live path and ``scripts/bake_model_omics.py`` must agree, or
        "we re-ran it live" quietly also means "we changed the aggregation".

        ★ This asserts against the IMPORTED bake implementation, not an inline
        expectation. An inline dict cannot fail when bake changes, which is the
        entire risk the parity claim is about.
        """
        vector, ids = [1.0, 2.0, 4.0], ["EG1", "EG1", "EG2"]
        bake_out, bake_unmapped, bake_collisions = self._bake_keyed()(vector, ids)

        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, vector)
            op = ops.run_operand(sweep, ids)

            self.assertEqual(op.values, bake_out)
            self.assertEqual(op.meta["n_collisions"], bake_collisions)
            self.assertEqual(op.meta["n_unmapped"], bake_unmapped)
            # pinned so a silent change on EITHER side is visible in the diff
            self.assertEqual(bake_out, {"EG1": 3.0, "EG2": 4.0})

    def test_the_one_way_the_two_keyed_implementations_deliberately_differ(self):
        """★ They are NOT byte-identical, and the difference is a hardening.

        ``operands._keyed`` normalises with ``k = "" if k is None else str(k)``;
        bake's copy does not. Consequences, which are the whole reason this is
        pinned rather than described:

        * ``None``   — identical (both fall to the unmapped branch);
        * ``"EG1"``  — identical (the real domain: entity ids are strings);
        * ``5``      — DIVERGENT. operands keys ``"5"``, bake keys ``5``.

        So "deliberately identical" is true on the domain either is used with,
        and false in general. Asserting equality unconditionally would fail here
        — which is how this divergence was found rather than assumed.
        """
        bake_keyed = self._bake_keyed()
        vector = [1.0, 2.0]

        for ids in (["EG1", "EG2"], [None, "EG2"], ["None", "EG2"]):
            with self.subTest(ids=ids):
                self.assertEqual(ops._keyed(vector, ids), bake_keyed(vector, ids))

        ours, _, _ = ops._keyed(vector, [5, "EG2"])
        theirs, _, _ = bake_keyed(vector, [5, "EG2"])
        self.assertEqual(set(ours), {"5", "EG2"})
        self.assertEqual(set(theirs), {5, "EG2"})
        self.assertNotEqual(ours, theirs)

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

    def test_declared_zeros_survives_a_consumer_substituting_the_centre(self):
        """★ The regression this exists for: keying on a NULL centre is wrong.

        Which statistic sits in `mean_geometric` is a property of the
        PRESENTATION, not of the record. `vs_experiment` grades the ARITHMETIC
        centre (matching the prior CD1 notebooks) and substitutes it into that
        column before grading -- after which no row is null, and a null-keyed
        implementation returns EMPTY on exactly the operand that needs it.

        Measured against the real payload when this was caught: on
        `cd1_ginkgo_viom5_dko_m9` the raw frame yields 145 declared zeros
        including trpR (EG11029); the substituted frame yielded 0.
        """
        op = self._measured([("EG10001", 71.7, 71.4, 6),
                             ("EG11029", 0.0, None, 0)])      # trpR, knocked out
        self.assertEqual(op.declared_zeros, {"EG11029"})

        # exactly what `vs_experiment._with_graded_centre` does
        frame = op.frame.copy()
        frame["mean_geometric"] = frame["mean_arithmetic"]
        substituted = dataclasses.replace(op, frame=frame)

        self.assertFalse(substituted.frame["mean_geometric"].isna().any(),
                         "precondition: the substitution leaves no nulls")
        self.assertEqual(
            substituted.declared_zeros, {"EG11029"},
            "declared_zeros must key on the COUNTS, which are invariant to a "
            "centre substitution, not on a null centre which is not")

    def test_a_promoted_operand_without_counts_RAISES_rather_than_reporting_none(self):
        """★ "I have no counts" is not "I have no zeros".

        `VectorObservationSchema` requires `n` and `n_pos`, so a promoted frame
        lacking them is a malformed payload. Returning an empty set would be
        indistinguishable from a payload that genuinely has no zeros — the exact
        silent inertness this accessor was fixed for.
        """
        op = self._measured([("A", 5.0, 4.8, 6)])
        stripped = dataclasses.replace(op, frame=op.frame.drop(columns=["n_pos"]))
        with self.assertRaises(KeyError) as ctx:
            stripped.declared_zeros
        self.assertIn("n_pos", str(ctx.exception))

    def test_a_synthesised_operand_without_counts_is_legitimately_empty(self):
        """The other side of the same rule — and why it is not symmetric.

        A fixture/run frame carries no replicate counts by construction, so
        empty is the honest answer rather than a swallowed error. This is a PATH
        distinction, not measured-vs-simulated: a promoted SIMULATION has counts
        and does report declared zeros.
        """
        op = self._measured([("A", 5.0, 4.8, 6)])
        for path in ("fixture", "in-investigation"):
            with self.subTest(path=path):
                synth = dataclasses.replace(
                    op, path=path, frame=op.frame.drop(columns=["n", "n_pos"]))
                self.assertEqual(synth.declared_zeros, set())

    def test_below_limit_is_not_a_declared_zero(self):
        """`below_limit` is a statement about the LIMIT, not a count of zero.

        Latent against today's payload (no row combines it with `n_pos == 0` and
        `n > 0`), but the payload carries 5,213 `below_limit` rows, so the rule
        is stated rather than left to the emitter.
        """
        op = self._measured([("A", 0.0, None, 0)])
        op.frame.loc[0, "detection"] = "below_limit"
        self.assertEqual(op.declared_zeros, set())

    def test_a_row_nobody_measured_is_not_a_declared_zero(self):
        """`n_pos == 0` alone is not the fact -- `n > 0` is what makes it one.

        A row with no measurements has no positives either; that is the absence
        of a measurement, not a measurement of absence.
        """
        op = self._measured([("A", None, None, 0)])
        op.frame.loc[0, "n"] = 0
        self.assertEqual(op.declared_zeros, set())

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


_REPO = Path(__file__).resolve().parents[1]
_BASAL_FIXTURES = _REPO / "tests" / "fixtures" / "population_phenotype_basal"
_BASAL_REFERENCE = _REPO / "tests" / "fixtures" / "population_phenotype_basal_reference.json"


def _column_means(per_cell: dict) -> dict:
    """``{entity: mean over cells}`` — the reduction that must reproduce
    ``Operand.values`` if the samples and the centre are the same measurement."""
    return {e: sum(vals) / len(vals) for e, vals in per_cell.items()}


class RunOperandPerCell(unittest.TestCase):
    """A live operand can carry the per-cell samples behind its mean.

    ★ Why this exists: a statistic that compares DISTRIBUTIONS has nothing to
    consume from a mean vector. Until the extractor emitted ``per_cell`` for the
    omics groups (it always did for fluxes), no such comparison could be written
    at all.
    """

    # 3 cells x 3 features. Column means are 2.0 / 20.0 / 200.0 exactly, and the
    # per-cell spread is real — both properties are load-bearing below.
    PER_CELL = [[1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0]]
    MEAN = [2.0, 20.0, 200.0]
    IDS = ["EG1", "EG2", "EG3"]

    def _resolve(self, d, per_cell, **kw):
        sweep = Path(d) / "sweep"
        sweep.mkdir()
        _run_cache(sweep, self.MEAN, per_cell=per_cell, n_cells=len(per_cell))
        return ops.run_operand(sweep, self.IDS, with_per_cell=True, **kw)

    def test_per_cell_is_absent_unless_asked_for(self):
        """WOULD CATCH: making it unconditional. ~900k floats at production
        ensemble sizes, and every caller that wants only the mean would pay for
        it. The flag is safe HERE precisely because no cache is involved — it is
        a pure function of the arguments, so it cannot serve a stale answer."""
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, self.MEAN, per_cell=self.PER_CELL, n_cells=3)
            op = ops.run_operand(sweep, self.IDS)          # default: opted out
        self.assertIsNone(op.meta["per_cell"])
        self.assertEqual(op.meta["n_per_cell"], 0)

    def test_per_cell_has_n_cells_rows_of_the_vectors_width(self):
        """WOULD CATCH: a transposed matrix, or one keyed by a different id set
        than ``values`` — either of which makes every per-entity sample list
        describe a different entity."""
        with TemporaryDirectory() as d:
            op = self._resolve(d, self.PER_CELL)
        self.assertEqual(set(op.meta["per_cell"]), set(op.values))
        self.assertEqual(op.meta["n_per_cell"], 3)
        for entity, vals in op.meta["per_cell"].items():
            self.assertEqual(len(vals), 3, entity)

    def test_the_column_mean_of_per_cell_IS_the_operands_centre(self):
        """The samples and the centre must be the same measurement.

        WOULD CATCH: keying the matrix with a different rule than the mean — in
        particular NOT summing collisions the same way, which is the one place
        the two could plausibly diverge (see the collision case below)."""
        with TemporaryDirectory() as d:
            op = self._resolve(d, self.PER_CELL)
        for entity, mean in _column_means(op.meta["per_cell"]).items():
            self.assertAlmostEqual(mean, op.values[entity], places=9, msg=entity)

    def test_collisions_are_summed_per_cell_exactly_as_they_are_in_the_mean(self):
        """Several source rows can map to one entity id; the mean sums them.

        WOULD CATCH: keying ``per_cell`` row-by-row with a rule that overwrites
        instead of summing. The identity above would then break for exactly the
        colliding entities and no others — which is why this is a separate case
        with ids chosen to collide, rather than trusted to the general test."""
        per_cell = [[1.0, 10.0, 100.0], [3.0, 30.0, 300.0]]
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [2.0, 20.0, 200.0], per_cell=per_cell, n_cells=2)
            op = ops.run_operand(sweep, ["EG1", "EG1", "EG2"],
                                 with_per_cell=True)
        self.assertEqual(op.meta["n_collisions"], 1)
        self.assertEqual(op.values, {"EG1": 22.0, "EG2": 200.0})
        self.assertEqual(op.meta["per_cell"], {"EG1": [11.0, 33.0],
                                               "EG2": [100.0, 300.0]})
        for entity, mean in _column_means(op.meta["per_cell"]).items():
            self.assertAlmostEqual(mean, op.values[entity], places=9, msg=entity)

    def test_the_ensemble_mean_repeated_is_NOT_accepted_as_per_cell(self):
        """★ THE RED CASE §5.1 ASKS FOR, CONSTRUCTED AND SHOWN.

        ⚠ **The mean-repeated matrix SATISFIES the column-mean identity.** That
        is the whole point: the identity is necessary and not sufficient, so a
        test that checks only the identity would pass against a `per_cell` that
        carries no per-cell information whatsoever — a degenerate matrix with
        zero dispersion, from which every distributional statistic returns a
        point mass and every t-test divides by zero.

        So the discriminating assertion is DISPERSION, and this asserts both
        halves: the identity cannot tell the two apart, and dispersion can.
        """
        degenerate = [list(self.MEAN) for _ in range(3)]
        with TemporaryDirectory() as d:
            good = self._resolve(d, self.PER_CELL)
        with TemporaryDirectory() as d:
            bad = self._resolve(d, degenerate)

        def spread(op):
            return {e: max(v) - min(v) for e, v in op.meta["per_cell"].items()}

        # Both satisfy the identity — the necessary-but-insufficient check.
        for op in (good, bad):
            for entity, mean in _column_means(op.meta["per_cell"]).items():
                self.assertAlmostEqual(mean, op.values[entity], places=9)
        # Only the real one carries dispersion.
        self.assertTrue(all(s > 0 for s in spread(good).values()))
        self.assertTrue(all(s == 0 for s in spread(bad).values()))

    def test_a_per_cell_row_of_the_wrong_width_RAISES(self):
        """WOULD CATCH: silent truncation. ``_keyed`` zips ids against values, so
        a short row would key fine and quietly describe the wrong entities — the
        same failure ``entity_ids``' width check exists to prevent, one level
        down."""
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, self.MEAN, n_cells=2,
                       per_cell=[[1.0, 10.0, 100.0], [3.0, 30.0]])
            with self.assertRaises(ValueError) as caught:
                ops.run_operand(sweep, self.IDS, with_per_cell=True)
        self.assertIn("per_cell row 1", str(caught.exception))


class RunOperandUnits(unittest.TestCase):
    """⛔ A live operand must declare what its numbers ARE.

    ★ THIS IS THE REGRESSION §5.2 ASKS FOR, AND IT FAILS AGAINST THE PARENT
    COMMIT — a live operand stamped no ``units`` at all.

    Why it matters downstream, stated here because the consumer is not in this
    repo: a grader that restricts a comparison to a counts-bearing subset
    ("species above N copies/cell") can only do so from a side that says it
    carries counts. With nothing declared, the live path contributed no counts,
    the subset silently widened to the whole panel, and the same sweep graded
    live rather than baked produced a different headline number with only a
    detail field to say so. The end-to-end assertion lives with that grader; what
    is assertable HERE is the interface it keys on — and that is what this pins.
    """

    CASES = [("transcriptome", "model_transcriptome.json", "by_gene_id"),
             ("proteome", "model_proteome.json", "by_id")]

    def test_a_live_operand_declares_the_same_units_the_baked_fixture_does(self):
        """WOULD CATCH: (a) declaring nothing, i.e. the parent commit; (b)
        declaring a near-miss spelling — ``counts_per_cell`` where the fixture
        says ``counts/cell`` — which is exactly what defeats a
        membership-test-based counts collector while looking correct in review.

        Asserted against the COMMITTED FIXTURE rather than an inline literal, so
        it also fails if the bake side changes its unit string unilaterally."""
        for observable, fixture, _map_key in self.CASES:
            with self.subTest(observable=observable):
                declared = json.loads(
                    (_BASAL_FIXTURES / fixture).read_text(encoding="utf-8"))["units"]
                self.assertTrue(declared)
                with TemporaryDirectory() as d:
                    sweep = Path(d) / "sweep"
                    sweep.mkdir()
                    _run_cache(sweep, [1.0, 2.0], observable=observable)
                    op = ops.run_operand(sweep, ["EG1", "EG2"],
                                         observable=observable)
                self.assertEqual(op.meta["units"], declared)

    def test_units_come_from_the_declaration_not_from_the_cached_node(self):
        """WOULD CATCH: reading ``node["units"]``, which would make an operand
        built from a pre-v2 or hand-written envelope declare ``None`` — silently,
        and only on the machines with an older cache. ``_run_cache`` deliberately
        writes no units, so this passes only if the resolver never consulted
        it."""
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            path = _run_cache(sweep, [1.0, 2.0])
            self.assertNotIn("units", json.loads(path.read_text())["vectors"]
                             ["omics"]["transcriptome"])
            op = ops.run_operand(sweep, ["EG1", "EG2"])
        self.assertEqual(op.meta["units"], "counts/cell")

    def test_a_fixture_operand_declares_its_units_too(self):
        """WOULD CATCH: the resolver-detail leak. A consumer that wants to know
        what a fixture's numbers are had to re-open the blob behind a
        ``path == "fixture"`` special case — a load-time branch on resolution
        mechanism sitting inside a grader, which is the split this module exists
        to remove."""
        for _observable, fixture, map_key in self.CASES:
            with self.subTest(fixture=fixture):
                blob = json.loads(
                    (_BASAL_FIXTURES / fixture).read_text(encoding="utf-8"))
                op = ops.fixture_operand(_BASAL_FIXTURES, fixture, map_key)
                self.assertEqual(op.meta["units"], blob["units"])

    def test_the_bake_path_declares_the_same_units_this_module_does(self):
        """★ The `live == baked` property of §5.4, asserted where the two paths
        GENUINELY DIVERGE.

        ⚠ Deliberately NOT asserted through ``_keyed``: live and bake read the
        same cached node through the same ``load_or_extract`` and key it with
        near-identical functions, so a test comparing their NUMBERS is comparing
        one measurement to itself and cannot fail. (The one real difference
        between the two ``_keyed`` implementations is already pinned by
        ``test_the_one_way_the_two_keyed_implementations_deliberately_differ``.)

        Units are different: bake hardcodes its string inline while this module
        reads ``card_vectors.VECTOR_UNITS``. Two independent sources, so this
        CAN fail — WOULD CATCH either side being changed alone."""
        import importlib.util
        from v2ecoli.library.card_vectors import VECTOR_UNITS

        src = _REPO / "scripts" / "bake_model_omics.py"
        spec = importlib.util.spec_from_file_location("_bake_for_units", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        baked = mod._node([1.0], {"gene_ids": ["EG1"]}, 1, "method", "source")
        self.assertEqual(baked["units"], VECTOR_UNITS[("omics", "transcriptome")])

    def test_the_id_key_is_the_callers_and_is_not_invented(self):
        """WOULD CATCH: hardcoding an id-space per observable. The positional
        vector is keyed by whatever labels the caller supplied, and only the
        caller knows which ``sim_data`` they came from — the same reasoning that
        makes ``entity_ids`` a parameter. A default that is right for the wild
        type is the kind that stays right until it silently isn't."""
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [1.0, 2.0])
            unstated = ops.run_operand(sweep, ["EG1", "EG2"])
            stated = ops.run_operand(sweep, ["EG1", "EG2"],
                                     id_key="EcoCyc gene id")
        self.assertIsNone(unstated.meta["id_key"])
        self.assertEqual(stated.meta["id_key"], "EcoCyc gene id")


class RunOperandExchange(unittest.TestCase):
    """The exchange observable resolves — and says loudly that it is SIGNED."""

    def _exchange_cache(self, d, vector, per_cell=None):
        sweep = Path(d) / "sweep"
        sweep.mkdir()
        _run_cache(sweep, vector, observable="exchange", group="fluxes",
                   node_name="exchange", per_cell=per_cell,
                   n_cells=len(per_cell) if per_cell else 20)
        return sweep

    def test_the_exchange_observable_resolves_instead_of_raising(self):
        """WOULD CATCH: the mapping being dropped. Before it existed, a study
        declaring this observable got a ``KeyError`` at resolution — which is the
        correct behaviour for an unmapped name and the wrong one for this."""
        with TemporaryDirectory() as d:
            sweep = self._exchange_cache(d, [-5.0, 0.0, 3.0])
            op = ops.run_operand(sweep, ["GLC[p]", "ACET[p]", "CARBON-DIOXIDE[p]"],
                                 observable="exchange")
        self.assertEqual(op.path, "in-investigation")
        self.assertEqual(op.values["GLC[p]"], -5.0)
        self.assertEqual(op.meta["units"], "mmol/gDCW/h")

    def test_an_unmapped_observable_still_raises_and_names_the_known_set(self):
        """WOULD CATCH: the mapping becoming permissive — a `.get` returning a
        default, or the guard being relaxed while exchange was added. An operand
        that resolves an observable nobody declared grades something nobody
        asked for."""
        with TemporaryDirectory() as d:
            sweep = self._exchange_cache(d, [1.0])
            with self.assertRaises(KeyError) as caught:
                ops.run_operand(sweep, ["X"], observable="metabolome")
        msg = str(caught.exception)
        self.assertIn("metabolome", msg)
        self.assertIn("exchange", msg)

    def test_exchange_declares_itself_signed_and_counts_the_signs(self):
        """⛔ THE SEAM. Uses the 87 REAL flux values pinned in the committed
        basal reference, so the numbers below are a property of the model rather
        than of this test's imagination.

        WOULD CATCH: an upstream change that rectifies exchange to magnitudes
        (``n_negative`` would fall to 0), or the flag being dropped — either of
        which would let a consumer treat fluxes as abundances with nothing
        objecting.
        """
        ref = json.loads(_BASAL_REFERENCE.read_text(encoding="utf-8"))
        crit = ref["axes"]["fluxes.exchange"]["criterion"]
        vector, flux_ids = crit["ref_vector"], crit["flux_ids"]
        self.assertEqual(len(vector), 87)

        with TemporaryDirectory() as d:
            sweep = self._exchange_cache(d, vector)
            op = ops.run_operand(sweep, flux_ids, observable="exchange")

        self.assertTrue(op.meta["signed_quantity"])
        self.assertEqual(op.meta["n_negative"], 17)
        self.assertEqual(op.meta["n_zero"], 46)
        self.assertEqual(op.meta["n_positive"], 24)
        # The consequence, spelled out so it cannot be rediscovered the hard way:
        # a log transform keeps only the positives, so 63 of 87 species — 72%,
        # INCLUDING GLUCOSE, the headline flux — leave the comparison silently.
        self.assertLess(op.values["GLC[p]"], 0)
        self.assertEqual(op.meta["n_negative"] + op.meta["n_zero"], 63)

    def test_an_abundance_observable_is_not_flagged_signed(self):
        """WOULD CATCH: flagging everything, which would make the flag mean
        nothing and train every consumer to ignore it."""
        with TemporaryDirectory() as d:
            sweep = Path(d) / "sweep"
            sweep.mkdir()
            _run_cache(sweep, [1.0, 2.0])
            op = ops.run_operand(sweep, ["EG1", "EG2"])
        self.assertFalse(op.meta["signed_quantity"])
        self.assertEqual((op.meta["n_negative"], op.meta["n_positive"]), (0, 2))


if __name__ == "__main__":
    unittest.main()
