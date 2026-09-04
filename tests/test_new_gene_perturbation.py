"""Unit tests for v2ecoli.perturbations.new_genes.

Hermetic: a duck-typed sim_data carrying only the five structures the module
touches. A real new-gene sim_data costs a full ParCa build, and none of the
logic under test (index resolution, weight pairing, the arithmetic, the error
paths) needs one.
"""
import numpy as np
import pytest

from v2ecoli.perturbations.new_genes import new_gene_indices, set_new_gene_expression


class _Struct:
    def __init__(self, arr):
        self.struct_array = arr


class _FakeSimData:
    """Two native genes and (by default) two new genes."""

    def __init__(self, n_new=2, rna_prefix="NG", with_monomers=True):
        cis = [("EG10001_RNA", False), ("EG10002_RNA", False)]
        cis += [(f"NG-RNA{i}", True) for i in range(n_new)]
        self.process = type("P", (), {})()
        self.process.transcription = type("T", (), {})()
        self.process.translation = type("L", (), {})()
        self.process.transcription.cistron_data = _Struct(
            np.array(cis, dtype=[("id", "U32"), ("is_new_gene", "?")]))

        mon = [("EG10001_RNA", "NATIVE-MONOMER-0"), ("EG10002_RNA", "NATIVE-MONOMER-1")]
        if with_monomers:
            mon += [(f"NG-RNA{i}", f"NG-MONOMER-{i}") for i in range(n_new)]
        self.process.translation.monomer_data = _Struct(
            np.array(mon, dtype=[("cistron_id", "U32"), ("id", "U32")]))
        # NOT all ones: with a baseline of 1.0 an "assign" and a "multiply into
        # the existing value" are numerically identical, so the efficiency test
        # would pass against either and prove nothing.
        self.process.translation.translation_efficiencies_by_monomer = np.array(
            [1.0, 1.0] + [2.0 * (i + 1) for i in range(len(mon) - 2)])

        rnas = ["EG10001_RNA[c]", "EG10002_RNA[c]"]
        rnas += [f"{rna_prefix}-RNA{i}[c]" for i in range(n_new)]
        self.process.transcription.rna_data = {"id": np.array(rnas, dtype="U32")}

        # rna_expression, and a per-gene baseline, so the fake can implement the
        # REAL semantics rather than just recording the call. Without this the
        # suite cannot see order-dependent renormalization at all.
        self.process.transcription.rna_expression = {
            "basal": np.array([0.5, 0.5] + [0.0] * n_new)}
        self._baseline = 1e-4
        self.adjust_calls = []

    def adjust_new_gene_final_expression(self, indices, factors):
        # Mirrors the reference: assign each target FROM ITS BASELINE (not from
        # the current value), then renormalize the whole transcriptome.
        self.adjust_calls.append((list(indices), list(factors)))
        arr = self.process.transcription.rna_expression["basal"]
        for i, f in zip(indices, factors):
            arr[i] = self._baseline * f
        arr /= arr.sum()


def test_new_gene_indices_finds_rnas_and_monomers():
    rna_ids, rna_idx, mon_ids, mon_idx = new_gene_indices(_FakeSimData())
    assert rna_ids == ["NG-RNA0", "NG-RNA1"]
    assert rna_idx == [2, 3]                    # after the two native RNAs
    assert list(mon_ids) == ["NG-MONOMER-0", "NG-MONOMER-1"]
    assert mon_idx == [2, 3]


def test_indices_raise_when_the_build_has_no_new_genes():
    with pytest.raises(ValueError, match="no new-gene cistrons"):
        new_gene_indices(_FakeSimData(n_new=0))


def test_indices_raise_when_the_rna_id_convention_does_not_hold():
    # The module identifies new-gene RNAs by id prefix (rna_data has no
    # is_new_gene flag). If that convention fails it must say so, not return [].
    with pytest.raises(ValueError, match="does not hold"):
        new_gene_indices(_FakeSimData(rna_prefix="XX"))


def test_indices_raise_when_a_new_gene_encodes_no_monomer():
    with pytest.raises(ValueError, match="every new gene should encode one"):
        new_gene_indices(_FakeSimData(with_monomers=False))


def test_expression_and_efficiency_apply_the_relative_weights():
    sd = _FakeSimData()
    applied = set_new_gene_expression(
        sd, expression=1000.0, translation_efficiency=0.5,
        rel_exp_adj=[1.0, 2.0], rel_trl_eff_adj=[0.4, 1.6])

    # expression: ONE batched call carrying every target, factor = expression * weight
    assert sd.adjust_calls == [([2, 3], [1000.0, 2000.0])]
    assert applied["expression_factors"] == [1000.0, 2000.0]

    # efficiency: ASSIGNED (not multiplied into the existing 1.0)
    te = sd.process.translation.translation_efficiencies_by_monomer
    # baseline te[2]=2.0, te[3]=4.0 -> a multiply would give 0.4 and 3.2
    assert te[2] == pytest.approx(0.2)
    assert te[3] == pytest.approx(0.8)
    assert applied["translation_efficiencies"] == pytest.approx([0.2, 0.8])
    # native monomers untouched
    assert te[0] == 1.0 and te[1] == 1.0


def test_weights_default_to_one_per_target():
    sd = _FakeSimData()
    applied = set_new_gene_expression(sd, expression=10.0, translation_efficiency=0.3)
    assert applied["expression_factors"] == [10.0, 10.0]
    assert applied["translation_efficiencies"] == pytest.approx([0.3, 0.3])


@pytest.mark.parametrize("kwargs,match", [
    ({"rel_exp_adj": [1.0]}, "rel_exp_adj has 1 entries but this build has 2"),
    ({"rel_trl_eff_adj": [1.0, 1.0, 1.0]},
     "rel_trl_eff_adj has 3 entries but this build has 2"),
])
def test_mispaired_design_vector_raises_rather_than_silently_truncating(kwargs, match):
    # zip() would silently drop the extra/missing entries, so a screen could rank
    # an arm whose design vector was never fully applied.
    with pytest.raises(ValueError, match=match):
        set_new_gene_expression(_FakeSimData(), expression=1.0,
                                translation_efficiency=1.0, **kwargs)


def test_returns_provenance_a_caller_can_record():
    applied = set_new_gene_expression(_FakeSimData(), expression=2.0,
                                      translation_efficiency=0.5)
    assert set(applied) == {"rna_ids", "rna_indices", "expression_factors",
                            "monomer_ids", "monomer_indices",
                            "translation_efficiencies",
                            "operon_structure", "is_polycistronic"}
    assert all(isinstance(m, str) for m in applied["monomer_ids"])


def test_equal_weights_give_equal_expression_regardless_of_gene_order():
    # adjust_new_gene_final_expression renormalizes the WHOLE transcriptome each
    # call, so calling it once per gene renormalizes N times and each gene a
    # different number of times: equal weights come out unequal, and the result
    # depends on call order. Measured on a real 5-gene insertion, that is 0.0014%
    # at expression 1e4 but 13.6% at 1e8 — i.e. it grows exactly as the construct
    # takes a larger share of the transcriptome, which is where a design screen
    # goes. One batched call renormalizes once and is exact at any level.
    sd = _FakeSimData(n_new=3)
    set_new_gene_expression(sd, expression=1e6, translation_efficiency=0.5,
                            rel_exp_adj=[1.0, 1.0, 1.0])
    vals = sd.process.transcription.rna_expression["basal"][2:]
    assert len(sd.adjust_calls) == 1, "must be one batched call, not one per gene"
    assert vals[0] == pytest.approx(vals[1]) == pytest.approx(vals[2])


def test_mismatched_rna_and_monomer_ordering_raises():
    # The two weight vectors are paired positionally against orderings resolved
    # two different ways. A wrong correspondence is not a wrong count, so the
    # length check cannot catch it.
    # The orderings at risk are rna_data order (NG prefix match) vs cistron_data
    # order (is_new_gene flag). Permuting the monomer->cistron map cannot break
    # it, because monomer_ids is derived THROUGH that map; permuting rna_data
    # against cistron_data is the real divergence.
    sd = _FakeSimData()
    rna = sd.process.transcription.rna_data["id"]
    rna[2], rna[3] = rna[3], rna[2]
    with pytest.raises(ValueError, match="do not correspond"):
        set_new_gene_expression(sd, expression=1.0, translation_efficiency=1.0)


# --------------------------------------------------------------------------- #
# Batching + order independence, asserted NUMERICALLY.
#
# ``test_equal_weights_give_equal_expression_regardless_of_gene_order`` above
# discriminates against a per-gene loop via ``len(sd.adjust_calls) == 1`` — a
# CALL-COUNT proxy for implementation shape. It would still pass if someone
# batched the call correctly and got the arithmetic wrong. The two tests below
# assert the consequence we actually care about instead: a loop produces the
# wrong numbers, and the real function produces the right ones.
# --------------------------------------------------------------------------- #

class _InterleavedFakeSimData:
    """A fake whose new genes sit at CHOSEN positions among the natives.

    ``_FakeSimData`` always appends the new genes after the natives, so the only
    reordering it can express is rna_data-vs-cistron_data divergence — which
    ``new_gene_indices`` rejects by design ("do not correspond"), meaning a test
    built on it asserts the guard rather than the arithmetic. The permutation a
    per-gene loop is sensitive to is which ARRAY POSITIONS the new genes occupy,
    because that fixes the order the loop writes and renormalizes in. This fake
    takes that layout as input and keeps rna_data and cistron_data in
    correspondence, so the guard does not fire first and the defect stays
    expressible.
    """

    def __init__(self, layout, baseline=1e-4, native_expression=0.5):
        is_new = [g.startswith("NG") for g in layout]
        self.process = type("P", (), {})()
        self.process.transcription = type("T", (), {})()
        self.process.translation = type("L", (), {})()
        self.process.transcription.cistron_data = _Struct(np.array(
            list(zip(layout, is_new)), dtype=[("id", "U32"), ("is_new_gene", "?")]))
        self.process.translation.monomer_data = _Struct(np.array(
            [(g, f"{g}-MONOMER") for g in layout],
            dtype=[("cistron_id", "U32"), ("id", "U40")]))
        self.process.transcription.rna_data = {
            "id": np.array([f"{g}[c]" for g in layout], dtype="U40")}
        self.process.transcription.rna_expression = {"basal": np.array(
            [0.0 if n else native_expression for n in is_new])}
        self.process.translation.translation_efficiencies_by_monomer = np.ones(
            len(layout))
        self._baseline = baseline
        self.adjust_calls = []

    def adjust_new_gene_final_expression(self, indices, factors):
        # Same semantics as the reference: assign each target FROM ITS BASELINE,
        # then renormalize the whole transcriptome once per call.
        self.adjust_calls.append((list(indices), list(factors)))
        arr = self.process.transcription.rna_expression["basal"]
        for i, f in zip(indices, factors):
            arr[i] = self._baseline * f
        arr /= arr.sum()

    def expression_by_gene(self):
        ids = [str(r)[:-3] for r in self.process.transcription.rna_data["id"]]
        arr = self.process.transcription.rna_expression["basal"]
        return {g: float(v) for g, v in zip(ids, arr)}


def _loop_apply(sd, expression, rel_exp_adj):
    """The fork's shape: ONE adjust call per gene, each renormalizing."""
    _, rna_indices, _, _ = new_gene_indices(sd)
    for idx, weight in zip(rna_indices, rel_exp_adj):
        sd.adjust_new_gene_final_expression([idx], [expression * weight])


_LAYOUT = ["EG10001_RNA", "NG-GFP-A", "NG-GFP-B", "EG10002_RNA", "NG-GFP-C"]


def test_a_per_gene_loop_corrupts_equal_weights_and_the_real_function_does_not():
    # Catches: a re-introduced per-gene loop, AND a batched call whose
    # arithmetic is wrong — neither of which the call-count assertion sees.
    weights = [1.0, 1.0, 1.0]
    new_genes = ["NG-GFP-A", "NG-GFP-B", "NG-GFP-C"]

    looped = _InterleavedFakeSimData(_LAYOUT)
    _loop_apply(looped, 1e6, weights)
    loop_vals = [looped.expression_by_gene()[g] for g in new_genes]

    batched = _InterleavedFakeSimData(_LAYOUT)
    set_new_gene_expression(batched, expression=1e6, translation_efficiency=1.0,
                            rel_exp_adj=weights)
    real_vals = [batched.expression_by_gene()[g] for g in new_genes]

    # Direction 1 — the stub really is broken, so the invariant is not vacuous.
    assert loop_vals[0] != pytest.approx(loop_vals[1], rel=1e-6)
    assert loop_vals[1] != pytest.approx(loop_vals[2], rel=1e-6)
    # Direction 2 — ... and the shipped function gets it exactly right.
    assert real_vals[0] == pytest.approx(real_vals[1]) == pytest.approx(real_vals[2])


def test_expression_is_independent_of_where_the_new_genes_sit_in_the_arrays():
    # Weights and assertions are keyed BY GENE ID, never by position, so the
    # permuted quantity is not derived through the map being permuted.
    weights = {"NG-GFP-A": 1.0, "NG-GFP-B": 2.0, "NG-GFP-C": 4.0}
    layouts = [
        ["EG10001_RNA", "EG10002_RNA", "NG-GFP-A", "NG-GFP-B", "NG-GFP-C"],
        ["NG-GFP-C", "EG10001_RNA", "NG-GFP-B", "EG10002_RNA", "NG-GFP-A"],
        ["NG-GFP-B", "NG-GFP-A", "EG10001_RNA", "NG-GFP-C", "EG10002_RNA"],
    ]

    per_layout = []
    for layout in layouts:
        sd = _InterleavedFakeSimData(layout)
        rna_ids, _, _, _ = new_gene_indices(sd)
        set_new_gene_expression(sd, expression=1e6, translation_efficiency=1.0,
                                rel_exp_adj=[weights[g] for g in rna_ids])
        got = {g: sd.expression_by_gene()[g] for g in weights}
        # Value ratios equal weight ratios in EVERY arrangement.
        assert got["NG-GFP-B"] / got["NG-GFP-A"] == pytest.approx(2.0)
        assert got["NG-GFP-C"] / got["NG-GFP-A"] == pytest.approx(4.0)
        per_layout.append(got)

    # ... and the values themselves are identical across arrangements.
    for other in per_layout[1:]:
        for gene in weights:
            assert other[gene] == pytest.approx(per_layout[0][gene])

    # Non-vacuity: a per-gene loop DOES depend on the layout, so this
    # permutation can express the defect (an earlier attempt in this area could
    # not — see test_mismatched_rna_and_monomer_ordering_raises).
    loop_results = []
    for layout in layouts[:2]:
        sd = _InterleavedFakeSimData(layout)
        rna_ids, _, _, _ = new_gene_indices(sd)
        _loop_apply(sd, 1e6, [weights[g] for g in rna_ids])
        loop_results.append({g: sd.expression_by_gene()[g] for g in weights})
    assert loop_results[0]["NG-GFP-A"] != pytest.approx(
        loop_results[1]["NG-GFP-A"], rel=1e-6)


class _FakeOperonSimData(_FakeSimData):
    """A POLYCISTRONIC insertion: one transcription unit, N cistrons.

    ⚠ The topology this fake exists for is not exotic — it is what an operon is,
    and it is the shape a real screen's design vectors are written against (one
    expression weight for the TU, one translation weight per gene). The default
    fake is monocistronic, so nothing in this file exercised it.

    Ids are invented. A fixture in a public repo must not carry a real construct's
    identifiers, and the assertions do not need them: what is under test is the
    1-TU-to-N-cistron shape, not which pathway it happens to encode.
    """

    def __init__(self, n_cistrons=5, tu_id="NG-TU-A", with_mapping=True,
                 orphan_cistron=False, extra_empty_tu=False):
        super().__init__(n_new=n_cistrons)
        cis = [("EG10001_RNA", False), ("EG10002_RNA", False)]
        cis += [(f"NG-CIS-{i}", True) for i in range(n_cistrons)]
        self.process.transcription.cistron_data = _Struct(
            np.array(cis, dtype=[("id", "U32"), ("is_new_gene", "?")]))

        mon = [("EG10001_RNA", "NATIVE-MONOMER-0"), ("EG10002_RNA", "NATIVE-MONOMER-1")]
        mon += [(f"NG-CIS-{i}", f"NG-MONOMER-{i}") for i in range(n_cistrons)]
        self.process.translation.monomer_data = _Struct(
            np.array(mon, dtype=[("cistron_id", "U32"), ("id", "U32")]))
        self.process.translation.translation_efficiencies_by_monomer = np.array(
            [1.0, 1.0] + [2.0 * (i + 1) for i in range(n_cistrons)])

        # ONE new RNA for N cistrons — the whole point.
        rnas = ["EG10001_RNA[c]", "EG10002_RNA[c]", f"{tu_id}[c]"]
        if extra_empty_tu:
            rnas.append("NG-TU-UNRELATED[c]")
        self.process.transcription.rna_data = {"id": np.array(rnas, dtype="U32")}
        self.process.transcription.rna_expression = {
            "basal": np.array([0.5, 0.5] + [0.0] * (len(rnas) - 2))}

        if with_mapping:
            m = np.zeros((len(cis), len(rnas)))
            tu_col = 2
            for i in range(n_cistrons):
                # `orphan_cistron` leaves the last cistron off every new TU.
                if orphan_cistron and i == n_cistrons - 1:
                    continue
                m[2 + i, tu_col] = 1.0
            self.process.transcription.cistron_tu_mapping_matrix = m


def test_a_polycistronic_insertion_is_read_not_rejected():
    # Catches: requiring the new-gene RNA list to equal the cistron list. An
    # operon has ONE TU and N cistrons, so the lists legitimately differ in
    # length — and the guard fired before any weight was applied, rejecting the
    # construct outright. This function's own docstring already says a cistron
    # does not map 1:1 to an RNA once operons are involved.
    rna_ids, rna_idx, monomer_ids, monomer_idx = new_gene_indices(
        _FakeOperonSimData(n_cistrons=5))
    assert rna_ids == ["NG-TU-A"], "the transcription unit was not identified"
    assert len(monomer_ids) == 5, "the five cistrons' monomers were not found"
    assert len(rna_idx) == 1 and len(monomer_idx) == 5


def test_operon_weight_vectors_pair_to_their_own_spaces():
    # Catches: pairing both vectors against one list, AND mis-pairing a weight to
    # the wrong monomer. Expression is a property of the TRANSCRIPTION UNIT (the
    # reference writes RNA-indexed arrays), translation efficiency is per MONOMER
    # — so an operon takes 1 expression weight and N efficiency weights.
    #
    # ⚠ Asserts on the EFFICIENCY ARRAY, not on the returned list. An earlier
    # version checked `applied["translation_efficiencies"]`, which the function
    # builds in weight order regardless of which index each value was written to
    # — so reversing the pairing left it green. A returned value is not evidence
    # about where it landed.
    sd = _FakeOperonSimData(n_cistrons=5)
    weights = [0.56, 0.94, 1.0, 1.73, 1.35]
    applied = set_new_gene_expression(
        sd, expression=1e6, translation_efficiency=0.285,
        rel_exp_adj=[1.0], rel_trl_eff_adj=weights)

    assert applied["expression_factors"] == [1e6]
    assert len(applied["rna_ids"]) == 1, "expression must pair against the ONE TU"

    te = sd.process.translation.translation_efficiencies_by_monomer
    for monomer_id, weight in zip([f"NG-MONOMER-{i}" for i in range(5)], weights):
        idx = applied["monomer_ids"].index(monomer_id)
        landed = te[applied["monomer_indices"][idx]]
        assert landed == pytest.approx(0.285 * weight), (
            f"{monomer_id} received {landed}, expected {0.285 * weight}")


def test_a_construct_cistron_on_a_native_tu_is_refused():
    # Catches: counting ANY transcription unit as coverage. A new cistron sitting
    # only on a NATIVE TU passes a naive coverage check while the expression
    # weight — applied to the new RNAs — never reaches it. That is exactly the
    # "part of the construct silent while reporting as fully induced" failure the
    # coverage check exists to catch, so accepting it makes the check decorative.
    sd = _FakeOperonSimData(n_cistrons=5)
    m = sd.process.transcription.cistron_tu_mapping_matrix
    m[2 + 4, 2] = 0.0     # last construct cistron off the new TU...
    m[2 + 4, 0] = 1.0     # ...and onto a native one
    with pytest.raises(ValueError, match="not transcribed from any"):
        new_gene_indices(sd)


def test_provenance_records_the_build_topology():
    # Catches: recording values without the topology they were applied against.
    # The same insertion can reconstruct as N monocistronic TUs or as one operon,
    # and one expression weight means something different against each. Without
    # this, an arm built on the wrong topology is invisible in the manifest.
    poly = set_new_gene_expression(_FakeOperonSimData(n_cistrons=5),
                                   expression=1.0, translation_efficiency=1.0,
                                   rel_exp_adj=[1.0])
    assert poly["is_polycistronic"] is True
    assert poly["operon_structure"] == {
        "NG-TU-A": [f"NG-CIS-{i}" for i in range(5)]}

    mono = set_new_gene_expression(_FakeSimData(n_new=2), expression=1.0,
                                   translation_efficiency=1.0)
    assert mono["is_polycistronic"] is False


def test_a_polycistronic_build_without_a_tu_mapping_is_refused():
    # Catches: pairing weights on an assumption. With differing list lengths and
    # no cistron->TU mapping, which cistrons sit on which TU is unknowable — and
    # guessing would apply the construct's expression weight to an RNA that may
    # not carry it. Refusing loudly beats a plausible wrong arm.
    with pytest.raises(ValueError, match="cistron_tu_mapping_matrix"):
        new_gene_indices(_FakeOperonSimData(n_cistrons=5, with_mapping=False))


def test_a_cistron_on_no_new_transcription_unit_is_refused():
    # Catches: assuming coverage. A new cistron transcribed from no new TU would
    # never receive the expression weight, so the arm would run with part of its
    # construct silent while reporting as fully induced.
    with pytest.raises(ValueError, match="not transcribed from any"):
        new_gene_indices(_FakeOperonSimData(n_cistrons=5, orphan_cistron=True))


def test_an_ng_rna_carrying_no_new_cistron_is_refused():
    # Catches: trusting the 'NG' id prefix. New RNAs are found by prefix because
    # rna_data has no is_new_gene flag, so an unrelated NG-prefixed RNA would
    # silently collect an expression weight meant for the construct.
    with pytest.raises(ValueError, match="carry no new-gene cistron"):
        new_gene_indices(_FakeOperonSimData(n_cistrons=5, extra_empty_tu=True))


def test_monocistronic_order_correspondence_is_still_enforced():
    # Catches: dropping the original guard while generalising. When the two lists
    # DO describe the same genes, a positional weight for RNA i and monomer i
    # must mean the same gene; that protection must survive the operon change.
    sd = _FakeSimData(n_new=3)
    rnas = list(sd.process.transcription.rna_data["id"])
    rnas[2], rnas[3] = rnas[3], rnas[2]          # scramble RNA order only
    sd.process.transcription.rna_data = {"id": np.array(rnas, dtype="U32")}
    with pytest.raises(ValueError, match="do not correspond"):
        new_gene_indices(sd)
