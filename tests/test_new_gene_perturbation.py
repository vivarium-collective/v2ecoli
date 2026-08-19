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
                            "translation_efficiencies"}
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
