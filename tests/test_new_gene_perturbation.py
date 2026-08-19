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

        self.adjust_calls = []

    def adjust_new_gene_final_expression(self, indices, factors):
        self.adjust_calls.append((list(indices), list(factors)))


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

    # expression: one adjust call per gene, factor = expression * weight
    assert sd.adjust_calls == [([2], [1000.0]), ([3], [2000.0])]
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
