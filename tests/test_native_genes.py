"""Tests for scaling native translation efficiency on sim_data before caching.

Each test names the defect it catches. The defects that matter here are quiet
ones: a perturbation applied to the wrong gene, or applied as an assignment
rather than a scaling, produces a complete run with wrong numbers.
"""
import numpy as np
import pytest

from v2ecoli.perturbations.native_genes import (
    resolve_native_targets,
    set_native_translation_efficiency,
)
from v2ecoli.perturbations.translation import UnknownPerturbationTarget


class _Array:
    """Minimal stand-in for a struct array: field access by name."""

    def __init__(self, **fields):
        self._f = {k: np.array(v) for k, v in fields.items()}

    def __getitem__(self, key):
        return self._f[key]


class _FakeSimData:
    """Four genes; ``EG9004`` is non-coding (its cistron encodes no monomer).

    Monomer order is deliberately NOT the gene order — a join that accidentally
    used position instead of the id mapping would still line up if they matched.
    """

    def __init__(self, efficiencies=(0.1, 0.2, 0.3)):
        cistrons = _Array(
            gene_id=["EG9001", "EG9002", "EG9003", "EG9004"],
            id=["EG9001_RNA", "EG9002_RNA", "EG9003_RNA", "EG9004_RNA"],
        )
        monomers = _Array(
            # monomer order: gene 3, gene 1, gene 2 — scrambled on purpose.
            id=["MON-3", "MON-1", "MON-2"],
            cistron_id=["EG9003_RNA", "EG9001_RNA", "EG9002_RNA"],
        )
        self.process = type("P", (), {})()
        self.process.transcription = type("T", (), {})()
        self.process.transcription.cistron_data = type("C", (), {})()
        self.process.transcription.cistron_data.struct_array = cistrons
        self.process.translation = type("L", (), {})()
        self.process.translation.monomer_data = type("M", (), {})()
        self.process.translation.monomer_data.struct_array = monomers
        self.process.translation.translation_efficiencies_by_monomer = np.array(
            efficiencies, dtype=float)


def test_targets_resolve_through_the_id_join_not_by_position():
    # Catches: joining gene index to monomer index positionally. The fake's
    # monomer order is scrambled relative to gene order, so a positional join
    # returns indices that exist and look plausible — and perturbs the wrong
    # genes. Nothing downstream would notice.
    sd = _FakeSimData()
    assert resolve_native_targets(sd, ["EG9001", "EG9002", "EG9003"]) == {
        "EG9001": 1, "EG9002": 2, "EG9003": 0}


def test_efficiency_is_scaled_from_its_baseline_not_assigned():
    # Catches: assigning the multiplier as the value. Both produce a changed
    # array, but assignment discards the gene's fitted efficiency — so a "2x
    # overexpression" would silently become "set to 2.0", which for a gene
    # fitted at 0.2 is a 10x change.
    sd = _FakeSimData(efficiencies=(0.1, 0.2, 0.3))
    set_native_translation_efficiency(sd, {"EG9001": 2.0})
    te = sd.process.translation.translation_efficiencies_by_monomer
    assert te[1] == pytest.approx(0.4)      # 0.2 baseline x 2.0
    assert te[0] == pytest.approx(0.1)      # untouched
    assert te[2] == pytest.approx(0.3)


def test_a_zero_multiplier_is_a_knockout():
    sd = _FakeSimData()
    set_native_translation_efficiency(sd, {"EG9003": 0.0})
    assert sd.process.translation.translation_efficiencies_by_monomer[0] == 0.0


def test_untargeted_genes_are_left_exactly_alone():
    # Catches: rebuilding or renormalising the whole array. A perturbation is
    # meant to be local; touching the others would change every gene's
    # efficiency while only one was declared.
    sd = _FakeSimData(efficiencies=(0.1, 0.2, 0.3))
    before = sd.process.translation.translation_efficiencies_by_monomer.copy()
    set_native_translation_efficiency(sd, {"EG9002": 0.5})
    after = sd.process.translation.translation_efficiencies_by_monomer
    assert after[2] == pytest.approx(before[2] * 0.5)
    untouched = [0, 1]
    assert np.allclose(after[untouched], before[untouched])


def test_provenance_records_as_assigned_values_in_stable_order():
    # Catches: provenance ordered by the caller's dict. A screen diffs these
    # manifests, so an ordering that follows input order makes two identical
    # perturbations produce different-looking records.
    sd = _FakeSimData(efficiencies=(0.1, 0.2, 0.3))
    out = set_native_translation_efficiency(sd, {"EG9003": 3.0, "EG9001": 2.0})
    assert out["gene_ids"] == ["EG9001", "EG9003"]
    assert out["multipliers"] == [2.0, 3.0]
    assert out["translation_efficiencies"] == pytest.approx([0.4, 0.3])
    assert out["monomer_indices"] == [1, 0]


def test_empty_perturbations_is_a_no_op_with_empty_provenance():
    # Catches: raising or returning None on the empty case. Callers apply this
    # unconditionally for the arms that declare no native perturbation.
    sd = _FakeSimData(efficiencies=(0.1, 0.2, 0.3))
    before = sd.process.translation.translation_efficiencies_by_monomer.copy()
    out = set_native_translation_efficiency(sd, {})
    assert out["gene_ids"] == []
    assert np.allclose(sd.process.translation.translation_efficiencies_by_monomer, before)


def test_every_unknown_gene_is_reported_not_just_the_first():
    # Catches: failing fast on the first bad id. A screen declares many genes at
    # once, so one-at-a-time failure turns a single fix into N build attempts.
    sd = _FakeSimData()
    with pytest.raises(UnknownPerturbationTarget) as exc:
        resolve_native_targets(sd, ["EG9001", "NOPE1", "NOPE2"])
    assert "NOPE1" in str(exc.value) and "NOPE2" in str(exc.value)


def test_a_non_coding_gene_is_reported_separately_from_a_typo():
    # Catches: collapsing both failures into one message. "this gene makes no
    # protein" is a modelling mistake; "this id does not exist" is a typo. The
    # fix differs, so the message must too.
    sd = _FakeSimData()
    with pytest.raises(UnknownPerturbationTarget) as exc:
        resolve_native_targets(sd, ["EG9004", "NOPE"])
    msg = str(exc.value)
    assert "non-coding" in msg and "EG9004" in msg
    assert "not in this sim_data" in msg and "NOPE" in msg


@pytest.mark.parametrize("bad", [{"EG9001": -1.0}, {"EG9001": float("nan")},
                                 {"EG9001": float("inf")}])
def test_negative_or_non_finite_multipliers_are_rejected(bad):
    # Catches: passing these through to the efficiency array, where they produce
    # a nonsensical rate rather than an error — and NaN in particular propagates
    # silently through the normalisation.
    sd = _FakeSimData()
    with pytest.raises(ValueError, match="finite and non-negative"):
        set_native_translation_efficiency(sd, bad)


def test_nothing_is_mutated_when_a_target_fails_to_resolve():
    # Catches: applying perturbations as they resolve. A partial application
    # would leave sim_data in a state matching neither the declaration nor the
    # baseline, and the exception would make it look like nothing happened.
    sd = _FakeSimData(efficiencies=(0.1, 0.2, 0.3))
    before = sd.process.translation.translation_efficiencies_by_monomer.copy()
    with pytest.raises(UnknownPerturbationTarget):
        set_native_translation_efficiency(sd, {"EG9001": 2.0, "NOPE": 0.0})
    assert np.allclose(
        sd.process.translation.translation_efficiencies_by_monomer, before)
