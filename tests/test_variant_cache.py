"""Tests for building one stage's cache from a design-variant plan.

Each test names the defect it catches. The failures that matter here are quiet:
a cache that is written before a perturbation is applied, or that carries only
one of the two halves, produces a complete run against the wrong strain.
"""
import copy
import os

import numpy as np
import pytest

from v2ecoli.perturbations.design_variant import CacheSpec, NewGeneInduction
from v2ecoli.perturbations.variant_cache import build_variant_cache


class _Struct:
    """Wraps a real numpy structured array so boolean masking works as it does
    on the genuine sim_data (``cistrons[cistrons["is_new_gene"]]``)."""

    def __init__(self, arr):
        self.struct_array = arr


class _Process:
    pass


class _FakeSimData:
    """Two native genes plus two new genes.

    ⚠ Monomer order is deliberately NOT cistron order — a join that used
    position instead of the id mapping would still line up if they matched, and
    would then perturb the wrong genes silently.
    """

    def __init__(self, baseline=1e-4):
        layout = ["EG10001_RNA", "EG10002_RNA", "NG-GFP0", "NG-GFP1"]
        genes = ["EG10001", "EG10002", "NG-GFP0-GENE", "NG-GFP1-GENE"]
        is_new = [g.startswith("NG") for g in layout]

        self.process = _Process()
        self.process.transcription = _Process()
        self.process.translation = _Process()

        self.process.transcription.cistron_data = _Struct(np.array(
            list(zip(layout, genes, is_new)),
            dtype=[("id", "U32"), ("gene_id", "U32"), ("is_new_gene", "?")]))
        # Monomer order reversed relative to cistron order, on purpose.
        self.process.translation.monomer_data = _Struct(np.array(
            [(g, f"{g}-MONOMER[c]") for g in reversed(layout)],
            dtype=[("cistron_id", "U32"), ("id", "U40")]))
        self.process.transcription.rna_data = {
            "id": np.array([f"{g}[c]" for g in layout], dtype="U40")}

        native = np.array([0.0 if n else 0.5 for n in is_new])
        self.process.transcription.rna_expression = {"basal": native.copy()}
        self.process.transcription.exp_free = native.copy()
        self.process.transcription.exp_ppgpp = native.copy()
        # NOT all ones: with a baseline of 1.0, "assign" and "scale" are
        # numerically identical and prove nothing.
        self.process.translation.translation_efficiencies_by_monomer = np.array(
            [4.0, 3.0, 0.2, 0.1])
        self._baseline = baseline

    def adjust_new_gene_final_expression(self, indices, factors):
        tx = self.process.transcription
        for i, f in zip(indices, factors):
            for exp in tx.rna_expression.values():
                exp[i] = self._baseline * f
            tx.exp_free[i] = self._baseline * f
            tx.exp_ppgpp[i] = self._baseline * f
        for exp in tx.rna_expression.values():
            exp /= exp.sum()
        tx.exp_free /= tx.exp_free.sum()
        tx.exp_ppgpp /= tx.exp_ppgpp.sum()


@pytest.fixture
def spy(monkeypatch):
    """Capture what save_sim_input sees, at the moment it is called."""
    seen = []

    def _fake(sd, bundle_dir, seed=0, condition=None, fixed_media=None, **kw):
        seen.append({
            "bundle_dir": bundle_dir, "seed": seed, "condition": condition,
            "fixed_media": fixed_media,
            "te": sd.process.translation.translation_efficiencies_by_monomer.copy(),
            "expr": sd.process.transcription.rna_expression["basal"].copy(),
        })

    monkeypatch.setattr("v2ecoli.core.save_sim_input", _fake)
    return seen


def _spec(label="s", native=None, new_gene=None, condition="basal"):
    return CacheSpec(label=label, condition=condition,
                     native_perturbations=native or {}, new_gene=new_gene)


def test_both_halves_are_in_sim_data_when_the_bundle_is_saved(spy, tmp_path):
    # Catches: saving before either perturbation is applied, and applying only
    # one half. Both produce a cache that is a pre-perturbation cache wearing a
    # perturbed cache's name — the run completes and the numbers are wrong.
    sd = _FakeSimData()
    build_variant_cache(
        sd, str(tmp_path / "c"),
        _spec(native={"EG10001": 2.0},
              new_gene=NewGeneInduction(expression=1e6, translation_efficiency=0.5)),
        seed=3)
    assert len(spy) == 1
    snap = spy[0]
    # native half: EG10001 -> monomer index 3, baseline 0.1, x2.0
    assert snap["te"][3] == pytest.approx(0.2)
    # new-gene half: both new-gene monomers (indices 1 and 0) assigned 0.5
    assert snap["te"][1] == pytest.approx(0.5)
    assert snap["te"][0] == pytest.approx(0.5)
    # and the new-gene expression (rna indices 2, 3) is non-zero at save time
    assert snap["expr"][2] > 0 and snap["expr"][3] > 0


def test_an_unperturbed_spec_is_valid_and_still_writes_a_cache(spy, tmp_path):
    # Catches: requiring at least one perturbation. The silent stage of an
    # induction plan has neither half, and refusing it would make the caller
    # special-case exactly the arm that is the control.
    sd = _FakeSimData()
    before = sd.process.translation.translation_efficiencies_by_monomer.copy()
    out = build_variant_cache(sd, str(tmp_path / "c"), _spec(label="uninduced"))
    assert len(spy) == 1
    assert np.allclose(spy[0]["te"], before)
    assert out["native"] == {"gene_ids": [], "monomer_indices": [],
                             "multipliers": [], "translation_efficiencies": []}
    assert out["new_gene"] == {}


def test_the_callers_sim_data_is_not_mutated(spy, tmp_path):
    # Catches: mutating in place. A grid loop reusing one loaded sim_data would
    # have stage k inherit stage k-1's perturbations — silently, and only the
    # later points would be wrong.
    sd = _FakeSimData()
    before = sd.process.translation.translation_efficiencies_by_monomer.copy()
    build_variant_cache(
        sd, str(tmp_path / "c"),
        _spec(native={"EG10001": 5.0},
              new_gene=NewGeneInduction(expression=1e6, translation_efficiency=0.9)))
    assert np.allclose(
        sd.process.translation.translation_efficiencies_by_monomer, before)


def test_stages_built_from_one_sim_data_do_not_contaminate_each_other(spy, tmp_path):
    # Catches: the same defect across a realistic call pattern — three stages of
    # one plan built from one loaded object, which is exactly how a screen runs.
    sd = _FakeSimData()
    for i, ng in enumerate([None,
                            NewGeneInduction(1e6, 0.5),
                            NewGeneInduction(0.0, 0.5)]):
        build_variant_cache(sd, str(tmp_path / f"c{i}"),
                            _spec(label=str(i), native={"EG10002": 0.5}, new_gene=ng))
    assert len(spy) == 3
    # stage 0 has no new-gene assignment; stages 1 and 2 do. If the copy leaked,
    # stage 0's snapshot would carry stage 1's values on a later run — instead
    # each snapshot reflects only its own spec.
    assert spy[0]["te"][1] == pytest.approx(3.0)   # untouched baseline
    assert spy[1]["te"][1] == pytest.approx(0.5)   # assigned by the induction
    # and the native half is identical across all three, applied from the same
    # baseline each time rather than compounding 0.5 -> 0.25 -> 0.125
    assert spy[0]["te"][2] == spy[1]["te"][2] == spy[2]["te"][2]
    assert spy[0]["te"][2] == pytest.approx(0.1)   # 0.2 baseline x 0.5, once


def test_the_two_halves_are_order_independent(spy, tmp_path):
    # Catches: an interaction between the halves that the fixed order hides.
    # They touch disjoint monomer indices and only the new-gene half renormalizes
    # the transcriptome, so the result must not depend on which runs first. If
    # that ever stops being true, the fixed order would silently pick a winner.
    from v2ecoli.perturbations.native_genes import set_native_translation_efficiency
    from v2ecoli.perturbations.new_genes import set_new_gene_expression

    native_first = _FakeSimData()
    set_native_translation_efficiency(native_first, {"EG10001": 2.0})
    set_new_gene_expression(native_first, 1e6, 0.5)

    new_gene_first = _FakeSimData()
    set_new_gene_expression(new_gene_first, 1e6, 0.5)
    set_native_translation_efficiency(new_gene_first, {"EG10001": 2.0})

    assert np.allclose(
        native_first.process.translation.translation_efficiencies_by_monomer,
        new_gene_first.process.translation.translation_efficiencies_by_monomer)
    assert np.allclose(
        native_first.process.transcription.rna_expression["basal"],
        new_gene_first.process.transcription.rna_expression["basal"])


def test_condition_seed_and_media_reach_save_and_the_provenance(spy, tmp_path):
    # Catches: dropping the condition on the floor. The cache would be a basal
    # fit labelled as something else — and every downstream number would be for
    # the wrong medium while the manifest claimed otherwise.
    sd = _FakeSimData()
    out = build_variant_cache(sd, str(tmp_path / "c"),
                              _spec(condition="acetate", label="induced"),
                              seed=7, fixed_media="minimal_acetate")
    assert (spy[0]["condition"], spy[0]["seed"], spy[0]["fixed_media"]) == (
        "acetate", 7, "minimal_acetate")
    assert out["condition"] == "acetate"
    assert out["label"] == "induced"
    assert out["seed"] == 7


def test_a_bad_native_target_fails_before_anything_is_written(spy, tmp_path):
    # Catches: writing the bundle anyway. A half-perturbed cache on disk that
    # looks complete is worse than no cache — the next run would use it.
    sd = _FakeSimData()
    with pytest.raises(Exception):
        build_variant_cache(sd, str(tmp_path / "c"),
                            _spec(native={"NOT_A_GENE": 1.0}))
    assert spy == []


def test_weight_vectors_reach_the_new_gene_half_in_order(spy, tmp_path):
    # Catches: dropping or reordering the weights between the plan and the
    # perturbation. They pair positionally with the new-gene targets, so a
    # reorder reassigns them while keeping the multiset intact.
    sd = _FakeSimData()
    build_variant_cache(
        sd, str(tmp_path / "c"),
        _spec(new_gene=NewGeneInduction(
            expression=1e6, translation_efficiency=1.0,
            rel_exp_adj=(1.0, 4.0), rel_trl_eff_adj=(1.0, 3.0))))
    te = spy[0]["te"]
    # new-gene monomers sit at indices 1 and 0, in new_gene_indices order
    assert te[1] == pytest.approx(1.0) and te[0] == pytest.approx(3.0)
    expr = spy[0]["expr"]
    assert expr[3] / expr[2] == pytest.approx(4.0)


def test_spec_mapping_is_not_mutated_by_the_build(spy, tmp_path):
    # Catches: consuming the spec's mapping in place. A plan is reused across
    # seeds and replicates; a spec that degraded after one build would give
    # different strains for the same declared point.
    sd = _FakeSimData()
    spec = _spec(native={"EG10001": 2.0})
    original = copy.deepcopy(dict(spec.native_perturbations))
    build_variant_cache(sd, str(tmp_path / "c"), spec)
    assert dict(spec.native_perturbations) == original


# --------------------------------------------------------------------------
# End-to-end, against a real new-gene ParCa state.
#
# Gated because the input cannot be committed — NOT because it is unverified.
# It has been executed; see the PR body. Everything above stops at the sim_data
# boundary by design, and a fake cannot exercise the one thing most likely to
# break in practice: the weight vectors pair POSITIONALLY against the real
# new-gene count, which a two-gene fake cannot disagree with.
# --------------------------------------------------------------------------

_E2E_STATE = os.environ.get("V2ECOLI_NEW_GENE_CACHE")


@pytest.mark.skipif(
    not _E2E_STATE,
    reason="set V2ECOLI_NEW_GENE_CACHE=/path/to/parca_state.pkl[.gz] from a "
           "`v2ecoli-parca --new-genes ...` build to run the real chain")
def test_a_three_stage_plan_produces_three_materially_different_caches(tmp_path):
    """Declaration -> plan -> three real caches, each differing as declared.

    Catches, as one chain: a plan whose stages are not actually distinct; an
    induction that does not reach the built cache; a knockout that does not
    switch the construct back off; and a chassis perturbation that fails to
    apply to every stage. None of these would raise — each produces a complete
    cache that is wrong.
    """
    from v2ecoli.core import load_cache_bundle
    from v2ecoli.perturbations import new_gene_indices, plan_design_variant
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state)

    sim_data = hydrate_sim_data_from_state(load_parca_state(_E2E_STATE))
    _, _, _, monomer_indices = new_gene_indices(sim_data)
    n = len(monomer_indices)

    # Unequal weights, one per REAL target — the length a fake cannot get wrong.
    weights = [float(i + 1) for i in range(n)]
    native_target = str(
        sim_data.process.transcription.cistron_data.struct_array["gene_id"][0])

    plan = plan_design_variant({
        "perturbations": {native_target: 0.5},
        "new_gene_internal_shift_variable_strength": {
            "induction_gen": 2,
            "knockout_gen": 4,
            "exp_trl_eff": {"exp": 1e6, "trl_eff": 0.285},
            "rel_adj": {"rel_exp_adj_list": weights,
                        "rel_trl_eff_adj_list": weights},
        }})
    assert [s.cache.label for s in plan.stages] == [
        "uninduced", "induced", "knocked_out"]

    built = {}
    for stage in plan.stages:
        out = build_variant_cache(
            sim_data, str(tmp_path / stage.cache.label), stage.cache)
        bundle = load_cache_bundle(out["cache_dir"])
        bulk = bundle["initial_state"]["bulk"]
        new_gene_counts = [int(row[1]) for row in bulk
                           if str(row[0]).startswith("NG-")]
        built[stage.cache.label] = {
            "counts": new_gene_counts,
            "te": np.asarray(bundle["configs"]["ecoli-polypeptide-initiation"]
                             ["translation_efficiencies"], dtype=float),
            "native": out["native"],
        }

    # The construct is SILENT before induction and PRODUCED after it. This is
    # the whole point of staging, and it is only visible on a real build:
    # ParCa inserts a new gene with expression exactly zero.
    assert sum(built["uninduced"]["counts"]) == 0, (
        "uninduced stage is not silent — the pre-induction control is wrong")
    assert sum(built["induced"]["counts"]) > 0, (
        "induction did not reach the built cache")
    assert sum(built["knocked_out"]["counts"]) == 0, (
        "knockout did not switch the construct back off")

    # Translation efficiency survives the knockout while expression does not —
    # the reference switches the construct off without disturbing the rest of
    # the declaration.
    ind_te = built["induced"]["te"][monomer_indices]
    ko_te = built["knocked_out"]["te"][monomer_indices]
    assert np.allclose(ind_te / ind_te[0], ko_te / ko_te[0], rtol=1e-6)

    # The declared weight ratios survive into the cache. Absolute values do not
    # — the cached array is L1-normalised — so ratios are the assertable thing.
    assert np.allclose(ind_te / ind_te[0],
                       np.array(weights) / weights[0], rtol=1e-6)

    # The chassis perturbation applies to EVERY stage, including the silent one.
    for label in ("uninduced", "induced", "knocked_out"):
        prov = built[label]["native"]
        assert prov["gene_ids"] == [native_target]
        assert prov["multipliers"] == [0.5]
