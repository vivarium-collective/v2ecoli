"""Tests for reading a design-screen variant declaration into a build plan.

Every test names the defect it catches. A test that passes against both the
correct and the broken reading is worthless here in a particular way: a
mis-read declaration does not crash, it produces a plausible plan for the
wrong experiment.
"""
import pytest

from v2ecoli.perturbations.design_variant import (
    DesignVariantError,
    plan_design_variant,
)


def _induction(gen, exp=1e6, trl=0.285, knockout_gen=None, weights=None):
    block = {"induction_gen": gen, "exp_trl_eff": {"exp": exp, "trl_eff": trl}}
    if knockout_gen is not None:
        block["knockout_gen"] = knockout_gen
    if weights is not None:
        block["rel_adj"] = {"rel_exp_adj_list": weights[0],
                            "rel_trl_eff_adj_list": weights[1]}
    return block


# --------------------------------------------------------------------------
# Stage structure — the part that encodes how many runs a declaration implies
# --------------------------------------------------------------------------

def test_no_induction_is_one_stage_covering_the_whole_lineage():
    # Catches: special-casing the unperturbed arm into an empty plan. A caller
    # would then have to branch on "did I get any stages", and the arm that
    # silently ran nothing is the one nobody checks.
    plan = plan_design_variant({"condition": "basal"})
    assert len(plan.stages) == 1
    assert plan.stages[0].first_generation == 1
    assert plan.stages[0].cache.new_gene is None
    assert plan.is_staged is False


def test_induction_at_generation_one_needs_no_uninduced_stage():
    # Catches: unconditionally emitting a silent stage. Induction at gen 1 means
    # the construct is on from birth, so a leading uninduced stage would insert
    # a generation the declaration never asked for.
    plan = plan_design_variant({"new_gene_shift": _induction(1)})
    assert [s.first_generation for s in plan.stages] == [1]
    assert plan.stages[0].cache.label == "induced"
    assert plan.is_staged is False


def test_induction_after_generation_one_puts_a_silent_stage_first():
    # Catches: dropping the pre-induction generations. They are the
    # within-lineage control, not padding — and a partially-installed construct
    # is not inert, so losing them changes the experiment rather than shortening
    # it. The resulting plan would still run and still look reasonable.
    plan = plan_design_variant({"new_gene_shift": _induction(3)})
    assert [s.first_generation for s in plan.stages] == [1, 3]
    assert [s.cache.label for s in plan.stages] == ["uninduced", "induced"]
    assert plan.stages[0].cache.new_gene is None
    assert plan.stages[1].cache.new_gene is not None
    assert plan.is_staged is True


def test_induction_plus_knockout_is_three_stages_not_two():
    # Catches: the two-stage assumption. A resumed run reads its process configs
    # once, so it cannot swap caches partway — a declaration carrying both
    # generations therefore needs THREE invocations. Reading it as two would
    # silently never switch the construct off, and the run would complete.
    plan = plan_design_variant({"new_gene_shift": _induction(2, knockout_gen=5)})
    assert [s.first_generation for s in plan.stages] == [1, 2, 5]
    assert [s.cache.label for s in plan.stages] == [
        "uninduced", "induced", "knocked_out"]
    assert plan.stages[2].cache.new_gene.expression == 0.0


def test_knockout_preserves_efficiency_and_weights_and_zeroes_only_expression():
    # Catches: implementing the knockout by dropping the induction block
    # entirely. That would revert efficiency and the weight vectors too, so the
    # final stage would differ from the reference in three ways while looking
    # like a knockout in the one place anyone checks.
    plan = plan_design_variant({
        "new_gene_internal_shift_variable_strength": _induction(
            2, exp=1e6, trl=0.285, knockout_gen=4,
            weights=([1.0, 2.0], [1.0, 3.0]))})
    ko = plan.stages[-1].cache.new_gene
    assert ko.expression == 0.0
    assert ko.translation_efficiency == 0.285
    assert ko.rel_exp_adj == (1.0, 2.0)
    assert ko.rel_trl_eff_adj == (1.0, 3.0)


def test_stages_are_ordered_by_generation():
    # Catches: emitting stages in declaration order rather than lineage order.
    # A consumer walking the list would run the wrong cache for a stretch of
    # generations and produce a complete, wrong result.
    plan = plan_design_variant({"new_gene_shift": _induction(4, knockout_gen=7)})
    gens = [s.first_generation for s in plan.stages]
    assert gens == sorted(gens) == [1, 4, 7]


# --------------------------------------------------------------------------
# The chassis half — applies to every stage, because it is not an event
# --------------------------------------------------------------------------

def test_native_perturbations_apply_to_every_stage():
    # Catches: treating the native perturbations as part of the induction. They
    # describe the chassis strain, so a knockout that only existed from the
    # induction generation onward would be a different strain before and after —
    # and the pre-induction generations would silently be the wrong control.
    plan = plan_design_variant({
        "perturbations": {"EG10527": 0.0, "EG11015": 2.5},
        "new_gene_shift": _induction(3)})
    assert len(plan.stages) == 2
    for stage in plan.stages:
        assert stage.cache.native_perturbations == {"EG10527": 0.0, "EG11015": 2.5}


def test_condition_applies_to_every_stage():
    # Catches: applying the condition only where it is nested. The reference
    # inherits a top-level condition into its induction blocks; a plan whose
    # stages disagreed about media would compare growth across two media and
    # attribute the difference to induction.
    plan = plan_design_variant({
        "condition": "with_aa", "new_gene_shift": _induction(2, knockout_gen=4)})
    assert {s.cache.condition for s in plan.stages} == {"with_aa"}


def test_stages_do_not_share_a_mutable_perturbation_mapping():
    # Catches: handing every stage the same dict object. A consumer patching one
    # stage's perturbations would silently patch all of them.
    plan = plan_design_variant({
        "perturbations": {"EG10527": 0.0}, "new_gene_shift": _induction(3)})
    a, b = plan.stages[0].cache, plan.stages[1].cache
    assert a.native_perturbations == b.native_perturbations
    assert a.native_perturbations is not b.native_perturbations


# --------------------------------------------------------------------------
# Reading the levels and weights
# --------------------------------------------------------------------------

def test_expression_and_efficiency_are_read_from_exp_trl_eff():
    plan = plan_design_variant({
        "new_gene_shift": _induction(1, exp=10 ** 6.07, trl=0.285)})
    ng = plan.stages[0].cache.new_gene
    assert ng.expression == pytest.approx(10 ** 6.07)
    assert ng.translation_efficiency == pytest.approx(0.285)


def test_weight_vectors_are_read_and_are_not_reordered():
    # Catches: normalising, sorting or de-duplicating the weights. They are
    # paired POSITIONALLY against the new-gene targets, so any reordering
    # reassigns weights to the wrong genes while keeping the multiset intact —
    # invisible to a length or sum check.
    plan = plan_design_variant({
        "new_gene_internal_shift_variable_strength": _induction(
            1, weights=([4.0, 1.0, 2.0], [1.0, 3.0, 1.0]))})
    ng = plan.stages[0].cache.new_gene
    assert ng.rel_exp_adj == (4.0, 1.0, 2.0)
    assert ng.rel_trl_eff_adj == (1.0, 3.0, 1.0)


def test_absent_weights_are_none_rather_than_ones():
    # Catches: substituting a default vector. Its length would have to be
    # guessed, and a wrong-length vector is rejected downstream against the real
    # target count — better to pass the absence through and let the layer that
    # knows the count decide.
    plan = plan_design_variant({"new_gene_shift": _induction(1)})
    ng = plan.stages[0].cache.new_gene
    assert ng.rel_exp_adj is None and ng.rel_trl_eff_adj is None


# --------------------------------------------------------------------------
# Refusals — each one a declaration that would otherwise run the wrong thing
# --------------------------------------------------------------------------

def test_both_induction_keys_is_an_error_not_a_precedence_rule():
    # Catches: silently preferring one key. They set the same fields, so a
    # precedence rule would make a declaration mean something its author did not
    # write — and the run would succeed.
    with pytest.raises(DesignVariantError, match="more than one induction block"):
        plan_design_variant({
            "new_gene_shift": _induction(1),
            "new_gene_internal_shift_variable_strength": _induction(3)})


def test_knockout_before_induction_is_rejected():
    with pytest.raises(DesignVariantError, match="must come after"):
        plan_design_variant({"new_gene_shift": _induction(4, knockout_gen=2)})


def test_knockout_equal_to_induction_is_rejected():
    # Catches: an off-by-one in the ordering guard. Equal generations would
    # produce two stages starting at the same generation — the second silently
    # shadowing the first.
    with pytest.raises(DesignVariantError, match="must come after"):
        plan_design_variant({"new_gene_shift": _induction(3, knockout_gen=3)})


def test_zero_or_negative_induction_generation_is_rejected():
    # Catches: accepting a 0-based generation. The declaration's convention is
    # 1-based; reading a 0 as "from the start" would silently shift the whole
    # protocol by one generation.
    with pytest.raises(DesignVariantError, match="1-based"):
        plan_design_variant({"new_gene_shift": _induction(0)})


def test_missing_exp_trl_eff_is_rejected():
    with pytest.raises(DesignVariantError, match="exp_trl_eff"):
        plan_design_variant({"new_gene_shift": {"induction_gen": 2}})


def test_negative_native_multiplier_is_rejected():
    # Catches: passing a negative multiplier through to the efficiency array,
    # where it would produce a nonsensical rate rather than an error.
    with pytest.raises(DesignVariantError, match="non-negative"):
        plan_design_variant({"perturbations": {"EG10527": -1.0}})


def test_non_mapping_perturbations_is_rejected():
    with pytest.raises(DesignVariantError, match="must be a mapping"):
        plan_design_variant({"perturbations": ["EG10527"]})


def test_string_weight_vector_is_rejected_rather_than_iterated_as_characters():
    # Catches: the classic Sequence check that accepts str. "123" would iterate
    # to three separate weights, giving a plausible-looking three-target vector.
    with pytest.raises(DesignVariantError, match="sequence of numbers"):
        plan_design_variant({
            "new_gene_internal_shift_variable_strength": _induction(
                1, weights=("123", [1.0]))})
