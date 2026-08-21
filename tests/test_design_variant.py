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


def _induction(gen, exp=1e6, trl=0.285, knockout_gen=None, weights=None,
               condition="basal"):
    """An induction block.

    ``condition`` defaults to a block-level one because that is where every real
    config puts it — the reference's own screens declare the media axis inside
    the induction block, never at the top level. Pass ``condition=None`` to test
    top-level inheritance.
    """
    block = {"induction_gen": gen, "exp_trl_eff": {"exp": exp, "trl_eff": trl}}
    if condition is not None:
        block["condition"] = condition
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
        "condition": "with_aa",
        "new_gene_shift": _induction(2, knockout_gen=4, condition=None)})
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
    # Catches: swapping exp and trl_eff, or coercing either through int(). They
    # are read positionally from a two-key mapping and differ by seven orders of
    # magnitude, so a swap does not raise — it plans a construct driven at 0.285
    # copies and a translation weight of a million.
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
    # Catches: dropping the ordering guard entirely. Stages are emitted in
    # declaration order, so a knockout before induction would plan a lineage
    # that switches the construct off before it was ever on and still run.
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
    # Catches: defaulting the induction levels. An induction block with no
    # levels is not "induce at the default strength" — nothing in the reference
    # supplies one — so a default would invent a dose the declaration never set
    # and the arm would report as a real point on the expression sweep.
    with pytest.raises(DesignVariantError, match="exp_trl_eff"):
        plan_design_variant({"new_gene_shift": {"induction_gen": 2,
                                                "condition": "basal"}})


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


# --------------------------------------------------------------------------
# Declaration SHAPE — the reader must handle both, because vEcoli dispatches on
# the single key under `variants:` and hands apply_variant the contents beneath
# it (runscripts/create_variants.py:398-412). The shapes below are the two the
# reference's own CD-screen configs actually use; earlier tests here were all
# built from synthetic declarations of one shape, which is how the other one
# went unread.
# --------------------------------------------------------------------------

def test_a_bare_induction_declaration_is_read_as_one():
    # Catches: reading only the composed (`strain_design`) shape. When the
    # variant module IS an induction variant, its declaration has no wrapper key
    # and its own keys are the block's. Looking only for a wrapper finds nothing,
    # and the induction is not partially read — it is not seen at all. The plan
    # comes back as a single unperturbed stage that looks entirely reasonable,
    # so an expression sweep would run as N copies of the same baseline.
    plan = plan_design_variant({
        "condition": "basal_with_trp", "induction_gen": 1,
        "exp_trl_eff": {"exp": 6.07, "trl_eff": 0.285},
        "rel_adj": {"rel_exp_adj_list": [1.0],
                    "rel_trl_eff_adj_list": [0.56, 0.94, 1.0, 1.73, 1.35]}})
    assert len(plan.stages) == 1
    ng = plan.stages[0].cache.new_gene
    assert ng is not None, "the induction was dropped entirely"
    assert ng.expression == pytest.approx(6.07)
    assert ng.rel_trl_eff_adj == (0.56, 0.94, 1.0, 1.73, 1.35)
    assert plan.stages[0].cache.condition == "basal_with_trp"


def test_a_block_level_condition_wins_over_the_top_level_one():
    # Catches: reading `condition` only at the top level. strain_design inherits
    # a top-level condition into the block ONLY if the block lacks one
    # (strain_design.py:81-82, "inherit if not given"), and the induction variant
    # then applies whatever it holds (new_gene_internal_shift.py:161) — so a
    # block-level condition is legal and wins. Every real screen declares the
    # media axis there, so dropping it does not fail: it plans the entire grid in
    # one medium and the media axis silently disappears.
    plan = plan_design_variant({
        "condition": "basal",
        "new_gene_internal_shift_variable_strength": _induction(
            3, condition="basal_with_trp")})
    assert {s.cache.condition for s in plan.stages} == {"basal_with_trp"}


def test_an_induction_without_any_condition_is_refused_not_defaulted():
    # Catches: falling through to the cache builder's default. The reference
    # reads params["condition"] unguarded (condition.apply_variant), so it raises
    # rather than choosing — and a default here would put the arm in a medium
    # nobody declared while every other axis matched.
    with pytest.raises(DesignVariantError, match="growth condition"):
        plan_design_variant({"new_gene_shift": _induction(2, condition=None)})


def test_a_misspelled_key_is_refused_rather_than_ignored():
    # Catches: silently ignoring keys the reader does not act on. This is the
    # module's whole purpose failing where it matters most: the declaration below
    # plans a completely unperturbed arm that builds, runs to completion and
    # reports as a data point indistinguishable from a real negative result.
    with pytest.raises(DesignVariantError, match="new_gene_shft"):
        plan_design_variant({"perturbation": {"EG10527": 0.0},
                             "new_gene_shft": {"induction_gen": 2}})


def test_an_unexpanded_grid_spec_is_refused():
    # Catches: reading a whole axis as a single grid point. parse_variants
    # expands {"value": [...]} into one declaration per point and pops "op"; a
    # declaration still carrying them skipped expansion. Both shapes are
    # mappings, so without this the media axis would be read as the literal dict.
    with pytest.raises(DesignVariantError, match="unexpanded grid spec"):
        plan_design_variant({
            "condition": {"value": ["basal", "basal_with_trp"]},
            "induction_gen": {"value": [1]},
            "exp_trl_eff": {"nested": {}}})


def test_a_mixed_shape_declaration_is_refused():
    # Catches: guessing which half carries the induction when a declaration has
    # both a nested block and block-level keys at the top level. Preferring
    # either silently discards the other.
    with pytest.raises(DesignVariantError, match="mixes"):
        plan_design_variant({
            "condition": "basal",
            "new_gene_shift": _induction(1),
            "induction_gen": 2})


@pytest.mark.parametrize("field,value", [
    ("induction_gen", 2.7),
    ("knockout_gen", 4.9),
])
def test_a_fractional_generation_is_refused_rather_than_truncated(field, value):
    # Catches: int() truncation. The reference fires on `generation >= gen`, so
    # 2.7 means generation 3 there and would mean generation 2 here — the whole
    # protocol shifted by one cell cycle, silently, with the run completing.
    block = _induction(2, knockout_gen=6)
    block[field] = value
    with pytest.raises(DesignVariantError, match="whole number"):
        plan_design_variant({"new_gene_shift": block})
