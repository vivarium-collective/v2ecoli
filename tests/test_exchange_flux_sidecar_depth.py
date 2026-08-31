"""The exchange-flux sidecar must record LINEAGE DEPTH, not an invocation's count.

A chained run is several invocations over one lineage (the induction seam: a
silent stage, then a stage resumed against a different cache). Every stage
rewrites the sidecar, so a stage recording its own ``max_generations``
understates the lineage — and the comparison card's arm-correspondence check
then refuses the grade as "one of them is stale" while both arms have every
generation on disk.

⚠ The asymmetry is the load-bearing part and is easy to "fix" wrongly: only the
v2ecoli arm accepts ``initial_generation``. The wrapped-reference arm has no
resume hook, so ``max_generations`` IS its depth and applying the offset there
would corrupt it.
"""
import inspect
import re

import scripts.run_comparison_ensemble as rce


def test_a_fresh_run_is_a_strict_no_op():
    # initial_generation is 1-based and inclusive, so the default path must be
    # byte-identical to recording max_generations.
    assert rce._lineage_depth(1, 3) == 3
    assert rce._lineage_depth(1, 1) == 1


def test_a_resumed_stage_reports_the_LINEAGE_not_its_own_count():
    # The real case: a 3-generation chain whose last stage ran generations 2-3.
    assert rce._lineage_depth(2, 2) == 3
    # And a per-generation chain, where every stage runs exactly one.
    assert rce._lineage_depth(3, 1) == 3
    assert rce._lineage_depth(8, 1) == 8


def test_depth_is_not_merely_max_generations():
    # Discriminating: this is the assertion that fails against the old code path,
    # where the sidecar recorded max_generations regardless of where the stage
    # started.
    assert rce._lineage_depth(2, 2) != 2
    assert rce._lineage_depth(5, 4) == 8


def _sidecar_calls():
    """Each ``_write_exchange_flux_sidecar(...)`` call, keyed by its arm prefix.

    Paren-matched rather than regex-delimited: the two call sites are wrapped
    differently, and a regex that happens to fit today's formatting would go
    quietly vacuous the first time someone reflows one of them.
    """
    src = inspect.getsource(rce)
    calls = {}
    needle = "_write_exchange_flux_sidecar("
    i = src.find(needle, src.find("def run_one"))
    while i != -1:
        j, depth = i + len(needle), 1
        while depth:
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
            j += 1
        body = src[i + len(needle):j - 1]
        m = re.search(r'"(\w+)"', body)
        assert m, body
        calls[m.group(1)] = body
        i = src.find(needle, j)
    return calls


def test_the_candidate_arm_records_depth_and_the_REFERENCE_ARM_DOES_NOT():
    """Both arms write this sidecar; only one of them can resume a lineage.

    Asserting the reference arm keeps ``max_generations`` is not redundant —
    it is the half a well-meaning follow-up is most likely to break.
    """
    calls = _sidecar_calls()
    assert set(calls) == {"vecoli", "v2ecoli"}, calls.keys()

    assert "_lineage_depth" in calls["v2ecoli"], (
        "the candidate arm can resume a lineage, so its sidecar must record "
        "lineage depth")
    assert "_lineage_depth" not in calls["vecoli"], (
        "the reference arm has no resume hook; max_generations IS its depth "
        "and offsetting it would corrupt the record")
    assert "max_generations" in calls["vecoli"]


def test_only_the_candidate_arm_is_even_given_a_resume_generation():
    """The premise behind the asymmetry above, asserted rather than assumed."""
    src = inspect.getsource(rce)
    assert "initial_generation=int(initial_generation)" in src
    # The reference arm's engine call takes no resume parameter.
    ref_call = re.search(r"run_vivarium_ecoli_pbg_multigen\((.*?)\)\n",
                         src, re.S)
    assert ref_call is not None
    assert "initial_generation" not in ref_call.group(1)
