"""workflow_nf: the campaign DAG, rendered rather than run.

The properties worth pinning are the ones that would otherwise fail silently --
a variant whose strain inputs never reach ParCa still produces N runs that look
right and share one genotype.
"""

from __future__ import annotations

import pytest

from v2ecoli.composites.workflow_nf import AnalysisTaskStep, ParcaTaskStep, build_workflow_nf


@pytest.fixture(scope="module")
def core():
    from v2ecoli.core import build_core
    from v2ecoli.workflow.meta_composite import register_workflow_processes

    c = build_core()
    register_workflow_processes(c)
    return c


def _render(core, doc):
    from process_bigraph import Composite
    from process_bigraph.nextflow import render_composite

    return render_composite(Composite(doc, core=core), {"workflow_name": ""})


def test_shape_is_one_parca_per_variant_feeding_its_own_seeds() -> None:
    doc = build_workflow_nf(n_seeds=2, variants=[{"variant_name": "a"}, {"variant_name": "b"}])
    state = doc["state"]
    assert sorted(state) == ["parca_v0", "parca_v1", "runs_v0", "runs_v1"]
    # each variant's sub-composite reads ITS OWN cache -- the point of a strain sweep
    assert state["runs_v0"]["inputs"]["cache"] == ["cache_v0"]
    assert state["runs_v1"]["inputs"]["cache"] == ["cache_v1"]
    # and the M lineages live inside it, wired to the take: port
    inner = state["runs_v0"]["config"]["state"]
    assert sorted(k for k in inner if k.startswith("lineage")) == ["lineage_s0", "lineage_s1"]
    assert inner["lineage_s1"]["inputs"]["cache_dir"] == ["cache"]


def test_strain_inputs_reach_PARCA_not_the_lineage() -> None:
    """The silent failure this guards: threading new_genes to the lineage instead
    (what lineage_ray_batch's variants collapse does today) gives ONE shared cache
    for the whole sweep -- N runs of the same genotype wearing different labels."""
    doc = build_workflow_nf(n_seeds=1, variants=[
        {"variant_name": "vio", "new_genes": "violacein_MG1655_M5", "bundle_overrides": "/m.json"}])
    parca_cfg = doc["state"]["parca_v0"]["config"]
    assert parca_cfg["new_genes"] == "violacein_MG1655_M5"
    assert parca_cfg["bundle_overrides"] == "/m.json"
    inner = doc["state"]["runs_v0"]["config"]["state"]
    assert "new_genes" not in inner["lineage_s0"]["config"]


def test_no_variants_means_one_baseline_not_zero() -> None:
    """Zero variants would render an empty workflow that exits 0."""
    doc = build_workflow_nf(n_seeds=1, variants=None)
    assert "parca_v0" in doc["state"] and "runs_v0" in doc["state"]


def test_renders_the_two_level_scatter(core) -> None:
    doc = build_workflow_nf(n_seeds=2, variants=[{"variant_name": "a"}, {"variant_name": "b"}])
    nf = _render(core, doc)
    # 2 parca + 2 lineage process blocks (one per node inside the sub-workflows)
    assert nf.count("workflow runs_v") == 2
    assert "ch_results_v0 = runs_v0(ch_cache_v0)" in nf
    assert "ch_results_v1 = runs_v1(ch_cache_v1)" in nf


def test_parca_script_carries_the_strain_flags_and_the_hydrate_step(core) -> None:
    """--new-genes/--bundle-overrides go to v2ecoli-parca ONLY: build_cache.py's CLI
    has neither (viva-api#410, a real crash). And the hydrate step is not optional --
    v2ecoli-parca emits only the raw parca_state.pkl."""
    step = ParcaTaskStep(config={"new_genes": "vio", "bundle_overrides": "/m.json"}, core=core)
    script = step.nextflow_script()
    parca_cmd, _, rest = script.partition("&&")
    assert "--new-genes vio" in parca_cmd and "--bundle-overrides /m.json" in parca_cmd
    assert "--new-genes" not in rest and "--bundle-overrides" not in rest
    assert "build_cache.py" in rest and "gzip" in rest


def test_parca_omits_the_off_sentinel(core) -> None:
    """'off' IS v2ecoli-parca's default for --new-genes, so passing it and omitting
    it are the same build. Omit, so the command stays byte-identical to a plain one."""
    assert "--new-genes" not in ParcaTaskStep(config={"new_genes": "off"}, core=core).nextflow_script()


def test_task_declarations_refuse_to_run_in_process(core) -> None:
    """These nodes carry no simulation logic. Running one would advance global_time
    and produce nothing while exiting 0 -- the silent-success class this plan is
    organised around."""
    with pytest.raises(RuntimeError, match="rendered"):
        ParcaTaskStep(config={}, core=core).update({})
    with pytest.raises(RuntimeError):
        AnalysisTaskStep(config={}, core=core).update({})


# --- the gather, and why it is gated ---------------------------------------


def test_gather_is_off_by_default() -> None:
    assert "analysis" not in build_workflow_nf(n_seeds=2)["state"]


def test_each_variant_collects_its_seeds_into_ONE_channel(core) -> None:
    """The fan-in, and why it needs nesting: M sibling channels cannot be wired to
    one port (a port maps to ONE store path -- the flat spelling raises
    `TypeError: unhashable type: 'list'` inside realize, before the renderer runs).
    A nested composite emits `take:`/`emit:` with CHAINED BINARY mixes, so M
    channels arrive at the parent as one."""
    nf = _render(core, build_workflow_nf(n_seeds=3, include_analysis=True))
    assert "workflow runs_v0 {" in nf
    assert "take:" in nf and "emit:" in nf
    assert nf.count("_merged = _merged.mix(") == 2      # 3 units -> 2 binary mixes
    assert "_merged.collect()" in nf


def test_analysis_takes_one_port_per_variant(core) -> None:
    """N named ports, each fed by a variant's already-collected channel. One port
    wired to N stores is what the document model cannot express."""
    nf = _render(core, build_workflow_nf(
        n_seeds=2, include_analysis=True,
        variants=[{"variant_name": "a"}, {"variant_name": "b"}]))
    assert "analysis(ch_results_v0, ch_results_v1" in nf


def test_renders_at_run4_scale_without_hitting_the_255_wall(core) -> None:
    """go/no-go 4: 84 variants x 4 seeds = 336 lineages. The unrolled form died at
    256 arguments (`bad parameter count 257`); nesting keeps the parent call at one
    argument per VARIANT and the mixes binary."""
    variants = [{"variant_name": f"v{i}"} for i in range(84)]
    nf = _render(core, build_workflow_nf(n_seeds=4, include_analysis=True, variants=variants))
    assert nf.count("workflow runs_v") == 84
    assert nf.count("= lineage_s") == 336
    # chained binary, never one n-ary call
    assert ".mix(" in nf and all("," not in seg[:seg.index(")")] for seg in nf.split(".mix(")[1:])
