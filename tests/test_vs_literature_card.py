# tests/test_vs_literature_card.py
"""Tests for the Gen 2 ``VsLiteratureCard(ReportCardStep)``.

Mirrors ``test_vs_vecoli_card`` / ``test_tests_card`` (registration, applies,
build returns a valid verdict) and adds the migration gate: the card's verdict
must be **identical** to the pre-migration golden pinned in
``test_basal_vs_literature`` — the correctness boundary for moving the basal
``vs_literature`` card off the standalone render script onto the per-study
report-card surface.

Needs the installed ecoli-sources validation bundle + the committed
``tests/fixtures/population_phenotype_basal/model_*.json`` fixtures (same inputs
the standalone card + golden test use).
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from v2ecoli.workflow.report_cards import StudyContext
from v2ecoli.workflow.report_cards.vs_literature_card import VsLiteratureCard

_FIXTURES_REL = "tests/fixtures/population_phenotype_basal"

# The pre-migration golden: the axis→verdict map produced by the standalone
# scripts/render_basal_vs_literature.build() → grade_card(). Behavior-preserving
# means the migrated card reproduces this EXACTLY (a 1% Δ still grades the same).
_GOLDEN_OVERALL = "mismatch"
_GOLDEN_AXES = {
    "physiology.growth_rate": "within_tol",
    "physiology.biomass_yield": "mismatch",
    "physiology.glucose_uptake": "mismatch",
    "metabolism.o2_uptake": "mismatch",
    "metabolism.co2_evolution": "mismatch",
    "metabolism.acetate_secretion": "mismatch",
    "metabolism.central_carbon_flux": "mismatch",
    "metabolism.glycolysis_split": "within_tol",
    "metabolism.isocitrate_split": "drift",
    "metabolism.accoa_split": "mismatch",
    "proteome.abundance": "drift",
    "composition.macromolecular": "drift",
    "pools.per_metabolite": "within_tol",
}


def _ctx(tmp_path, refs=None):
    """A StudyContext whose ws_root is the real repo (so the fixtures resolve),
    with a study.yaml written under tmp_path only to exercise spec loading."""
    sd = tmp_path / "workspace" / "studies" / "demo"
    sd.mkdir(parents=True)
    spec = {"name": "Demo"}
    if refs:
        spec["report_card_refs"] = refs
    (sd / "study.yaml").write_text(yaml.safe_dump(spec))
    # ws_root must be the real repo for the fixtures pointer to resolve.
    return StudyContext(study_name="demo", study_dir=sd, spec=spec, ws_root=REPO)


def _axis_verdicts(vjson) -> dict:
    """Flatten a report_card_verdict/v1 dict to {axis_id: verdict}."""
    return {ax["id"]: ax["verdict"]
            for grp in vjson.get("groups", {}).values()
            for ax in grp.get("axes", [])}


def test_registered():
    from v2ecoli.workflow.report_cards import REPORT_CARD_REGISTRY
    assert REPORT_CARD_REGISTRY.get("vs_literature") is VsLiteratureCard


def test_absent_without_ref(core, tmp_path):
    assert VsLiteratureCard({}, core=core).applies(_ctx(tmp_path)) is False


def test_absent_when_ref_dir_missing_fixture(core, tmp_path):
    # A ref that doesn't hold model_physiology.json → does not apply (no throw).
    ctx = _ctx(tmp_path, refs={"vs_literature": "workspace/studies"})
    assert VsLiteratureCard({}, core=core).applies(ctx) is False


def test_applies_for_basal_fixtures(core, tmp_path):
    ctx = _ctx(tmp_path, refs={"vs_literature": _FIXTURES_REL})
    assert VsLiteratureCard({}, core=core).applies(ctx) is True


def test_build_returns_valid_verdict(core, tmp_path):
    ctx = _ctx(tmp_path, refs={"vs_literature": _FIXTURES_REL})
    vjson, html = VsLiteratureCard({}, core=core).build(ctx)
    assert vjson["schema"] == "report_card_verdict/v1"
    assert vjson["reference_model"].startswith("experimental literature")
    assert vjson["model_ref"].startswith("v2ecoli baseline")
    # rich HTML (plots + coverage), not the minimal render_verdict_html
    assert "Basal-condition physiology" in html
    assert "Scope &amp; coverage" in html


def test_verdict_matches_pre_migration_golden(core, tmp_path):
    # THE GATE: the migrated card grades identically to the standalone script.
    ctx = _ctx(tmp_path, refs={"vs_literature": _FIXTURES_REL})
    vjson, _ = VsLiteratureCard({}, core=core).build(ctx)
    assert vjson["overall"] == _GOLDEN_OVERALL
    assert _axis_verdicts(vjson) == _GOLDEN_AXES
