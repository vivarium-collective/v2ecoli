import textwrap

import pytest

import scripts._compare.runner as runner
from scripts._compare.study_spec import StudySpec


def _spec(name="basal", condition="basal", seeds=1, gens=4, cards=("config", "parca", "standard")):
    return StudySpec(name=name, condition=condition, seeds=seeds, gens=gens,
                     cards=list(cards), invest_name="v2ecoli-vecoli-comparison",
                     v2_cache="out/cache_full", ve_cache="out/compare_harness/vecoli_parca",
                     study_path="/x/study.yaml")


def test_run_engines_passes_condition_seeds_gens_and_store_dir(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **k: calls.append(argv))
    runner._run_engines(_spec(name="basal_4x4", condition="basal", seeds=4, gens=4),
                        out="out/x", mode="ray")
    assert len(calls) == 2
    v2, ve = calls
    # both engines simulate the BIOLOGICAL condition, store under the study NAME
    assert v2[v2.index("--composite") + 1] == "v2ecoli"
    assert v2[v2.index("--condition") + 1] == "basal"
    assert v2[v2.index("--n-seeds") + 1] == "4"
    assert v2[v2.index("--out-root") + 1] == "out/x/basal_4x4"
    assert "--match-initial-state" in v2 and v2[v2.index("--mode") + 1] == "ray"
    assert ve[ve.index("--composite") + 1] == "vecoli"
    assert ve[ve.index("--vecoli-source") + 1] == "vivarium-process"
    assert ve[ve.index("--out-root") + 1] == "out/x/basal_4x4"
    # v2 max-steps = gens*PER_GEN (total); vEcoli = PER_GEN (per-gen)
    assert v2[v2.index("--max-steps") + 1] == str(4 * runner.PER_GEN_STEPS)
    assert ve[ve.index("--max-steps") + 1] == str(runner.PER_GEN_STEPS)


def test_run_study_renders_single_study_and_materializes(monkeypatch):
    seq = []
    monkeypatch.setattr(runner, "_run_engines", lambda spec, out, mode: seq.append(("engines", spec.name)))
    monkeypatch.setattr(runner, "_render",
                        lambda inv, out, ms, study=None: seq.append(("render", inv, study)))
    monkeypatch.setattr(runner, "materialize_study", lambda spec: seq.append(("materialize", spec.name)))
    rc = runner.run_study(_spec(name="basal"), out="out/x", mode="serial")
    assert rc == 0
    assert seq == [("engines", "basal"),
                   ("render", "v2ecoli-vecoli-comparison", "basal"),  # render ONLY this study
                   ("materialize", "basal")]


def test_run_study_render_only_skips_engines(monkeypatch):
    seq = []
    monkeypatch.setattr(runner, "_run_engines", lambda *a, **k: seq.append("engines"))
    monkeypatch.setattr(runner, "_render", lambda *a, **k: seq.append("render"))
    monkeypatch.setattr(runner, "materialize_study", lambda spec: seq.append("materialize"))
    runner.run_study(_spec(), out="out/x", render_only=True)
    assert "engines" not in seq
    assert seq == ["render", "materialize"]


def test_run_investigation_loops_studies(monkeypatch, tmp_path):
    # config-is-the-unit model (Task 4): run_investigation builds specs from
    # comparison.configs[], not from a `members:`/`studies:` name list.
    ws = tmp_path / "workspace"
    inv = ws / "investigations/v2ecoli-vecoli-comparison"
    inv.mkdir(parents=True)
    (inv / "investigation.yaml").write_text(textwrap.dedent("""
        name: v2ecoli-vecoli-comparison
        comparison:
          defaults: {cards: [config, parca, standard], seeds: 1, gens: 4}
          configs:
          - {name: basal, config: basal}
          - {name: with_aa, config: with_aa}
    """), encoding="utf-8")
    ran, rendered, mat = [], [], []
    monkeypatch.setattr(runner, "_run_engines", lambda spec, out, mode: ran.append(spec.name))
    monkeypatch.setattr(runner, "_render", lambda inv_ref, out, ms, study=None: rendered.append((inv_ref, ms)))
    monkeypatch.setattr(runner, "materialize_study", lambda spec: mat.append(spec.name))
    rc = runner.run_investigation(str(inv), out="out/x", mode="ray")
    assert rc == 0
    assert ran == ["basal", "with_aa"]
    assert rendered == [(str(inv), 1)]          # whole-investigation render once
    assert mat == ["basal", "with_aa"]


def test_run_investigation_empty_raises(monkeypatch, tmp_path):
    inv = tmp_path / "workspace/investigations/v2ecoli-vecoli-comparison"
    inv.mkdir(parents=True)
    (inv / "investigation.yaml").write_text("name: x\nstudies: []\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        runner.run_investigation(str(inv))
