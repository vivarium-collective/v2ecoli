import textwrap

import pytest

from scripts._compare.study_spec import (
    load_investigation, load_study, _spec_from_study, _context, StudySpec)


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")


def _make_invest(tmp_path):
    inv = tmp_path / "workspace/investigations/v2ecoli-vecoli-comparison"
    _write(inv / "investigation.yaml", """
        schema_version: 4
        name: v2ecoli-vecoli-comparison
        comparison:
          vecoli_dir_env: V2E_TEST_FORK
          v2_cache: out/cache_full
          ve_cache: out/compare_harness/vecoli_parca
          defaults: {cards: [config, parca, standard]}
        studies: [basal, basal_4x4, missing_one]
    """)
    _write(inv / "studies/basal/study.yaml", """
        name: basal
        investigation: v2ecoli-vecoli-comparison
        condition: basal
        comparison: {seeds: 1, generations: 4}
    """)
    _write(inv / "studies/basal_4x4/study.yaml", """
        name: basal_4x4
        investigation: v2ecoli-vecoli-comparison
        condition: basal
        comparison: {seeds: 4, generations: 4, cards: [config, parca, statistical]}
    """)
    return inv


def test_load_investigation_yields_specs_in_order_skipping_missing(tmp_path):
    inv = _make_invest(tmp_path)
    ctx, specs = load_investigation(str(inv))
    assert [s.name for s in specs] == ["basal", "basal_4x4"]   # missing_one skipped
    assert ctx["invest_name"] == "v2ecoli-vecoli-comparison"


def test_store_key_vs_condition_separation(tmp_path):
    inv = _make_invest(tmp_path)
    _, specs = load_investigation(str(inv))
    s44 = next(s for s in specs if s.name == "basal_4x4")
    assert s44.name == "basal_4x4"      # store key
    assert s44.condition == "basal"     # biological condition simulated
    assert (s44.seeds, s44.gens) == (4, 4)


def test_study_inherits_default_cards_when_omitted(tmp_path):
    inv = _make_invest(tmp_path)
    _, specs = load_investigation(str(inv))
    basal = next(s for s in specs if s.name == "basal")
    assert basal.cards == ["config", "parca", "standard"]   # from investigation defaults
    assert basal.graded_cards == ["parca", "standard"]      # parca + standard gate; config informational


def test_study_card_override_and_graded_subset(tmp_path):
    inv = _make_invest(tmp_path)
    _, specs = load_investigation(str(inv))
    s44 = next(s for s in specs if s.name == "basal_4x4")
    assert s44.cards == ["config", "parca", "statistical"]
    assert s44.graded_cards == ["parca", "statistical"]


def test_context_reads_fork_from_named_env(tmp_path, monkeypatch):
    inv = _make_invest(tmp_path)
    monkeypatch.setenv("V2E_TEST_FORK", "/some/vEcoli")
    ctx = _context(inv)
    assert ctx["fork"] == "/some/vEcoli"
    assert ctx["v2_cache"] == "out/cache_full"
    assert ctx["ve_cache"] == "out/compare_harness/vecoli_parca"


def test_load_study_by_path_resolves_context(tmp_path):
    inv = _make_invest(tmp_path)
    spec = load_study(str(inv / "studies/basal_4x4"))
    assert isinstance(spec, StudySpec)
    assert spec.name == "basal_4x4" and spec.condition == "basal"


def test_load_investigation_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_investigation(str(tmp_path / "nope"))


def test_load_study_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_study(str(tmp_path / "studies/ghost"))


def test_study_without_condition_fails_loud(tmp_path):
    sp = tmp_path / "studies/x/study.yaml"
    _write(sp, "name: x\ncomparison: {seeds: 1, generations: 4}\n")  # no condition
    with pytest.raises(ValueError, match="no `condition`"):
        load_study(str(sp))


def test_study_with_nonpositive_seeds_fails_loud(tmp_path):
    sp = tmp_path / "studies/x/study.yaml"
    _write(sp, "name: x\ncondition: basal\ncomparison: {seeds: 0, generations: 4}\n")
    with pytest.raises(ValueError, match=">= 1"):
        load_study(str(sp))
