import pytest
from v2ecoli.workflow.lineage import LineageProcess


def _make(monkeypatch, generations, divide_after=2):
    """Build a LineageProcess whose _build_generation/_run_until_division are
    stubbed so we can test generation counting without a real cell composite."""
    lp = LineageProcess.__new__(LineageProcess)
    # Minimal config + state normally set by Process.__init__/initialize.
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "baseline", "config_overrides": {}, "generations": generations,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0,
    }
    lp.initialize(lp.config)
    calls = {"built": 0}

    def fake_build():
        calls["built"] += 1
        lp._gen_elapsed = 0.0

    def fake_run_until_division(interval):
        lp._gen_elapsed += interval
        divided = lp._gen_elapsed >= divide_after
        daughter = {"bulk": {}, "unique": {}} if divided else None
        return divided, daughter, 100.0 + lp._generation

    monkeypatch.setattr(lp, "_build_generation", fake_build)
    monkeypatch.setattr(lp, "_run_until_division", fake_run_until_division)
    return lp, calls


def test_completes_after_generations(monkeypatch):
    lp, calls = _make(monkeypatch, generations=3, divide_after=2)
    out = {}
    for _ in range(20):
        out = lp.update({}, 1.0)
        if out.get("complete"):
            break
    assert out["complete"] is True
    assert len(lp._summaries) == 3            # 3 generations recorded
    assert [s["generation"] for s in lp._summaries] == [0, 1, 2]
    # agent_id advanced one phylogeny step per completed generation: 0 -> 00 -> 000
    assert [s["agent_id"] for s in lp._summaries] == ["0", "00", "000"]
    assert all(s["divided"] for s in lp._summaries)


def test_single_daughters_false_not_implemented(monkeypatch):
    lp, _ = _make(monkeypatch, generations=2)
    lp.config["single_daughters"] = False
    with pytest.raises(NotImplementedError):
        lp.update({}, 1.0)


def test_daughter_carry_forward_orchestration(monkeypatch):
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "baseline", "config_overrides": {}, "generations": 2,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0,
    }
    lp.initialize(lp.config)

    builds = []  # (generation, agent_id, carry_state) seen at each build

    def fake_build():
        builds.append((lp._generation, lp._agent_id, lp._carry_state))
        lp._gen_elapsed = 0.0

    daughter = {"bulk": {"marker": 1}, "unique": {}}

    def fake_run_until_division(interval):
        lp._gen_elapsed += interval
        # Always "divide" after one tick, handing back a synthetic daughter.
        return True, daughter, 100.0

    monkeypatch.setattr(lp, "_build_generation", fake_build)
    monkeypatch.setattr(lp, "_run_until_division", fake_run_until_division)

    out = {}
    for _ in range(10):
        out = lp.update({}, 1.0)
        if out.get("complete"):
            break

    assert out["complete"] is True
    # Generation 0 built with no carry; generation 1 built carrying the daughter.
    assert len(builds) == 2
    assert builds[0] == (0, "0", None)
    assert builds[1][0] == 1
    assert builds[1][1] == "00"            # agent_id advanced via daughter_phylogeny_id
    assert builds[1][2] is daughter        # carry_state handed to the next build
