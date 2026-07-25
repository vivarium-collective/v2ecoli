import pytest


def test_run_bench_simple_free_colony_smoke():
    from v2ecoli.colony_bench.harness import run_bench
    out = run_bench("free_colony", "simple", n_ticks=5, dt=1.0, seed=0,
                    builder_kwargs={"n_cells": 2, "env_size": 30})
    assert out["n_final"] >= 2
    assert len(out["trajectory"]) == 5
    assert "phenotypes" in out and "n_division_events" in out["phenotypes"]


@pytest.mark.wcm
def test_run_bench_wcm_daughter_machine_smoke():
    from v2ecoli.colony_bench.harness import run_bench
    out = run_bench("daughter_machine", "wcm", n_ticks=2, dt=1.0, seed=0)
    assert out["n_final"] >= 1
    assert len(out["trajectory"]) == 2
