"""Integration tests for the representative-growing-population runner (#225 #1).

These exercise the full path that breaks the single-lineage biomass plateau:

  run_multigen_sqlite (single_daughters) ── injects ──▶ lineage.doublings store
                                                          │ reads
                          PopulationAggregator (representative_doubling) ◀──┘
                                  │ writes population.biomass_concentration_gL
                                  ▼
                        ACCUMULATES 2x per generation (vs fixed-mode plateau)

A tiny synthetic "growing" composite stands in for the full ParCa whole-cell
model (which takes ~one generation per ~2500 ticks → minutes/gen). It runs the
REAL :class:`PopulationAggregator` each tick over its own ``agents`` /
``population`` / ``lineage`` stores and divides on a fixed schedule with a
mass-conserving halving (mother at 2x birth mass → two daughters at birth mass),
so the continuity-across-division invariant is exercised with the real
aggregator + the real runner, not a re-implementation.

We assert on the recorded biomass series + the injected doubling count directly,
so the test does not depend on reading the SQLite db back.
"""
from __future__ import annotations

import pytest

from v2ecoli.core import build_core
from v2ecoli.library.sqlite_run import run_multigen_sqlite
from v2ecoli.steps.population_aggregator import (
    GROWTH_MODE_DOUBLING,
    GROWTH_MODE_FIXED,
    PopulationAggregator,
)


@pytest.fixture(scope="module")
def core():
    return build_core()


# Synthetic-cell parameters: birth mass 1e15 fg (1 g), grows +1e14 fg/tick, so
# at age=10 the cell reaches 2e15 fg (2x birth) and divides into two daughters
# of 1e15 fg each (mass-conserving halving).
BIRTH_MASS_FG = 1.0e15
GROWTH_FG_PER_TICK = 1.0e14
DIVIDE_PERIOD = 10


class _GrowingPopComposite:
    """Single-lineage composite that grows + divides, running the real aggregator.

    Mirrors the v2ecoli inner Division convention (mother always named "0", so
    every division yields daughters "00"/"01" — the followed id is REUSED for
    "…0"). Records ``population.biomass_concentration_gL`` per tick so the test
    can inspect the trajectory the coupled reactor would see.
    """

    def __init__(self, aggregator: PopulationAggregator):
        self.aggregator = aggregator
        self._age = 0
        self._t = 0
        self.state = {
            "agents": {"0": self._cell(BIRTH_MASS_FG)},
            "population": {
                "total_biomass_gDW": 0.0, "cell_count": 0.0,
                "biomass_concentration_gL": 0.0, "OD600": 0.0,
            },
            "lineage": {"generation": 1.0, "doublings": 0.0},
            "global_time": 0.0,
        }
        self.biomass_series: list[float] = []
        self.doublings_series: list[float] = []
        self._aggregate()

    @staticmethod
    def _cell(mass_fg: float) -> dict:
        return {"listeners": {"mass": {"cell_mass": float(mass_fg)}}}

    def run(self, n: int) -> None:
        for _ in range(int(n)):
            self._age += 1
            self._t += 1
            agents = self.state["agents"]
            cur = next(iter(agents))
            if self._age >= DIVIDE_PERIOD:
                # Division: mother (at 2x birth) → two daughters at birth mass.
                self._age = 0
                del agents[cur]
                agents["00"] = self._cell(BIRTH_MASS_FG)
                agents["01"] = self._cell(BIRTH_MASS_FG)
            else:
                agents[cur]["listeners"]["mass"]["cell_mass"] = (
                    BIRTH_MASS_FG + self._age * GROWTH_FG_PER_TICK
                )
            self._aggregate()
        self.state["global_time"] = float(self._t)

    def _aggregate(self) -> None:
        out = self.aggregator.next_update(1.0, self.state)
        self.state["population"].update(out["population"])
        self.biomass_series.append(self.state["population"]["biomass_concentration_gL"])
        self.doublings_series.append(float(self.state["lineage"]["doublings"]))

    # prune_to_followed_lineage pokes this; a no-op fake is fine.
    def find_instance_paths(self, state):
        return {}


def _drive(core, mode: str, tmp_path, *, max_generations: int = 4):
    agg = PopulationAggregator(
        config={"cells_per_agent": 1.0, "population_growth_mode": mode}, core=core,
    )
    comp = _GrowingPopComposite(agg)
    db_file = tmp_path / f"{mode}.db"
    result = run_multigen_sqlite(
        comp,
        run_id=f"doubling-test-{mode}",
        db_file=str(db_file),
        emit_paths=["listeners.mass.cell_mass"],
        extra_root_paths=["population/biomass_concentration_gL"],
        max_steps=DIVIDE_PERIOD * (max_generations + 1),  # cross all divisions
        max_generations=max_generations,
        chunk=5,
        initial_agent_id="0",
        single_daughters=True,
        core=core,
    )
    return comp, result


def _generation_peaks(biomass: list[float], doublings: list[float]) -> dict[float, float]:
    """Max biomass observed at each distinct doubling level (≈ per-generation peak)."""
    peaks: dict[float, float] = {}
    for b, d in zip(biomass, doublings):
        peaks[d] = max(peaks.get(d, 0.0), b)
    return peaks


def test_runner_injects_doubling_count_per_generation(core, tmp_path):
    """The runner advances lineage.doublings once per generation (= gen - 1),
    counting EVERY division including the id-reuse gen>=2 case (#225 #2 fix in
    sqlite_run)."""
    comp, result = _drive(core, GROWTH_MODE_DOUBLING, tmp_path, max_generations=4)
    assert result["generations"] == [1, 2, 3, 4], result["generations"]
    # Final injected doubling count = max_generations - 1.
    assert comp.state["lineage"]["doublings"] == pytest.approx(3.0)
    # The doubling count actually reached the aggregator at each level.
    assert set(comp.doublings_series) == {0.0, 1.0, 2.0, 3.0}


def test_doubling_mode_accumulates_across_generations(core, tmp_path):
    """representative_doubling: per-generation peak biomass GROWS ~2x each gen
    (exponential accumulation), and the trajectory is continuous (never drops by
    more than a single growth step — no halving at division)."""
    comp, _ = _drive(core, GROWTH_MODE_DOUBLING, tmp_path, max_generations=4)
    peaks = _generation_peaks(comp.biomass_series, comp.doublings_series)
    # gen1..gen4 peaks at doublings 0,1,2,3 ≈ 2.0, 4.0, 8.0, 16.0 g/L.
    assert peaks[0.0] == pytest.approx(2.0)
    assert peaks[1.0] == pytest.approx(2.0 * peaks[0.0])
    assert peaks[2.0] == pytest.approx(2.0 * peaks[1.0])
    assert peaks[3.0] == pytest.approx(2.0 * peaks[2.0])

    # Continuity: no consecutive halving (the fixed-mode signature). The biomass
    # series strictly trends up — every consecutive ratio stays well above 0.5.
    series = comp.biomass_series
    min_ratio = min(b / a for a, b in zip(series, series[1:]) if a > 0)
    assert min_ratio > 0.9, f"discontinuous drop in doubling biomass: {min_ratio}"


def test_fixed_mode_plateaus_on_same_run(core, tmp_path):
    """Same synthetic run in the DEFAULT fixed mode: per-generation peak biomass
    is CONSTANT (plateau) and the series OSCILLATES (halves at each division) —
    the documented architectural gap the doubling mode breaks."""
    comp, _ = _drive(core, GROWTH_MODE_FIXED, tmp_path, max_generations=4)
    peaks = _generation_peaks(comp.biomass_series, comp.doublings_series)
    # In fixed mode the runner still injects doublings (harmless), but the
    # aggregator ignores it — every generation peaks at the SAME ~2.0 g/L.
    for d, peak in peaks.items():
        assert peak == pytest.approx(2.0), f"fixed-mode peak at doublings={d} drifted: {peak}"
    # Oscillation: the series halves at each division (a real <0.6 drop exists).
    series = comp.biomass_series
    min_ratio = min(b / a for a, b in zip(series, series[1:]) if a > 0)
    assert min_ratio < 0.6, f"fixed mode unexpectedly continuous: {min_ratio}"


def test_doubling_final_biomass_far_exceeds_fixed(core, tmp_path):
    """End-to-end headline: after the same number of generations the doubling
    mode's peak biomass is 2^(G-1)x the fixed-mode plateau (here 8x at G=4)."""
    comp_d, _ = _drive(core, GROWTH_MODE_DOUBLING, tmp_path, max_generations=4)
    comp_f, _ = _drive(core, GROWTH_MODE_FIXED, tmp_path, max_generations=4)
    assert max(comp_d.biomass_series) == pytest.approx(8.0 * max(comp_f.biomass_series))
