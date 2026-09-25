"""run_multigen_parquet in multi-founder mode, driven by a fake composite.

The fake holds two founder lineages that divide on DIFFERENT schedules, the
case one scalar generation counter cannot represent. Each division replaces the
lineage's agent with its ``...0`` daughter, as the in-composite LineageBookkeeper
leaves it. Checks the hive layout (one partition per founder per generation,
keyed by the real phylogeny id, plus one ``population`` partition for the
document roots), the per-founder generation and row counts, the generation cap,
and the refusals. No ParCa cell needed.
"""
from __future__ import annotations

import glob

import pytest

from v2ecoli.core import build_core
from v2ecoli.library.parquet_run import (
    POPULATION_PARTITION_AGENT_ID,
    run_multigen_parquet,
)
from v2ecoli.steps.population_aggregator import LINEAGE_FOUNDER_ID_LENGTH_KEY

PERIODS = {"0": 250, "1": 150}   # ticks between divisions, per founder


@pytest.fixture(scope="module")
def core():
    return build_core()


class _FakeFounders:
    """Two founder lineages; each divides every PERIODS[founder] ticks."""

    def __init__(self, multi: bool = True):
        self._t = 0
        self.state = {
            "agents": {f: self._cell(f, 0) for f in PERIODS},
            "lineage": ({LINEAGE_FOUNDER_ID_LENGTH_KEY: 1.0} if multi else {}),
            "population": {"cell_count": 2.0},
            "global_time": 0.0,
        }

    @staticmethod
    def _cell(agent_id: str, age: int) -> dict:
        return {"listeners": {"mass": {"dry_mass": 300.0 + age}},
                "tag": {"agent_id": agent_id}}

    def run(self, n: int) -> None:
        agents = self.state["agents"]
        for _ in range(int(n)):
            self._t += 1
            for aid in list(agents):
                founder = aid[0]
                if self._t % PERIODS[founder] == 0:
                    del agents[aid]
                    agents[aid + "0"] = self._cell(aid + "0", 0)
                else:
                    agents[aid]["listeners"]["mass"]["dry_mass"] += 1.0
        self.state["population"]["cell_count"] = float(self._t)
        self.state["global_time"] = float(self._t)

    def find_instance_paths(self, state):
        return {}

    core = None


def _run(core, tmp_path, comp=None, **kw):
    args = dict(experiment_id="mf", out_dir=str(tmp_path),
                emit_paths=["listeners.mass.dry_mass"],
                extra_root_paths=["population/cell_count"],
                max_steps=600, max_generations=10, chunk=10,
                single_daughters=True, threaded=False, core=core)
    args.update(kw)
    return run_multigen_parquet(comp or _FakeFounders(), **args)


def _partitions(tmp_path) -> dict[tuple[int, str], int]:
    import polars as pl
    out = {}
    for d in glob.glob(f"{tmp_path}/mf/history/**/generation=*/agent_id=*", recursive=True):
        gen = int(d.split("generation=")[1].split("/")[0])
        aid = d.split("agent_id=")[1].split("/")[0]
        files = glob.glob(d + "/*.pq")
        out[(gen, aid)] = pl.read_parquet(files).height if files else 0
    return out


def test_each_founder_gets_its_own_generations_and_partitions(core, tmp_path):
    res = _run(core, tmp_path)
    # "0" divides at 250 and 500; "1" at 150, 300, 450 and 600.
    assert res["generations"] == {"0": [1, 2, 3], "1": [1, 2, 3, 4, 5]}
    assert res["stopped"] == "max_steps" and res["steps"] == 600
    parts = _partitions(tmp_path)
    expected = {(1, "0"), (2, "00"), (3, "000"),
                (1, "1"), (2, "10"), (3, "100"), (4, "1000"), (5, "10000"),
                (0, POPULATION_PARTITION_AGENT_ID)}
    assert set(parts) == expected
    # One row per chunk per founder; the population rows are written once.
    assert res["rows"] == {"0": 60, "1": 60} and res["population_rows"] == 60
    assert sum(n for (g, a), n in parts.items() if a.startswith("0")) == 60
    assert sum(n for (g, a), n in parts.items() if a.startswith("1")) == 60
    assert parts[(0, POPULATION_PARTITION_AGENT_ID)] == 60


def test_population_rows_carry_the_document_roots_not_agent_state(core, tmp_path):
    import polars as pl
    _run(core, tmp_path)
    pop = pl.read_parquet(glob.glob(
        f"{tmp_path}/mf/history/**/agent_id={POPULATION_PARTITION_AGENT_ID}/*.pq",
        recursive=True))
    assert "population__cell_count" in pop.columns
    assert not any(c.startswith("listeners__") for c in pop.columns)
    founder = pl.read_parquet(glob.glob(f"{tmp_path}/mf/history/**/agent_id=1/*.pq",
                                        recursive=True))
    assert "listeners__mass__dry_mass" in founder.columns
    assert "population__cell_count" not in founder.columns


def test_stops_when_any_founder_divides_past_the_cap(core, tmp_path):
    res = _run(core, tmp_path, max_generations=2)
    # "1" reaches gen 2 at 150 and would pass the cap at 300; "0" reached gen 2 at 250.
    assert res["steps"] == 300
    assert res["generations"] == {"0": [1, 2], "1": [1, 2]}
    assert "'1'" in res["stopped"] and "max_generations=2" in res["stopped"]


def test_multi_founder_needs_explicit_emit_paths(core, tmp_path):
    with pytest.raises(ValueError, match="explicit, reduced emit_paths"):
        _run(core, tmp_path, emit_paths=[])


def test_single_lineage_runners_refuse_multi_founder_at_start(tmp_path):
    from v2ecoli.library.sqlite_run import run_multigen_sqlite
    comp = _FakeFounders()
    with pytest.raises(NotImplementedError, match="run_multigen_parquet"):
        run_multigen_sqlite(comp, run_id="mf", db_file=str(tmp_path / "x.db"),
                            emit_paths=["listeners.mass.dry_mass"], max_steps=10)
    assert comp._t == 0, "refused before running a single tick"


def test_without_the_lineage_key_the_single_founder_path_runs(core, tmp_path):
    """Control: the same two-agent fake WITHOUT founder_id_length takes the
    unchanged single-lineage path, which follows one agent only."""
    res = _run(core, tmp_path, comp=_FakeFounders(multi=False), max_steps=100,
               single_daughters=False)
    assert isinstance(res["generations"], list)
    assert "rows" not in res
