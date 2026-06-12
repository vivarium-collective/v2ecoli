"""Regression: the multigen runners must end EVERY generation at its division.

Root cause (showcase-2 baseline, 2026-06): the inner Division step always names
its mother ``"0"`` (``agent_id`` defaults to "0" in both
``v2ecoli/steps/division.py`` and ``composites/_helpers.py``), so a division
produces daughters ``"00"`` / ``"01"`` regardless of the mother's lineage id.

Generation 1 (followed ``"0"``) divides into ``"00"`` / ``"01"`` — the followed
id ``"0"`` vanishes, so ``run_multigen_parquet`` / ``run_multigen_xarray``
caught it.  But generation 2 (followed ``"00"``) divides into ``"00"`` / ``"01"``
— the followed id ``"00"`` is REUSED by a daughter, so ``followed in agents``
stayed True, the division-handler branch never fired, and the gen-3 daughter was
emitted under gen-2's generation/agent_id partition (a mid-partition dry-mass
fold-change reset).

These tests drive the runners with a tiny fake composite that reproduces the
id-reuse division so the bug is exercised without a real ParCa cell.
"""
from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.library.parquet_run import run_multigen_parquet


class _FakeComposite:
    """Single-cell composite that "divides" on a schedule, reusing the mother
    id for daughter ``…0`` exactly like the real inner Division step.

    Each ``run(n)`` advances ``n`` ticks; dry_mass ramps linearly and a division
    fires when the followed cell's local age crosses ``divide_period``.  At a
    division the followed agent ``m`` is removed and ``m0`` / ``m1`` are added —
    so for mother ``"0"`` the daughter is ``"00"`` (vanishes) and for mother
    ``"00"`` the daughter is ``"000"``… NO: the real step uses agent_id="0"
    always, so daughters are ALWAYS ``"00"`` / ``"01"``.  We model that exactly.
    """

    def __init__(self, divide_period: int = 200, dry0: float = 350.0):
        self.divide_period = divide_period
        self.dry0 = dry0
        self._age = 0  # ticks since this cell's birth
        self._t = 0
        # The inner composite always names its sole cell "0".
        self.state = {"agents": {"0": self._cell(dry0)}, "global_time": 0.0}

    @staticmethod
    def _cell(dry_mass: float) -> dict:
        return {
            "listeners": {"mass": {"dry_mass": float(dry_mass)}},
            "bulk": np.array([(b"X", 10)], dtype=[("id", "S1"), ("count", "i8")]),
        }

    def run(self, n: int) -> None:
        for _ in range(int(n)):
            self._age += 1
            self._t += 1
            agents = self.state["agents"]
            # Only one agent is ever present (single-cell composite + pruning).
            cur_id = next(iter(agents))
            if self._age >= self.divide_period:
                # DIVISION: remove mother, add daughters "00"/"01" (agent_id="0").
                self._age = 0
                del agents[cur_id]
                agents["00"] = self._cell(self.dry0)
                agents["01"] = self._cell(self.dry0)
            else:
                dry = self.dry0 + self._age * 2.0
                agents[cur_id]["listeners"]["mass"]["dry_mass"] = dry
        self.state["global_time"] = float(self._t)

    # prune_to_followed_lineage pokes this; a no-op fake is fine.
    def find_instance_paths(self, state):
        return {}

    core = None


def _gens_in_partitions(out_dir: str, experiment_id: str) -> dict[int, dict]:
    """Read the hive parquet and return {generation: {agent_id, t_min, t_max,
    dry_min, dry_max, n_fold_resets}} where a "fold reset" is a dry_mass drop
    between consecutive rows (the mid-partition daughter signature)."""
    import glob
    import polars as pl

    base = f"{out_dir}/{experiment_id}/history/experiment_id={experiment_id}"
    out: dict[int, dict] = {}
    for gdir in sorted(glob.glob(base + "/variant=*/lineage_seed=*/generation=*")):
        gen = int(gdir.split("generation=")[1].split("/")[0])
        files = glob.glob(gdir + "/agent_id=*/*.pq")
        if not files:
            continue
        agent_id = files[0].split("agent_id=")[1].split("/")[0]
        df = pl.read_parquet(files).sort("global_time")
        dry = df["listeners__mass__dry_mass"].to_numpy()
        resets = int(np.sum(np.diff(dry) < -50.0))  # a halving is a big drop
        out[gen] = {
            "agent_id": agent_id,
            "t_min": float(df["global_time"].min()),
            "t_max": float(df["global_time"].max()),
            "dry_min": float(dry.min()),
            "dry_max": float(dry.max()),
            "n_fold_resets": resets,
        }
    return out


def test_every_generation_ends_at_its_division(tmp_path):
    """PASS criteria: each generation partition holds exactly ONE cell — a
    monotonic dry_mass ramp with NO mid-partition fold-change reset — and the
    3rd cell is its own generation=3 partition, never folded into gen-2."""
    from v2ecoli.core import build_core
    core = build_core()
    comp = _FakeComposite(divide_period=200, dry0=350.0)
    res = run_multigen_parquet(
        comp,
        experiment_id="regress",
        out_dir=str(tmp_path),
        emit_paths=["listeners.mass.dry_mass"],
        max_steps=650,            # crosses 3 divisions (at 200/400/600)
        max_generations=4,
        chunk=20,
        single_daughters=True,
        threaded=False,
        core=core,
    )

    # Before the fix: generations == [1, 2] (gen-2 ran to max_steps with the
    # gen-3 daughter folded in).  After: at least [1, 2, 3].
    assert res["generations"][:3] == [1, 2, 3], res["generations"]

    gens = _gens_in_partitions(str(tmp_path), "regress")
    assert set(gens) >= {1, 2, 3}, f"missing generations: {sorted(gens)}"

    # Each generation partition is a single cell: no mid-partition halving.
    for g, info in gens.items():
        assert info["n_fold_resets"] == 0, (
            f"generation={g} partition contains a mid-partition dry_mass reset "
            f"(daughter folded in): {info}")

    # agent_id increments per generation: 0 -> 00 -> 000 ...
    assert gens[1]["agent_id"] == "0"
    assert gens[2]["agent_id"] == "00"
    assert gens[3]["agent_id"] == "000"
