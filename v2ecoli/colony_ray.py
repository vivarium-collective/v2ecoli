"""Ray-actor-per-cell colony driver — one whole-cell E. coli per Ray actor.

This is the GIL-aware counterpart to the single-process colony in
``v2ecoli/colony.py``. The HPC-readiness study (colonies-01) found that the
process-bigraph composite engine walks all N cells' 55-step EcoliWCM updates
SEQUENTIALLY on one Python thread, so one colony process uses ~1 CPU core
regardless of cell count (finding F-02: "scaling means more PROCESSES, not
more cells/process"; followup ``gil-aware-engine-research``). This module is
that follow-up: each cell runs as its OWN ``@ray.remote`` actor — a separate
worker PROCESS — so N cells advance on N cores concurrently.

Design (mirrors the proven fan-out conventions in
``v2ecoli/library/parallel_seeds.py``):

  * ``CellActor`` (``@ray.remote``) wraps a single-cell v2ecoli ``baseline``
    composite. It builds the heavy composite LAZILY inside the worker (the
    actor class is pickled into a fresh process, so v2ecoli imports must run
    there, not at driver import time). ``step_chunk(seconds)`` advances the
    inner composite and returns a small picklable summary
    (mass / volume / length / divided?) — never the composite itself.
  * ``RayColony`` is the driver. It spins up one actor per initial cell, and
    each tick fires ``actor.step_chunk.remote(dt)`` on EVERY live actor, then
    ``ray.get``s the whole batch — so the wall per tick is the SLOWEST cell,
    not the SUM (the whole point). On a division it kills the mother actor and
    spawns a fresh daughter actor (one daughter keeps the lineage growing; the
    inner WCM's own division produces the genuinely-divided half-mass state).

Core affinity: per the parallel_seeds convention, each actor is pinned to
``cores // peak_actors`` BLAS/OpenMP threads (``num_cpus`` + the OMP/MKL env
knobs) so N actors don't each grab all the box's cores and thrash. We size for
the EXPECTED peak actor count (initial + a generation of divisions), capped to
the machine, so a mid-run division doesn't oversubscribe.

The existing serial path (``v2ecoli.colony.make_colony`` /
``studies/.../sims/run.py``) is untouched; this is an additive runner.

CLI::

    python -m v2ecoli.colony_ray --n-cells 2 --seconds 30 --chunk 5 \
        --cache-dir out/cache --force-divide-after 5
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Thread-cap env keys (same set parallel_seeds pins) — must be set in the
# worker env BEFORE numpy imports so N actors don't oversubscribe the box.
_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)

_RADIUS_UM = 0.5  # E. coli capsule radius, matches EcoliWCM._read_outputs


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce a listener value to a plain float, stripping pint units.

    Mass/volume listeners can be pint ``Quantity`` (e.g. femtogram) at some
    points in the composite lifecycle — ``float(Quantity)`` raises
    ``DimensionalityError``. ``.magnitude`` gives the bare number; otherwise
    a plain ``float()`` works."""
    if value is None:
        return float(default)
    mag = getattr(value, "magnitude", None)
    if mag is not None:
        try:
            return float(mag)
        except (TypeError, ValueError):
            return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _length_from_volume(volume_fl: float) -> float:
    """Capsule length (µm) from volume (fL≈µm³): V=(4/3)πr³+πr²a, l=a+2r."""
    if volume_fl <= 0:
        return 2.0
    a = (volume_fl - (4 / 3) * math.pi * _RADIUS_UM ** 3) / (math.pi * _RADIUS_UM ** 2)
    return max(2 * _RADIUS_UM, a + 2 * _RADIUS_UM)


# ---------------------------------------------------------------------------
# The per-cell actor body. Defined as a plain class so it pickles cleanly into
# the Ray worker; RayColony decorates it with ``ray.remote`` at runtime (after
# ``import ray``) so importing this module never requires Ray to be installed.
# ---------------------------------------------------------------------------
class _CellActorImpl:
    """One whole-cell E. coli model, owned by a single Ray worker process.

    Lazily builds the v2ecoli ``baseline`` composite on first ``step_chunk``
    (heavy imports happen in the worker). Each ``step_chunk`` advances the
    inner composite by ``seconds`` and returns a small picklable dict — the
    composite never crosses the Ray boundary.
    """

    def __init__(self, cell_id: str, seed: int, cache_dir: str,
                 location: tuple[float, float] = (15.0, 15.0),
                 angle: float = 0.0):
        # Light work only — the worker is already pinned via runtime_env, and
        # the heavy composite is built lazily so a just-spawned daughter actor
        # doesn't block the driver's ray.get on the rest of the batch.
        self.cell_id = cell_id
        self.seed = int(seed)
        self.cache_dir = cache_dir or "out/cache"
        self.location = tuple(location)
        self.angle = float(angle)

        self._composite = None
        self._prev_mass = 0.0
        self._prev_volume = 0.0
        self._sim_time = 0.0
        self.pid = os.getpid()

    # -- lazy build (worker-side heavy imports) ----------------------------
    def _build(self) -> None:
        from process_bigraph import Composite
        from v2ecoli.core import build_core
        from v2ecoli.composites.baseline import baseline
        import v2ecoli.types  # noqa: F401  -- registers resolve dispatch

        core = build_core()
        document = baseline(core=core, seed=self.seed, cache_dir=self.cache_dir)
        self._composite = Composite(document, core=core)

        cell = self._composite.state.get("agents", {}).get("0", self._composite.state)
        mass = cell.get("listeners", {}).get("mass", {})
        self._prev_mass = _to_float(mass.get("dry_mass", 0.0))
        self._prev_volume = _to_float(mass.get("volume", 0.0))

    def _inner_cell(self) -> dict:
        return self._composite.state.get("agents", {}).get("0", self._composite.state)

    # -- driver-callable methods ------------------------------------------
    def ping(self) -> dict:
        """Return identity + worker pid so the driver can prove the actor is a
        distinct process. Triggers the lazy build so timing is fair."""
        if self._composite is None:
            self._build()
        return {"cell_id": self.cell_id, "pid": self.pid, "seed": self.seed}

    def request_force_divide(self) -> None:
        """Arm the WCM divide flag on the next step (test hook — natural
        division takes ~2500s sim time). Mirrors run.py's --force-divide."""
        if self._composite is None:
            self._build()
        self._inner_cell()["divide"] = True

    def step_chunk(self, seconds: float) -> dict:
        """Advance the inner composite by ``seconds`` and return a summary.

        Returns a picklable dict:
            cell_id, pid, sim_time, dry_mass, volume, length,
            d_mass, divided (bool), error (str|None)
        ``divided`` True signals the driver to retire this actor and spawn a
        daughter; the inner WCM has produced the divided (half-mass) state.
        """
        t_wall = time.perf_counter()
        if self._composite is None:
            try:
                self._build()
            except Exception as e:  # noqa: BLE001
                return self._summary(error=f"build failed: {type(e).__name__}: {e}")

        agents_before = set((self._composite.state.get("agents") or {}).keys())
        divided = False
        try:
            self._composite.run(seconds)
        except Exception as e:  # noqa: BLE001
            es = str(e).lower()
            if "divide" in es or "division" in es:
                divided = True
            else:
                return self._summary(error=f"run failed: {type(e).__name__}: {e}",
                                     wall_s=time.perf_counter() - t_wall)

        self._sim_time += seconds

        cell = self._inner_cell()
        if cell.get("divide", False):
            divided = True
        agents_after = set((self._composite.state.get("agents") or {}).keys())
        if agents_before and agents_before != agents_after:
            divided = True

        return self._summary(divided=divided, wall_s=time.perf_counter() - t_wall)

    # -- helpers -----------------------------------------------------------
    def _summary(self, *, divided: bool = False, error: str | None = None,
                 wall_s: float = 0.0) -> dict:
        if error is not None or self._composite is None:
            return {
                "cell_id": self.cell_id, "pid": self.pid,
                "sim_time": self._sim_time, "dry_mass": 0.0, "volume": 0.0,
                "length": 2.0, "d_mass": 0.0, "divided": False,
                "location": list(self.location), "wall_s": wall_s, "error": error,
            }
        cell = self._inner_cell()
        mass = cell.get("listeners", {}).get("mass", {})
        dry_mass = _to_float(mass.get("dry_mass", 0.0))
        volume = _to_float(mass.get("volume", 0.0))
        d_mass = dry_mass - self._prev_mass
        self._prev_mass = dry_mass
        self._prev_volume = volume
        return {
            "cell_id": self.cell_id, "pid": self.pid,
            "sim_time": self._sim_time,
            "dry_mass": dry_mass, "volume": volume,
            "length": _length_from_volume(volume),
            "d_mass": d_mass, "divided": divided,
            "location": list(self.location),
            "wall_s": wall_s, "error": None,
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class ColonyTickResult:
    tick: int
    sim_time: float
    wall_s: float
    live_cells: int
    divisions: int
    per_cell: dict = field(default_factory=dict)  # cell_id -> summary


class RayColony:
    """Drives one Ray actor per cell, advancing them concurrently each tick.

    Args:
      n_cells: initial cell count (one actor each).
      seed: base seed; cell ``i`` gets ``seed + i``.
      cache_dir: ParCa cache dir (absolute path recommended — Ray workers
        start in their own cwd).
      env_size: 2D placement extent for initial cell locations (cosmetic; the
        spatial physics layer is not modeled in the Ray driver — cells are
        independent WCMs, which is exactly the embarrassingly-parallel regime
        the GIL finding identified).
      num_threads: BLAS/OMP threads per actor; default ``cores // peak_actors``.
      max_actors: cap on concurrent actors used to size thread affinity so a
        mid-run division doesn't oversubscribe (default: 2× n_cells, capped to
        cpu count).
    """

    def __init__(self, n_cells: int = 2, *, seed: int = 0,
                 cache_dir: str = "out/cache", env_size: float = 30.0,
                 num_threads: int | None = None, max_actors: int | None = None):
        import ray  # local import — Ray only needed for the parallel driver
        self._ray = ray

        self.n_cells = int(n_cells)
        self.seed = int(seed)
        self.cache_dir = os.path.abspath(cache_dir)
        self.env_size = float(env_size)

        cores = os.cpu_count() or 1
        peak = max_actors if max_actors else min(cores, max(1, 2 * self.n_cells))
        self.threads = (int(num_threads) if num_threads and num_threads > 0
                        else max(1, cores // max(1, peak)))

        env_vars = {k: str(self.threads) for k in _THREAD_ENV_KEYS}
        ray.init(ignore_reinit_error=True, log_to_driver=False,
                 runtime_env={"env_vars": env_vars})

        # Decorate the impl with the resolved CPU affinity.
        self._Actor = ray.remote(num_cpus=self.threads)(_CellActorImpl)

        self.actors: dict[str, Any] = {}
        self._next_lineage = 0
        self.tick = 0
        self.sim_time = 0.0
        self.total_divisions = 0
        self.history: list[ColonyTickResult] = []

        self._spawn_initial()

    # -- lifecycle ---------------------------------------------------------
    def _new_id(self) -> str:
        cid = f"cell{self._next_lineage}"
        self._next_lineage += 1
        return cid

    def _spawn_initial(self) -> None:
        for i in range(self.n_cells):
            cid = self._new_id()
            angle = (2 * math.pi * i) / max(1, self.n_cells)
            loc = (self.env_size / 2 + 3 * math.cos(angle),
                   self.env_size / 2 + 3 * math.sin(angle))
            self.actors[cid] = self._Actor.remote(
                cid, self.seed + i, self.cache_dir, loc, angle)

    def _spawn_daughter(self, mother_id: str, mother_summary: dict) -> str:
        """Spawn ONE daughter actor continuing the mother's lineage. The inner
        WCM already produced the half-mass divided state; the daughter starts a
        fresh single-cell composite seeded distinctly so the lineage diverges.
        """
        cid = self._new_id()
        # Distinct seed per daughter so lineages aren't identical RNG streams.
        d_seed = self.seed + 1000 * (self.total_divisions + 1) + len(self.actors)
        loc = tuple(mother_summary.get("location")
                    or (self.env_size / 2, self.env_size / 2))
        self.actors[cid] = self._Actor.remote(
            cid, d_seed, self.cache_dir, loc, 0.0)
        return cid

    def warmup(self) -> dict:
        """Force all actors to build their inner composite up front (so the
        first real tick's wall time is steady-state, not build-dominated) and
        return the worker pids — proof of distinct processes."""
        futs = {cid: a.ping.remote() for cid, a in self.actors.items()}
        pings = {cid: self._ray.get(f) for cid, f in futs.items()}
        return pings

    def request_force_divide(self) -> None:
        """Arm the divide flag on every live actor (test hook)."""
        self._ray.get([a.request_force_divide.remote()
                       for a in self.actors.values()])

    # -- the concurrent step ----------------------------------------------
    def step(self, seconds: float) -> ColonyTickResult:
        """Advance ALL live actors by ``seconds`` concurrently, gather results,
        spawn a daughter for each divided cell. Wall ≈ slowest cell, not sum."""
        ray = self._ray
        t0 = time.perf_counter()

        # Fire every actor's chunk at once, then block on the whole batch so
        # they run concurrently across workers (wall = slowest, not sum).
        futs = {cid: a.step_chunk.remote(seconds) for cid, a in self.actors.items()}
        summaries = {cid: ray.get(f) for cid, f in futs.items()}
        wall = time.perf_counter() - t0

        self.tick += 1
        self.sim_time += seconds

        # Handle divisions: retire mother actor, spawn one daughter.
        divisions = 0
        for cid, s in list(summaries.items()):
            if s.get("error"):
                continue
            if s.get("divided"):
                divisions += 1
                self.total_divisions += 1
                mother = self.actors.pop(cid, None)
                if mother is not None:
                    try:
                        ray.kill(mother)
                    except Exception:  # noqa: BLE001
                        pass
                new_id = self._spawn_daughter(cid, s)
                summaries[cid]["daughter_id"] = new_id

        result = ColonyTickResult(
            tick=self.tick, sim_time=self.sim_time, wall_s=wall,
            live_cells=len(self.actors), divisions=divisions,
            per_cell=summaries)
        self.history.append(result)
        return result

    def run(self, total_seconds: float, chunk: float = 5.0,
            *, force_divide_after: float | None = None,
            verbose: bool = True) -> list[ColonyTickResult]:
        """Run the colony for ``total_seconds`` in ``chunk``-second ticks."""
        n_ticks = max(1, int(math.ceil(total_seconds / chunk)))
        forced = False
        for _ in range(n_ticks):
            if (force_divide_after is not None and not forced
                    and self.sim_time >= force_divide_after):
                self.request_force_divide()
                forced = True
                if verbose:
                    print(f"  [force-divide armed at sim_time={self.sim_time:.0f}s]")
            r = self.step(chunk)
            if verbose:
                pids = sorted({s["pid"] for s in r.per_cell.values()})
                masses = ", ".join(
                    f"{cid}:{s['dry_mass']:.0f}fg" for cid, s in
                    sorted(r.per_cell.items()))
                errs = [s["error"] for s in r.per_cell.values() if s.get("error")]
                print(f"  tick {r.tick:3d} t={r.sim_time:6.0f}s "
                      f"wall={r.wall_s:5.1f}s cells={r.live_cells} "
                      f"div={r.divisions} pids={pids} | {masses}"
                      + (f" ERR:{errs}" if errs else ""))
        return self.history

    def shutdown(self) -> None:
        for a in list(self.actors.values()):
            try:
                self._ray.kill(a)
            except Exception:  # noqa: BLE001
                pass
        self.actors.clear()
        try:
            self._ray.shutdown()
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ray-actor-per-cell colony driver")
    p.add_argument("--n-cells", type=int, default=2)
    p.add_argument("--seconds", type=float, default=30.0,
                   help="total sim seconds")
    p.add_argument("--chunk", type=float, default=5.0,
                   help="sim seconds advanced per concurrent tick")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default="out/cache")
    p.add_argument("--env-size", type=float, default=30.0)
    p.add_argument("--num-threads", type=int, default=None,
                   help="BLAS/OMP threads per actor (default cores//peak_actors)")
    p.add_argument("--force-divide-after", type=float, default=None,
                   help="arm WCM divide on all actors once sim_time >= this "
                        "(test hook; natural division is ~2500s)")
    args = p.parse_args(argv)

    cache_dir = os.path.abspath(args.cache_dir)
    if not os.path.isdir(cache_dir):
        print(f"[colony_ray] cache_dir not found: {cache_dir}", file=sys.stderr)
        return 2

    print(f"[colony_ray] n_cells={args.n_cells} seconds={args.seconds} "
          f"chunk={args.chunk} cache_dir={cache_dir}")
    colony = RayColony(n_cells=args.n_cells, seed=args.seed,
                       cache_dir=cache_dir, env_size=args.env_size,
                       num_threads=args.num_threads)
    print(f"[colony_ray] cores={os.cpu_count()} threads/actor={colony.threads}")

    try:
        print("[colony_ray] warmup (building inner composites in workers)…")
        t0 = time.perf_counter()
        pings = colony.warmup()
        pids = sorted({p["pid"] for p in pings.values()})
        print(f"  warmup done in {time.perf_counter()-t0:.1f}s; "
              f"distinct worker pids={pids}")
        if len(pids) < len(pings):
            print(f"  WARNING: {len(pings)} actors but only {len(pids)} "
                  f"distinct pids — actors are NOT all on separate processes")

        print(f"[colony_ray] running {args.seconds}s in {args.chunk}s chunks…")
        t0 = time.perf_counter()
        colony.run(args.seconds, chunk=args.chunk,
                   force_divide_after=args.force_divide_after)
        wall = time.perf_counter() - t0

        final = colony.history[-1] if colony.history else None
        serial_est = sum(
            sum(s.get("wall_s", 0.0) for s in r.per_cell.values())
            for r in colony.history)
        print(f"\n[colony_ray] DONE in {wall:.1f}s")
        print(f"  final live cells: {final.live_cells if final else 0}")
        print(f"  total divisions:  {colony.total_divisions}")
        print(f"  Σ per-cell wall (serial-equivalent): {serial_est:.1f}s")
        if wall > 0:
            print(f"  parallel speedup vs serial-equivalent: {serial_est/wall:.2f}×")
        return 0
    finally:
        colony.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
