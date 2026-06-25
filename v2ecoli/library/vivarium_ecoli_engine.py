"""Single-process, faithful-by-construction vEcoli engine for the comparison harness.

Instead of decomposing upstream vEcoli into ~50 individual process-bigraph steps and
reimplementing vivarium's Engine semantics (partition/allocation, update reconciliation,
division) in process-bigraph — which introduced a class of faithfulness bugs (mass
explosion from a missing bulk reconcile; a division-handoff crash from `_add` dropping
live process instances) — this module runs the GENUINE vEcoli on its OWN vivarium Engine
inside one wrapper, advancing it incrementally with ``Engine.run_for``.

Faithful by construction: partition, allocation, and per-tick update reconciliation are
all handled by vivarium-core exactly as upstream intends. Multi-generation is single-
lineage (matching v2ecoli's ``run_multigen_xarray``): each generation runs a genuine
vEcoli Engine to its division point, vEcoli's own ``divide_cell`` splits the state, and
the followed daughter seeds the next generation's Engine. No process-bigraph division.

Fork-robust: the vEcoli fork is selected by ``V2E_VECOLI_DIR`` (or the ``fork_dir`` arg),
imported through the same compiled-``wholecell``-pinning shim the step-decomposed wrapper
uses (:func:`v2ecoli.library.vecoli_pbg_upstream._ensure_upstream`). Swapping forks is a
one-knob change: point ``V2E_VECOLI_DIR`` at a different checkout + its matching ParCa
``sim_data``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

from v2ecoli.library.vecoli_pbg_upstream import _ensure_upstream
from v2ecoli.library.upstream_division import _n_chromosomes, _inc_to_fg
from v2ecoli.library.division import divide_cell


# ---------------------------------------------------------------------------
# Build a genuine vEcoli vivarium Engine (fork-parameterized)
# ---------------------------------------------------------------------------

@dataclass
class EngineHandle:
    """A built genuine-vEcoli vivarium Engine plus the metadata needed to drive it."""
    engine: object
    sim: object
    dry_mass_inc_dict: dict = field(default_factory=dict)
    condition: str = "basal"
    media_id: str = "minimal"
    time_step: float = 1.0


def build_vivarium_ecoli(
    *,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    time_step: float = 1.0,
    exclude_processes: list | None = None,
    fork_dir: str | None = None,
    initial_overlay: dict | None = None,
) -> EngineHandle:
    """Build the genuine upstream vEcoli composite and wrap its vivarium Engine.

    ``fork_dir`` (or ``$V2E_VECOLI_DIR``) selects the vEcoli checkout; ``sim_data_path``
    is its matching upstream ParCa ``simData.cPickle``. ``initial_overlay`` (a daughter's
    divided ``bulk``/``unique``/``environment``/``boundary``) seeds a non-founder
    generation; ``None`` builds a fresh founder.
    """
    if fork_dir:
        os.environ["V2E_VECOLI_DIR"] = fork_dir
    up = _ensure_upstream()
    EcoliSim = up["EcoliSim"]
    from vivarium.core.engine import Engine

    _argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        sim = EcoliSim.from_cli()
    finally:
        sys.argv = _argv

    sim.config["condition"] = condition
    sim.config["seed"] = int(seed)
    sim.config["sim_data_path"] = sim_data_path
    sim.config["time_step"] = float(time_step)
    # Division is handled by THIS module's single-lineage loop (genuine divide_cell
    # between generations), not vivarium's in-Engine Division roundtrip — so each
    # generation is one clean genuine-vEcoli Engine.
    sim.config["divide"] = False
    sim.config["d_period"] = False
    sim.config["generations"] = None
    sim.config["progress_bar"] = False
    sim.config["emit_paths"] = []
    if exclude_processes:
        existing = list(sim.config.get("exclude_processes", []) or [])
        sim.config["exclude_processes"] = existing + list(exclude_processes)

    # Capture the per-media expectedDryMassIncreaseDict (drives the division
    # threshold) the same way upstream_division does — via the composer's sim_data.
    dry_mass_inc_dict: dict = {}
    import ecoli.composites.ecoli_master as _em
    _orig_init = _em.Ecoli.__init__
    _captured: dict = {}

    def _capturing_init(self, config, *a, **k):
        _orig_init(self, config, *a, **k)
        _captured["composer"] = self

    _em.Ecoli.__init__ = _capturing_init
    try:
        sim.build_ecoli()
    finally:
        _em.Ecoli.__init__ = _orig_init
    composer = _captured.get("composer")
    if composer is not None:
        try:
            dry_mass_inc_dict = dict(
                composer.load_sim_data.sim_data.expectedDryMassIncreaseDict)
        except Exception:
            dry_mass_inc_dict = {}

    init_state = sim.generated_initial_state
    if initial_overlay:
        for key in ("bulk", "unique", "environment", "boundary"):
            if key in initial_overlay:
                init_state[key] = initial_overlay[key]

    engine = Engine(
        processes=sim.ecoli.processes,
        steps=sim.ecoli.steps,
        flow=sim.ecoli.flow,
        topology=sim.ecoli.topology,
        initial_state=init_state,
        emitter={"type": "null"},
        progress_bar=False,
    )

    media_id = "minimal"
    try:
        env = init_state.get("environment", {})
        media_id = env.get("media_id", "minimal") if isinstance(env, dict) else "minimal"
    except Exception:
        pass

    return EngineHandle(
        engine=engine, sim=sim, dry_mass_inc_dict=dry_mass_inc_dict,
        condition=condition, media_id=media_id, time_step=float(time_step))


# ---------------------------------------------------------------------------
# Read observables out of a live Engine state
# ---------------------------------------------------------------------------

def _state(engine):
    return engine.state.get_value()


# The mass observables compared by the report card (scripts/comparison_report_card.py
# MASS_OBS). Emitted into the v2ecoli-format zarr so BOTH engines read identically.
MASS_OBS = ("cell_mass", "dry_mass", "protein_mass", "rna_mass")


def cell_observables(engine) -> dict:
    """Pull the comparison observables from the live Engine state. Single-cell, no
    agents wrapper (divide=False). Scalar mass axes + the raw bulk/unique for division."""
    st = _state(engine)
    mass = (st.get("listeners", {}) or {}).get("mass", {}) or {}
    obs = {k: float(mass.get(k, 0.0) or 0.0) for k in MASS_OBS}
    obs.update({
        "bulk": st.get("bulk"),
        "unique": st.get("unique"),
        "environment": st.get("environment", {}),
        "boundary": st.get("boundary", {}),
    })
    return obs


# ---------------------------------------------------------------------------
# pbg Process: genuine vEcoli as ONE process-bigraph node (vivarium Engine inside)
# ---------------------------------------------------------------------------

from process_bigraph import Process


class VivariumEcoliProcess(Process):
    """Genuine upstream vEcoli as a SINGLE process-bigraph node, with vivarium-core's
    own Engine running inside.

    This PRESERVES the process-bigraph design: v2ecoli and vEcoli compare on the SAME
    pbg runtime (one pbg node each at ``agents/0``), so the comparison stays a true
    pbg-vs-pbg test — but vivarium handles partition / update-reconciliation / division
    *inside* this node, so none of the Engine-reimplementation bugs (the bulk-reconcile
    mass explosion; the ``_add`` division crash) can arise.

    ``update(state, interval)`` advances the inner vivarium Engine by ``interval`` via
    ``Engine.run_for`` and writes the cell's mass observables back to the pbg store
    (``listeners.mass.*``), so the standard ``XArrayEmitter`` view reads them exactly as
    it does for v2ecoli. REST-ready: the in-process Engine can later be swapped for an
    out-of-process vivarium service behind this same port interface.
    """

    config_schema = {
        "sim_data_path": "string",
        "condition": {"_type": "string", "_default": "basal"},
        "seed": {"_type": "integer", "_default": 0},
        "time_step": {"_type": "float", "_default": 1.0},
        "exclude_processes": {"_type": "list[string]", "_default": []},
        "fork_dir": {"_type": "string", "_default": ""},
    }

    # Set by build_vivarium_ecoli_composite to inject a pre-built (possibly daughter-
    # overlaid) engine so the process doesn't rebuild EcoliSim. Consumed once at
    # construction. Single build is sequential; Ray seeds are separate processes.
    _PENDING_HANDLE = None

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        if VivariumEcoliProcess._PENDING_HANDLE is not None:
            self._handle = VivariumEcoliProcess._PENDING_HANDLE
            VivariumEcoliProcess._PENDING_HANDLE = None
        else:
            self._handle = build_vivarium_ecoli(
                sim_data_path=self.config["sim_data_path"],
                condition=self.config["condition"],
                seed=int(self.config["seed"]),
                time_step=float(self.config["time_step"]),
                exclude_processes=list(self.config.get("exclude_processes") or []) or None,
                fork_dir=(self.config.get("fork_dir") or None),
            )

    def inputs(self):
        return {}

    def outputs(self):
        # Mass observables are recomputed-absolute each tick → 'set' semantics
        # (overwrite), matching vivarium's Mass listener _updater='set'.
        return {"listeners": {"mass": {k: "overwrite[float]" for k in MASS_OBS}}}

    def update(self, state, interval):
        self._handle.engine.run_for(float(interval))
        obs = cell_observables(self._handle.engine)
        return {"listeners": {"mass": {k: obs[k] for k in MASS_OBS}}}

    def divide(self) -> dict:
        """Split the inner cell with vEcoli's faithful ``divide_cell``; return
        daughter-0's overlay (bulk/unique/environment/boundary) to seed the next
        generation's Engine. The split is vivarium-native — no pbg ``_add``."""
        obs = cell_observables(self._handle.engine)
        d1, _d2 = divide_cell({
            "bulk": obs["bulk"], "unique": obs["unique"],
            "environment": obs["environment"], "boundary": obs["boundary"]})
        return d1

    def division_signals(self) -> tuple:
        """``(dry_mass_fg, n_chromosomes)`` for the lineage driver's division gate."""
        obs = cell_observables(self._handle.engine)
        return obs["dry_mass"], _n_chromosomes(obs["unique"])


def build_vivarium_ecoli_composite(
    *,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    time_step: float = 1.0,
    exclude_processes: list | None = None,
    fork_dir: str | None = None,
    core=None,
    agent_id: str = "0",
    initial_overlay: dict | None = None,
):
    """Wrap a single :class:`VivariumEcoliProcess` as a one-node pbg Composite under
    ``agents/<agent_id>`` — the genuine-vEcoli analogue of the v2ecoli agent composite,
    so the SAME ``run_multigen_xarray`` / ``XArrayEmitter`` path serves both engines.

    ``initial_overlay`` (a daughter's divided bulk/unique/env/boundary) seeds a non-
    founder generation. Returns ``(composite, info)``. The process writes
    ``listeners.mass.*`` (overwrite/set semantics) into the agent store each tick.
    """
    from process_bigraph import Composite
    if core is None:
        from v2ecoli.core import build_core
        core = build_core()

    # Build the (optionally daughter-seeded) engine once and inject it so the process
    # doesn't rebuild EcoliSim.
    VivariumEcoliProcess._PENDING_HANDLE = build_vivarium_ecoli(
        sim_data_path=sim_data_path, condition=condition, seed=int(seed),
        time_step=float(time_step), exclude_processes=list(exclude_processes or []) or None,
        fork_dir=fork_dir or None, initial_overlay=initial_overlay)
    proc = VivariumEcoliProcess(config={
        "sim_data_path": sim_data_path, "condition": condition, "seed": int(seed),
        "time_step": float(time_step),
        "exclude_processes": list(exclude_processes or []),
        "fork_dir": fork_dir or "",
    }, core=core)
    iface = proc.interface()

    cell_state = {
        "vivarium_ecoli": {
            "_type": "process",
            "instance": proc,
            "_inputs": iface.get("inputs", {}),
            "_outputs": iface.get("outputs", {}),
            "inputs": {},
            "outputs": {"listeners": ["listeners"]},
            "interval": float(time_step),
        }
    }
    state = {"agents": {agent_id: cell_state}, "global_time": 0.0}
    composite = Composite(dict(schema=dict(), state=state), core=core)
    return composite, {"core": core, "agent_root": "agents",
                       "agent_id": agent_id, "process": proc}


def run_vivarium_ecoli_pbg_multigen(
    *,
    store_path,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    max_generations: int = 2,
    max_steps_per_gen: int = 9000,
    time_step: float = 1.0,
    chunk: int = 20,
    exclude_processes: list | None = None,
    fork_dir: str | None = None,
    mass_multiplier: float = 1.0,
    core=None,
    experiment_id: str = "vecoli",
    variant: int = 0,
    lineage_seed: int = 0,
) -> dict:
    """Single-lineage multigen for the vEcoli **pbg node**, emitting the v2ecoli-format zarr.

    Each generation is a one-node pbg ``Composite`` (``VivariumEcoliProcess``) driven by
    ``composite.run``; a per-generation ``XArrayEmitter`` writes a ``generation=N``
    partition into the shared store (the SAME emitter v2ecoli uses). At the division
    criterion (dry_mass ≥ birth + expectedDryMassIncrease AND ≥2 chromosomes) vEcoli's own
    ``divide_cell`` splits the inner cell and the followed daughter seeds the next
    generation's Composite. No pbg ``_add`` — the division-handoff crash cannot occur.
    """
    import shutil
    from pathlib import Path
    from v2ecoli.library.xarray_run import _build_emitter, _filter_agent_state
    from v2ecoli.library.upstream_division import daughter_phylogeny_id

    if core is None:
        from v2ecoli.core import build_core
        core = build_core()
    store_path = str(store_path)
    if Path(store_path).exists():
        shutil.rmtree(store_path)

    view = [{"root": ("listeners",),
             "variables": {"mass": {k: [{"path": k, "dtype": "<f8"}] for k in MASS_OBS}}}]
    metadata_base = {
        "experiment_id": experiment_id, "variant": int(variant),
        "lineage_seed": int(lineage_seed), "time_step": float(time_step),
        "max_duration": float(max_generations * max_steps_per_gen),
    }

    overlay = None
    composite_agent_id = "0"            # the inner cell's key in the pbg agents map
    partition_agent_id = "0"            # the emitter's phylogeny key ("0"->"00"->...),
                                        # distinct per generation so each writes its own
                                        # zarr partition (avoids a same-store collision).
    done_global = 0
    divisions = 0
    gens_done = 0
    final_cell_mass = None

    for gen in range(max_generations):
        # gen 0 is a fresh founder (overlay=None); later generations seed the inner
        # Engine from the previous generation's daughter (overlay set below).
        comp, info = build_vivarium_ecoli_composite(
            sim_data_path=sim_data_path, condition=condition, seed=seed + gen,
            time_step=time_step, exclude_processes=exclude_processes,
            fork_dir=fork_dir, core=core, agent_id=composite_agent_id,
            initial_overlay=overlay)
        proc = info["process"]
        comp.run(1)  # warm-up tick so listeners materialise
        em = _build_emitter(
            core=core, store_path=store_path, view=view, metadata_base=metadata_base,
            generation=gen, agent_id=partition_agent_id, output_metadata={}, buffer_size=3)

        steps = 1
        threshold = None
        divided = False
        while steps < max_steps_per_gen:
            comp.run(chunk)
            steps += chunk
            done_global += chunk
            agent_state = comp.state["agents"][composite_agent_id]
            payload = _filter_agent_state(agent_state, view)
            # Relabel the payload to the emitter's phylogeny key (the emitter strips
            # the agent prefix via get_in(data, ("agents", partition_agent_id))).
            em.update({"time": float(done_global), "global_time": float(done_global),
                       "agents": {partition_agent_id: payload}})
            mass = agent_state["listeners"]["mass"]
            final_cell_mass = float(mass.get("cell_mass", 0.0) or 0.0)
            dry = float(mass.get("dry_mass", 0.0) or 0.0)
            if threshold is None and dry > 0:
                raw = (proc._handle.dry_mass_inc_dict.get(proc._handle.media_id)
                       or proc._handle.dry_mass_inc_dict.get("minimal"))
                inc = _inc_to_fg(raw) if raw is not None else dry
                if not inc or inc <= 0:
                    inc = dry  # fallback: mass doubling
                threshold = dry + inc * mass_multiplier
            _dry, nchrom = proc.division_signals()
            if threshold is not None and dry >= threshold and nchrom >= 2:
                divided = True
                break

        try:
            em.close(success=True)
        except AssertionError:
            pass  # F5: trailing-buffer include_static assert; generation already on disk
        gens_done += 1
        if not divided:
            break
        overlay = proc.divide()
        partition_agent_id = daughter_phylogeny_id(partition_agent_id)[0]
        divisions += 1

    return {"generations": gens_done, "divisions": divisions,
            "store": store_path, "final_cell_mass": final_cell_mass}


# ---------------------------------------------------------------------------
# Single-lineage multi-generation driver (standalone, non-pbg — local utility)
# ---------------------------------------------------------------------------

def run_vivarium_ecoli_multigen(
    *,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    max_generations: int = 2,
    max_steps_per_gen: int = 9000,
    time_step: float = 1.0,
    chunk: int = 20,
    exclude_processes: list | None = None,
    fork_dir: str | None = None,
    mass_multiplier: float = 1.0,
    on_emit=None,
) -> dict:
    """Drive genuine vEcoli single-lineage for ``max_generations`` generations.

    Each generation: build a genuine-vEcoli Engine (seeded with the previous daughter's
    divided state), advance with ``run_for(chunk)`` until the division criterion is met
    (dry_mass >= birth_dry_mass + expectedDryMassIncrease AND >=2 chromosomes), then split
    with vEcoli's ``divide_cell`` and carry one daughter into the next generation.

    ``on_emit(gen, t_global, obs)`` is called once per chunk with the live observables
    (the caller writes them wherever it likes — e.g. a matched-timepoint zarr).

    Returns a summary dict ``{generations, divisions, steps_per_gen, final_cell_mass}``.
    """
    overlay = None
    divisions = 0
    steps_per_gen: list[int] = []
    t_global = 0.0
    last_obs = None

    for gen in range(max_generations):
        h = build_vivarium_ecoli(
            sim_data_path=sim_data_path, condition=condition, seed=seed + gen,
            time_step=time_step, exclude_processes=exclude_processes,
            fork_dir=fork_dir, initial_overlay=overlay)
        engine = h.engine
        threshold = None
        gen_steps = 0
        divided = False

        while gen_steps < max_steps_per_gen:
            engine.run_for(float(chunk))
            gen_steps += chunk
            obs = cell_observables(engine)
            last_obs = obs
            if on_emit is not None:
                on_emit(gen, t_global + gen_steps, obs)

            if threshold is None and obs["dry_mass"] > 0:
                inc = _inc_to_fg(h.dry_mass_inc_dict.get(h.media_id))
                if inc is None:
                    inc = _inc_to_fg(h.dry_mass_inc_dict.get("minimal"))
                if inc is None:
                    inc = obs["dry_mass"]  # fallback: mass doubling
                threshold = obs["dry_mass"] + inc * mass_multiplier

            n_chrom = _n_chromosomes(obs["unique"])
            if threshold is not None and obs["dry_mass"] >= threshold and n_chrom >= 2:
                divided = True
                break

        steps_per_gen.append(gen_steps)
        t_global += gen_steps

        if not divided:
            # ran out of steps before dividing — stop the lineage here
            break
        # vEcoli's faithful split; follow daughter 0 into the next generation
        d1, _d2 = divide_cell({
            "bulk": last_obs["bulk"], "unique": last_obs["unique"],
            "environment": last_obs["environment"], "boundary": last_obs["boundary"]})
        overlay = d1
        divisions += 1

    return {
        "generations": len(steps_per_gen),
        "divisions": divisions,
        "steps_per_gen": steps_per_gen,
        "final_cell_mass": (last_obs or {}).get("cell_mass"),
    }
