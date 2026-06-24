"""Process-bigraph-native cell division for the *pristine upstream* vEcoli wrap.

This is the last piece that lets the unmodified ``CovertLab/vEcoli`` model
(imported + wrapped by :mod:`v2ecoli.library.vecoli_pbg_upstream`) run
**multi-generationally** on the process-bigraph runtime, with **zero edits to
the upstream checkout**.

Why upstream's own division can't be used
-----------------------------------------
Upstream's :class:`ecoli.processes.cell_division.Division` divides by emitting a
vivarium ``{'agents': {'_divide': ...}}`` sentinel that the vivarium ``Engine``
consumes by invoking the ``Ecoli`` *composer* to regenerate daughters. The
process-bigraph runtime has no such Engine hook, so that update is inert (which
is why the prototype runs with ``divide=False``).

What this module provides
-------------------------
A harness-side pbg ``Step`` (:class:`UpstreamDivision`) that mirrors how
**v2ecoli** divides its own pbg composite (``v2ecoli/steps/division.py``):

  * **Trigger** — the same condition upstream's ``Division`` uses: the cell's
    dry mass (``listeners.mass.dry_mass``) reaches a per-cell threshold AND the
    chromosome has finished replication (``unique.full_chromosome`` has >= 2
    active entries). The threshold is derived ``mass_distribution``-style on the
    first populated tick: ``birth_dry_mass + expectedDryMassIncrease`` (the
    increment captured from upstream ``sim_data``), exactly as upstream's
    ``Division.next_update`` first-tick branch does.
  * **Split** — the mother's ``bulk`` (binomial p=0.5) and ``unique`` molecules
    (domain-based) are partitioned by :func:`v2ecoli.library.division.divide_cell`
    (the faithful numpy port of vEcoli's ``ecoli/library/schema.py`` dividers).
  * **Create daughters** — each daughter is a FRESH upstream-wrapped cell
    document (a new call to
    :func:`~v2ecoli.library.vecoli_pbg_upstream.build_upstream_cell_document`,
    i.e. a fresh upstream ``EcoliSim`` build with independent process
    instances) whose biological stores are overlaid with the divided state —
    the direct analogue of v2ecoli's ``baseline()`` + overlay. The step returns
    a structural ``{'agents': {'_remove': [...], '_add': [...]}}`` update so the
    daughters appear in the colony ``('agents',)`` map and
    :func:`~v2ecoli.library.xarray_run.run_multigen_xarray` follows the lineage
    by agent-count growth.

The mother is built under ``agents/0`` and the division step is wired with its
``agents`` output to ``('..',)`` (the colony map), exactly mirroring v2ecoli's
``_helpers`` division topology. Nothing under the upstream checkout is touched.
"""

from __future__ import annotations

import copy

import numpy as np

from bigraph_schema.schema import Float

from v2ecoli.library.ecoli_step import EcoliStep
from v2ecoli.types.stores import InPlaceDict


# Daughters built fully (real process tree) vs. as a data-only stub. The
# multigen runner follows a single lineage (the survivor whose id ends in
# ``0``); the sibling only needs to EXIST so the colony agent-count grows by one
# (which is how ``run_multigen_xarray`` detects a division). Building it as a
# data-only stub (divided stores, no process edges → never ticks) avoids
# accumulating a full 50-edge process tree per generation in RAM.
_FULL_DAUGHTER_SUFFIX = "0"


def daughter_phylogeny_id(mother_id):
    return [str(mother_id) + "0", str(mother_id) + "1"]


def _dry_mass(states) -> float:
    listeners = states.get("listeners") or {}
    mass = listeners.get("mass") or {} if isinstance(listeners, dict) else {}
    dm = mass.get("dry_mass", 0.0) if isinstance(mass, dict) else 0.0
    try:
        return float(dm)
    except (TypeError, ValueError):
        return 0.0


def _n_chromosomes(unique) -> int:
    if not isinstance(unique, dict):
        return 0
    fc = unique.get("full_chromosome")
    if fc is None or not hasattr(fc, "dtype") or fc.dtype.names is None:
        return 0
    if "_entryState" in fc.dtype.names:
        return int(fc["_entryState"].view(np.bool_).sum())
    return len(fc)


def _inc_to_fg(inc) -> float:
    """Coerce an ``expectedDryMassIncreaseDict`` value to a plain fg float.

    Upstream stores these as Unum quantities (``333.8 [fg]``); convert via
    ``asNumber(units.fg)`` when available, else assume already-fg float.
    """
    if hasattr(inc, "asNumber"):
        try:
            from wholecell.utils import units
            return float(inc.asNumber(units.fg))
        except Exception:
            try:
                return float(inc.magnitude)  # pint fallback
            except Exception:
                return 0.0
    try:
        return float(inc)
    except (TypeError, ValueError):
        return 0.0


class UpstreamDivision(EcoliStep):
    """pbg division Step for the upstream-wrapped cell. See module docstring."""

    name = "upstream-division"

    def initialize(self, config):
        cfg = config or {}
        self.agent_id = str(cfg.get("agent_id", "0"))
        self.dry_mass_inc_dict = cfg.get("dry_mass_inc_dict", {}) or {}
        self.mass_multiplier = float(cfg.get("mass_multiplier", 1.0))
        self.seed = int(cfg.get("seed", 0))
        # Kwargs to rebuild a fresh upstream cell document for each daughter
        # (seed/condition/sim_data_path/time_step/exclude_processes/core).
        self.builder_kwargs = dict(cfg.get("builder_kwargs", {}) or {})
        # Per-cell-instance state that persists across ticks (the Step object is
        # constructed once and re-invoked each tick): the lazily-derived mass
        # threshold and a one-shot latch so we emit the structural divide once.
        self._threshold = None
        self._divided = False

    # -- ports ---------------------------------------------------------------
    def inputs(self):
        return {
            "bulk": "bulk_array",
            "unique": InPlaceDict(),
            "listeners": InPlaceDict(),
            "environment": InPlaceDict(),
            "boundary": InPlaceDict(),
            "global_time": Float(_default=0.0),
        }

    def outputs(self):
        # The only thing this step writes is the colony agents map (structural
        # _add/_remove). Typed as ``map[node]`` and wired to ('..',) so the
        # daughter docs land as sibling agents next to the mother.
        return {"agents": {"_type": "map", "_value": "node"}}

    # -- division logic ------------------------------------------------------
    def update(self, states, interval=None):
        if self._divided:
            return {}

        dry_mass = _dry_mass(states)

        # First-tick threshold derivation (mass_distribution semantics). Wait
        # until the mass listener has populated dry_mass (>0) this generation.
        if self._threshold is None:
            if dry_mass <= 0.0:
                return {}
            env = states.get("environment") or {}
            media_id = env.get("media_id", "minimal") if isinstance(env, dict) else "minimal"
            inc = self.dry_mass_inc_dict.get(media_id)
            if inc is None:
                inc = self.dry_mass_inc_dict.get("minimal")
            inc_fg = _inc_to_fg(inc)
            if inc_fg <= 0.0:
                # No usable increment captured — fall back to mass doubling.
                inc_fg = dry_mass
            self._threshold = dry_mass + inc_fg * self.mass_multiplier
            return {}

        n_chrom = _n_chromosomes(states.get("unique"))
        if dry_mass < self._threshold or n_chrom < 2:
            return {}

        # --- DIVISION -------------------------------------------------------
        self._divided = True
        division_time = float(states.get("global_time", 0.0) or 0.0)
        d1_id, d2_id = daughter_phylogeny_id(self.agent_id)
        print(
            f"[upstream-division] DIVIDE at t={division_time:.0f}s "
            f"mother={self.agent_id} -> {d1_id},{d2_id} "
            f"(dry_mass={dry_mass:.1f} fg >= threshold={self._threshold:.1f} fg, "
            f"chromosomes={n_chrom})",
            flush=True,
        )

        try:
            from v2ecoli.library.division import divide_cell

            cell_data = {
                "bulk": states["bulk"],
                "unique": states["unique"],
                "environment": states.get("environment", {}),
                "boundary": states.get("boundary", {}),
            }
            d1_data, d2_data = divide_cell(cell_data)

            m_bulk = float(np.asarray(states["bulk"]["count"]).sum())
            b1 = float(np.asarray(d1_data["bulk"]["count"]).sum())
            b2 = float(np.asarray(d2_data["bulk"]["count"]).sum())
            print(
                f"[upstream-division]   bulk split: mother={m_bulk:.0f} -> "
                f"{d1_id}={b1:.0f} ({100*b1/m_bulk:.1f}%) + "
                f"{d2_id}={b2:.0f} ({100*b2/m_bulk:.1f}%)",
                flush=True,
            )

            # Full daughter (followed lineage) + data-only stub sibling.
            d1_cell = self._build_full_daughter(d1_data, d1_id, division_time)
            d2_cell = self._build_stub_daughter(d2_data, d2_id, division_time)

            return {
                "agents": {
                    "_remove": [self.agent_id],
                    "_add": [(d1_id, d1_cell), (d2_id, d2_cell)],
                }
            }
        except Exception as e:  # noqa: BLE001
            # process-bigraph silently swallows exceptions raised from a step's
            # update; surface the full traceback and re-raise.
            import traceback

            print(f"[upstream-division] DIVISION FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            raise

    # -- daughter construction ----------------------------------------------
    def _daughter_seed(self, daughter_id: str) -> int:
        suffix = daughter_id[len(self.agent_id):] or "0"
        return (self.seed + 1 + int(suffix, 2) if suffix.isdigit() else self.seed + 1) % (2 ** 31)

    def _build_full_daughter(self, d_data, daughter_id, division_time):
        """Fresh upstream-wrapped cell document with the divided state overlaid.

        Mirrors v2ecoli ``Division._build_daughter_doc``: a fresh build supplies
        independent process instances + correctly-sized partition stores; the
        biological stores are then replaced with this daughter's divided slice.
        """
        from v2ecoli.library.vecoli_pbg_upstream import build_upstream_cell_document

        bk = dict(self.builder_kwargs)
        bk["seed"] = self._daughter_seed(daughter_id)
        cell_state, info = build_upstream_cell_document(verbose=False, **bk)

        # Overlay the divided biological stores onto the fresh newborn doc.
        for key in ("bulk", "unique", "environment", "boundary"):
            if key in d_data:
                cell_state[key] = d_data[key]

        # Reset the mass listener so the daughter re-derives its own threshold
        # from its own birth mass on the next populated tick.
        listeners = cell_state.setdefault("listeners", {})
        listeners["mass"] = {"dry_mass": 0.0, "cell_mass": 0.0}
        cell_state["global_time"] = division_time

        # Give the daughter its OWN division step (so it divides in turn),
        # parameterised with its lineage id.
        add_division_edge(
            cell_state,
            agent_id=daughter_id,
            dry_mass_inc_dict=self.dry_mass_inc_dict,
            mass_multiplier=self.mass_multiplier,
            seed=bk["seed"],
            builder_kwargs=self.builder_kwargs,
        )
        return cell_state

    def _build_stub_daughter(self, d_data, daughter_id, division_time):
        """Data-only sibling: divided stores, NO process edges (never ticks).

        Exists solely so the colony agent count grows by one at division (the
        signal ``run_multigen_xarray`` uses to detect a division and advance a
        generation). The followed lineage is always the ``…0`` daughter, so this
        ``…1`` sibling is never stepped.
        """
        stub = {
            "bulk": d_data.get("bulk"),
            "unique": d_data.get("unique"),
            "environment": copy.deepcopy(d_data.get("environment", {})),
            "boundary": copy.deepcopy(d_data.get("boundary", {})),
            "listeners": {"mass": {"dry_mass": 0.0, "cell_mass": 0.0}},
            "global_time": division_time,
        }
        return stub


def add_division_edge(
    cell_state: dict,
    *,
    agent_id: str,
    dry_mass_inc_dict: dict,
    mass_multiplier: float,
    seed: int,
    builder_kwargs: dict,
    core=None,
):
    """Insert an :class:`UpstreamDivision` step edge into a cell document.

    Wires (relative to the cell root, i.e. ``agents/<id>``) mirror v2ecoli's
    division topology: biological stores read in-place; the ``agents`` output
    routed to ``('..',)`` so structural daughter ``_add``/``_remove`` land in
    the colony map one level up.
    """
    if core is None:
        core = builder_kwargs.get("core")

    inst = UpstreamDivision(
        {
            "agent_id": agent_id,
            "dry_mass_inc_dict": dry_mass_inc_dict,
            "mass_multiplier": mass_multiplier,
            "seed": seed,
            "builder_kwargs": builder_kwargs,
        },
        core=core,
    )
    interface = inst.interface()
    decl = {
        "_type": "step",
        "instance": inst,
        "_inputs": interface.get("inputs", {}),
        "_outputs": interface.get("outputs", {}),
        "inputs": {
            "bulk": ["bulk"],
            "unique": ["unique"],
            "listeners": ["listeners"],
            "environment": ["environment"],
            "boundary": ["boundary"],
            "global_time": ["global_time"],
        },
        "outputs": {"agents": [".."]},
        # Lowest priority so the divide check runs AFTER every per-tick listener
        # (notably the mass listener that populates listeners.mass.dry_mass) in
        # the cyclic step-network's same-tick ordering.
        "priority": 0.5,
    }
    cell_state["division"] = decl
    return cell_state


def build_upstream_agents_composite(
    *,
    seed: int = 0,
    condition: str = "basal",
    sim_data_path: str = (
        "/Users/eranagmon/code/v2ecoli/out/compare_harness/vecoli_parca/"
        "kb/simData.cPickle"),
    time_step: float = 1.0,
    exclude_processes: list | None = None,
    mass_multiplier: float = 1.0,
    core=None,
    verbose: bool = True,
):
    """Build the gen-1 upstream cell wrapped under ``agents/0`` + a division step.

    Returns ``(composite, info)``. The composite has a colony ``('agents',)``
    map with the single founder cell ``'0'``; ticking it past a full cell cycle
    fires :class:`UpstreamDivision`, which adds daughters to that map. Drive it
    with :func:`v2ecoli.library.xarray_run.run_multigen_xarray`.
    """
    from process_bigraph import Composite

    from v2ecoli.library.vecoli_pbg_upstream import build_upstream_cell_document

    cell_state, info = build_upstream_cell_document(
        seed=seed,
        condition=condition,
        sim_data_path=sim_data_path,
        time_step=time_step,
        exclude_processes=exclude_processes,
        core=core,
        verbose=verbose,
    )
    core = info["core"]

    builder_kwargs = {
        "condition": condition,
        "sim_data_path": sim_data_path,
        "time_step": time_step,
        "exclude_processes": list(exclude_processes or []),
        "core": core,
    }
    add_division_edge(
        cell_state,
        agent_id="0",
        dry_mass_inc_dict=info.get("dry_mass_inc_dict", {}),
        mass_multiplier=mass_multiplier,
        seed=seed,
        builder_kwargs=builder_kwargs,
        core=core,
    )

    state = {"agents": {"0": cell_state}, "global_time": 0.0}
    composite = Composite(dict(schema=dict(), state=state), core=core)
    info["agent_root"] = "agents"
    return composite, info


# ---------------------------------------------------------------------------
# Smoke driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import time

    os.environ.setdefault("PYTHONHASHSEED", "0")
    max_steps = int(os.environ.get("N", "3000"))
    composite, info = build_upstream_agents_composite(
        seed=0, condition="basal",
        exclude_processes=["monomer_counts_listener"],
    )
    print("BUILD OK; agents:", list(composite.state["agents"].keys()), flush=True)
    t0 = time.time()
    last_n = 1
    for i in range(max_steps):
        composite.run(1)
        agents = composite.state.get("agents", {})
        if len(agents) != last_n:
            print(f"  tick {i+1}: agents now {sorted(agents.keys())} "
                  f"(t={time.time()-t0:.0f}s wall)", flush=True)
            last_n = len(agents)
        if (i + 1) % 200 == 0:
            cell = agents.get(sorted(agents)[0], {})
            dm = (cell.get("listeners", {}) or {}).get("mass", {}).get("dry_mass")
            print(f"  tick {i+1}: agents={len(agents)} dry_mass={dm}", flush=True)
    print(f"=== ran {max_steps} steps in {time.time()-t0:.0f}s; "
          f"final agents={sorted(composite.state.get('agents', {}).keys())} ===",
          flush=True)
