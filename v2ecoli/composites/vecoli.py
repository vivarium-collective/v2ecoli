"""Genuine vEcoli, registered as a template-made process-bigraph Composite.

Wraps the vivarium-process vEcoli engine (``v2ecoli.library.vivarium_ecoli_engine``)
— the same engine ``scripts/run_comparison_ensemble.py`` drives for
``--composite vecoli --vecoli-source vivarium-process`` ("genuine vEcoli as a
single composite, its own Engine inside") — as a registered ``@composite_generator``
so the general ``vivarium-workbench`` runner can build/run it uniformly, the same
way it already does ``ecoli_baseline``. See
docs/superpowers/specs/2026-08-01-comparison-general-runner-convergence-design.md
§2/§5 and docs/superpowers/plans/2026-08-01-comparison-convergence-phase-1.md Task 1.

Does NOT reimplement vEcoli's partition/allocation/division — that logic lives
entirely in ``vivarium_ecoli_engine.build_vivarium_ecoli`` (a genuine upstream
vEcoli ``EcoliSim`` wrapped in vivarium-core's own ``Engine``, run inside ONE
process-bigraph node). This module only assembles the standard
``agents/<agent_id>/vivarium_ecoli`` document envelope around that engine build
— the same envelope
:func:`v2ecoli.library.vivarium_ecoli_engine.build_vivarium_ecoli_composite`
assembles, inlined here so the ``@composite_generator`` contract (return a
document dict, not a pre-built ``Composite``) is honoured without constructing
a throwaway ``Composite`` first.

``reference_repo`` is the fork as an explicit, declared composite param (spec
§5) — a repo/commit locally, or a git ref for the sms-api build — threaded into
the engine builder's ``fork_dir`` and recorded in run provenance, instead of
relying on the ambient ``$V2E_VECOLI_DIR`` env var. Empty (default) falls back
to ``$V2E_VECOLI_DIR`` for interactive/local convenience.
"""

from __future__ import annotations

import os
from typing import Any

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.core import build_core

# Local fallback for the genuine-vEcoli ParCa simData when ``cache_dir`` has no
# simData.cPickle. Mirrors scripts/run_comparison_ensemble.py's
# ``_UPSTREAM_SIMDATA_FALLBACK`` (the same last-resort path the vivarium-process
# vEcoli loader and matched-initial-state reference both use there).
_UPSTREAM_SIMDATA_FALLBACK = (
    "/Users/eranagmon/code/v2ecoli/out/compare_harness/vecoli_parca/"
    "kb/simData.cPickle")


def _resolve_sim_data_path(cache_dir: str) -> str:
    """<cache_dir>/simData.cPickle if present, else the upstream local fallback."""
    candidate = os.path.abspath(os.path.join(cache_dir, "simData.cPickle"))
    return candidate if os.path.exists(candidate) else _UPSTREAM_SIMDATA_FALLBACK


def _resolve_fork_config(reference_repo: str, fork_config: str | None):
    """Resolve an optional vEcoli fork config into ``(swap_processes, flow)``.

    Mirrors the ``--from-vecoli-config`` vecoli-side resolution in
    ``scripts/run_comparison_ensemble.py`` (``ve_swap_processes``/``ve_flow``),
    read-only reuse of ``scripts/_compare/config_adapter.py`` — not duplicated
    logic. Returns ``(None, None)`` when no ``fork_config`` is given.
    """
    if not fork_config:
        return None, None
    from scripts._compare.config_adapter import resolve_vecoli_config_local
    fork_dir = reference_repo or os.environ.get("V2E_VECOLI_DIR", "")
    resolved = resolve_vecoli_config_local(fork_config, fork_dir)
    return resolved.get("swap_processes") or None, resolved.get("flow") or None


@composite_generator(
    name="vecoli",
    description=(
        "Genuine upstream vEcoli, run on its own vivarium-core Engine inside "
        "ONE process-bigraph node (agents/<agent_id>/vivarium_ecoli) — the "
        "REFERENCE model for the v2ecoli<->vEcoli comparison. Wraps "
        "v2ecoli.library.vivarium_ecoli_engine.build_vivarium_ecoli (the same "
        "engine builder scripts/run_comparison_ensemble.py drives for "
        "--composite vecoli --vecoli-source vivarium-process); faithful by "
        "construction — no reimplemented partition/allocation/division."
    ),
    parameters={
        "reference_repo": {
            "type": "string",
            "default": "",
            "description": (
                "vEcoli fork checkout path (local) or git ref (sms-api build) "
                "— an EXPLICIT, declared param (not the ambient $V2E_VECOLI_DIR "
                "env var), so retargeting a fork is one param that lands in the "
                "run's provenance. Empty = fall back to $V2E_VECOLI_DIR."
            ),
        },
        "condition": {
            "type": "string",
            "default": "basal",
            "description": "vEcoli growth condition (EcoliSim.config['condition']).",
        },
        "seed": {
            "type": "integer",
            "default": 0,
            "description": "RNG seed for vEcoli's stochastic initialization.",
        },
        "fork_config": {
            "type": "string",
            "default": "",
            "description": (
                "Optional vEcoli config JSON (path relative to reference_repo, "
                "or absolute) driving a process swap (e.g. FBA metabolism -> "
                "MetabolismRedux), resolved via "
                "scripts._compare.config_adapter.resolve_vecoli_config_local. "
                "Empty (default) = no swap."
            ),
        },
        "cache_dir": {
            "type": "string",
            "default": "out/cache",
            "description": (
                "Directory holding the matching ParCa simData.cPickle for this "
                "reference vEcoli run: <cache_dir>/simData.cPickle if present, "
                "else the upstream local fallback."
            ),
        },
        "time_step": {
            "type": "float",
            "default": 1.0,
            "description": "Simulation time step in seconds.",
        },
        "agent_id": {
            "type": "string",
            "default": "0",
            "description": "Agent key under 'agents' the vEcoli node lives at.",
        },
        "whole_config": {
            "type": "string",
            "default": "",
            "description": (
                "Optional full fork config (path relative to reference_repo or "
                "absolute) loaded NATIVELY by EcoliSim (its add_processes / "
                "spatial_environment_config / variants applied) instead of the "
                "swap-only fork_config path. Empty = swap/baseline behavior."
            ),
        },
        "variant": {
            "type": "integer",
            "default": 0,
            "description": (
                "1-based index into the loaded config's 'variants' grid "
                "(0 = unperturbed baseline). Requires whole_config."
            ),
        },
        "observable_bulk_ids": {
            "type": "list",
            "default": [],
            "description": (
                "Bulk molecule ids to emit as observables (path 'bulk.<id>') "
                "for downstream sweep/phenotype extraction. Empty = mass/count "
                "observables only."
            ),
        },
        "observables": {
            "type": "list",
            "default": [],
            "description": (
                "Arbitrary listener leaves ('group.leaf', e.g. "
                "'peptidoglycan_shape.lysed') to emit under "
                "'listeners.<group>.<leaf>' for downstream sweep/phenotype "
                "extraction. Empty = the default mass/count observables only. "
                "The engine already supports this (see VivariumEcoliProcess); "
                "this exposes it as a node param, mirroring observable_bulk_ids."
            ),
        },
    },
    default_n_steps=2700,
)
def vecoli(
    core: Any = None,
    *,
    reference_repo: str = "",
    condition: str = "basal",
    seed: int = 0,
    fork_config: str | None = None,
    cache_dir: str = "out/cache",
    time_step: float = 1.0,
    agent_id: str = "0",
    whole_config: str = "",
    variant: int = 0,
    observable_bulk_ids: list | None = None,
    observables: list | None = None,
) -> dict:
    """Build the process-bigraph document for genuine vEcoli as one node.

    Delegates the actual engine construction (upstream ``EcoliSim`` build +
    vivarium-core ``Engine`` wrap) entirely to
    :func:`v2ecoli.library.vivarium_ecoli_engine.build_vivarium_ecoli` — this
    function only resolves ``reference_repo``/``cache_dir``/``fork_config``
    into that builder's kwargs and assembles the standard
    ``agents/<agent_id>/vivarium_ecoli`` document envelope (matching
    :func:`v2ecoli.library.vivarium_ecoli_engine.build_vivarium_ecoli_composite`,
    inlined here so this returns a document dict rather than a pre-built
    ``Composite`` — the ``@composite_generator`` contract).

    Args:
        core: bigraph-schema core. If None, one is created via build_core().
        reference_repo: vEcoli fork path/git ref (spec §5's explicit fork
            param). Empty falls back to ``$V2E_VECOLI_DIR``.
        condition: vEcoli growth condition.
        seed: RNG seed.
        fork_config: optional vEcoli config JSON path driving a process swap.
        cache_dir: directory holding the matching ParCa simData.cPickle.
        time_step: simulation time step (seconds).
        agent_id: agent key under 'agents' the vEcoli node lives at, and the
            lineage phylogeny key ("0" -> "00" -> ...). Its LENGTH is the
            generation index the wrapped fork applies staged shifts on.
        whole_config: optional full fork config loaded NATIVELY by EcoliSim
            (its own add_processes / spatial_environment_config / variants
            applied) instead of the swap-only fork_config path. Empty
            (default) preserves the existing swap/baseline behavior.
        variant: 1-based index into the loaded config's 'variants' grid
            (0 = unperturbed baseline). Only meaningful with whole_config.
        observable_bulk_ids: bulk molecule ids to emit as observables for
            downstream sweep/phenotype extraction.
        observables: arbitrary listener leaves ('group.leaf') to emit under
            'listeners.<group>.<leaf>' (e.g. 'peptidoglycan_shape.lysed' for a
            cell-shape/lysis phenotype). Threaded into VivariumEcoliProcess,
            which already supports it; lands under the wired 'listeners' port.

    Returns:
        Process-bigraph document dict with keys ``schema``/``state``.
    """
    from v2ecoli.library.vivarium_ecoli_engine import (
        build_vivarium_ecoli, VivariumEcoliProcess, set_ecolisim_config_file)

    if core is None:
        core = build_core()

    sim_data_path = _resolve_sim_data_path(cache_dir)

    if whole_config:
        # Native whole-config load: EcoliSim reads add_processes / spatial /
        # variants from this file. Resolve relative to the fork checkout.
        cfg_path = whole_config
        if not os.path.isabs(cfg_path):
            base = reference_repo or os.environ.get("V2E_VECOLI_DIR", "")
            cfg_path = os.path.join(base, cfg_path)
        set_ecolisim_config_file(cfg_path)
        swap_processes, flow = None, None      # native path, not swap
    else:
        swap_processes, flow = _resolve_fork_config(reference_repo, fork_config)

    # Build the (fork-parameterized) genuine-vEcoli engine and hand it to the
    # process via the same PENDING_HANDLE injection
    # build_vivarium_ecoli_composite uses, so the process doesn't rebuild
    # EcoliSim a second time.
    try:
        VivariumEcoliProcess._PENDING_HANDLE = build_vivarium_ecoli(
            sim_data_path=sim_data_path,
            condition=condition,
            seed=int(seed),
            time_step=float(time_step),
            swap_processes=swap_processes,
            flow=flow,
            fork_dir=(reference_repo or None),
            variant=int(variant),
            # ⛔ The wrapped fork reads its GENERATION INDEX off this key
            # (``LoadSimData``: ``generation = len(agent_id)``), which is what
            # makes a config's staged induction fire. A declared composite that
            # dropped it here ran as the founder whatever generation it was.
            agent_id=str(agent_id),
        )
        proc = VivariumEcoliProcess(config={
            "sim_data_path": sim_data_path,
            "condition": condition,
            "seed": int(seed),
            "time_step": float(time_step),
            "fork_dir": reference_repo or "",
            "variant": int(variant),
            "agent_id": str(agent_id),
            "observable_bulk_ids": list(observable_bulk_ids or []),
            "observables": list(observables or []),
        }, core=core)
    finally:
        if whole_config:
            set_ecolisim_config_file(None)   # deterministic isolation
    iface = proc.interface()

    # Wire the process's output ports to the agent's stores. The process only
    # DECLARES a ``bulk`` output port when observable ids are configured, so only
    # then do we wire ``bulk`` -> ``["bulk"]`` (agents/<id>/bulk); otherwise pbg
    # would drop an unmapped-but-declared port and the observables never land.
    _outputs = {"listeners": ["listeners"]}
    if list(observable_bulk_ids or []):
        _outputs["bulk"] = ["bulk"]
    cell_state = {
        "vivarium_ecoli": {
            "_type": "process",
            "instance": proc,
            "_inputs": iface.get("inputs", {}),
            "_outputs": iface.get("outputs", {}),
            "inputs": {},
            "outputs": _outputs,
            "interval": float(time_step),
        }
    }
    state = {"agents": {agent_id: cell_state}, "global_time": 0.0}

    return {"schema": {}, "state": state}
