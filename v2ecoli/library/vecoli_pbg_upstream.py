"""Run *pristine, unmodified* upstream ``CovertLab/vEcoli`` as a
process-bigraph composite by importing its ``EcoliSim`` and wrapping its
vivarium processes externally.

Strategy
--------
1. Import upstream ``EcoliSim`` from the pristine checkout (no edits there).
   ``EcoliSim.build_ecoli()`` drives the upstream ``Ecoli`` composer, which
   already encodes ALL the vEcoli-specific assembly: the partitioned-process
   split into ``Requester``/``Evolver`` halves, the ``Allocator`` and
   ``UniqueUpdate`` infrastructure steps, the four-layer execution ``flow``,
   the full ``topology``, and the ``generated_initial_state`` (including the
   shared ``PartitionedProcess`` instances placed in the ``('process',)``
   store as 1-tuples).

2. Wrap each vivarium process/step *instance* via
   :func:`v2ecoli.library.vivarium_bridge.wrap_vivarium_instance`, which
   synthesizes typed ``inputs()``/``outputs()`` from each process's
   ``ports_schema()`` WITHOUT modifying the class, routes ``update`` →
   ``next_update``, and honors the ``update_condition`` partition gate.

3. Assemble a process-bigraph composite *document* from the wrapped instances
   + the upstream topology (port→wire) + the upstream initial state, declaring
   each edge with a pre-built ``instance`` (process-bigraph's ``realize_link``
   honors a supplied instance — bigraph_schema/methods/realize.py:626 — so no
   address/config re-realization and no SharedProcess machinery is needed).

4. Apply the hard-won step-flow PRIORITY fix (strictly-decreasing priority by
   execution-layer index) so the cyclic step network's cycle-fallback runs the
   requester→allocator→evolver cascade in topological order each tick.

NOTHING under the upstream checkout is modified. The only code that lives here
is the external wrapper + the harness converter.

Entry point: :func:`build_upstream_pbg_composite`.
"""

from __future__ import annotations

import os
import sys


# ---------------------------------------------------------------------------
# Import shimming (upstream vEcoli ships uncompiled Cython)
# ---------------------------------------------------------------------------

_UPSTREAM: dict = {}

# Default vEcoli simData for the upstream builders. Resolved from
# $V2E_UPSTREAM_SIMDATA (empty when unset) rather than a hardcoded developer
# path, so a caller on any machine must supply one explicitly or set the env
# var; the builders raise a clear error when it is empty (issue #131).
_UPSTREAM_SIMDATA_DEFAULT: str = os.environ.get("V2E_UPSTREAM_SIMDATA", "")

# Pure-python vEcoli source packages that both the upstream checkout AND the
# installed ``vEcoli`` dist provide, so the installed copy shadows the checkout
# in ``sys.modules``. These must be dropped (after the checkout goes first on
# ``sys.path``) so imports resolve from the checkout. ``wholecell`` is
# deliberately EXCLUDED: it must keep resolving to the installed Cython-compiled
# tree, not the source checkout.
_SHADOW_SOURCE_PKGS = ("ecoli", "configs", "reconstruction", "validation")


def _purge_shadow_source_modules() -> None:
    """Evict cached ``_SHADOW_SOURCE_PKGS`` modules from ``sys.modules`` so a
    subsequent import resolves from the upstream checkout rather than a
    pre-imported copy from the installed ``vEcoli`` dist.

    Concretely, if ``reconstruction`` stays cached from the installed dist,
    ``reconstruction.ecoli.dataclasses.process.two_component_system`` keeps a
    pre-``2208460b`` copy whose stoich matrix has 41 rows, while the checkout's
    sim_data (loaded fresh) reports 45 ``modified_molecules`` -> MonomerCounts
    crashes with ``operands could not be broadcast together with shapes
    (45,) (41,)``.
    """
    for _pkg in _SHADOW_SOURCE_PKGS:
        for _m in [m for m in sys.modules if m == _pkg or m.startswith(_pkg + ".")]:
            del sys.modules[_m]


def _ensure_upstream() -> dict:
    """Import upstream ``EcoliSim`` once; cache its symbols.

    Imports ONLY ``EcoliSim`` (the pristine upstream checkout has no
    ``ecoli.composites.ecoli_composite`` and no migrated bigraph processes).
    """
    if _UPSTREAM:
        return _UPSTREAM

    # Resolve the vEcoli SOURCE checkout from $V2E_VECOLI_DIR. When it is unset
    # (or points at a directory that does not exist), fall THROUGH to whatever
    # ``ecoli``/``configs`` the installed environment provides (the public vEcoli
    # PyPI package) rather than a hardcoded developer path — and say so, so a run
    # is never ambiguous about which vEcoli it used (issue #131).
    vecoli_dir = os.environ.get("V2E_VECOLI_DIR", "")

    # NumPy 2.0: chromosome-replication uses np.in1d (~tick 1851). It is only
    # deprecated->removed upstream of where we monkeypatch; alias it so a pure
    # external wrapper survives without editing the vEcoli source.
    import numpy as _np
    if not hasattr(_np, "in1d"):
        _np.in1d = _np.isin

    # Pin the INSTALLED (Cython-compiled) wholecell tree BEFORE the source-only
    # upstream checkout goes on sys.path, so wholecell.* resolves to compiled
    # builds while ecoli.* loads from the upstream checkout.
    import importlib as _importlib
    import pkgutil as _pkgutil
    import wholecell as _wholecell
    for _mi in _pkgutil.walk_packages(_wholecell.__path__, "wholecell."):
        try:
            _importlib.import_module(_mi.name)
        except Exception:
            pass

    if vecoli_dir and os.path.isdir(vecoli_dir):
        sys.path.insert(0, vecoli_dir)
        print(f"[vecoli] using vEcoli SOURCE checkout: {vecoli_dir} "
              f"($V2E_VECOLI_DIR)")
    else:
        try:
            import importlib.metadata as _md
            _ver = _md.version("vEcoli")
        except Exception:
            _ver = "unknown"
        if vecoli_dir:
            print(f"[vecoli] $V2E_VECOLI_DIR={vecoli_dir!r} does not exist; "
                  f"falling back to the installed vEcoli package (v{_ver})")
        else:
            print(f"[vecoli] $V2E_VECOLI_DIR unset; using the installed vEcoli "
                  f"package (v{_ver})")
    # configs/ is a top-level package in the upstream checkout (ECOLI_DEFAULT_*)
    # that ecoli.composites.ecoli_master imports as ``from configs import ...``.

    # Make vivarium Registry.register idempotent (re-importing ecoli.* would
    # otherwise raise on duplicate parquet/updater/divider keys).
    import vivarium.core.registry as _vreg
    if not getattr(_vreg.Registry, "_idempotent_patched", False):
        _orig_register = _vreg.Registry.register

        def _idempotent_register(self, key, item, alternate_keys=tuple()):
            if key in self.registry:
                return None
            return _orig_register(self, key, item, alternate_keys)

        _vreg.Registry.register = _idempotent_register
        _vreg.Registry._idempotent_patched = True

    # Inject cloud-URI helpers the installed wholecell.utils.filepath may lack.
    import posixpath as _posixpath
    import wholecell.utils.filepath as _wfp
    if not hasattr(_wfp, "is_cloud_uri"):
        _wfp.is_cloud_uri = lambda path: str(path).startswith(
            ("s3://", "gs://", "gcs://"))
    if not hasattr(_wfp, "cloud_path_join"):
        _wfp.cloud_path_join = lambda *parts: _posixpath.join(*parts)

    # Drop any cached (possibly shadow) pure-python vEcoli source modules so
    # imports resolve from the upstream checkout (now first on sys.path), not
    # from the installed ``vEcoli`` dist in site-packages. See
    # _purge_shadow_source_modules for why ``reconstruction``/``validation``
    # matter (the MonomerCounts 45-vs-41 shape crash) and why ``wholecell`` is
    # excluded (it must stay the installed Cython-compiled tree).
    _purge_shadow_source_modules()

    # CRITICAL: clear vivarium's GLOBAL process_registry before the fresh
    # upstream import. The v2ecoli harness (imported first) has already
    # registered its *vendored* ecoli process classes (e.g. 'ecoli-complexation')
    # into this global registry — and those vendored classes inherit from a
    # DIFFERENT ``PartitionedProcess`` object than the one upstream's
    # ecoli_master sees. EcoliSim._retrieve_processes pulls classes from this
    # registry, so without clearing it, upstream's
    # ``issubclass(cls, PartitionedProcess)`` partition-split test returns
    # False for every partitioned process (they silently fall through to the
    # non-partitioned branch → no Requester/Evolver/Allocator → broken model).
    # The idempotent-register patch above would otherwise PRESERVE the stale
    # vendored entries. Clearing lets the fresh upstream import repopulate the
    # registry with its own, internally-consistent classes.
    _vreg.process_registry.registry.clear()

    from ecoli.experiments.ecoli_master_sim import EcoliSim

    _UPSTREAM.update(vecoli_dir=vecoli_dir, EcoliSim=EcoliSim)
    return _UPSTREAM


# ---------------------------------------------------------------------------
# vivarium ``_path`` topology expansion
# ---------------------------------------------------------------------------


def _expand_vivarium_wire(sub_schema, topo):
    """Expand a vivarium ``_path`` port topology into a process-bigraph wire.

    vivarium lets a single port wire to a base path while redirecting specific
    sub-stores elsewhere::

        {'_path': ('environment',), 'exchange': ('exchange',)}

    means "the port lives at ('environment',), except its ``exchange``
    sub-store is at ('exchange',)". process-bigraph has no ``_path`` key, so we
    materialize the equivalent nested wire dict: every typed sub-port of the
    port maps to ``base + [subkey]`` unless an explicit override redirects it.
    """
    base = topo.get('_path', ())
    base = list(base) if isinstance(base, (list, tuple)) else []
    overrides = {k: v for k, v in topo.items() if k != '_path'}

    subkeys = set(overrides)
    if isinstance(sub_schema, dict):
        subkeys |= {k for k in sub_schema if not str(k).startswith('_')}

    if not subkeys:
        # Opaque/scalar port — wire wholesale to the base path.
        return base

    result = {}
    for k in subkeys:
        if k in overrides:
            ov = overrides[k]
            if isinstance(ov, dict) and '_path' in ov:
                child = sub_schema.get(k) if isinstance(sub_schema, dict) else None
                result[k] = _expand_vivarium_wire(child, ov)
            else:
                result[k] = list(ov) if isinstance(ov, (list, tuple)) else ov
        else:
            result[k] = base + [k]
    return result


def _resolve_port_wires(wires, interface):
    """Post-process a topology dict: expand any vivarium ``_path`` ports."""
    inputs = interface.get('inputs', {})
    outputs = interface.get('outputs', {})
    for port, wire in list(wires.items()):
        if isinstance(wire, dict) and '_path' in wire:
            sub_schema = inputs.get(port)
            if sub_schema is None:
                sub_schema = outputs.get(port)
            wires[port] = _expand_vivarium_wire(sub_schema, wire)
    return wires


# ---------------------------------------------------------------------------
# Composite assembly
# ---------------------------------------------------------------------------


def build_upstream_cell_document(
    *,
    seed: int = 0,
    condition: str = "basal",
    sim_data_path: str = _UPSTREAM_SIMDATA_DEFAULT,
    time_step: float = 1.0,
    exclude_processes: list | None = None,
    core=None,
    verbose: bool = True,
):
    """Build the upstream-wrapped *cell document* WITHOUT constructing a Composite.

    Returns ``(cell_state, info)`` where ``cell_state`` is a flat dict carrying
    the wrapped process/step edges (each with a pre-built ``instance``), the
    upstream initial state, and the seeded partition runtime stores; ``info`` is
    a dict of diagnostics (process/step counts, partitioned names, the ``core``,
    the captured ``dry_mass_inc_dict`` if available, etc.).

    This is the reusable unit for cell division: a fresh daughter cell is just
    another call to this function whose biological stores (``bulk``/``unique``)
    are then overlaid with the mother's divided state. See
    :mod:`v2ecoli.library.upstream_division`.
    """
    from copy import deepcopy

    from bigraph_schema import (
        make_arrays_writeable as _make_arrays_writeable,
        tuples_to_lists as _tuples_to_lists,
    )
    from process_bigraph import wire_step_layers

    from v2ecoli.core import build_core
    from v2ecoli.library.vivarium_bridge import wrap_vivarium_instance

    up = _ensure_upstream()
    EcoliSim = up["EcoliSim"]
    if core is None:
        core = build_core()

    prev_cwd = os.getcwd()
    os.chdir(up["vecoli_dir"])
    try:
        # --- 1. Drive the upstream Ecoli composer via EcoliSim -------------
        _saved_argv = sys.argv
        sys.argv = sys.argv[:1]
        try:
            sim = EcoliSim.from_cli()
        finally:
            sys.argv = _saved_argv

        if not sim_data_path:
            raise ValueError(
                "build_upstream_cell_document requires a sim_data_path: pass it "
                "explicitly or set $V2E_UPSTREAM_SIMDATA to your vEcoli "
                "simData.cPickle (there is no hardcoded default path).")
        sim.config["condition"] = condition
        sim.config["seed"] = seed
        sim.config["sim_data_path"] = sim_data_path
        sim.config["time_step"] = time_step
        # Division uses a vivarium Composer roundtrip (Division step) that has
        # no process-bigraph equivalent; disable it so we exercise a single
        # generation. (Pure external wrapping of upstream division is a
        # separate, harder problem — see the report.)
        sim.config["divide"] = False
        sim.config["d_period"] = False
        sim.config["generations"] = None
        sim.config["progress_bar"] = False
        sim.config["emit_paths"] = []
        # Optional read-only listeners to drop (e.g. ones tripping on a
        # sim_data provenance skew). Not a vEcoli source edit — a config knob.
        if exclude_processes:
            existing = list(sim.config.get("exclude_processes", []) or [])
            sim.config["exclude_processes"] = existing + list(exclude_processes)

        # Capture the upstream ``Ecoli`` composer instance build_ecoli()
        # constructs internally (it is not stored on ``sim``) so we can read
        # ``load_sim_data.sim_data.expectedDryMassIncreaseDict`` — the per-media
        # dry-mass increment the faithful ``mass_distribution`` division
        # threshold needs. Harness-side monkeypatch (records the live instance);
        # NOT an upstream source edit.
        import ecoli.composites.ecoli_master as _emaster
        _captured: dict = {}
        _orig_ecoli_init = _emaster.Ecoli.__init__

        def _capturing_init(self, config, *a, **k):
            _orig_ecoli_init(self, config, *a, **k)
            _captured["composer"] = self

        _emaster.Ecoli.__init__ = _capturing_init
        try:
            sim.build_ecoli()
        finally:
            _emaster.Ecoli.__init__ = _orig_ecoli_init

        dry_mass_inc_dict = {}
        try:
            dry_mass_inc_dict = dict(
                _captured["composer"].load_sim_data.sim_data
                .expectedDryMassIncreaseDict)
        except Exception:
            dry_mass_inc_dict = {}

        processes = dict(sim.ecoli.processes)
        steps = dict(sim.ecoli.steps)
        flow = dict(sim.ecoli.flow)
        topology = sim.ecoli.topology
        cell_state = sim.generated_initial_state
        if verbose:
            print(f"[upstream] processes={len(processes)} steps={len(steps)} "
                  f"flow={len(flow)}", flush=True)

        # Shared PartitionedProcess instances live here as 1-tuples.
        partitioned = list((cell_state.get("process") or {}).keys())

        _make_arrays_writeable(cell_state)

        # --- 2. Wrap every instance + declare it as a composite edge -------
        # Upstream already split processes (continuous) vs steps (event), so
        # classify by dict membership.
        edges: dict = {}
        for name, inst in processes.items():
            edges[name] = (inst, False)   # process
        for name, inst in steps.items():
            edges[name] = (inst, True)    # step

        n_bulk = 0
        bulk = cell_state.get("bulk")
        if bulk is not None and hasattr(bulk, "__len__"):
            n_bulk = len(bulk)

        import numpy as _np

        wrapped_count = 0
        for name, (inst, as_step) in edges.items():
            wrapped = wrap_vivarium_instance(core, inst, name=name, as_step=as_step)
            interface = wrapped.interface()
            input_ports = set(interface.get("inputs", {}).keys())
            output_ports = set(interface.get("outputs", {}).keys())

            wires = _tuples_to_lists(topology.get(name, {})) or {}
            # Expand vivarium ``_path`` ports into nested process-bigraph wires.
            wires = _resolve_port_wires(wires, interface)
            for port in input_ports | output_ports:
                wires.setdefault(port, [port])
            output_wires = {k: v for k, v in wires.items() if k in output_ports}

            decl = {
                "_type": "step" if as_step else "process",
                "instance": wrapped,
                "_inputs": interface.get("inputs", {}),
                "_outputs": interface.get("outputs", {}),
                "inputs": deepcopy(wires),
                "outputs": deepcopy(output_wires),
            }
            if as_step:
                decl["priority"] = 1.0
            else:
                decl["interval"] = float(time_step)
            cell_state[name] = decl
            wrapped_count += 1

        # --- 3. Seed partition runtime stores ------------------------------
        cell_state.setdefault("global_time", 0.0)
        cell_state.setdefault("timestep", int(time_step))
        for proc_name in partitioned:
            cell_state.setdefault("next_update_time", {}).setdefault(
                proc_name, float(time_step))
            cell_state.setdefault("request", {}).setdefault(
                proc_name, {"bulk": []})
            cell_state.setdefault("allocate", {}).setdefault(
                proc_name, {"bulk": _np.zeros(n_bulk, dtype=_np.int64)})
        # Also seed next_update_time for non-partitioned steps (e.g. Metabolism,
        # TfBinding, ChromosomeStructure) whose topology wires next_update_time to
        # a process-specific subkey like ("next_update_time", "metabolism"). The
        # partitioned loop above only covers PartitionedProcess instances stored in
        # cell_state["process"]; direct Steps use different subkeys and would have
        # their update_condition raise KeyError("next_update_time") on every tick if
        # those subkeys are absent — causing them to be skipped (fail-closed gate),
        # which starves PolypeptideElongation of metabolite data → NaN crash.
        for _step_name, _step_topo in topology.items():
            _nup_wire = _step_topo.get("next_update_time")
            if isinstance(_nup_wire, (tuple, list)) and len(_nup_wire) == 2:
                _subkey = _nup_wire[1]
                cell_state.setdefault("next_update_time", {}).setdefault(
                    _subkey, float(time_step))
        # Allocator needs an RNG store (upstream wires allocator_rng).
        if "allocator_rng" not in cell_state:
            cell_state["allocator_rng"] = _np.random.RandomState(seed=int(seed))

        # --- 4. Step-flow priorities (THE key ordering fix) ----------------
        if flow:
            step_layers = wire_step_layers(cell_state, flow)
            ordered = [n for lvl in sorted(step_layers) for n in step_layers[lvl]]
            total = len(ordered)
            for idx, name in enumerate(ordered):
                edge = cell_state.get(name)
                if isinstance(edge, dict) and (
                        "priority" in edge or "interval" in edge):
                    edge["priority"] = float(total - idx)

        info = {
            "n_processes": len(processes),
            "n_steps": len(steps),
            "n_wrapped": wrapped_count,
            "n_partitioned": len(partitioned),
            "partitioned": partitioned,
            "process_names": list(processes.keys()),
            "step_names": list(steps.keys()),
            "core": core,
            "n_bulk": n_bulk,
            "dry_mass_inc_dict": dry_mass_inc_dict,
            "vecoli_dir": up["vecoli_dir"],
            "sim_data_path": sim_data_path,
            "condition": condition,
            "time_step": time_step,
            "seed": seed,
            "exclude_processes": list(exclude_processes or []),
        }
        return cell_state, info
    finally:
        os.chdir(prev_cwd)


def build_upstream_pbg_composite(
    *,
    seed: int = 0,
    condition: str = "basal",
    sim_data_path: str = _UPSTREAM_SIMDATA_DEFAULT,
    time_step: float = 1.0,
    exclude_processes: list | None = None,
    core=None,
    verbose: bool = True,
):
    """Build a process-bigraph ``Composite`` from pristine upstream vEcoli.

    Thin wrapper over :func:`build_upstream_cell_document`: it builds the flat
    single-cell document and constructs a ``Composite`` directly from it (no
    ``agents`` map — one generation only). Returns ``(composite, info)``.
    """
    from process_bigraph import Composite

    cell_state, info = build_upstream_cell_document(
        seed=seed, condition=condition, sim_data_path=sim_data_path,
        time_step=time_step, exclude_processes=exclude_processes, core=core,
        verbose=verbose)
    composite = Composite(
        dict(schema=dict(), state=cell_state), core=info["core"])
    return composite, info


# ---------------------------------------------------------------------------
# Smoke driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    os.environ.setdefault("PYTHONHASHSEED", "0")
    n = int(os.environ.get("N", "5"))
    composite, info = build_upstream_pbg_composite(
        seed=0, condition="basal",
        # The v2ecoli-built out/kb pickle predates upstream master's TCS
        # ``modified_molecules`` and fails to unpickle cleanly under master;
        # the vEcoli-built parca pickle is upstream-compatible. monomer_counts
        # is a read-only listener that trips on a TCS 41-vs-45 sim_data skew.
        exclude_processes=["monomer_counts_listener"],
    )
    print("BUILD OK", {k: info[k] for k in
                       ("n_processes", "n_steps", "n_partitioned")}, flush=True)
    t0 = time.time()
    done = 0
    try:
        for i in range(n):
            composite.run(1)
            done = i + 1
            print(f"  tick {done} OK global_time={composite.state.get('global_time')}",
                  flush=True)
        print(f"=== TICKED {done} steps in {time.time() - t0:.1f}s ===", flush=True)
    except Exception:
        import traceback
        print(f"=== STALLED at tick {done + 1} "
              f"global_time={composite.state.get('global_time')} ===", flush=True)
        traceback.print_exc()
