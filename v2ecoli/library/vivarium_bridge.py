"""
Automatic vivarium-1.0 -> process-bigraph conversion.

Extracted and adapted from Ryan Spangler's vEcoli ``composite`` branch
(https://github.com/vivarium-collective/vEcoli/tree/composite) — specifically
``ecoli/library/bigraph_types.py::translate_ports`` and the bridge pattern in
``ecoli/library/bigraph_bridge.py``.

The vEcoli branch migrated each process by *hand-rewriting* it to inherit from a
dual-API bridge base class (vivarium ``Process`` + process-bigraph ``Process``)
and to declare both the old ``ports_schema()`` and the new typed
``inputs()``/``outputs()``.  v2ecoli already absorbed that bridge as
``v2ecoli/library/ecoli_step.py`` (``EcoliStep`` / ``EcoliProcess``).

The one genuinely *automatic* piece — the part that lets you take an unmodified
vivarium-1.0 process and run it on the process-bigraph runtime without rewriting
its ports by hand — is ``translate_ports``: it reads a vivarium ``ports_schema()``
dict and infers a bigraph-schema typed-port tree from the ``_default`` values and
``_updater`` flags.  That is what this module provides, plus a thin
``wrap_vivarium_process`` adapter that uses it.

Two entry points:

- ``translate_ports(core, ports)`` — convert a vivarium ``ports_schema()`` dict
  into a v2ecoli typed-port dict (``{'_type': ..., '_default': ...}`` / nested
  dicts / type strings).
- ``wrap_vivarium_process(v1_cls)`` — build an ``EcoliProcess`` / ``EcoliStep``
  subclass whose ``inputs()`` / ``outputs()`` are derived automatically from the
  wrapped process's ``ports_schema()`` and whose ``update()`` delegates to the
  v1 ``next_update()``.

No dependency on ``vivarium-core``: the wrapped class is duck-typed (it only
needs ``ports_schema()`` and ``next_update()`` / ``update()``), matching the
"no vivarium-core import" contract of ``ecoli_step.py``.
"""

import os
import sys

from bigraph_schema.methods import render
from bigraph_schema.schema import Overwrite

from v2ecoli.library.ecoli_step import EcoliProcess, EcoliStep
from v2ecoli.library.schema_types import UNIQUE_TYPES


def _has_serializer_tags(obj):
    """True if any leaf string looks like a vEcoli serializer tag (``!Tag[...]``)."""
    if isinstance(obj, str):
        return obj.startswith("!") and "[" in obj
    if isinstance(obj, dict):
        return any(_has_serializer_tags(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_serializer_tags(v) for v in obj)
    return False


def _walk_deserialize(params, sers):
    """Recursively deserialize tagged strings with the given serializers."""
    def walk(v):
        if isinstance(v, str):
            for s in sers:
                try:
                    if s.can_deserialize(v):
                        return s.deserialize(v)
                except Exception:  # noqa: BLE001 — a serializer that mis-claims
                    continue
            return v
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(walk(x) for x in v)
        return v
    return walk(params)


def _deserialize_v1_params(params):
    """Resolve a wrapped vEcoli process's serialized config tags into real values
    before it reads them.

    vEcoli configs may carry serialized values as tagged strings —
    ``!ParameterSerializer[mecillinam>pbp2>binding_kf]`` (a parameter read from the
    fork's ``ecoli.library.parameters.param_store``) or ``!units[0 1 / second]`` (a
    pint Quantity). Without deserialization a process like ``gillespie`` receives
    ``"!ParameterSerializer[...]"`` as a bare string and crashes comparing it to a
    number.

    ``!units`` resolves against whatever ecoli is loaded, but ``!ParameterSerializer``
    reads the FORK's ``param_store`` (the installed vEcoli's lacks e.g. mecillinam
    params). So briefly activate the fork's ``ecoli.*`` (evict installed, fork first
    on ``sys.path``) for the deserialization, then restore — mirroring
    ``inject._fork_registry``. Best-effort: any string no serializer claims is left
    untouched; if the fork can't be resolved we fall back to the loaded serializers."""
    # Cheap fast path: most wrapped processes carry no serializer tags — skip the
    # (expensive, module-swapping) fork activation entirely for them.
    if not _has_serializer_tags(params):
        return params
    fork = os.environ.get("V2E_VECOLI_DIR")
    fork = os.path.abspath(os.path.expanduser(fork)) if fork else None
    saved, fork_added = {}, False
    try:
        if fork and os.path.isdir(fork):
            for k in [k for k in list(sys.modules)
                      if k == "ecoli" or k.startswith("ecoli.")]:
                saved[k] = sys.modules.pop(k)
            # Force the fork to the FRONT so a fresh ``import ecoli.*`` resolves to
            # it, not the installed vEcoli (which the run already has on sys.path).
            sys.path.insert(0, fork)
            fork_added = True
        from vivarium.core.registry import (Registry, Serializer,
                                             serializer_registry)
        sers = list(serializer_registry.registry.values())
        try:
            # Build FRESH serializer instances from the fork's serialize module:
            # the registry's instances were bound to the installed param_store at
            # registration; a fresh ParameterSerializer() binds the fork's (which
            # actually has e.g. mecillinam params). Put them first.
            # Importing the fork's ecoli re-registers emitters/processes into
            # vivarium's GLOBAL singleton registries (already populated by the
            # installed ecoli), which errors on duplicate keys — so make
            # Registry.register idempotent for the duration of the fork import.
            _orig_reg = Registry.register

            def _idem(self, key, item, *a, **k):
                if key in getattr(self, "registry", {}):
                    return
                return _orig_reg(self, key, item, *a, **k)

            Registry.register = _idem
            try:
                import ecoli.library.serialize as _fs  # noqa: F401
            finally:
                Registry.register = _orig_reg
            fork_sers = [c() for c in vars(_fs).values()
                         if isinstance(c, type) and issubclass(c, Serializer)
                         and c is not Serializer]
            sers = fork_sers + sers
        except Exception:  # noqa: BLE001 — fall back to registry serializers
            pass
        return _walk_deserialize(params, sers) if sers else params
    finally:
        if saved:
            for k in [k for k in list(sys.modules)
                      if k == "ecoli" or k.startswith("ecoli.")]:
                del sys.modules[k]
            sys.modules.update(saved)
        if fork_added:
            try:
                sys.path.remove(fork)
            except ValueError:
                pass


def _strip_pint_magnitudes(obj):
    """Recursively replace pint Quantities with their raw magnitude.

    The boundary unit-bridge for wrapped vivarium/vEcoli processes: they read raw
    numbers from stores and attach their own (unum) units in-process, but v2's
    stores hold pint Quantities. ndarrays/scalars pass through untouched, so this
    is a cheap no-op on states that carry no pint Quantities.
    """
    from v2ecoli.types.quantity import ureg
    if isinstance(obj, ureg.Quantity):
        return obj.magnitude
    if isinstance(obj, dict):
        return {k: _strip_pint_magnitudes(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_strip_pint_magnitudes(v) for v in obj)
    return obj


def _backfill_schema_defaults(state, ports_schema):
    """Ensure every leaf a v1 process DECLARES is present in ``state`` before its
    ``next_update`` runs — backfilling absent leaves with their schema
    ``_default`` (including ``None``), recursively.

    This is what vivarium's Engine guarantees a process: it always sees its full
    declared port state. The process-bigraph store layer drops a leaf whose value
    is ``None`` (it cannot type it), so a sentinel-``None`` field like the
    cell-wall model's ``wall_state.lattice`` (read as ``states[...]["lattice"]``,
    ``is None`` → build) would otherwise KeyError. Backfilling at call time
    presents the sentinel without needing the store to persist ``None``; the
    value the process WRITES back (e.g. the sampled ndarray) types and persists
    normally. Never overwrites a value already present in ``state``."""
    if not isinstance(state, dict) or not isinstance(ports_schema, dict):
        return state
    for port, schema in ports_schema.items():
        if not isinstance(schema, dict):
            continue
        if '_default' in schema or '_updater' in schema or '_emit' in schema:
            # leaf metadata dict
            if port not in state:
                state[port] = schema.get('_default')
        else:
            # nested ports subtree
            if port not in state or not isinstance(state[port], dict):
                if port not in state:
                    state[port] = {}
            if isinstance(state.get(port), dict):
                _backfill_schema_defaults(state[port], schema)
    return state


def _attach_pint_units(obj, unit):
    """Recursively wrap raw numbers as pint Quantities of ``unit``.

    The reverse of :func:`_strip_pint_magnitudes`: some wrapped vEcoli processes
    read a store with pint's API (e.g. metabolism_redux's
    ``boundary.external[x].to("mM")``) but v2 delivers a raw magnitude there.
    Existing Quantities and non-numbers pass through untouched.
    """
    from v2ecoli.types.quantity import ureg
    if isinstance(obj, ureg.Quantity):
        return obj
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return ureg.Quantity(obj, unit)
    if isinstance(obj, dict):
        return {k: _attach_pint_units(v, unit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_attach_pint_units(v, unit) for v in obj)
    return obj


def _special_type(key):
    """Return the v2ecoli type string for a well-known port name, or None.

    vEcoli's bulk and unique-molecule stores carry numpy structured-array
    defaults that should not be inferred field-by-field — they map to the
    registered ``bulk_array`` / ``unique_array[...]`` types instead. Mirrors
    ``v2ecoli/steps/partition.py::_typed_ports``.
    """
    if key in ('bulk', 'bulk_total'):
        return 'bulk_array'
    if key in UNIQUE_TYPES:
        return UNIQUE_TYPES[key]
    return None


# Parent store names under which a child port literally named ``bulk`` does
# NOT mean the structured bulk-molecule array. Under the partition machinery,
# ``request.<proc>.bulk`` holds a list of ``(index, count)`` request tuples and
# ``allocate.<proc>.bulk`` holds a per-process integer allocation vector —
# neither is the ``bulk_array`` structured type. Mapping them to bulk_array
# (the default for the name) produces a BulkNumpyUpdate schema that conflicts
# with the plain list/array values flowing through those stores.
_NON_BULK_PARENTS = ('request', 'allocate')


def translate_ports(core, ports, key=None, parent=None, in_partition=False):
    """Convert a vivarium ``ports_schema`` dict into a v2ecoli typed-port tree.

    Args:
        core: bigraph-schema core (``build_core()``) used for type inference.
        ports: a dict from a vivarium process's ``ports_schema()`` — leaves are
            metadata dicts (keys ``_default``, ``_updater``, ``_emit``, ...);
            interior nodes are dicts of sub-ports keyed by store name.
        key: the port name this subtree is bound to (used for special-name
            handling); supplied automatically during recursion.

    Returns:
        Either a type string (``'bulk_array'``, ``'overwrite[boolean]'``), a
        ``{'_type': ..., '_default': ...}`` dict for an inferred leaf, or a
        nested dict mirroring the port hierarchy.

    Type inference rules (faithful to vEcoli's ``translate_ports``):
      - A leaf with ``_default`` -> ``{'_type': render(core.infer(default)),
        '_default': default}``.
      - ``_updater == 'set'`` wraps the inferred type in ``overwrite[...]``
        (matching vivarium's ``set`` updater / ``_divider: set`` semantics).
      - The empty tuple ``()`` default is normalized to ``[]``.
      - Well-known store names (``bulk``, unique molecules) map straight to
        their registered array types, ignoring the raw numpy default.
    """
    if isinstance(ports, str):
        return ports
    if not isinstance(ports, dict):
        return 'node'

    # Once we descend into a ``request``/``allocate`` subtree (the partition
    # machinery), ``bulk`` no longer means the structured bulk-molecule array
    # at ANY depth — the Allocator nests it as request→{proc}→{bulk}. Track
    # this with a propagating flag rather than just the immediate parent.
    in_partition = in_partition or (parent in _NON_BULK_PARENTS)
    special = None if in_partition else _special_type(key)

    # A vivarium leaf schema is identified by any underscore-prefixed key
    # (_default / _updater / _emit / _divider / _serializer / ...). Store
    # names (interior nodes) never start with an underscore.
    is_leaf = any(k.startswith('_') for k in ports)
    if is_leaf:
        if special is not None:
            return special
        # vivarium updater-dict literal: ``{'_value': v, '_updater': n}``.
        # Some processes embed these directly as a port "default". Normalize
        # to the bare value ``v`` so the schema key soup ('_value'/'_updater')
        # doesn't leak into the inferred type / state (else, e.g., media_id
        # would become a dict instead of a string). Faithful to the fork's
        # build_ecoli_document converter gap fix.
        if '_value' in ports and '_default' not in ports:
            value = ports['_value']
            schema = core.infer(value)
            if ports.get('_updater') == 'set':
                schema = Overwrite(_value=schema)
            return {'_type': render(schema), '_default': value}
        if '_default' in ports:
            value = ports['_default']
            if isinstance(value, tuple) and value == ():
                value = []
            # Empty-collection defaults carry no element-type information, so
            # ``core.infer([])`` would pin an over-specific ``list[?]`` that
            # rejects the real runtime payload (e.g. the shared-process tuple
            # ``(process,)`` in the ('process',) store, or bulk request lists).
            # Map them to a permissive ``node`` instead (fork converter gap).
            if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
                # Keep the permissive ``node`` TYPE (don't core.infer an empty
                # collection — it over-specifies), but still carry the empty
                # ``_default`` so the store is seeded with it. Returning a bare
                # type string here dropped the default and regressed
                # test_translate_ports_empty_tuple_default_normalized.
                type_str = 'overwrite[node]' if ports.get('_updater') == 'set' else 'node'
                return {'_type': type_str, '_default': value}
            schema = core.infer(value)
            if ports.get('_updater') == 'set':
                schema = Overwrite(_value=schema)
            return {'_type': render(schema), '_default': value}
        # Metadata-only leaf (updater but no default).
        type_str = 'node'
        if ports.get('_updater') == 'set':
            type_str = 'overwrite[node]'
        return type_str

    if special is not None:
        return special

    # vivarium glob port: ``{'*': subschema}`` declares "any number of
    # dynamically-named children of this shape" — the bigraph equivalent is
    # ``map[<subschema>]`` (fork converter gap). Without this the literal
    # '*' key leaks into the typed schema and no real child store is created.
    non_underscore = [k for k in ports if not k.startswith('_')]
    if non_underscore == ['*']:
        sub = translate_ports(core, ports["*"], key="*", parent=key, in_partition=in_partition)
        if isinstance(sub, str):
            return f'map[{sub}]' if sub not in ('node', 'any') else 'map'
        # Non-trivial subschema → keep permissive map of nodes (the children
        # are typed structurally when their real values land).
        return 'map'

    result = {}
    for subkey, subports in ports.items():
        if subkey.startswith('_'):
            continue
        if subkey == '*':
            # Mixed concrete + glob keys: fold the glob into a permissive
            # passthrough so the concrete siblings still get typed.
            continue
        result[subkey] = translate_ports(
            core, subports, key=subkey, parent=key, in_partition=in_partition)
    return result


def wrap_vivarium_process(
    v1_cls,
    *,
    name=None,
    as_step=False,
    output_ports=None,
    defer_ports=None,
    strip_pint_ports=None,
    attach_pint_ports=None,
):
    """Build an ``EcoliProcess`` / ``EcoliStep`` subclass from a vivarium-1.0 class.

    The wrapped class is run *unmodified*: this adapter instantiates it,
    derives ``inputs()`` / ``outputs()`` from its ``ports_schema()`` via
    :func:`translate_ports`, and routes the process-bigraph ``update(state,
    interval)`` call to the vivarium ``next_update(timestep, states)`` method.

    Args:
        v1_cls: a vivarium-style process class. Must define ``ports_schema()``
            and ``next_update(timestep, states)`` (or ``update``). May define a
            ``defaults`` dict and a ``name`` — vivarium's own ``__init__``
            merges its ``defaults``, so config flows through normally.
        name: process-bigraph ``name`` for the wrapper (defaults to
            ``v1_cls.name`` or the class name).
        as_step: wrap as an ``EcoliStep`` (runs to convergence within a tick)
            instead of the default time-driven ``EcoliProcess``.
        output_ports: optional iterable of top-level port names that the
            process writes. vivarium ``ports_schema`` is bidirectional, so by
            default every port is declared in both ``inputs()`` and
            ``outputs()`` (safe over-declaration). Pass this to restrict the
            write surface, mirroring vEcoli's ``_output_ports`` convention.

    Returns:
        A new subclass of ``EcoliProcess`` (or ``EcoliStep``) ready to drop
        into a composite. Instantiate it like any v2ecoli process:
        ``WrappedCls(parameters, core=core)``.
    """
    base = EcoliStep if as_step else EcoliProcess
    proc_name = name or getattr(v1_cls, 'name', v1_cls.__name__)
    write_ports = set(output_ports) if output_ports is not None else None
    # defer_ports: declare these top-level ports as the top type ``any`` so the
    # composite's EXISTING store type governs. Needed when a swapped/injected
    # process shares a store another process already types (e.g. v2's mass deriver
    # owns listeners.mass as quantity[fg]; the bridged vEcoli process infers a
    # bare unitless float and the two won't subtype-resolve). ``any`` defers.
    defer = set(defer_ports) if defer_ports is not None else set()
    # strip_pint_ports: top-level ports whose pint Quantities should be stripped
    # to raw magnitudes on input. Needed where the wrapped process attaches its
    # OWN (unum) units to a raw number (e.g. metabolism_redux's
    # ``listeners.mass.cell_mass * units.fg``). Ports it reads as pint (e.g.
    # ``boundary.external`` via ``.to("mM")``) must be LEFT pint — so this is
    # per-port, declared in the injection spec, not a blanket strip.
    strip_pint = set(strip_pint_ports) if strip_pint_ports is not None else set()
    # attach_pint_ports: {port: unit} — wrap raw magnitudes as pint Quantities of
    # that unit for ports the process reads with pint's API (e.g. metabolism_redux
    # ``boundary.external[x].to("mM")``) where v2 delivers raw numbers.
    attach_pint = dict(attach_pint_ports) if attach_pint_ports is not None else {}

    class _VivariumBridge(base):
        name = proc_name

        def initialize(self, config):
            # vivarium processes take ``parameters`` positionally and merge
            # their own ``defaults`` in __init__, so self.parameters (user
            # overrides) is enough. First resolve vEcoli's serialized config
            # tags (!ParameterSerializer[...] / !units[...]) into real values —
            # e.g. gillespie's rate constants — which the wrapped v1 expects.
            self._v1 = v1_cls(_deserialize_v1_params(self.parameters))
            self._typed_ports = translate_ports(self.core, self._v1.ports_schema())

        def inputs(self):
            return {k: ({'_type': 'node'} if k in defer else v)
                    for k, v in self._typed_ports.items()}

        def outputs(self):
            ports = (self._typed_ports if write_ports is None
                     else {k: v for k, v in self._typed_ports.items()
                           if k in write_ports})
            return {k: ({'_type': 'node'} if k in defer else v) for k, v in ports.items()}

        def update(self, state, interval=None):
            v1 = self._v1
            # Present the v1 process its FULL declared state: backfill any leaf the
            # pbg store dropped (notably sentinel-None fields it can't type, e.g.
            # cell-wall's wall_state.lattice) with the schema default — mirroring
            # vivarium's Engine guarantee. Without it a `states[...]["lattice"]`
            # read KeyErrors on tick 0.
            if isinstance(state, dict):
                _backfill_schema_defaults(state, v1.ports_schema())
            # Per-port boundary unit-bridge: for ports the wrapped process attaches
            # its own (unum) units to, strip v2's pint Quantities to raw magnitudes
            # (e.g. metabolism_redux's ``cell_mass * units.fg``). Ports it reads as
            # pint (``boundary.external`` via ``.to("mM")``) are left untouched.
            if (strip_pint or attach_pint) and isinstance(state, dict):
                state = {
                    k: (_strip_pint_magnitudes(v) if k in strip_pint
                        else _attach_pint_units(v, attach_pint[k]) if k in attach_pint
                        else v)
                    for k, v in state.items()
                }
            if hasattr(v1, 'next_update'):
                return v1.next_update(interval or 0, state)
            return v1.update(state, interval)

    _VivariumBridge.__name__ = f'{v1_cls.__name__}Bridge'
    _VivariumBridge.__qualname__ = _VivariumBridge.__name__
    _VivariumBridge.__doc__ = (
        f'Auto-generated process-bigraph bridge for vivarium process '
        f'``{v1_cls.__module__}.{v1_cls.__name__}``.'
    )
    return _VivariumBridge


def wrap_vivarium_instance(
    core,
    v1_instance,
    *,
    name=None,
    as_step=False,
    output_ports=None,
):
    """Wrap an ALREADY-INSTANTIATED vivarium process/step for process-bigraph.

    Unlike :func:`wrap_vivarium_process` (which takes a *class* and re-builds
    the v1 object from config), this adapter wraps a live instance unchanged.
    That is essential for the partitioning machinery: an upstream vEcoli
    ``Requester`` and ``Evolver`` share a single ``PartitionedProcess`` object
    (passed through the ``('process',)`` store as a 1-tuple), and re-building
    them from config would shatter that shared identity. By wrapping the exact
    instances the upstream ``Ecoli`` composer produced, the shared-instance
    coordination is preserved verbatim.

    The wrapper derives ``inputs()``/``outputs()`` from the instance's
    ``ports_schema()`` via :func:`translate_ports`, exposes the instance's
    ``ports_schema()`` (so process-bigraph's ``realize_link`` can enrich port
    defaults), routes ``update(state, interval)`` to the v1
    ``next_update(timestep, states)``, and — for steps — honors the v1
    ``update_condition`` gate via ``perform_update`` (this is what implements
    the per-process ``next_update_time <= global_time`` partition timing).

    The wrapped class subclasses :class:`EcoliProcess` / :class:`EcoliStep`
    but is constructed via ``__new__`` so the v1 instance is injected directly
    (no re-instantiation, no config round-trip).
    """
    base = EcoliStep if as_step else EcoliProcess
    proc_name = name or getattr(v1_instance, 'name', type(v1_instance).__name__)
    typed = translate_ports(core, v1_instance.ports_schema())
    write = set(output_ports) if output_ports is not None else None

    class _InstanceBridge(base):
        name = proc_name

        def inputs(self):
            return dict(self._typed_ports)

        def outputs(self):
            if self._write_ports is None:
                return dict(self._typed_ports)
            return {k: v for k, v in self._typed_ports.items()
                    if k in self._write_ports}

        def ports_schema(self):
            return self._v1.ports_schema()

        def update(self, state, interval=None):
            v1 = self._v1
            # Backfill dropped/sentinel-None declared leaves (see process bridge).
            if isinstance(state, dict):
                _backfill_schema_defaults(state, v1.ports_schema())
            # Per-port boundary unit-bridge: for ports the wrapped process attaches
            # its own (unum) units to, strip v2's pint Quantities to raw magnitudes
            # (e.g. metabolism_redux's ``cell_mass * units.fg``). Ports it reads as
            # pint (``boundary.external`` via ``.to("mM")``) are left untouched.
            if (strip_pint or attach_pint) and isinstance(state, dict):
                state = {
                    k: (_strip_pint_magnitudes(v) if k in strip_pint
                        else _attach_pint_units(v, attach_pint[k]) if k in attach_pint
                        else v)
                    for k, v in state.items()
                }
            if hasattr(v1, 'next_update'):
                return v1.next_update(interval or 0, state)
            return v1.update(state, interval)

        def perform_update(self, state):
            # Honor the vivarium update_condition gate (Requester/Evolver
            # variable-timestep gate keyed on next_update_time/global_time).
            v1 = self._v1
            if hasattr(v1, 'update_condition'):
                try:
                    return bool(v1.update_condition(state.get('timestep', 1), state))
                except Exception as _gate_exc:
                    # FAIL CLOSED. The Evolver/Requester variable-timestep gate is
                    # the ONLY thing that stops a partitioned process from
                    # re-applying its bulk delta more than once per scheduled
                    # step. If the gate can't be evaluated, returning True (run
                    # anyway) lets the evolver re-apply every tick → runaway
                    # synthesis / mass explosion (observed: cell_mass 5k→98k over
                    # one generation on the upstream wrapper). A gate we cannot
                    # evaluate must SKIP, not run. Surface the cause, don't swallow.
                    import sys as _sys
                    print(f"[vivarium_bridge] update_condition raised for "
                          f"{type(v1).__name__}; skipping (fail-closed): {_gate_exc!r}",
                          file=_sys.stderr, flush=True)
                    return False
            return True

    _InstanceBridge.__name__ = f'{type(v1_instance).__name__}InstanceBridge'
    _InstanceBridge.__qualname__ = _InstanceBridge.__name__

    inst = _InstanceBridge.__new__(_InstanceBridge)
    inst.parameters = getattr(v1_instance, 'parameters', {}) or {}
    inst.core = core
    inst._v1 = v1_instance
    inst._typed_ports = typed
    inst._write_ports = write
    return inst
