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

from bigraph_schema.methods import render
from bigraph_schema.schema import Overwrite

from v2ecoli.library.ecoli_step import EcoliProcess, EcoliStep
from v2ecoli.library.schema_types import UNIQUE_TYPES


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


def translate_ports(core, ports, key=None):
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

    special = _special_type(key)

    # A vivarium leaf schema is identified by any underscore-prefixed key
    # (_default / _updater / _emit / _divider / _serializer / ...). Store
    # names (interior nodes) never start with an underscore.
    is_leaf = any(k.startswith('_') for k in ports)
    if is_leaf:
        if special is not None:
            return special
        if '_default' in ports:
            value = ports['_default']
            if isinstance(value, tuple) and value == ():
                value = []
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

    result = {}
    for subkey, subports in ports.items():
        if subkey.startswith('_'):
            continue
        result[subkey] = translate_ports(core, subports, key=subkey)
    return result


def wrap_vivarium_process(
    v1_cls,
    *,
    name=None,
    as_step=False,
    output_ports=None,
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

    class _VivariumBridge(base):
        name = proc_name

        def initialize(self, config):
            # vivarium processes take ``parameters`` positionally and merge
            # their own ``defaults`` in __init__, so self.parameters (user
            # overrides) is enough.
            self._v1 = v1_cls(self.parameters)
            self._typed_ports = translate_ports(self.core, self._v1.ports_schema())

        def inputs(self):
            return dict(self._typed_ports)

        def outputs(self):
            if write_ports is None:
                return dict(self._typed_ports)
            return {k: v for k, v in self._typed_ports.items() if k in write_ports}

        def update(self, state, interval=None):
            v1 = self._v1
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
