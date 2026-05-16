import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, String, Float, Link, Integer, Array, List, Tuple
from bigraph_schema.methods import infer, set_default, default, realize, render, wrap_default, resolve, reify_schema
from bigraph_schema.methods.serialize import serialize
from bigraph_schema.methods.handle_parameters import align_parameters

# bigraph-schema 1.1.2+ ships its own Quantity in BASE_TYPES. Subclass it
# when present so v2ecoli's parameterized version passes isinstance checks
# against the upstream type and resolves cleanly during register_types.
try:
    from bigraph_schema.schema import Quantity as _BaseQuantity
except ImportError:
    _BaseQuantity = Node

import pint
# Share bigraph-schema's UnitRegistry so v2ecoli, bigraph-schema, and
# anything else on the bigraph stack use the same registry — and
# register it as pint's application registry so pint Quantities survive
# dill round-trips (cache.dill, save-state). The application registry
# is pickled by reference; unpickled Quantities resolve to whatever
# app-registry the loading process has configured, so arithmetic
# between cache-built and runtime-built Quantities stays consistent
# (otherwise pint raises "different registries" or silently produces
# garbage units — the pre-fix behavior caused ~8x mRNA accumulation in
# daughter sims because Kms/rates lost their registry binding on
# cache reload).
from bigraph_schema.units import units as ureg
pint.set_application_registry(ureg)

@dataclass(kw_only=True)
class Quantity(_BaseQuantity):
    """Pint-backed unit-bearing schema.

    Supports parameterized string syntax ``quantity[<magnitude>,<unit>]``
    or ``quantity[<unit>]`` (magnitude defaults to float). For example::

        'quantity[float,1/mol]'     # Avogadro-style scalar
        'quantity[fg]'              # shorthand — float magnitude
        'quantity[array[float],mmol/L]'  # array-valued concentration

    The ``_units`` field carries the unit string set at parse time;
    ``units`` is the dict form inferred from a pint Quantity instance.
    Both may be present — a runtime pint Quantity populates ``units``
    (via :func:`infer`), while a declared string populates ``_units``
    (via :func:`align_parameters` + :func:`reify_schema`).
    """
    _schema_keys = _BaseQuantity._schema_keys | frozenset({'_units'})
    units: typing.Dict = field(default_factory=dict)
    magnitude: Node = field(default_factory=lambda: Float())
    _units: str = ''

    def _serialize_state(self, state):
        if isinstance(state, dict):
            return state
        if isinstance(state, int):
            return {
                'units': self.units,
                'magnitude': serialize(self.magnitude, state),
            }
        return {
            'units': self.units,
            'magnitude': serialize(self.magnitude, state.magnitude),
        }


def units_dict(value):
    return {
        key: subvalue
        for key, subvalue in value.unit_items()}


@infer.dispatch
def infer(core, value: pint.Quantity, path: tuple = ()):
    units = units_dict(value)
    magnitude, _ = infer(
        core,
        value.magnitude,
        path+('magnitude',))

    data = {
        'units': units,
        'magnitude': magnitude}

    schema = Quantity(**data)
    schema = set_default(schema, value)

    return schema, []

@default.dispatch
def default(schema: Quantity):
    if schema._default:
        return schema._default
    else:
        return {
            'units': schema.units,
            'magnitude': default(schema.magnitude)}

@resolve.dispatch
def resolve(schema: Integer, update: Array, path=()):
    return update

@resolve.dispatch
def resolve(schema: Quantity, update: Quantity, path=()):
    if schema.units == update.units:
        # TODO: transfer default?
        return update


if _BaseQuantity is not Node:
    @resolve.dispatch
    def _resolve_quantity_classes(
        schema: typing.Type[_BaseQuantity],
        update: typing.Type['Quantity'],
        path=(),
    ):
        # Core.update_type passes the stored class and the new class
        # (not instances) to resolve. bigraph-schema 1.1.2+ seeds
        # BASE_TYPES['quantity'] with its own Quantity class, so our
        # register_types(ECOLI_TYPES) call hits this path. The v2ecoli
        # subclass extends the upstream one (adds `_units` and the
        # parameterized `quantity[...]` form) — supersede with `update`.
        return update

@resolve.dispatch
def resolve(schema: Quantity, update: Integer, path=()):
    return schema

@resolve.dispatch
def resolve(schema: Tuple, update: List, path=()):
    # TODO: expand on this
    return schema

@realize.dispatch
def realize(core, schema: Quantity, encode, path=()):
    if isinstance(encode, pint.Quantity):
        return schema, encode, []
    elif isinstance(encode, dict):
        _, magnitude, _ = realize(
            core,
            schema.magnitude,
            encode['magnitude'],
            path+('magnitude',))

        decode = (
            magnitude,
            tuple([(key, value)
                for key, value in schema.units.items()]))

    else:
        decode = (
            encode,
            tuple([(key, value)
                for key, value in schema.units.items()]))

    return schema, ureg.Quantity.from_tuple(
        decode), []


@render.dispatch
def render(schema: Quantity, defaults=False):
    # Prefer the parameterized string form when the schema was declared
    # via `quantity[<magnitude>,<unit>]`; fall back to the dict form for
    # inferred schemas (where only the `units` dict was ever populated).
    if schema._units:
        mag_render = render(schema.magnitude)
        if mag_render == 'float':
            result = f'quantity[{schema._units}]'
        else:
            result = f'quantity[{mag_render},{schema._units}]'
        return wrap_default(schema, result) if defaults else result
    data = {
        '_type': 'quantity',
        'units': schema.units,
        'magnitude': render(schema.magnitude)}
    return wrap_default(schema, data) if defaults else data


@align_parameters.dispatch
def align_parameters(schema: Quantity, parameters):
    """Map ``quantity[<magnitude>,<unit>]`` or ``quantity[<unit>]`` to
    schema fields. A single parameter is treated as the unit string;
    two parameters are magnitude type + unit string."""
    if len(parameters) == 2:
        return {'magnitude': parameters[0], '_units': parameters[1]}
    if len(parameters) == 1:
        return {'_units': parameters[0]}
    return {}


@reify_schema.dispatch
def reify_schema(core, schema: Quantity, parameters):
    """Populate `magnitude`, `_units`, and `units` on the schema from
    parsed parameters. The magnitude parameter may be a type name
    (resolved via `core.access`) or an already-constructed Node.

    The `units` dict is derived from `_units` so that downstream
    `realize` (which reads `schema.units`) can wrap bare numeric
    values with the correct pint unit — otherwise a default like
    `0.0` on a `quantity[float,1/mol]` field realizes to a
    dimensionless Quantity, silently losing the declared unit.
    """
    if 'magnitude' in parameters:
        mag_param = parameters['magnitude']
        if isinstance(mag_param, str):
            schema.magnitude = core.access(mag_param)
        elif isinstance(mag_param, Node):
            schema.magnitude = mag_param
    if '_units' in parameters:
        units_param = parameters['_units']
        if isinstance(units_param, str):
            schema._units = units_param
            try:
                schema.units = dict(ureg.Quantity(1, units_param).unit_items())
            except Exception:
                # Unparseable unit string — leave `units` empty; the
                # `_units` field still records the declared annotation.
                pass
    return schema

