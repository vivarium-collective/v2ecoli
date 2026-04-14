import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, String, Float, Link, Integer, Array, List, Tuple
from bigraph_schema.methods import infer, set_default, default, realize, render, wrap_default, resolve
from bigraph_schema.methods.serialize import serialize

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
class Quantity(Node):
    units: typing.Dict = field(default_factory=dict)
    magnitude: Node = field(default_factory=Node)

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
    data = {
        '_type': 'quantity',
        'units': schema.units,
        'magnitude': render(schema.magnitude)}

    return wrap_default(schema, data) if defaults else data
    
