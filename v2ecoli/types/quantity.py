import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, String, Float, Link, Integer, Array, List, Tuple
from bigraph_schema.methods import infer, set_default, default, serialize, realize, render, wrap_default, resolve

import pint
ureg = pint.UnitRegistry()

@dataclass(kw_only=True)
class Quantity(Node):
    units: typing.Dict = field(default_factory=dict)
    magnitude: Node = field(default_factory=Node)


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

@serialize.dispatch
def serialize(schema: Quantity, state):
    if isinstance(state, dict):
        return state

    elif isinstance(state, int):
        return {
            'units': schema.units,
            'magnitude': serialize(
                schema.magnitude,
                state)}

    else:
        magnitude = serialize(
            schema.magnitude,
            state.magnitude)

        return {
            'units': schema.units,
            'magnitude': magnitude}

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
    
