import numpy as np
import typing

from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, Integer, Array
from bigraph_schema.methods import infer, set_default, serialize, realize, render, wrap_default

from v2ecoli.types.unum import Unum

from wholecell.utils.unit_struct_array import UnitStructArray


@dataclass(kw_only=True)
class UnitsArray(Node):
    struct: Array = field(default_factory=Array)
    units: Unum = field(default_factory=Unum)


@infer.dispatch
def infer(core, value: UnitStructArray, path: tuple = ()):
    data = {
        'struct': infer(core, value.struct_array, path=path + ('struct',))[0],
        'units': infer(core, value.units, path=path + ('units',))[0]}

    schema = UnitsArray(**data)
    schema = set_default(schema, value)

    return schema, []


@serialize.dispatch
def serialize(schema: UnitsArray, state):
    if isinstance(state, dict):
        encode = state
    else:
        encode = {
            'struct': serialize(schema.struct, state.struct_array),
            'units': serialize(schema.units, state.units)}

    return encode


@realize.dispatch
def realize(core, schema: UnitsArray, encode, path=()):
    if isinstance(encode, UnitStructArray):
        return schema, encode, []
    else:
        inner = tuple(
            realize(
                core,
                getattr(schema, key),
                encode[key],
                path+(key,))[1]
            for key in ['struct', 'units'])

        return schema, UnitStructArray(
            *inner), []

@render.dispatch
def render(schema: UnitsArray, defaults=False):
    data = {
        'struct': render(schema.struct),
        'units': render(schema.units)}

    return wrap_default(schema, data) if defaults else data
    
