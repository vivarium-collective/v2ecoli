import numpy as np
import typing

from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, Integer, Array
from bigraph_schema.methods import infer, set_default, realize, render, wrap_default
from bigraph_schema.methods.serialize import serialize

from wholecell.utils.unit_struct_array import UnitStructArray
# UnitStructArray comes from upstream wholecell as an Unum-tagged structured
# array. v2ecoli internals don't introspect the unit field — it is
# round-tripped opaquely — so we use ``typing.Any`` as the annotation rather
# than carrying an Unum import here.
from typing import Any


@dataclass(kw_only=True)
class UnitsArray(Node):
    struct: Array = field(default_factory=Array)
    units: Any = field(default=None)

    def _serialize_state(self, state):
        if isinstance(state, dict):
            return state
        return {
            'struct': serialize(self.struct, state.struct_array),
            'units': serialize(self.units, state.units),
        }


@infer.dispatch
def infer(core, value: UnitStructArray, path: tuple = ()):
    data = {
        'struct': infer(core, value.struct_array, path=path + ('struct',))[0],
        'units': infer(core, value.units, path=path + ('units',))[0]}

    schema = UnitsArray(**data)
    schema = set_default(schema, value)

    return schema, []


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
    
