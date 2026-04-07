import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node
from bigraph_schema.methods import infer, set_default, default, serialize, realize, render, wrap_default, resolve

from unum import Unum


def unum_dimension(value):
    dimension = {}
    for unit, scale in value._unit.items():
        entry = value._unitTable[unit]
        base_unit = {
            unit: scale}
        if entry[0]:
            dimension_unit = entry[0]._unit
            base_key = list(dimension_unit.keys())[0]
            base_unit = {base_key: scale}

        dimension.update(
            base_unit)

    return dimension


@dataclass(kw_only=True)
class UnumUnits(Node):
    _dimension: typing.Dict = field(default_factory=dict)
    units: typing.Dict = field(default_factory=dict)
    magnitude: Node = field(default_factory=Node)


@infer.dispatch
def infer(core, value: Unum, path: tuple = ()):
    dimension = unum_dimension(value)
    magnitude, _ = infer(
        core,
        value._value,
        path+(value.strUnit(),))

    unum_data = {
        '_dimension': dimension,
        'units': value._unit,
        'magnitude': magnitude}

    schema = UnumUnits(**unum_data)
    schema = set_default(schema, value)

    return schema, []

@default.dispatch
def default(schema: UnumUnits):
    if schema._default:
        return schema._default
    else:
        return Unum(
            schema.units,
            default(schema.magnitude))

@serialize.dispatch
def serialize(schema: UnumUnits, state):
    if isinstance(state, dict):
        return state
    else:
        if state is None:
            if schema._default:
                return schema._default
            else:
                return default(schema)
        else:
            magnitude = serialize(
                schema.magnitude,
                state._value)

            return {
                'units': state._unit,
                'magnitude': magnitude}

@resolve.dispatch
def resolve(schema: UnumUnits, update: UnumUnits, path=()):
    # TODO: expand on this
    return schema

@realize.dispatch
def realize(core, schema: UnumUnits, encode, path=()):
    if isinstance(encode, Unum):
        return schema, encode, []
    else:
        _, magnitude, _ = realize(
            core,
            schema.magnitude,
            encode['magnitude'],
            path=path)

        return schema, Unum(
            encode['units'],
            magnitude), []

@render.dispatch
def render(schema: UnumUnits, defaults=False):
    data = {
        '_type': 'unum',
        '_dimension': schema._dimension,
        'units': schema.units,
        'magnitude': render(schema.magnitude)}

    return wrap_default(schema, data) if defaults else data
    
