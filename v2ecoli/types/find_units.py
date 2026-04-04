from plum import dispatch

from bigraph_schema.schema import Node
from bigraph_schema.methods import default
from bigraph_schema import Core

from genEcoli.types.unum import UnumUnits
from genEcoli.types.quantity import Quantity



@dispatch
def find_units(schema: UnumUnits):
    return str(default(schema))

@dispatch
def find_units(schema: Quantity):
    return str(default(schema))

@dispatch
def find_units(schema: Node):
    found = {}
    for key in schema.__dataclass_fields__:
        print(f'{key} - Node')
        subunits = find_units(getattr(schema, key))
        if not subunits is None:
            found[key] = subunits
    if found:
        return found

@dispatch
def find_units(schema: dict):
    found = {}
    for key in schema.keys():
        print(f'{key} - dict')
        subunits = find_units(schema[key])
        if not subunits is None:
            found[key] = subunits
    if found:
        return found

@dispatch
def find_units(schema: object):
    if isinstance(schema, Core):
        return

    found = {}
    if hasattr(schema, '__dict__'):
        for key in schema.__dict__:
            if not key.startswith('_'):
                print(f'{key} - object')
                subunits = find_units(getattr(schema, key))
                if not subunits is None:
                    found[key] = subunits
        if found:
            return found

@dispatch
def find_units(schema):
    pass
