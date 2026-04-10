import numpy as np
import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, Integer, Array
from bigraph_schema.methods import infer, set_default, realize, render, wrap_default, reify_schema, validate
from bigraph_schema.methods.serialize import serialize

from scipy.sparse._csr import csr_matrix



@dataclass(kw_only=True)
class CSRMatrix(Node):
    _shape: typing.Tuple[int] = field(default_factory=tuple)
    _data: np.dtype = field(default_factory=lambda: np.dtype('float64'))
    data: Array = field(default_factory=Array)
    indices: Array = field(default_factory=Array)
    pointers: Array = field(default_factory=Array)

    def _serialize_state(self, state):
        if isinstance(state, dict):
            return state
        return {
            'data': serialize(self.data, state.data),
            'indices': serialize(self.indices, state.indices),
            'pointers': serialize(self.pointers, state.indptr),
        }


@infer.dispatch
def infer(core, value: csr_matrix, path: tuple = ()):
    data = {
        '_shape': value.shape,
        '_data': infer(core, value.dtype, path=path + ('_data',))[0],
        'data': infer(core, value.data, path=path + ('data',))[0],
        'indices': infer(core, value.indices, path=path + ('indices',))[0],
        'pointers': infer(core, value.indptr, path=path + ('pointers',))[0]}

    schema = CSRMatrix(**data)
    schema = set_default(schema, value)

    return schema, []


@realize.dispatch
def realize(core, schema: CSRMatrix, encode, path=()):
    if isinstance(encode, csr_matrix):
        return schema, encode, []
    else:
        inner = tuple([
            realize(
                core,
                getattr(schema, key),
                encode[key],
                path+(key,))[1]
            for key in ['data', 'indices', 'pointers']])

        return schema, csr_matrix(
            inner,
            shape=schema._shape), []

@reify_schema.dispatch
def reify_schema(core, schema: CSRMatrix, parameters):
    for key, parameter in parameters.items():
        subkey = core.access(parameter)
        setattr(schema, key, subkey)

    return schema
        

@render.dispatch
def render(schema: CSRMatrix, defaults=False):
    data = {
        '_type': 'csr_matrix',
        '_shape': schema._shape,
        '_data': render(schema._data),
        'data': render(schema.data),
        'indices': render(schema.indices),
        'pointers': render(schema.pointers)}

    return wrap_default(schema, data) if defaults else data
    
@validate.dispatch
def validate(core, schema: CSRMatrix, state):
    # TODO: validate csr_matrix
    return
