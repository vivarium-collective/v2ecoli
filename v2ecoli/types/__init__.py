# Patch bigraph-schema's serialize to call _serialize_state on custom Node
# subclasses that define it. This replaces the old @serialize.dispatch pattern
# that broke when serialize became a plain function (no longer plum-dispatched).
import importlib as _importlib
_ser_mod = _importlib.import_module('bigraph_schema.methods.serialize')

_original_serialize_leaf = _ser_mod._serialize_leaf

def _patched_serialize_leaf(schema, state, path):
    if hasattr(schema, '_serialize_state') and not isinstance(state, dict):
        return schema._serialize_state(state)
    return _original_serialize_leaf(schema, state, path)

_ser_mod._serialize_leaf = _patched_serialize_leaf

from v2ecoli.types.unum import UnumUnits
from v2ecoli.types.quantity import Quantity
from v2ecoli.types.csr_matrix import CSRMatrix
from v2ecoli.types.units_array import UnitsArray
from v2ecoli.types.method import Method
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.process import StepInstance, ProcessInstance
from v2ecoli.types.stores import InPlaceDict, SetStore, ListenerStore, AccumulateFloat

from process_bigraph import StepLink, ProcessLink

# Register align_parameters for custom array types so that string type
# expressions like 'bulk_array[id:string|count:integer|...]' and
# 'unique_array[domain_index:integer|...]' can be parsed by bigraph-schema.
from bigraph_schema.methods.handle_parameters import align_parameters as _align

@_align.dispatch
def _align_bulk(schema: BulkNumpyUpdate, parameters):
    if len(parameters) == 1 and isinstance(parameters[0], dict):
        return {'_data': parameters[0]}
    return {}

@_align.dispatch
def _align_unique(schema: UniqueNumpyUpdate, parameters):
    if len(parameters) == 1 and isinstance(parameters[0], dict):
        return {'_data': parameters[0]}
    return {}


# Register resolve dispatches for cross-type resolution.
# When multiple steps wire to the same store with different schema types
# (e.g. InPlaceDict vs Float for 'timestep'), bigraph-schema needs to know
# how to merge them. The more specific type wins.
from bigraph_schema.methods.resolve import resolve as _resolve
from bigraph_schema.schema import (
    Float as _Float, Node as _Node, Array as _Array, Map as _Map,
    Integer as _Integer, List as _List, Tuple as _Tuple, String as _String,
    Overwrite as _Overwrite, Boolean as _Boolean, Quote as _Quote,
)

# vEcoli-style type relaxation dispatches (from bigraph_types.py)
@_resolve.dispatch
def _resolve_integer_array(current: _Integer, update: _Array, path=None):
    return update

@_resolve.dispatch
def _resolve_array_integer(current: _Array, update: _Integer, path=None):
    return current

@_resolve.dispatch
def _resolve_tuple_list(current: _Tuple, update: _List, path=None):
    return current

@_resolve.dispatch
def _resolve_list_tuple(current: _List, update: _Tuple, path=None):
    return update

@_resolve.dispatch
def _resolve_list_map(current: _List, update: _Map, path=None):
    return update

@_resolve.dispatch
def _resolve_map_list(current: _Map, update: _List, path=None):
    return current

@_resolve.dispatch
def _resolve_overwrite_float(current: _Overwrite, update: _Float, path=None):
    return current

@_resolve.dispatch
def _resolve_float_overwrite(current: _Float, update: _Overwrite, path=None):
    return update

@_resolve.dispatch
def _resolve_overwrite_inplace(current: _Overwrite, update: InPlaceDict, path=None):
    return current

@_resolve.dispatch
def _resolve_inplace_overwrite(current: InPlaceDict, update: _Overwrite, path=None):
    return update

@_resolve.dispatch
def _resolve_overwrite_listener(current: _Overwrite, update: ListenerStore, path=None):
    return current

@_resolve.dispatch
def _resolve_listener_overwrite(current: ListenerStore, update: _Overwrite, path=None):
    return update

@_resolve.dispatch
def _resolve_string_map(current: _String, update: _Map, path=None):
    return update

@_resolve.dispatch
def _resolve_map_string(current: _Map, update: _String, path=None):
    return current

@_resolve.dispatch
def _resolve_quote_quote(current: _Quote, update: _Quote, path=None):
    return current

@_resolve.dispatch
def _resolve_list_integer(current: _List, update: _Integer, path=None):
    return current

@_resolve.dispatch
def _resolve_integer_list(current: _Integer, update: _List, path=None):
    return update

@_resolve.dispatch
def _resolve_list_float(current: _List, update: _Float, path=None):
    return current

@_resolve.dispatch
def _resolve_float_list(current: _Float, update: _List, path=None):
    return update

@_resolve.dispatch
def _resolve_list_array(current: _List, update: _Array, path=None):
    return update

@_resolve.dispatch
def _resolve_array_list(current: _Array, update: _List, path=None):
    return current

# Self-resolves for subtypes (needed to avoid ambiguity since
# ListenerStore is a subclass of InPlaceDict)
@_resolve.dispatch
def _resolve_listener_listener(current: ListenerStore, update: ListenerStore, path=None):
    return current

@_resolve.dispatch
def _resolve_set_set(current: SetStore, update: SetStore, path=None):
    return current

@_resolve.dispatch
def _resolve_bulk_bulk(current: BulkNumpyUpdate, update: BulkNumpyUpdate, path=None):
    return current

@_resolve.dispatch
def _resolve_unique_unique(current: UniqueNumpyUpdate, update: UniqueNumpyUpdate, path=None):
    return current

@_resolve.dispatch
def _resolve_inplace_inplace(current: InPlaceDict, update: InPlaceDict, path=None):
    return current

# Cross-type resolves: InPlaceDict is the generic fallback; more specific types win
_SPECIFIC_TYPES = (
    _Float, SetStore, ListenerStore, BulkNumpyUpdate, UniqueNumpyUpdate)

@_resolve.dispatch
def _resolve_inplace_float(current: InPlaceDict, update: _Float, path=None):
    return update

@_resolve.dispatch
def _resolve_float_inplace(current: _Float, update: InPlaceDict, path=None):
    return current

@_resolve.dispatch
def _resolve_inplace_set(current: InPlaceDict, update: SetStore, path=None):
    return update

@_resolve.dispatch
def _resolve_set_inplace(current: SetStore, update: InPlaceDict, path=None):
    return current

@_resolve.dispatch
def _resolve_inplace_bulk(current: InPlaceDict, update: BulkNumpyUpdate, path=None):
    return update

@_resolve.dispatch
def _resolve_bulk_inplace(current: BulkNumpyUpdate, update: InPlaceDict, path=None):
    return current

@_resolve.dispatch
def _resolve_inplace_unique(current: InPlaceDict, update: UniqueNumpyUpdate, path=None):
    return update

@_resolve.dispatch
def _resolve_unique_inplace(current: UniqueNumpyUpdate, update: InPlaceDict, path=None):
    return current

# ListenerStore vs other specific types
@_resolve.dispatch
def _resolve_listener_set(current: ListenerStore, update: SetStore, path=None):
    return current

@_resolve.dispatch
def _resolve_set_listener(current: SetStore, update: ListenerStore, path=None):
    return update

@_resolve.dispatch
def _resolve_listener_float(current: ListenerStore, update: _Float, path=None):
    return update

@_resolve.dispatch
def _resolve_float_listener(current: _Float, update: ListenerStore, path=None):
    return current

@_resolve.dispatch
def _resolve_listener_bulk(current: ListenerStore, update: BulkNumpyUpdate, path=None):
    return update

@_resolve.dispatch
def _resolve_bulk_listener(current: BulkNumpyUpdate, update: ListenerStore, path=None):
    return current

@_resolve.dispatch
def _resolve_listener_unique(current: ListenerStore, update: UniqueNumpyUpdate, path=None):
    return update

@_resolve.dispatch
def _resolve_unique_listener(current: UniqueNumpyUpdate, update: ListenerStore, path=None):
    return current

# Array (from emitter) vs specific types
@_resolve.dispatch
def _resolve_bulk_array(current: BulkNumpyUpdate, update: _Array, path=None):
    return current

@_resolve.dispatch
def _resolve_array_bulk(current: _Array, update: BulkNumpyUpdate, path=None):
    return update

@_resolve.dispatch
def _resolve_unique_array(current: UniqueNumpyUpdate, update: _Array, path=None):
    return current

@_resolve.dispatch
def _resolve_array_unique(current: _Array, update: UniqueNumpyUpdate, path=None):
    return update

# str (from emitter) vs typed stores
@_resolve.dispatch
def _resolve_str_inplace(current: str, update: InPlaceDict, path=None):
    return update

@_resolve.dispatch
def _resolve_inplace_str(current: InPlaceDict, update: str, path=None):
    return current

# str vs specific types
@_resolve.dispatch
def _resolve_str_float(current: str, update: _Float, path=None):
    return update

@_resolve.dispatch
def _resolve_float_str(current: _Float, update: str, path=None):
    return current

@_resolve.dispatch
def _resolve_str_bulk(current: str, update: BulkNumpyUpdate, path=None):
    return update

@_resolve.dispatch
def _resolve_bulk_str(current: BulkNumpyUpdate, update: str, path=None):
    return current

@_resolve.dispatch
def _resolve_str_unique(current: str, update: UniqueNumpyUpdate, path=None):
    return update

@_resolve.dispatch
def _resolve_unique_str(current: UniqueNumpyUpdate, update: str, path=None):
    return current

@_resolve.dispatch
def _resolve_str_listener(current: str, update: ListenerStore, path=None):
    return update

@_resolve.dispatch
def _resolve_listener_str(current: ListenerStore, update: str, path=None):
    return current

# Map vs InPlaceDict/ListenerStore
@_resolve.dispatch
def _resolve_map_inplace(current: _Map, update: InPlaceDict, path=None):
    return current

@_resolve.dispatch
def _resolve_inplace_map(current: InPlaceDict, update: _Map, path=None):
    return update

@_resolve.dispatch
def _resolve_map_listener(current: _Map, update: ListenerStore, path=None):
    return current

@_resolve.dispatch
def _resolve_listener_map(current: ListenerStore, update: _Map, path=None):
    return update


ECOLI_TYPES = {
    'unum': UnumUnits,
    'quantity': Quantity,
    'csr_matrix': CSRMatrix,
    'units_array': UnitsArray,
    'method': Method,
    'bulk_numpy': BulkNumpyUpdate,
    'unique_numpy': UniqueNumpyUpdate,
    # vEcoli-compatible aliases (used in schema_types.py type expressions)
    'bulk_array': BulkNumpyUpdate,
    'unique_array': UniqueNumpyUpdate,
    'step_instance': StepInstance,
    'process_instance': ProcessInstance,
    'step': StepLink,
    'process': ProcessLink,
    'inplace_dict': InPlaceDict,
    'set_store': SetStore,
    'listener_store': ListenerStore,
    'accumulate_float': AccumulateFloat,
}
