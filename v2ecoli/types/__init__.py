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

# Register resolve dispatches for cross-type resolution.
# When multiple steps wire to the same store with different schema types
# (e.g. InPlaceDict vs Float for 'timestep'), bigraph-schema needs to know
# how to merge them. The more specific type wins.
from bigraph_schema.methods.resolve import resolve as _resolve
from bigraph_schema.schema import Float as _Float, Node as _Node, Array as _Array

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


ECOLI_TYPES = {
    'unum': UnumUnits,
    'quantity': Quantity,
    'csr_matrix': CSRMatrix,
    'units_array': UnitsArray,
    'method': Method,
    'bulk_numpy': BulkNumpyUpdate,
    'unique_numpy': UniqueNumpyUpdate,
    'step_instance': StepInstance,
    'process_instance': ProcessInstance,
    'step': StepLink,
    'process': ProcessLink,
    'inplace_dict': InPlaceDict,
    'set_store': SetStore,
    'listener_store': ListenerStore,
    'accumulate_float': AccumulateFloat,
}
