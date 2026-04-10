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
