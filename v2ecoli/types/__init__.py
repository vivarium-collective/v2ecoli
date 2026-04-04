from v2ecoli.types.unum import UnumUnits
from v2ecoli.types.quantity import Quantity
from v2ecoli.types.csr_matrix import CSRMatrix
from v2ecoli.types.units_array import UnitsArray
from v2ecoli.types.method import Method
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.process import StepInstance, ProcessInstance

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
}
