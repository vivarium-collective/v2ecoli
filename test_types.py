"""Unit tests for v2ecoli type system."""

import numpy as np
from bigraph_schema import allocate_core
from v2ecoli.types import ECOLI_TYPES
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from bigraph_schema.methods.apply import apply


def test_bulk_numpy_apply():
    """Test BulkNumpyUpdate apply: index-add on structured array."""
    dt = np.dtype([('id', 'U10'), ('count', 'i8')])
    state = np.array([('ATP', 100), ('GTP', 50)], dtype=dt)
    state.flags.writeable = True

    schema = BulkNumpyUpdate()
    update = [(np.array([0]), np.array([10])), (np.array([1]), np.array([-5]))]

    result, merges = apply(schema, state, update, ())
    assert result['count'][0] == 110
    assert result['count'][1] == 45
    assert merges == []
    print('test_bulk_numpy_apply PASSED')


def test_bulk_numpy_apply_none():
    """Test BulkNumpyUpdate apply with None update returns state."""
    dt = np.dtype([('id', 'U10'), ('count', 'i8')])
    state = np.array([('ATP', 100)], dtype=dt)

    schema = BulkNumpyUpdate()
    result, _ = apply(schema, state, None, ())
    assert np.array_equal(result, state)
    print('test_bulk_numpy_apply_none PASSED')


def test_unique_numpy_accumulate():
    """Test UniqueNumpyUpdate accumulates set/add/delete and flushes on update=True."""
    from v2ecoli.types.unique_numpy import clear_updater_registry
    clear_updater_registry()

    schema = UniqueNumpyUpdate()
    path = ('test', 'unique')

    dt = np.dtype([('_entryState', 'i1'), ('unique_index', 'i8'), ('mass', 'f8')])
    state = np.zeros(5, dtype=dt)
    state['_entryState'][:3] = 1
    state['unique_index'][:3] = [10, 11, 12]
    state['mass'][:3] = [1.0, 2.0, 3.0]
    state.flags.writeable = True

    # Accumulate a set update (should not flush yet)
    update1 = {'set': {'mass': np.array([10.0, 20.0, 30.0])}}
    result1, _ = apply(schema, state, update1, path)
    # Not flushed yet — masses should be unchanged
    assert result1['mass'][0] == 1.0

    # Now flush
    update2 = {'update': True}
    result2, _ = apply(schema, result1, update2, path)
    # After flush, masses should be updated
    assert result2['mass'][0] == 10.0
    assert result2['mass'][1] == 20.0
    assert result2['mass'][2] == 30.0
    print('test_unique_numpy_accumulate PASSED')


def test_type_registration():
    """Test that all types can be registered with a core."""
    core = allocate_core()
    core.register_types(ECOLI_TYPES)
    print('test_type_registration PASSED')


if __name__ == '__main__':
    test_bulk_numpy_apply()
    test_bulk_numpy_apply_none()
    test_unique_numpy_accumulate()
    test_type_registration()
