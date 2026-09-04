"""Save-state JSON round-trip tests.

AGENTS.md:43 and the `feedback_save_state_format` memory pin the
save-state format to bigraph-schema JSON:

    serialize(state) -> JSON -> deserialize  must reproduce the original

Two layers:

1. Low-level encoder — `v2ecoli.cache.save_json` / `load_json` must round-
   trip the full tricky-type vocabulary: numpy arrays (1D and structured),
   pint Quantities (scalar and array), sets, bytes, tuples. This runs
   always (no fixture required) so a broken encoder fails fast in CI.

2. Whole-state — `save_initial_state` / `load_initial_state` must round-
   trip the blessed pre-division fixture. Skipped if the fixture is not
   present (via the `predivision_state` fixture in conftest.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.cache import (
    load_initial_state, load_json, save_initial_state, save_json,
)
from v2ecoli.types.quantity import ureg

from _state_equal import deep_equal


pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# 1. Encoder round-trip over the tricky-type vocabulary.
# ---------------------------------------------------------------------------

def _synthetic_state():
    """A dict that exercises every branch of NumpyJSONEncoder."""
    struct_dtype = np.dtype([
        ('id', 'i8'),
        ('name', 'U16'),
        ('pos', 'f8', (3,)),
    ])
    struct = np.array(
        [(1, 'alpha', [0.0, 1.0, 2.0]), (2, 'beta', [3.0, 4.0, 5.0])],
        dtype=struct_dtype,
    )
    return {
        'scalar_int': 42,
        'scalar_float': 3.14,
        'scalar_bool': True,
        'scalar_str': 'hello',
        'none': None,
        'plain_array_f64': np.arange(10, dtype=np.float64),
        'plain_array_i32': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'structured': struct,
        'pint_scalar': 2.5 * ureg.fg,
        'pint_array': ureg.Quantity(np.arange(5, dtype=np.float64), 'mmol / L'),
        'a_set': {'a', 'b', 'c'},
        'some_bytes': b'\x00\x01\x02\xff',
        'nested': {
            'deep': {
                'arr': np.array([1.0, 2.0, 3.0]),
                'tags': ['x', 'y'],
            },
        },
    }


def test_json_encoder_roundtrips_tricky_types(tmp_path):
    """save_json → load_json reproduces numpy arrays, pint Quantities,
    structured dtypes, sets, bytes, and nested dicts without loss."""
    state = _synthetic_state()
    path = tmp_path / 'synth.json'
    save_json(state, str(path))
    reloaded = load_json(str(path))

    ok, reason = deep_equal(state, reloaded)
    assert ok, f'round-trip mismatch: {reason}'


def test_json_encoder_roundtrips_gzipped(tmp_path):
    """The .gz extension activates gzip transparently. Mid-pipeline
    artifacts (pre_division_state.json.gz) rely on this."""
    state = _synthetic_state()
    path = tmp_path / 'synth.json.gz'
    save_json(state, str(path))
    reloaded = load_json(str(path))

    ok, reason = deep_equal(state, reloaded)
    assert ok, f'gzipped round-trip mismatch: {reason}'


# ---------------------------------------------------------------------------
# 2. Whole-state round-trip on the blessed pre-division fixture.
# ---------------------------------------------------------------------------

def test_predivision_state_roundtrips(predivision_state, tmp_path):
    """The checkpoint that tests/test_model_behavior.py resumes from must
    round-trip losslessly through save_initial_state / load_initial_state.
    A break here means any resumed behavior test can silently start from a
    mutated state. Skipped by the fixture if the checkpoint is absent."""
    out_path = tmp_path / 'predivision.json'
    save_initial_state(predivision_state, str(out_path))
    reloaded = load_initial_state(str(out_path))

    ok, reason = deep_equal(predivision_state, reloaded)
    assert ok, f'pre-division round-trip mismatch: {reason}'


# ---------------------------------------------------------------------------
# 3. Binary (pickle) checkpoint round-trip — the per-generation lineage
#    checkpoint (gen_XXXX.pkl) writes binary, not JSON, to avoid the
#    tolist()/pretty-print stall on growing unique-molecule state
#    (sms-ecoli#210 / dispatch 313).
# ---------------------------------------------------------------------------

def _metadata_state():
    """A whole-state dict with a MetadataArray under 'unique', as the real
    checkpoint has. `unique_index` + `_entryState` fields are required by the
    MetadataArray constructor."""
    from v2ecoli.library.schema import MetadataArray
    dt = np.dtype([('unique_index', 'i8'), ('_entryState', 'i1'), ('x', 'f8')])
    base = np.array([(0, 1, 1.5), (1, 1, 2.5), (2, 1, 3.5)], dtype=dt)
    ma = MetadataArray(base, metadata={'next_unique_index': 3})
    return {
        'bulk': np.arange(6, dtype=np.int64),
        'unique': {'active_ribosome': ma},
        'environment': {'media_id': 'minimal'},
        'boundary': {'volume': 1.2},
    }


def test_pickle_checkpoint_roundtrips_metadata_array(tmp_path):
    """A `.pkl` path is written as binary pickle and round-trips the
    MetadataArray (values AND metadata) that the JSON path would have to
    reconstruct by hand."""
    state = _metadata_state()
    out_path = tmp_path / 'gen_0003.pkl'
    save_initial_state(state, str(out_path))

    # The file is a real binary pickle, not JSON text.
    head = out_path.read_bytes()[:1]
    assert head == b'\x80', f'expected a binary pickle, got first byte {head!r}'

    reloaded = load_initial_state(str(out_path))
    ma = reloaded['unique']['active_ribosome']
    assert ma.metadata == {'next_unique_index': 3}
    ok, reason = deep_equal(state, reloaded)
    assert ok, f'pickle round-trip mismatch: {reason}'


def test_json_extension_still_writes_json(tmp_path):
    """A non-pickle extension (the ParCa cache's initial_state.json) stays
    JSON text — build_cache.py and the cache-verify tests read it as JSON."""
    out_path = tmp_path / 'initial_state.json'
    save_initial_state(_metadata_state(), str(out_path))
    assert out_path.read_bytes().lstrip()[:1] == b'{'


def test_load_sniffs_json_in_pkl_named_file(tmp_path):
    """An in-flight checkpoint written as JSON before this change — a
    `gen_XXXX.pkl` that actually holds JSON — must still resume. load sniffs
    content, so it parses the JSON regardless of the .pkl name."""
    state = _metadata_state()
    legacy = tmp_path / 'gen_0002.pkl'
    # Write JSON the old way, then hand load_initial_state the .pkl name.
    save_initial_state(state, str(tmp_path / 'legacy.json'))
    legacy.write_bytes((tmp_path / 'legacy.json').read_bytes())

    reloaded = load_initial_state(str(legacy))
    ok, reason = deep_equal(state, reloaded)
    assert ok, f'legacy JSON-in-.pkl mismatch: {reason}'
