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
