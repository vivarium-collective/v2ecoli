"""Regression test for the bulk-update reconcile fix.

Root cause (upstream-wrapper cell_mass explosion): ``BulkNumpyUpdate`` had no
``reconcile`` dispatch, so when several steps in one execution layer each emitted
a ``bulk`` delta-list, they reconciled via ``reconcile(Node, ...)`` whose non-dict
branch is "last non-None wins" — silently discarding every bulk writer but the
last in that layer. In the upstream wrapper this dropped PolypeptideInitiation's
ribosomal-subunit consumption (lost to PolypeptideElongation's release in the same
layer) → subunits accumulated unconsumed → runaway ribosome initiation → cell_mass
explosion (5k→83k/gen).

The fix gives ``BulkNumpyUpdate`` a ``reconcile`` that CONCATENATES the delta-lists
so the additive ``apply`` (``count[idx] += value``) sums every writer's deltas.
"""
import numpy as np
import pytest

from bigraph_schema.methods.apply import apply
from bigraph_schema.methods.reconcile import reconcile
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate


@pytest.fixture
def schema():
    return BulkNumpyUpdate()


def test_reconcile_concatenates_same_index_deltas(schema):
    # Two writers touch the SAME molecule index in one layer: both must survive.
    # (Pre-fix Node last-wins returned only [(2, 3)], dropping the -10.)
    out = reconcile(schema, [[(2, -10)], [(2, 3)], None, [(5, 1)]])
    assert out == [(2, -10), (2, 3), (5, 1)]


def test_reconcile_single_writer_unchanged(schema):
    # A layer with one bulk writer is a no-op for the combine (the common case,
    # so non-step architectures like v2ecoli's per-process apply are untouched).
    assert reconcile(schema, [[(7, 4)]]) == [(7, 4)]


def test_reconcile_all_none(schema):
    assert reconcile(schema, [None, None]) is None


def test_combined_deltas_apply_additively(schema):
    # End-to-end: reconcile two opposing same-index deltas, then apply, and
    # confirm the store nets them (consumption + release), not last-wins.
    state = np.zeros(6, dtype=[("id", "U8"), ("count", "<i8")])
    state["count"][2] = 100
    combined = reconcile(schema, [[(2, -30)], [(2, 12)]])   # consume 30, release 12
    new_state, _ = apply(schema, state, combined, ())
    assert int(new_state["count"][2]) == 82                  # 100 - 30 + 12, not 112
