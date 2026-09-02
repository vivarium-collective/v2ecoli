"""Regression tests for item 101's n_workers durable fix.

An earlier concrete default of ``n_workers=2`` silently shadowed the cluster-derived
``RAY_SHARDS_DEFAULT`` env var viva-api's own dispatch code computes (``RayProtocolRuntime``
only reads it when given ``None`` explicitly) — see ``batch_lineage_ray``'s own module
docstring. These tests pin the fix: the default is ``None``, negative/zero overrides still
reject, and ``prewarm_lineage_pool`` forwards whatever it's given to the protocol runtime
unchanged.
"""
import inspect
from unittest.mock import patch

import pytest

from v2ecoli.composites.lineage_ray_batch import lineage_ray_batch
from v2ecoli.workflow.batch_lineage_ray import prewarm_lineage_pool


def test_lineage_ray_batch_n_workers_defaults_to_none():
    sig = inspect.signature(lineage_ray_batch)
    assert sig.parameters["n_workers"].default is None


def test_prewarm_lineage_pool_rejects_zero():
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        prewarm_lineage_pool(core=object(), n_workers=0)


def test_prewarm_lineage_pool_rejects_negative():
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        prewarm_lineage_pool(core=object(), n_workers=-1)


def test_prewarm_lineage_pool_forwards_none_to_runtime():
    # None must NOT be rejected by the >= 1 check, and must reach the runtime
    # unchanged so it can fall through to RAY_SHARDS_DEFAULT — the actual fix.
    core = object()
    with patch("process_bigraph.protocols.ray.get_or_create_runtime") as mock_runtime:
        prewarm_lineage_pool(core=core, n_workers=None)
    mock_runtime.assert_called_once_with(core, n_shards_default=None)


def test_prewarm_lineage_pool_forwards_explicit_override():
    # An explicit override must still reach the runtime unchanged (regression
    # guard: the fix must not break the deliberate-cap use case).
    core = object()
    with patch("process_bigraph.protocols.ray.get_or_create_runtime") as mock_runtime:
        prewarm_lineage_pool(core=core, n_workers=5)
    mock_runtime.assert_called_once_with(core, n_shards_default=5)
