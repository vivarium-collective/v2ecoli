"""Phase 6 — DDAH backup DnaA-ATP hydrolysis at the datA locus.

Pairs with Phase 5 (RIDA): RIDA dominates while replisomes are
active, DDAH provides a constitutive background drain on DnaA-ATP
that runs whenever IHF binds datA. This first-cut implementation
models DDAH as a first-order hydrolysis at a small rate constant;
IHF gating + datA-specific box coordinates are deferred follow-ups.

These tests pin down:
  * The DDAH Step is wired into the cell_state.
  * Its listener emits a non-zero rate constant.
  * Over a multi-minute window, DDAH produces some non-zero flux
    (it does *something*; quantitative tuning is out of scope).
  * Disabling DDAH via the architecture flag falls back to a
    composite without it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


CACHE_DIR = 'out/cache'

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(
        not os.path.isdir(CACHE_DIR) and not os.environ.get('CI'),
        reason=f'cache dir {CACHE_DIR!r} not present; '
               f'rebuild with `python scripts/build_cache.py`',
    ),
]


@pytest.fixture(scope='module')
def composite():
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    return make_replication_initiation_composite(cache_dir=CACHE_DIR, seed=0)


def test_ddah_step_in_cell_state(composite):
    cell = composite.state['agents']['0']
    assert 'ddah' in cell, (
        f'ddah step missing from cell_state; keys: '
        f'{[k for k in cell if not k.startswith("_")][:10]}...')


def test_ddah_listener_emits_rate_constant(composite):
    composite.run(60.0)
    listener = composite.state['agents']['0'].get('listeners', {})
    ddah = listener.get('ddah', {})
    assert 'flux_atp_to_adp' in ddah
    assert 'rate_constant' in ddah
    assert float(ddah['rate_constant']) > 0


def test_ddah_produces_nonzero_cumulative_flux(composite):
    """Over ~5 minutes, DDAH should have hydrolyzed at least a few
    DnaA-ATP molecules. The default rate is small (Poisson process
    on a pool of ~30 molecules at 0.0005/s/molecule), so the expected
    cumulative count over this window is ~4-5; the assertion is a
    loose ``> 0`` to keep the test a property check rather than a
    point estimate.

    The ``flux_atp_to_adp`` listener reports the *current tick's*
    flux only — sparsely-sampled summing of that field consistently
    underreports. Use the cumulative-flux field instead, which
    monotonically accumulates across ticks."""
    composite.run(300.0)
    ddah = composite.state['agents']['0']['listeners'].get('ddah', {})
    cumulative = int(ddah.get('cumulative_flux_atp_to_adp') or 0)
    assert cumulative > 0, (
        f'DDAH cumulative flux over 5 min = {cumulative}; expected > 0. '
        f'Listener: {dict(ddah)}')


def test_disabling_ddah_removes_step(composite):
    """``enable_ddah=False`` skips the splice."""
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    c = make_replication_initiation_composite(
        cache_dir=CACHE_DIR, seed=0, enable_ddah=False)
    cell = c.state['agents']['0']
    assert 'ddah' not in cell, (
        f'ddah step present despite enable_ddah=False')
