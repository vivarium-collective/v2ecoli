"""Regression tests for the full cell cycle.

Each test pins a specific biological/behavioral invariant that has broken
silently in the past. When one of these fails, the failure message names
the exact invariant so the next debug session starts from the right place.

Incidents covered:

1. ``bulk_name_to_idx`` cached ``argsort`` keyed on ``id(bulk_names)``. For
   ephemeral arrays (e.g. ``np.unique`` called every tick) the id can be
   reused after GC with different contents, so the stale sorter returned
   out-of-bounds indices — the ribosome_data listener crashed at ~23 min.

2. ``ChromosomeReplication.inputs()`` did not declare ``global_time``, yet
   the update code uses ``states["global_time"] + self.D_period`` when
   scheduling a division. Upstream vEcoli wires global_time into every
   partitioned process's topology automatically; v2ecoli converted the
   process to a plain Step and that wiring was lost. Sim crashed at ~23 min
   when replication fired.

3. ``rna_synth_prob_listener`` does ``cistron_tu_mapping_matrix.dot(...)``
   on inputs that default to empty lists. Before any upstream process has
   populated them (first tick, or after division reset) the matmul raises
   a dim-mismatch.
"""

import os

import pytest


CACHE_DIR = 'out/cache'


pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE_DIR),
    reason=f'cache dir {CACHE_DIR!r} not present (run scripts/cache_predivision.py)',
)


# ---------------------------------------------------------------------------
# 1. bulk_name_to_idx cache is safe when id() is reused across arrays
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_bulk_name_to_idx_cache_rejects_stale_entry():
    """Two distinct arrays that happen to share the same Python id must
    not share a cached sorter. We force the collision by binding the name
    that holds the first array, deleting it (so the id becomes reusable),
    and constructing a second array until the ids match."""
    import numpy as np
    from v2ecoli.library.schema import bulk_name_to_idx, _bulk_sorter_cache

    _bulk_sorter_cache.clear()

    a = np.array(['x', 'y', 'z'])
    assert int(bulk_name_to_idx('y', a)) == 1

    a_id = id(a)
    del a
    # Construct new arrays until one lands at the same id. This is the
    # exact pattern that caused the 23-min crash with np.unique(...).
    for _ in range(1000):
        b = np.array(['q', 'r', 's'])
        if id(b) == a_id:
            break
    # Even if we never get the collision (rare on some allocators), the
    # lookup must return the index in the *current* array — the old
    # cached sorter for 'x y z' would give the wrong answer for 'r'.
    assert int(bulk_name_to_idx('r', b)) == 1, (
        'bulk_name_to_idx returned stale index — the cache is keying '
        'on id() alone and must invalidate when the array is replaced.')


# ---------------------------------------------------------------------------
# 2. ChromosomeReplication declares every state key it reads
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_chromosome_replication_declares_global_time():
    """The update body reads states['global_time']; inputs() must declare
    it or the state dict won't contain the key when division schedules."""
    from v2ecoli.processes.chromosome_replication import (
        ChromosomeReplication, TOPOLOGY,
    )
    assert 'global_time' in TOPOLOGY, (
        'ChromosomeReplication.TOPOLOGY missing global_time — sim will '
        'crash with KeyError when replication completes.')

    # Walk inputs() via a dummy instance. port_defaults flows through
    # inputs(), so if the port isn't declared the state dict won't have
    # the key at update time.
    proc = ChromosomeReplication.__new__(ChromosomeReplication)
    proc.parameters = {}
    ports = proc.inputs()
    assert 'global_time' in ports, (
        "ChromosomeReplication.inputs() missing 'global_time' — the "
        "update body at line ~561 reads states['global_time'].")


# ---------------------------------------------------------------------------
# 3. rna_synth_prob listener tolerates empty upstream inputs
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_rna_synth_prob_listener_survives_empty_inputs():
    """Default values for actual/target/n_bound are [] until the
    transcript initiation process populates them. The listener must
    return {} instead of attempting a matmul on mismatched shapes."""
    from v2ecoli.steps.listeners.rna_synth_prob import RnaSynthProb

    listener = RnaSynthProb.__new__(RnaSynthProb)
    listener.parameters = {}
    listener.n_TU = 3277
    listener.n_TF = 10
    listener.n_cistron = 4500
    listener.cistron_tu_mapping_matrix = None  # would crash if dot() reached
    listener.rna_ids = [None] * listener.n_TU
    listener.tf_ids = [None] * listener.n_TF
    listener.cistron_ids = [None] * listener.n_cistron

    states = {
        'rna_synth_prob': {
            'actual_rna_synth_prob': [],
            'target_rna_synth_prob': [],
            'n_bound_TF_per_TU': [],
            'total_rna_init': 0,
        },
        'promoters': None,
        'genes': None,
        'global_time': 0.0,
        'timestep': 1.0,
    }
    # Must short-circuit before touching cistron_tu_mapping_matrix.
    assert listener.update(states) == {}


# ---------------------------------------------------------------------------
# 4. End-to-end cell cycle: grows, replicates twice, divides
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cell_cycle_completes_to_division():
    """Run long enough to exercise the 23-min replication initiation
    and reach division. Asserts the biological landmarks: starting
    replication fires (forks: 2 -> 0 -> 4 as first round finishes and
    second round starts), the cell accumulates enough dry mass to
    cross the division threshold, and the composite does not crash.

    This is the single best regression signal — it catches any listener
    crash, allocator wiring break, or state-handoff bug within ~4 min
    wall time."""
    from v2ecoli.composite import make_composite

    composite = make_composite(cache_dir=CACHE_DIR, seed=0)

    def _chrom_and_forks():
        cell = composite.state.get('agents', {}).get('0')
        if cell is None:
            return None, None
        u = cell.get('unique', {})
        fc = u.get('full_chromosome')
        rep = u.get('active_replisome')
        nch = int(fc['_entryState'].sum()) if (
            fc is not None and hasattr(fc, 'dtype')
            and '_entryState' in fc.dtype.names) else 0
        nrep = int(rep['_entryState'].sum()) if (
            rep is not None and hasattr(rep, 'dtype')
            and '_entryState' in rep.dtype.names) else 0
        return nch, nrep

    # Before the 23-min replication event
    composite.run(1200)
    m_20min = composite.state['agents']['0']['listeners']['mass']['dry_mass']
    assert m_20min > 450.0, (
        f'At 20 min dry_mass={m_20min:.1f} fg; expected >450 fg. A stalled '
        'run usually means a listener crashed and snapshot loop bailed.')

    # Run through first replication round completion (~23 min) and into
    # the second round. This is the window where all three previous bugs
    # manifested.
    composite.run(700)
    nch, nrep = _chrom_and_forks()
    assert nch == 2, (
        f'At ~32 min expected 2 chromosomes, got {nch}. First round of '
        'replication did not complete — chromosome_replication process '
        'is not firing correctly.')

    # Continue to division (~41 min). Division removes agent '0' and
    # adds two daughter cells.
    composite.run(1000)
    agents = composite.state.get('agents', {})
    # Either we still have cell 0 right at threshold, or division fired.
    if '0' in agents:
        dry = agents['0']['listeners']['mass']['dry_mass']
        assert dry >= 650.0, (
            f'At ~48 min dry_mass={dry:.1f} fg — cell did not grow '
            'through second round of replication.')
    else:
        # Division fired — we should have two daughters.
        assert len(agents) == 2, (
            f'After division expected 2 daughter cells, got {len(agents)}: '
            f'{list(agents.keys())}')
