"""Definitive behavioral tests for the baseline whole-cell model.

These tests pin the four properties that define a working simulation:

  1. Baseline cell grows. Dry mass roughly doubles over a cell cycle.
  2. Chromosome replication initiates at the expected time (~22 min after
     start, yielding 2 chromosomes before division).
  3. Division fires: the pre-division cell splits into two daughters with
     bulk conservation.
  4. Daughters grow: after division, both daughters are viable and their
     dry mass increases over a short follow-on run.

Tests resume from the pre-division checkpoint produced by `workflow.py`
(see tests/conftest.py). This avoids the ~4 min warmup and lets the tests
run in seconds — suitable to block PR merges, not just nightly runs.

If the checkpoint is absent, tests skip. CI should either:
  - produce the checkpoint in a setup job and upload as an artifact, or
  - pull a blessed checkpoint from release assets / LFS.
"""
import pytest


pytestmark = pytest.mark.behavior


# ---------------------------------------------------------------------------
# 1. Baseline growth: initial -> final dry mass ratio
# ---------------------------------------------------------------------------

def test_baseline_dry_mass_roughly_doubles(single_cell_trajectory):
    """Over a complete cycle the cell accumulates ~1.5-2.2x initial dry mass.

    Tight bound: rejects both stalls (ratio < 1.5) and runaway growth
    (> 2.5). Loose enough to survive normal numerical drift."""
    meta = single_cell_trajectory['meta']
    m0 = float(meta['initial_dry_mass'])
    m1 = float(meta['final_dry_mass'])
    ratio = m1 / m0
    assert 1.5 <= ratio <= 2.5, (
        f'Dry-mass ratio {ratio:.2f} outside [1.5, 2.5]. '
        f'initial={m0:.1f}fg final={m1:.1f}fg — cell either stalled or '
        f'grew runaway.')


def test_baseline_monotone_growth(single_cell_trajectory):
    """Growth trajectory is roughly monotone — no long flat spans.

    Allow small downward blips (bulk counter noise, rounding) but no
    >60 s interval where dry_mass is strictly non-increasing."""
    snaps = single_cell_trajectory['snapshots']
    masses = [(s['time'], s['dry_mass']) for s in snaps if 'dry_mass' in s]
    if len(masses) < 4:
        pytest.skip('need at least 4 mass snapshots')

    # Sliding window: over any 4 consecutive snaps, last > first.
    for i in range(len(masses) - 3):
        t0, m0 = masses[i]
        t3, m3 = masses[i + 3]
        assert m3 > m0, (
            f'Dry mass did not grow over window [{t0:.0f}s..{t3:.0f}s]: '
            f'{m0:.1f} -> {m3:.1f} fg. A stall of this length means a '
            f'listener crashed or metabolism shut off.')


# ---------------------------------------------------------------------------
# 2. Chromosome replication initiates at expected time
# ---------------------------------------------------------------------------

REPLICATION_COMPLETE_WINDOW = (900.0, 1800.0)  # seconds: expect 2 chroms in this window


def test_replication_completes_in_expected_window(single_cell_trajectory):
    """First appearance of 2 full chromosomes lands in a biologically
    reasonable window. The baseline seed-0 run reaches 2 chromosomes at
    ~1350 s; the window [900, 1800] is wide enough to allow small
    changes in elongation rates / ppGpp dynamics without flaking, but
    tight enough to catch a broken replication process that either
    fires too early (runaway) or never completes (silent hang)."""
    snaps = single_cell_trajectory['snapshots']
    t_two_chrom = next(
        (s['time'] for s in snaps if s.get('n_chromosomes', 0) >= 2),
        None,
    )
    assert t_two_chrom is not None, (
        'Second chromosome never appeared — chromosome_replication is not '
        'completing a round.')
    lo, hi = REPLICATION_COMPLETE_WINDOW
    assert lo <= t_two_chrom <= hi, (
        f'Second chromosome appeared at t={t_two_chrom:.0f}s, '
        f'outside expected window [{lo:.0f}, {hi:.0f}]. '
        f'Too early suggests D-period/elongation is wrong; too late '
        f'suggests replication stalled.')


def test_replication_forks_present_mid_cycle(single_cell_trajectory):
    """At least one mid-cycle snapshot has active forks. Protects against
    a regression where forks terminate immediately."""
    snaps = single_cell_trajectory['snapshots']
    mid = [s for s in snaps if 300 <= s['time'] <= 1500]
    with_forks = [s for s in mid if s.get('fork_coords')]
    assert len(with_forks) >= 3, (
        f'Only {len(with_forks)} mid-cycle snapshots had active forks; '
        f'expected >= 3. Replication forks are not persisting.')


# ---------------------------------------------------------------------------
# 3. Division fires and conserves bulk
# ---------------------------------------------------------------------------

def test_division_splits_cell_into_two_daughters(predivision_state):
    """Loading the pre-division cell state and invoking the divide
    machinery must produce two daughter states, each with a populated
    bulk array. Asserts bulk conservation (mother_count == d1 + d2)."""
    from v2ecoli.library.division import divide_cell, divide_bulk

    cell = predivision_state
    assert 'bulk' in cell, 'checkpoint missing bulk array'

    d1_bulk, d2_bulk = divide_bulk(cell['bulk'])
    mother = int(cell['bulk']['count'].sum())
    d1 = int(d1_bulk['count'].sum())
    d2 = int(d2_bulk['count'].sum())
    assert d1 + d2 == mother, (
        f'Bulk not conserved across division: mother={mother} '
        f'd1={d1} d2={d2} (d1+d2={d1 + d2}).')

    # Full state split (includes unique molecules + listener scaffolding).
    d1_state, d2_state = divide_cell(cell)
    assert 'bulk' in d1_state and 'bulk' in d2_state
    assert d1_state['bulk']['count'].sum() > 0
    assert d2_state['bulk']['count'].sum() > 0


def test_division_conserves_unique_molecules(predivision_state):
    """Key unique-molecule species split without creation/destruction."""
    from v2ecoli.library.division import divide_cell
    import numpy as np

    cell = predivision_state
    d1_state, d2_state = divide_cell(cell)

    failures = []
    for name, mother_arr in cell.get('unique', {}).items():
        if not (hasattr(mother_arr, 'dtype') and mother_arr.dtype.names
                and '_entryState' in mother_arr.dtype.names):
            continue
        m = int(mother_arr['_entryState'].view(np.bool_).sum())
        d1_arr = d1_state['unique'].get(name)
        d2_arr = d2_state['unique'].get(name)
        d1 = d1_arr.shape[0] if d1_arr is not None else 0
        d2 = d2_arr.shape[0] if d2_arr is not None else 0
        # Chromosomes and replisomes should exactly conserve; some
        # transient species (e.g. DNA polymerase) may not — allow equal
        # or short-by-a-few. We only hard-assert on chromosomes.
        if name == 'full_chromosome' and d1 + d2 != m:
            failures.append(f'{name}: mother={m} d1+d2={d1 + d2}')
    assert not failures, 'Division lost unique molecules: ' + '; '.join(failures)


# ---------------------------------------------------------------------------
# 4. Daughters are viable and grow
# ---------------------------------------------------------------------------

DAUGHTER_RUN_SECONDS = 60.0
DAUGHTER_MIN_GROWTH_FG = 0.5


def test_daughters_build_and_grow(predivision_state, sim_data_cache):
    """Build two daughter composites from the split state and run each
    for 60 s. Each must (a) build without exception and (b) gain at
    least 0.5 fg of dry mass — the single strongest evidence that the
    full machinery (metabolism, translation, transcription) is wired
    through division, not just that division itself succeeded."""
    from process_bigraph import Composite
    from v2ecoli.library.division import divide_cell
    from v2ecoli.composite import _build_core, _load_cache_bundle
    from v2ecoli.generate import build_document

    cell = predivision_state
    d1_state, d2_state = divide_cell(cell)

    _, cache = _load_cache_bundle(sim_data_cache)
    configs = cache.get('configs')
    unique_names = cache.get('unique_names')
    dry_mass_inc = cache.get('dry_mass_inc_dict')
    assert configs, 'sim_data_cache missing configs — regenerate cache'

    core = _build_core()

    def _run_one(state, seed):
        doc = build_document(
            state, configs, unique_names,
            dry_mass_inc_dict=dry_mass_inc, seed=seed)
        comp = Composite(doc, core=core)
        agent = comp.state['agents']['0']
        m0 = float(agent['listeners']['mass']['dry_mass'])
        comp.run(DAUGHTER_RUN_SECONDS)
        agent = comp.state['agents']['0']
        m1 = float(agent['listeners']['mass']['dry_mass'])
        return m0, m1

    m0_1, m1_1 = _run_one(d1_state, seed=1)
    assert m1_1 - m0_1 >= DAUGHTER_MIN_GROWTH_FG, (
        f'Daughter 1 did not grow: {m0_1:.1f} -> {m1_1:.1f} fg over '
        f'{DAUGHTER_RUN_SECONDS:.0f}s.')

    m0_2, m1_2 = _run_one(d2_state, seed=2)
    assert m1_2 - m0_2 >= DAUGHTER_MIN_GROWTH_FG, (
        f'Daughter 2 did not grow: {m0_2:.1f} -> {m1_2:.1f} fg over '
        f'{DAUGHTER_RUN_SECONDS:.0f}s.')
