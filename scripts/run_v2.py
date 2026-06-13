"""Run a v2ecoli composite and collect snapshots. Called as a subprocess by
reports/v1_v2_report.py and reports/composite_comparison.py.

Args: <duration> <interval> <result_path> [composite_name] [seed]
The optional 4th arg selects which registered composite to build (default
'baseline'); the optional 5th is the RNG seed (default 0)."""
import os, sys, json, time, warnings, resource
import numpy as np

warnings.filterwarnings('ignore')

duration = int(sys.argv[1])
interval = int(sys.argv[2])
result_path = sys.argv[3]
composite_name = sys.argv[4] if len(sys.argv) > 4 else "baseline"
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0

v2ecoli_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(v2ecoli_dir)
sys.path.insert(0, v2ecoli_dir)

# Suppress C-level warnings
fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(fd, 2)

from v2ecoli import build_composite


def mag(x):
    # Mass/volume/rate listeners are now pint Quantities (units-on-ports work);
    # strip to the stored-unit magnitude (same numbers as the pre-units floats).
    return float(getattr(x, 'magnitude', x))


def _unique_counts(unique):
    """Active-molecule count for every unique-molecule type present."""
    out = {}
    for name, arr in (unique or {}).items():
        if hasattr(arr, 'dtype') and getattr(arr.dtype, 'names', None) \
                and '_entryState' in arr.dtype.names:
            out[name] = int(arr['_entryState'].sum())
    return out


def _bulk_summary(cell):
    """Total bulk molecule count + number of distinct bulk species (>0)."""
    bulk = cell.get('bulk')
    if bulk is None or not hasattr(bulk, 'dtype'):
        return {'bulk_total': 0, 'bulk_species_nonzero': 0, 'bulk_n_species': 0}
    counts = bulk['count'] if (bulk.dtype.names and 'count' in bulk.dtype.names) else bulk
    counts = np.asarray(counts)
    return {
        'bulk_total': int(counts.sum()),
        'bulk_species_nonzero': int((counts > 0).sum()),
        'bulk_n_species': int(counts.size),
    }


def _fba_summary(cell):
    """FBA objective value + (for the bridge harness) flux-pin diagnostics.

    Only the millard_fba_bridge_harness composite populates the bridge stores;
    for every other composite these keys are simply absent and the report
    renders them as not-applicable.
    """
    out = {}
    fba = cell.get('listeners', {}).get('fba_results', {})
    if 'objective_value' in fba:
        out['fba_objective'] = mag(fba.get('objective_value', 0))

    # FBA-bridge diagnostics (millard_fba_bridge_harness only).
    diag = cell.get('bridge_diagnostics')
    if isinstance(diag, dict) and 'n_pinned' in diag:
        out['n_pinned'] = int(diag.get('n_pinned', 0) or 0)
    relaxed = cell.get('listeners', {}).get('fba_bridge', {}).get('relaxed_reactions')
    if relaxed is not None:
        out['n_relaxed'] = int(len(relaxed))
    cf = cell.get('central_fluxes')
    if isinstance(cf, dict) and cf:
        vals = [abs(float(v)) for v in cf.values() if v == v]
        nz = [v for v in vals if v > 0]
        out['central_flux_n'] = len(cf)
        out['central_flux_nonzero'] = len(nz)
        out['central_flux_absmean'] = (sum(vals) / len(vals)) if vals else 0.0
    pft = cell.get('pinned_flux_targets')
    if isinstance(pft, dict) and pft:
        out['pin_n_nonzero'] = int(sum(1 for v in pft.values() if v))
    return out


def _base_reaction_ids(cell):
    """Labels for the base_reaction_fluxes vector, read off the live metabolism
    process instance (config-only, not in the state tree). None if absent."""
    node = cell.get('ecoli-metabolism')
    inst = node.get('instance') if isinstance(node, dict) else None
    ids = getattr(inst, 'base_reaction_ids', None)
    return [str(x) for x in ids] if ids is not None and len(ids) else None


def _accum_flux(cell, acc):
    """Running (sum, count) of the per-reaction base flux vector, for a
    time-mean over the run. Guards on shape so a malformed tick is skipped."""
    brf = cell.get('listeners', {}).get('fba_results', {}).get('base_reaction_fluxes')
    if brf is None:
        return acc
    arr = np.asarray(brf, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        return acc
    s, n = acc
    if s is None:
        s = np.zeros_like(arr)
    if s.shape != arr.shape:
        return (s, n)
    return (s + arr, n + 1)


def snap(t, cell):
    mass = cell.get('listeners', {}).get('mass', {})
    unique = cell.get('unique', {})
    uc = _unique_counts(unique)
    s = {
        'time': t,
        'dry_mass': mag(mass.get('dry_mass', 0)),
        'cell_mass': mag(mass.get('cell_mass', 0)),
        'protein_mass': mag(mass.get('protein_mass', 0)),
        'rna_mass': mag(mass.get('rRna_mass', 0)) + mag(mass.get('tRna_mass', 0)) + mag(mass.get('mRna_mass', 0)),
        'rRna_mass': mag(mass.get('rRna_mass', 0)),
        'tRna_mass': mag(mass.get('tRna_mass', 0)),
        'mRna_mass': mag(mass.get('mRna_mass', 0)),
        'dna_mass': mag(mass.get('dna_mass', 0)),
        'smallMolecule_mass': mag(mass.get('smallMolecule_mass', 0)),
        'water_mass': mag(mass.get('water_mass', 0)),
        'volume': mag(mass.get('volume', 0)),
        'instantaneous_growth_rate': mag(mass.get('instantaneous_growth_rate', 0)),
        # legacy convenience keys (kept for the older v1_v2 report)
        'n_chromosomes': uc.get('full_chromosome', 0),
        'n_forks': uc.get('active_replisome', 0),
        # rich molecular-species detail
        'unique_counts': uc,                  # per-type active unique molecules
    }
    s.update(_bulk_summary(cell))
    s.update(_fba_summary(cell))
    return s


# Use a minimal in-memory emitter instead of the default ParquetEmitter: this
# runner reads composite.state directly (never the on-disk history), and the
# Parquet emitter intermittently crashes writing post-division partitions
# (FileNotFoundError on .pq.tmp), which the loop below would mis-report as an
# early division — truncating the trajectory well before the requested duration.
from v2ecoli.composites._helpers import set_null_emitter_override
set_null_emitter_override(True)

t0 = time.time()
composite = build_composite(composite_name, cache_dir='out/cache', seed=seed)
load_time = time.time() - t0

cell = composite.state['agents']['0']
snapshots = [snap(0, cell)]
base_reaction_ids = _base_reaction_ids(cell)
flux_acc = (None, 0)  # (sum_vector, n_ticks) → time-mean per base reaction

t0 = time.time()
total = 0
# Per-chunk wall timing: one entry per composite.run(chunk) call so the report
# can show the step-time distribution (median/p95) and realtime-factor drift,
# not just the single aggregate wall_time.
step_times = []
while total < duration:
    chunk = min(interval, duration - total)
    c0 = time.time()
    try:
        composite.run(chunk)
    except Exception:
        # composite.run raises when the cell divides — mother is removed
        # and daughters are spawned under agents['00'] / agents['01']. We
        # don't follow a daughter here (that's what multigeneration_report
        # is for), so stop cleanly and report the sim-time at division.
        break
    chunk_wall = time.time() - c0
    total += chunk

    cell = composite.state.get('agents', {}).get('0')
    if cell is None:
        # Division event with no exception path — same handling.
        break
    step_times.append({
        'time': total,
        'sim_s': chunk,                                    # sim-seconds advanced
        'wall_s': chunk_wall,                              # wall-seconds taken
        'ms_per_sim_s': 1000.0 * chunk_wall / chunk if chunk else 0.0,
        'realtime_x': chunk / chunk_wall if chunk_wall else 0.0,
    })
    snapshots.append(snap(total, cell))
    flux_acc = _accum_flux(cell, flux_acc)

wall_time = time.time() - t0
# ru_maxrss is bytes on macOS (darwin), kilobytes on Linux.
_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_rss_mb = _maxrss / (1024 * 1024) if sys.platform == 'darwin' else _maxrss / 1024

result = {
    'engine': f'v2ecoli:{composite_name} (process-bigraph)',
    'composite': composite_name,
    'load_time': load_time,
    'wall_time': wall_time,
    'sim_time': total,
    'speed': total / wall_time if wall_time > 0 else 0,
    'peak_rss_mb': peak_rss_mb,
    'step_times': step_times,
    'snapshots': snapshots,
}

# Per-reaction time-mean FBA flux + labels, for the flux-divergence section
# (additive; absent on engines without a metabolism listener).
_fsum, _fn = flux_acc
if base_reaction_ids and _fsum is not None and _fn:
    _mean = (_fsum / _fn)
    if len(base_reaction_ids) == _mean.size:
        result['base_reaction_ids'] = base_reaction_ids
        result['base_reaction_flux_mean'] = [float(x) for x in _mean]

with open(result_path, 'w') as f:
    json.dump(result, f)
