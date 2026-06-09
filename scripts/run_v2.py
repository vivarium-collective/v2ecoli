"""Run a v2ecoli composite and collect snapshots. Called as a subprocess by
reports/v1_v2_report.py and reports/composite_comparison.py.

Args: <duration> <interval> <result_path> [composite_name] [seed]
The optional 4th arg selects which registered composite to build (default
'baseline'); the optional 5th is the RNG seed (default 0)."""
import os, sys, json, time, warnings
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

t0 = time.time()
total = 0
while total < duration:
    chunk = min(interval, duration - total)
    try:
        composite.run(chunk)
    except Exception:
        # composite.run raises when the cell divides — mother is removed
        # and daughters are spawned under agents['00'] / agents['01']. We
        # don't follow a daughter here (that's what multigeneration_report
        # is for), so stop cleanly and report the sim-time at division.
        break
    total += chunk

    cell = composite.state.get('agents', {}).get('0')
    if cell is None:
        # Division event with no exception path — same handling.
        break
    snapshots.append(snap(total, cell))

wall_time = time.time() - t0

result = {
    'engine': f'v2ecoli:{composite_name} (process-bigraph)',
    'composite': composite_name,
    'load_time': load_time,
    'wall_time': wall_time,
    'sim_time': total,
    'speed': total / wall_time if wall_time > 0 else 0,
    'snapshots': snapshots,
}

with open(result_path, 'w') as f:
    json.dump(result, f)
