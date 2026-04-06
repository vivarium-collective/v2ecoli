"""
v2ecoli Benchmark Suite

Runs a series of benchmarks measuring performance and correctness
of the whole-cell E. coli simulation through process-bigraph.

Benchmarks:
1. Cache loading — time to build document from cached simData
2. Short simulation (60s) — step execution, mass growth
3. v1 comparison (60s) — correlation with vEcoli
4. Long simulation (to division) — growth trajectory, division detection
5. Step-level diagnostics — per-step timing and error rates
"""

import os
import io
import json
import time
import base64
import html as html_lib
import copy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from contextlib import chdir

try:
    from wholecell.utils.filepath import ROOT_PATH
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    from ecoli.library.schema import not_a_process
    V1_AVAILABLE = True
except ImportError:
    ROOT_PATH = os.getcwd()
    V1_AVAILABLE = False

from v2ecoli.composite import make_composite, _build_core, save_cache
from process_bigraph import Composite
from v2ecoli.generate import build_document, DEFAULT_FLOW
from v2ecoli.cache import NumpyJSONEncoder, load_initial_state
from v2ecoli.steps.base import _translate_schema

from bigraph_viz import plot_bigraph
from bigraph_schema import get_path, strip_schema_keys


# Try to find simData — may not exist in CI (cache used instead)
_sim_data_candidates = [
    os.path.join(ROOT_PATH, 'out', 'kb', 'simData.cPickle'),
    'out/kb/simData.cPickle',
]
SIM_DATA_PATH = next((p for p in _sim_data_candidates if os.path.exists(p)), None)
OUT_DIR = 'out/benchmark'
CACHE_DIR = 'out/cache'
COMPARISON_DURATION = 60.0
LONG_DURATION = 500.0  # Longer run showing growth (not full division)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_cache_load():
    """Benchmark: load document from cache."""
    cache_file = os.path.join(CACHE_DIR, 'sim_data_cache.dill')
    if os.path.exists(cache_file):
        print("  Using existing cache")
        cache_time = 0.0
    else:
        t0 = time.time()
        save_cache(SIM_DATA_PATH, CACHE_DIR)
        cache_time = time.time() - t0

    t0 = time.time()
    composite = make_composite(cache_dir=CACHE_DIR)
    build_time = time.time() - t0

    return {
        'cache_generation': cache_time,
        'document_build': build_time,
        'n_steps': len(composite.step_paths),
        'n_processes': len(composite.process_paths),
        'composite': composite,
    }


def bench_short_sim(composite, duration=60.0):
    """Benchmark: short simulation with per-step diagnostics."""
    cell = composite.state['agents']['0']
    bulk_before = np.array(cell['bulk']['count'], copy=True)

    t0 = time.time()
    composite.run(duration)
    wall_time = time.time() - t0

    bulk_after = cell['bulk']['count']
    changed = (bulk_before != bulk_after).sum()

    # Get emitter data
    em_edge = cell.get('emitter')
    history = em_edge['instance'].history if isinstance(em_edge, dict) and 'instance' in em_edge else []

    mass = cell.get('listeners', {}).get('mass', {})

    return {
        'duration': duration,
        'wall_time': wall_time,
        'bulk_changed': int(changed),
        'total_bulk': len(bulk_before),
        'final_dry_mass': mass.get('dry_mass', 0),
        'final_cell_mass': mass.get('cell_mass', 0),
        'final_volume': mass.get('volume', 0),
        'history': history,
        'rate': duration / wall_time,
    }


def _collect_v2_timeseries(composite, duration):
    """Run v2, collect per-timestep bulk, mass, and unique data from emitter."""
    from v2ecoli.library.schema import attrs as ecoli_attrs
    cell = composite.state['agents']['0']

    v2_initial_bulk = np.array(cell['bulk']['count'], copy=True)

    t0 = time.time()
    composite.run(duration)
    v2_time = time.time() - t0

    cell = composite.state['agents']['0']
    em = cell.get('emitter', {}).get('instance')
    v2_history = em.history if em else []

    v2_bulk_ts = {}
    v2_mass_ts = {}
    for snap in v2_history:
        t = int(snap.get('global_time', 0))
        bulk = snap.get('bulk')
        if bulk is not None and hasattr(bulk, 'dtype') and 'count' in bulk.dtype.names:
            v2_bulk_ts[t] = bulk['count'].copy()
        mass = snap.get('listeners', {}).get('mass', {})
        if isinstance(mass, dict) and mass:
            v2_mass_ts[t] = dict(mass)

    # Unique molecule counts from final state
    unique = cell.get('unique', {})
    v2_unique = {}
    for name, arr in unique.items():
        if hasattr(arr, 'dtype') and '_entryState' in arr.dtype.names:
            v2_unique[name] = int(arr['_entryState'].view(np.bool_).sum())

    # Full chromosome details
    fc = unique.get('full_chromosome')
    v2_chrom = {}
    if fc is not None and hasattr(fc, 'dtype'):
        active = fc[fc['_entryState'].view(np.bool_)]
        v2_chrom['n_chromosomes'] = len(active)
        if 'division_time' in fc.dtype.names:
            dt, htd = ecoli_attrs(fc, ['division_time', 'has_triggered_division'])
            v2_chrom['division_times'] = dt.tolist()
            v2_chrom['has_triggered'] = htd.tolist()
        if 'domain_index' in fc.dtype.names:
            (di,) = ecoli_attrs(fc, ['domain_index'])
            v2_chrom['domain_indexes'] = di.tolist()

    return {
        'time': v2_time,
        'initial_bulk': v2_initial_bulk,
        'final_bulk': cell['bulk']['count'].copy(),
        'bulk_ts': v2_bulk_ts,
        'mass_ts': v2_mass_ts,
        'unique_counts': v2_unique,
        'chromosome': v2_chrom,
        'final_mass': dict(cell.get('listeners', {}).get('mass', {})),
    }


def _collect_v1_timeseries(duration):
    """Run v1, collect per-timestep bulk, mass, and unique data."""
    import sys
    try:
        # Monkey-patch np.in1d for numpy compat
        if not hasattr(np, 'in1d'):
            np.in1d = np.isin

        saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        with chdir(ROOT_PATH):
            sim = EcoliSim.from_file()
            sim.max_duration = int(duration)
            sim.emitter = 'timeseries'
            sim.divide = False
            sim.build_ecoli()
            v1_initial = sim.generated_initial_state['bulk']['count'].copy()
            t0 = time.time()
            sim.run()
            v1_time = time.time() - t0
        sys.argv = saved_argv

        v1_state = sim.ecoli_experiment.state.get_value(condition=not_a_process)
        v1_final = v1_state['bulk']['count'].copy()

        # v1 timeseries: top-level keys are time floats {0.0: snapshot, 1.0: snapshot, ...}
        v1_ts = sim.query()
        v1_bulk_ts = {}
        v1_mass_ts = {}
        for t_key in sorted(v1_ts.keys()):
            if not isinstance(t_key, (int, float)):
                continue
            t = int(t_key)
            snapshot = v1_ts[t_key]
            if not isinstance(snapshot, dict):
                continue

            bulk = snapshot.get('bulk')
            if bulk is not None:
                if hasattr(bulk, 'dtype') and 'count' in (bulk.dtype.names or []):
                    v1_bulk_ts[t] = np.array(bulk['count'], dtype=float)
                elif isinstance(bulk, (list, np.ndarray)):
                    v1_bulk_ts[t] = np.array(bulk, dtype=float)

            # Mass from listeners — each value is a scalar at this timestep
            mass = snapshot.get('listeners', {}).get('mass', {})
            if isinstance(mass, dict) and mass:
                entry = {}
                for k, v in mass.items():
                    try:
                        entry[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                v1_mass_ts[t] = entry

        # Unique molecule counts from final state
        v1_unique = {}
        unique = v1_state.get('unique', {})
        for name, arr in unique.items():
            if hasattr(arr, 'dtype') and '_entryState' in arr.dtype.names:
                v1_unique[name] = int(arr['_entryState'].view(np.bool_).sum())

        # Full chromosome
        v1_chrom = {}
        fc = unique.get('full_chromosome')
        if fc is not None and hasattr(fc, 'dtype'):
            from v2ecoli.library.schema import attrs as ecoli_attrs
            active = fc[fc['_entryState'].view(np.bool_)]
            v1_chrom['n_chromosomes'] = len(active)
            if 'division_time' in fc.dtype.names:
                dt, htd = ecoli_attrs(fc, ['division_time', 'has_triggered_division'])
                v1_chrom['division_times'] = dt.tolist()
                v1_chrom['has_triggered'] = htd.tolist()

        return {
            'time': v1_time,
            'initial_bulk': v1_initial,
            'final_bulk': v1_final,
            'bulk_ts': v1_bulk_ts,
            'mass_ts': v1_mass_ts,
            'unique_counts': v1_unique,
            'chromosome': v1_chrom,
            'final_mass': v1_mass_ts.get(int(duration), {}),
        }
    except Exception as e:
        print(f"  v1 comparison skipped: {e}")
        return None


def bench_v1_comparison(duration=60.0):
    """Benchmark: comprehensive v1 vs v2 comparison.

    Compares at every timestep:
    - Bulk molecule counts (correlation, exact match, RMSE)
    - Mass components (dry, protein, RNA, DNA, small molecule)
    - Chromosome replication state
    - Unique molecule counts
    """
    # v2
    composite = make_composite(cache_dir=CACHE_DIR)
    v2 = _collect_v2_timeseries(composite, duration)

    # v1
    v1 = _collect_v1_timeseries(duration)
    v1_available = v1 is not None

    if not v1_available:
        # Create empty v1 data for compatibility
        v1 = {
            'time': 0,
            'initial_bulk': v2['initial_bulk'].copy(),
            'final_bulk': v2['initial_bulk'].copy(),
            'bulk_ts': {},
            'mass_ts': {},
            'unique_counts': {},
            'chromosome': {},
            'final_mass': {},
        }

    # Per-timestep bulk comparison
    common_times = sorted(set(v1['bulk_ts'].keys()) & set(v2['bulk_ts'].keys()))
    per_ts_corr = []
    per_ts_exact = []
    per_ts_rmse = []
    for t in common_times:
        v1_c = v1['bulk_ts'][t].astype(float)
        v2_c = v2['bulk_ts'][t].astype(float)
        if len(v1_c) == len(v2_c) and len(v1_c) > 0:
            corr = np.corrcoef(v1_c, v2_c)[0, 1]
            exact = np.mean(v1_c == v2_c)
            rmse = np.sqrt(np.mean((v1_c - v2_c) ** 2))
            per_ts_corr.append(corr)
            per_ts_exact.append(exact)
            per_ts_rmse.append(rmse)

    # Final-state delta correlation
    both = (v1['initial_bulk'] != v1['final_bulk']) & (v2['initial_bulk'] != v2['final_bulk'])
    d1 = v1['final_bulk'][both] - v1['initial_bulk'][both]
    d2 = v2['final_bulk'][both] - v2['initial_bulk'][both]
    delta_corr = np.corrcoef(d1.astype(float), d2.astype(float))[0, 1] if both.sum() > 0 else 0.0

    # Per-timestep mass comparison
    mass_keys = ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallMolecule_mass']
    mass_comparison = {k: {'v1': [], 'v2': [], 'times': []} for k in mass_keys}
    v2_times = sorted(v2['mass_ts'].keys())
    v1_times = sorted(v1['mass_ts'].keys())
    for t in v2_times:
        for k in mass_keys:
            mass_comparison[k]['v2'].append(v2['mass_ts'][t].get(k, 0))
            mass_comparison[k]['times'].append(t)
    for t in v1_times:
        for k in mass_keys:
            mass_comparison[k]['v1'].append(v1['mass_ts'][t].get(k, 0))

    # --- Per-category mass accuracy metrics ---
    mass_metrics = {}
    for k in mass_keys:
        v1_vals = np.array(mass_comparison[k]['v1'])
        v2_vals = np.array(mass_comparison[k]['v2'])
        n = min(len(v1_vals), len(v2_vals))
        if n > 0:
            v1_v = v1_vals[:n]
            v2_v = v2_vals[:n]
            abs_err = np.abs(v2_v - v1_v)
            pct_err = np.abs(v2_v - v1_v) / np.maximum(np.abs(v1_v), 1e-10) * 100
            # Time-series R²
            ss_res = np.sum((v2_v - v1_v) ** 2)
            ss_tot = np.sum((v1_v - v1_v.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mass_metrics[k] = {
                'mean_abs_err': float(abs_err.mean()),
                'max_abs_err': float(abs_err.max()),
                'mean_pct_err': float(pct_err.mean()),
                'max_pct_err': float(pct_err.max()),
                'final_pct_err': float(pct_err[-1]) if len(pct_err) > 0 else 0,
                'r2': float(r2),
                'v1_final': float(v1_v[-1]) if len(v1_v) > 0 else 0,
                'v2_final': float(v2_v[-1]) if len(v2_v) > 0 else 0,
            }
        else:
            mass_metrics[k] = {
                'mean_abs_err': 0, 'max_abs_err': 0,
                'mean_pct_err': 0, 'max_pct_err': 0,
                'final_pct_err': 0, 'r2': 0,
                'v1_final': 0, 'v2_final': 0,
            }

    # Overall accuracy score: worst-case % error across all mass categories
    worst_pct = max(m['max_pct_err'] for m in mass_metrics.values()) if mass_metrics else 0

    return {
        'duration': duration,
        'v1_available': v1_available,
        'v1_time': v1['time'],
        'v2_time': v2['time'],
        'v1_changed': int((v1['initial_bulk'] != v1['final_bulk']).sum()),
        'v2_changed': int((v2['initial_bulk'] != v2['final_bulk']).sum()),
        'both_changed': int(both.sum()),
        'delta_correlation': delta_corr,
        'common_timesteps': len(common_times),
        'per_ts_corr': per_ts_corr,
        'per_ts_exact': per_ts_exact,
        'per_ts_rmse': per_ts_rmse,
        'mean_correlation': np.mean(per_ts_corr) if per_ts_corr else 0.0,
        'mean_exact_match': np.mean(per_ts_exact) if per_ts_exact else 0.0,
        'mean_rmse': np.mean(per_ts_rmse) if per_ts_rmse else 0.0,
        'v1_initial': v1['initial_bulk'],
        'v1_final': v1['final_bulk'],
        'v2_initial': v2['initial_bulk'],
        'v2_final': v2['final_bulk'],
        'mass_comparison': mass_comparison,
        'mass_metrics': mass_metrics,
        'worst_pct_error': worst_pct,
        'v1_unique': v1['unique_counts'],
        'v2_unique': v2['unique_counts'],
        'v1_chromosome': v1['chromosome'],
        'v2_chromosome': v2['chromosome'],
        'v1_final_mass': v1['final_mass'],
        'v2_final_mass': v2['final_mass'],
    }


def bench_division():
    """Benchmark: division state splitting and daughter generation.

    Tests divide_cell() on the initial state (not at actual division time)
    to verify conservation and daughter viability.
    """
    import time as time_mod
    from v2ecoli.library.division import divide_cell, divide_bulk

    composite = make_composite(cache_dir=CACHE_DIR)
    cell = composite.state['agents']['0']

    # Test bulk conservation
    d1_bulk, d2_bulk = divide_bulk(cell['bulk'])
    mother_count = int(cell['bulk']['count'].sum())
    d1_count = int(d1_bulk['count'].sum())
    d2_count = int(d2_bulk['count'].sum())
    conserved = (d1_count + d2_count == mother_count)

    # Test full cell division
    t0 = time_mod.time()
    d1_state, d2_state = divide_cell(cell)
    split_time = time_mod.time() - t0

    # Unique molecule counts
    unique_conservation = {}
    for name in d1_state.get('unique', {}):
        d1_arr = d1_state['unique'][name]
        d2_arr = d2_state['unique'][name]
        mother_arr = cell['unique'][name]
        if hasattr(d1_arr, 'shape') and hasattr(mother_arr, 'dtype'):
            if '_entryState' in mother_arr.dtype.names:
                m = int(mother_arr['_entryState'].view(np.bool_).sum())
                d1 = d1_arr.shape[0]
                d2 = d2_arr.shape[0]
                unique_conservation[name] = {
                    'mother': m, 'd1': d1, 'd2': d2,
                    'conserved': d1 + d2 == m
                }

    # Test daughter document build (if configs available)
    div_step = cell.get('division', {}).get('instance')
    can_build_daughters = div_step is not None and bool(getattr(div_step, '_configs', {}))
    daughter_build_time = 0
    daughter_viable = False

    if can_build_daughters:
        from v2ecoli.generate import build_document_from_configs
        t0 = time_mod.time()
        try:
            d1_doc = build_document_from_configs(
                d1_state, div_step._configs, div_step._unique_names,
                dry_mass_inc_dict=div_step.dry_mass_inc_dict,
                seed=1)
            daughter_build_time = time_mod.time() - t0
            # Quick viability check: can we create a composite?
            d1_composite = Composite(d1_doc, core=_build_core())
            d1_composite.run(1.0)
            daughter_viable = True
        except Exception as e:
            daughter_build_time = time_mod.time() - t0
            print(f"  Daughter build error: {e}")

    return {
        'mother_bulk_count': mother_count,
        'd1_bulk_count': d1_count,
        'd2_bulk_count': d2_count,
        'bulk_conserved': conserved,
        'split_time': split_time,
        'unique_conservation': unique_conservation,
        'can_build_daughters': can_build_daughters,
        'daughter_build_time': daughter_build_time,
        'daughter_viable': daughter_viable,
    }


def bench_step_diagnostics(composite):
    """Benchmark: per-step analysis."""
    cell = composite.state['agents']['0']
    core = composite.core

    diagnostics = []
    for step_name in DEFAULT_FLOW:
        path = ('agents', '0', step_name)
        edge = cell.get(step_name)
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue

        inst = edge['instance']
        proc = getattr(inst, 'process', inst)

        # Step info
        info = {
            'name': step_name,
            'class': type(inst).__name__,
            'inner_class': type(proc).__name__ if proc is not inst else None,
            'has_ports_schema': hasattr(proc, 'ports_schema') and callable(getattr(proc, 'ports_schema')),
            'has_inputs': hasattr(inst, 'inputs') and 'inputs' in type(inst).__dict__,
            'has_config_schema': bool(getattr(inst, 'config_schema', {})),
            'has_defaults': bool(getattr(proc, 'defaults', {})),
            'n_config_keys': len(getattr(proc, 'parameters', {})),
            'priority': edge.get('priority', 0),
        }

        # Port info
        if info['has_ports_schema']:
            try:
                ps = proc.ports_schema()
                info['input_ports'] = sorted(ps.keys())
            except Exception:
                info['input_ports'] = []
        else:
            info['input_ports'] = []

        # Wire info
        wires = edge.get('inputs', {})
        info['wires'] = {k: v for k, v in wires.items() if not k.startswith('_flow')}

        diagnostics.append(info)

    return diagnostics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

MASS_KEYS = {
    'Protein': 'protein_mass', 'tRNA': 'tRna_mass', 'rRNA': 'rRna_mass',
    'mRNA': 'mRna_mass', 'DNA': 'dna_mass', 'Small Mol': 'smallMolecule_mass',
    'Dry Mass': 'dry_mass',
}
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']


def plot_mass(history, title=''):
    if not history or len(history) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig_to_b64(fig)

    times = np.array([s.get('global_time', 0) for s in history])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (label, key) in enumerate(MASS_KEYS.items()):
        vals = np.array([s.get('listeners', {}).get('mass', {}).get(key, 0) for s in history])
        if len(vals) > 0 and vals[0] > 0:
            axes[0].plot(times / 60, vals / vals[0], color=COLORS[i], lw=1.5, label=label)
            axes[1].plot(times / 60, vals, color=COLORS[i], lw=1.5, label=label)

    axes[0].set_xlabel('Time (min)')
    axes[0].set_ylabel('Fold change')
    axes[0].set_title('Fold Change (normalized to t=0)')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.15)
    # Don't clamp y-axis — show decreases below 1.0

    axes[1].set_xlabel('Time (min)')
    axes[1].set_ylabel('Mass (fg)')
    axes[1].set_title('Absolute Mass')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.15)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_growth(history):
    if not history or len(history) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig_to_b64(fig)

    times = np.array([s.get('global_time', 0) for s in history])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gr = np.array([s.get('listeners', {}).get('mass', {}).get('instantaneous_growth_rate', 0) for s in history])
    vol = np.array([s.get('listeners', {}).get('mass', {}).get('volume', 0) for s in history])
    pf = np.array([s.get('listeners', {}).get('mass', {}).get('protein_mass_fraction', 0) for s in history])
    rf = np.array([s.get('listeners', {}).get('mass', {}).get('rna_mass_fraction', 0) for s in history])

    axes[0].plot(times / 60, gr * 3600, color='#2563eb', lw=1)
    axes[0].set_ylabel('Growth rate (1/h)')
    axes[0].set_title('Instantaneous Growth Rate')

    axes[1].plot(times / 60, vol, color='#16a34a', lw=1)
    axes[1].set_ylabel('Volume (fL)')
    axes[1].set_title('Cell Volume')

    axes[2].plot(times / 60, pf, color='#e41a1c', lw=1, label='Protein')
    axes[2].plot(times / 60, rf, color='#377eb8', lw=1, label='RNA')
    axes[2].set_ylabel('Mass fraction')
    axes[2].set_title('Mass Fractions')
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel('Time (min)')
        ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_comparison(comp):
    """Plot per-timestep correlation, scatter of final deltas, and RMSE."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Per-timestep correlation
    if comp['per_ts_corr']:
        axes[0].plot(range(1, len(comp['per_ts_corr']) + 1), comp['per_ts_corr'],
                     color='#2563eb', lw=1.5)
        axes[0].set_xlabel('Timestep (s)')
        axes[0].set_ylabel('Pearson correlation')
        axes[0].set_title(f'Per-Timestep Correlation (mean={comp["mean_correlation"]:.6f})')
        axes[0].set_ylim(min(0.999, min(comp['per_ts_corr']) - 0.0001), 1.0001)
        axes[0].grid(True, alpha=0.15)
    else:
        axes[0].text(0.5, 0.5, 'No per-timestep data', ha='center', va='center')

    # Scatter of final deltas
    v1_d = comp['v1_final'].astype(float) - comp['v1_initial'].astype(float)
    v2_d = comp['v2_final'].astype(float) - comp['v2_initial'].astype(float)
    both = (v1_d != 0) & (v2_d != 0)
    if both.sum() > 0:
        axes[1].scatter(v1_d[both], v2_d[both], alpha=0.5, s=8, c='#2563eb')
        lim = max(abs(v1_d[both]).max(), abs(v2_d[both]).max()) * 1.1
        axes[1].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
        axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
    axes[1].set_xlabel('v1 count change')
    axes[1].set_ylabel('v2 count change')
    axes[1].set_title(f'Final Delta (r={comp["delta_correlation"]:.4f}, n={both.sum()})')
    axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.2)

    # Per-timestep RMSE
    if comp['per_ts_rmse']:
        axes[2].plot(range(1, len(comp['per_ts_rmse']) + 1), comp['per_ts_rmse'],
                     color='#dc2626', lw=1.5)
        axes[2].set_xlabel('Timestep (s)')
        axes[2].set_ylabel('RMSE (counts)')
        axes[2].set_title(f'Per-Timestep RMSE (mean={comp["mean_rmse"]:.2f})')
        axes[2].grid(True, alpha=0.15)
    else:
        axes[2].text(0.5, 0.5, 'No per-timestep data', ha='center', va='center')

    fig.suptitle('v1 vs v2 Comparison — Bulk Molecules', fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_mass_comparison(comp):
    """Plot per-timestep mass components for v1 vs v2."""
    mc = comp.get('mass_comparison', {})
    mass_keys = ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallMolecule_mass']
    labels = ['Dry Mass', 'Protein', 'RNA', 'DNA', 'Small Molecules']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (key, label) in enumerate(zip(mass_keys, labels)):
        ax = axes[i]
        data = mc.get(key, {})
        v2_vals = data.get('v2', [])
        v1_vals = data.get('v1', [])
        v2_times = data.get('times', [])
        v1_times = list(range(1, len(v1_vals) + 1))

        if v2_vals:
            ax.plot(v2_times, v2_vals, 'b-', lw=1.5, label='v2', alpha=0.8)
        if v1_vals:
            ax.plot(v1_times, v1_vals, 'r--', lw=1.5, label='v1', alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mass (fg)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    # Mass delta per timestep (v2)
    ax = axes[5]
    dm = mc.get('dry_mass', {}).get('v2', [])
    if len(dm) > 1:
        deltas = np.diff(dm)
        ax.plot(range(1, len(deltas) + 1), deltas, 'b-o', markersize=2, lw=1)
        v1_dm = mc.get('dry_mass', {}).get('v1', [])
        if len(v1_dm) > 1:
            v1_deltas = np.diff(v1_dm)
            ax.plot(range(1, len(v1_deltas) + 1), v1_deltas, 'r--x', markersize=2, lw=1)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_title('Dry Mass Δ per Timestep')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mass change (fg)')
        ax.grid(True, alpha=0.15)

    fig.suptitle('v1 vs v2 — Mass Components Over Time', fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_unique_comparison(comp):
    """Plot unique molecule counts and chromosome state for v1 vs v2."""
    v1_unique = comp.get('v1_unique', {})
    v2_unique = comp.get('v2_unique', {})
    v1_chrom = comp.get('v1_chromosome', {})
    v2_chrom = comp.get('v2_chromosome', {})

    all_names = sorted(set(list(v1_unique.keys()) + list(v2_unique.keys())))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of unique molecule counts
    if all_names:
        x = np.arange(len(all_names))
        v1_vals = [v1_unique.get(n, 0) for n in all_names]
        v2_vals = [v2_unique.get(n, 0) for n in all_names]
        width = 0.35
        axes[0].bar(x - width/2, v1_vals, width, label='v1', color='#dc2626', alpha=0.7)
        axes[0].bar(x + width/2, v2_vals, width, label='v2', color='#2563eb', alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([n.replace('_', '\n') for n in all_names], fontsize=7, rotation=45, ha='right')
        axes[0].set_ylabel('Active Count')
        axes[0].set_title('Unique Molecule Counts (active)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.15, axis='y')

    # Chromosome state text
    ax = axes[1]
    ax.axis('off')
    lines = ['Chromosome State at t=60s', '']
    lines.append('v2:')
    lines.append(f'  Full chromosomes: {v2_chrom.get("n_chromosomes", "?")}')
    if 'division_times' in v2_chrom:
        lines.append(f'  Division times: {v2_chrom["division_times"]}')
        lines.append(f'  Has triggered: {v2_chrom["has_triggered"]}')
    if 'domain_indexes' in v2_chrom:
        lines.append(f'  Domain indexes: {v2_chrom["domain_indexes"]}')

    if v1_chrom:
        lines.append('')
        lines.append('v1:')
        lines.append(f'  Full chromosomes: {v1_chrom.get("n_chromosomes", "?")}')
        if 'division_times' in v1_chrom:
            lines.append(f'  Division times: {v1_chrom["division_times"]}')
            lines.append(f'  Has triggered: {v1_chrom["has_triggered"]}')

    # Final mass comparison
    v1_fm = comp.get('v1_final_mass', {})
    v2_fm = comp.get('v2_final_mass', {})
    lines.append('')
    lines.append('Final Mass (fg):')
    for k in ['dry_mass', 'protein_mass', 'rna_mass', 'dna_mass', 'smallMolecule_mass']:
        v1v = v1_fm.get(k, 0)
        v2v = v2_fm.get(k, 0)
        diff = v2v - v1v if v1v else 0
        label = k.replace('_mass', '').replace('_', ' ').title()
        if v1v:
            lines.append(f'  {label}: v1={v1v:.2f}  v2={v2v:.2f}  diff={diff:+.2f}')
        else:
            lines.append(f'  {label}: v2={v2v:.2f}')

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('v1 vs v2 — Unique Molecules & Chromosome State', fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Bigraph
# ---------------------------------------------------------------------------

SKIP_STEPS = {'unique_update', 'global_clock', 'mark_d_period', 'division',
              'exchange_data', 'media_update', 'post-division-mass-listener', 'emitter'}
SKIP_PORTS = {'timestep', 'global_time', 'next_update_time', 'process'}
BIO_COLORS = {
    'dna': ('#FFB6C1', lambda n: 'chromosome' in n),
    'rna': ('#ADD8E6', lambda n: any(s in n for s in ('transcript', 'rna-', 'RNA', 'rnap'))),
    'protein': ('#90EE90', lambda n: any(s in n for s in ('polypeptide', 'protein', 'ribosome'))),
    'meta': ('#FFD700', lambda n: any(s in n for s in ('metabolism', 'equilibrium', 'complexation', 'two-component'))),
    'reg': ('#DDA0DD', lambda n: any(s in n for s in ('tf-', 'tf_'))),
    'listen': ('#D3D3D3', lambda n: 'listener' in n),
}


def make_bigraph_svg(state):
    cell = state.get('agents', {}).get('0', state)
    viz = {}
    for name, edge in cell.items():
        if not isinstance(edge, dict): continue
        if '_type' in edge:
            if any(s in name for s in SKIP_STEPS): continue
            if '_requester' in name: continue
            inputs = {p: w for p, w in edge.get('inputs', {}).items()
                      if not p.startswith('_flow') and p not in SKIP_PORTS
                      and not (isinstance(w, list) and w and w[0] in ('request', 'allocate'))}
            clean = name.replace('ecoli-', '').replace('_evolver', '')
            viz[clean] = {'_type': edge['_type'], 'inputs': inputs}
        elif name == 'unique' and isinstance(edge, dict):
            viz[name] = {k: {} for k in edge.keys()}
        elif name in ('bulk', 'listeners', 'environment', 'boundary'):
            viz[name] = {}

    viz_state = {'agents': {'0': viz}}
    prefix = ('agents', '0')
    colors, groups = {}, {k: [] for k in BIO_COLORS}
    for n in viz:
        if '_type' not in viz.get(n, {}): continue
        p = prefix + (n,)
        for gk, (c, m) in BIO_COLORS.items():
            if m(n): colors[p] = c; groups[gk].append(p); break

    try:
        plot_bigraph(viz_state, remove_process_place_edges=True,
                     node_groups=[g for g in groups.values() if g],
                     node_fill_colors=colors, rankdir='TB',
                     dpi='72', port_labels=False, node_label_size='16pt',
                     label_margin='0.05', out_dir=OUT_DIR,
                     filename='bigraph', file_format='svg')
        with open(os.path.join(OUT_DIR, 'bigraph.svg')) as f:
            svg = f.read()
        # Remove fixed width/height from SVG root so CSS can control sizing
        import re
        svg = re.sub(r'width="[^"]*pt"', '', svg, count=1)
        svg = re.sub(r'height="[^"]*pt"', '', svg, count=1)
        return svg
    except Exception as e:
        return f'<p>Failed: {html_lib.escape(str(e))}</p>'


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_benchmarks():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    # 1. Cache + build
    print("=" * 50)
    print("Benchmark 1: Cache & Document Build")
    r = bench_cache_load()
    results['build'] = r
    composite = r['composite']
    print(f"  Cache: {r['cache_generation']:.1f}s, Build: {r['document_build']:.2f}s")
    print(f"  {r['n_steps']} steps, {r['n_processes']} processes")

    # 2. Bigraph
    print("Generating bigraph...")
    bigraph_svg = make_bigraph_svg(composite.state)

    # 3. Short sim
    print(f"Benchmark 2: Simulation ({COMPARISON_DURATION}s)")
    r = bench_short_sim(composite, COMPARISON_DURATION)
    results['short'] = r
    print(f"  {r['wall_time']:.1f}s wall, {r['bulk_changed']} changed, {r['rate']:.1f}x realtime")

    # 4. Step diagnostics
    print("Benchmark 3: Step Diagnostics")
    diag = bench_step_diagnostics(composite)
    results['diagnostics'] = diag
    print(f"  {len(diag)} steps analyzed")

    # 5. v1 comparison
    print(f"Benchmark 4: v1 Comparison ({COMPARISON_DURATION}s)")
    comp = bench_v1_comparison(COMPARISON_DURATION)
    results['comparison'] = comp
    mm = comp.get('mass_metrics', {})
    print(f"  v1: {comp['v1_time']:.2f}s, v2: {comp['v2_time']:.2f}s, "
          f"worst_pct_err: {comp['worst_pct_error']:.2f}%")
    for k, m in mm.items():
        label = k.replace('_mass', '').replace('_', ' ').title()
        print(f"    {label}: mean_err={m['mean_pct_err']:.2f}%, max_err={m['max_pct_err']:.2f}%, R2={m['r2']:.4f}")

    # 6. Division
    print("Benchmark 5: Division")
    div = bench_division()
    results['division'] = div
    print(f"  Bulk conserved: {div['bulk_conserved']}, split time: {div['split_time']:.3f}s")
    print(f"  Can build daughters: {div['can_build_daughters']}, viable: {div['daughter_viable']}")
    if div['daughter_build_time'] > 0:
        print(f"  Daughter build time: {div['daughter_build_time']:.1f}s")

    # 7. Long sim
    print(f"Benchmark 6: Long Simulation ({LONG_DURATION}s = {LONG_DURATION/60:.0f}min)")
    long_composite = make_composite(cache_dir=CACHE_DIR)
    long_r = bench_short_sim(long_composite, LONG_DURATION)
    results['long'] = long_r
    print(f"  {long_r['wall_time']:.1f}s wall, {long_r['bulk_changed']} changed")

    # Plots
    print("Generating plots...")
    mass_short = plot_mass(results['short']['history'], f'Mass Components ({COMPARISON_DURATION}s)')
    growth_short = plot_growth(results['short']['history'])
    mass_long = plot_mass(results['long']['history'], f'Mass Components ({LONG_DURATION/60:.0f} min)')
    growth_long = plot_growth(results['long']['history'])
    comparison_plot = plot_comparison(comp)
    mass_comp_plot = plot_mass_comparison(comp)
    unique_comp_plot = plot_unique_comparison(comp)

    # Build HTML
    b = results['build']
    s = results['short']
    c = results['comparison']
    l = results['long']
    div = results['division']

    # Division unique molecule rows
    div_unique_rows = ''
    for name, info in div.get('unique_conservation', {}).items():
        ok = 'green' if info['conserved'] else 'red'
        div_unique_rows += f"""<tr>
          <td>{name}</td><td>{info['mother']}</td>
          <td>{info['d1']}</td><td>{info['d2']}</td>
          <td class="{ok}">{'Yes' if info['conserved'] else 'No'}</td></tr>"""

    # Mass accuracy table rows
    mass_accuracy_rows = ''
    mass_labels = {
        'dry_mass': 'Dry Mass', 'protein_mass': 'Protein',
        'rna_mass': 'RNA', 'dna_mass': 'DNA', 'smallMolecule_mass': 'Small Molecules',
    }
    for k, label in mass_labels.items():
        m = c.get('mass_metrics', {}).get(k, {})
        err_color = 'green' if m.get('max_pct_err', 0) < 1 else ('red' if m.get('max_pct_err', 0) > 5 else 'purple')
        r2_color = 'green' if m.get('r2', 0) > 0.99 else ('red' if m.get('r2', 0) < 0.9 else 'purple')
        mass_accuracy_rows += f"""<tr>
          <td><strong>{label}</strong></td>
          <td>{m.get('v1_final', 0):.2f}</td>
          <td>{m.get('v2_final', 0):.2f}</td>
          <td class="{err_color}">{m.get('mean_pct_err', 0):.2f}%</td>
          <td class="{err_color}">{m.get('max_pct_err', 0):.2f}%</td>
          <td class="{r2_color}">{m.get('r2', 0):.4f}</td>
        </tr>"""

    # Step diagnostics table
    step_rows = ''
    for d in diag:
        inner = f' ({d["inner_class"]})' if d['inner_class'] else ''
        v1_markers = []
        if d['has_ports_schema']: v1_markers.append('ports_schema')
        if d['has_defaults']: v1_markers.append('defaults')
        v2_markers = []
        if d['has_inputs']: v2_markers.append('inputs()')
        if d['has_config_schema']: v2_markers.append('config_schema')
        ports = ', '.join(d.get('input_ports', [])[:5])
        if len(d.get('input_ports', [])) > 5:
            ports += f' +{len(d["input_ports"])-5}'

        step_rows += f"""<tr>
          <td>{d['name']}</td>
          <td>{d['class']}{inner}</td>
          <td>{d['n_config_keys']}</td>
          <td>{ports}</td>
          <td>{d['priority']:.0f}</td>
        </tr>"""

    report_path = os.path.join(OUT_DIR, 'benchmark_report.html')
    with open(report_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v2ecoli Benchmark Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 1.8em; margin: 15px 0; color: #0f172a; }}
  h2 {{ font-size: 1.3em; margin: 25px 0 10px; color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
  h3 {{ font-size: 1.05em; margin: 15px 0 8px; color: #475569; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 8px; margin: 10px 0; }}
  .metric {{ background: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
  .metric .label {{ font-size: 0.7em; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric .value {{ font-size: 1.3em; font-weight: 700; margin-top: 2px; }}
  .green {{ color: #16a34a; }} .blue {{ color: #2563eb; }} .red {{ color: #dc2626; }} .purple {{ color: #7c3aed; }}
  .plot {{ background: white; border-radius: 8px; padding: 12px; margin: 10px 0;
           box-shadow: 0 1px 2px rgba(0,0,0,0.08); text-align: center; }}
  .plot img {{ max-width: 100%; }}
  .section {{ background: white; border-radius: 8px; padding: 15px; margin: 10px 0;
              box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
  details {{ margin: 5px 0; }}
  details > summary {{ cursor: pointer; font-weight: 600; color: #475569; padding: 5px 0; }}
  details > summary:hover {{ color: #1e293b; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.82em; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 5px 8px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  .bigraph {{ overflow: auto; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px;
              background: white; }}
  .bigraph svg {{ min-width: 100%; height: auto; }}
  .bench-bar {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
  .bench-bar .bar {{ height: 20px; border-radius: 3px; min-width: 2px; }}
  .bench-bar .label {{ font-size: 0.8em; min-width: 100px; }}
  .timing {{ display: inline-block; background: #dbeafe; color: #1e40af; padding: 1px 6px;
             border-radius: 3px; font-size: 0.8em; font-weight: 500; }}
  footer {{ margin-top: 30px; padding: 15px 0; border-top: 1px solid #e2e8f0;
            color: #94a3b8; font-size: 0.75em; text-align: center; }}
</style>
</head>
<body>

<h1>v2ecoli Benchmark Report</h1>
<p style="color: #64748b; font-size: 0.9em;">{time.strftime('%Y-%m-%d %H:%M')} &middot; All benchmarks run through process-bigraph <code>Composite.run()</code></p>

<div class="section">
  <p><strong>v2ecoli</strong> is a whole-cell <em>E. coli</em> model running natively on
  <a href="https://github.com/vivarium-collective/process-bigraph">process-bigraph</a>.
  It migrates all 55 biological steps from
  <a href="https://github.com/CovertLab/vEcoli">vEcoli</a> to run through the standard
  <code>Composite.run()</code> pipeline with custom bigraph-schema types for bulk molecules,
  unique molecules, and listener stores.</p>
  <p>This report benchmarks the simulation's performance, correctness against vEcoli (v1),
  and biological accuracy across multiple timescales.</p>
</div>

<nav style="background: white; border-radius: 8px; padding: 12px 20px; margin: 10px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);">
  <strong style="font-size: 0.9em; color: #475569;">Contents</strong>
  <ol style="margin: 6px 0 0 0; padding-left: 20px; font-size: 0.88em; columns: 2; column-gap: 30px;">
    <li><a href="#sec-build">Document Build</a></li>
    <li><a href="#sec-sim">Simulation ({COMPARISON_DURATION:.0f}s)</a></li>
    <li><a href="#sec-compare">v1 Comparison</a></li>
    <li><a href="#sec-division">Division</a></li>
    <li><a href="#sec-long">Long Simulation</a></li>
    <li><a href="#sec-steps">Step Diagnostics</a></li>
    <li><a href="#sec-bigraph">Network Visualization</a></li>
    <li><a href="#sec-timing">Timing Summary</a></li>
  </ol>
</nav>

<!-- ===== Benchmark 1: Build ===== -->
<h2 id="sec-build">1. Document Build</h2>
<div class="metrics">
  <div class="metric"><div class="label">Cache Gen</div><div class="value blue">{b['cache_generation']:.1f}s</div></div>
  <div class="metric"><div class="label">Doc Build</div><div class="value blue">{b['document_build']:.2f}s</div></div>
  <div class="metric"><div class="label">Steps</div><div class="value">{b['n_steps']}</div></div>
  <div class="metric"><div class="label">Processes</div><div class="value">{b['n_processes']}</div></div>
</div>

<details>
<summary>Input Data (133 TSV files → simData → initial state)</summary>
<div class="section">
  <p>Raw data: <code>v2ecoli/reconstruction/ecoli/flat/</code> (133 TSV files, 13MB)</p>
  <p>Cached: <code>{CACHE_DIR}/initial_state.json</code> (10MB) + <code>sim_data_cache.dill</code> (190MB)</p>
</div>
</details>

<!-- ===== Benchmark 2: Short Sim ===== -->
<h2 id="sec-sim">2. Simulation ({COMPARISON_DURATION:.0f}s)</h2>
<div class="metrics">
  <div class="metric"><div class="label">Wall Time</div><div class="value blue">{s['wall_time']:.1f}s</div></div>
  <div class="metric"><div class="label">Sim/Wall</div><div class="value green">{s['rate']:.1f}x</div></div>
  <div class="metric"><div class="label">Bulk Changed</div><div class="value purple">{s['bulk_changed']}</div></div>
  <div class="metric"><div class="label">Dry Mass</div><div class="value">{s['final_dry_mass']:.1f} fg</div></div>
  <div class="metric"><div class="label">Volume</div><div class="value">{s['final_volume']:.4f} fL</div></div>
</div>
<div class="plot"><img src="data:image/png;base64,{mass_short}" alt="Mass"></div>
<div class="plot"><img src="data:image/png;base64,{growth_short}" alt="Growth"></div>

<!-- ===== Benchmark 3: v1 Comparison ===== -->
<h2 id="sec-compare">3. v1 Comparison ({COMPARISON_DURATION:.0f}s)</h2>

<div class="section">
  <h3>Methodology</h3>
  <p>Both v1 (vEcoli) and v2 (v2ecoli) simulations run for {COMPARISON_DURATION:.0f} seconds with identical
     initial states from the same simData. Accuracy is measured by comparing mass components
     (dry mass, protein, RNA, DNA, small molecules) at each simulated second. These are the
     biologically meaningful quantities — bulk molecule correlation alone is misleading because
     most of the 16,321 molecules don't change.</p>
</div>

<h3>Mass Component Accuracy</h3>
<div class="section" style="overflow-x: auto;">
  <table>
    <thead><tr>
      <th>Component</th><th>v1 Final (fg)</th><th>v2 Final (fg)</th>
      <th>Mean % Error</th><th>Max % Error</th><th>R&sup2;</th>
    </tr></thead>
    <tbody>{mass_accuracy_rows}</tbody>
  </table>
</div>

<div class="metrics">
  <div class="metric"><div class="label">Worst % Error</div><div class="value {'green' if c['worst_pct_error'] < 1 else 'red'}">{c['worst_pct_error']:.2f}%</div></div>
  <div class="metric"><div class="label">v1 Runtime</div><div class="value red">{c['v1_time']:.2f}s</div></div>
  <div class="metric"><div class="label">v2 Runtime</div><div class="value blue">{c['v2_time']:.2f}s</div></div>
  <div class="metric"><div class="label">Timesteps</div><div class="value">{c['common_timesteps']}</div></div>
</div>

<div class="plot"><img src="data:image/png;base64,{mass_comp_plot}" alt="Mass Comparison"></div>

<h3>Unique Molecules & Chromosome State</h3>
<div class="section">
  <p>Comparison of unique molecule populations and chromosome replication state at the end of
  the {COMPARISON_DURATION:.0f}s simulation.</p>
</div>
<div class="plot"><img src="data:image/png;base64,{unique_comp_plot}" alt="Unique Comparison"></div>

<details>
<summary>Bulk Molecule Correlation (supplementary)</summary>
<div class="section">
  <p><em>Note: Bulk correlation across all 16,321 molecules is dominated by invariant counts
  and does not reflect accuracy of biologically active molecules. Use mass component metrics above instead.</em></p>
  <div class="metrics">
    <div class="metric"><div class="label">Mean Correlation</div><div class="value">{c['mean_correlation']:.6f}</div></div>
    <div class="metric"><div class="label">Mean Exact Match</div><div class="value">{c['mean_exact_match']*100:.2f}%</div></div>
    <div class="metric"><div class="label">Mean RMSE</div><div class="value">{c['mean_rmse']:.2f}</div></div>
  </div>
  <div class="plot"><img src="data:image/png;base64,{comparison_plot}" alt="Bulk Comparison"></div>
</div>
</details>

<!-- ===== Benchmark 4: Division ===== -->
<h2 id="sec-division">4. Division</h2>

<div class="section">
  <h3>How Division Works</h3>
  <p>The Division step uses process-bigraph's native <code>_add</code>/<code>_remove</code> structural
  updates. When division is triggered (dry mass &ge; threshold with &ge; 2 chromosomes):</p>
  <ol style="margin: 8px 0 8px 20px; font-size: 0.9em;">
    <li><strong>State splitting</strong> — <code>divide_cell()</code> partitions the mother cell's state:
      <ul>
        <li>Bulk molecules: binomial distribution (p=0.5) on each molecule's count</li>
        <li>Chromosomes: alternating assignment (even→D1, odd→D2) with descendant domain tracking</li>
        <li>Chromosome-attached molecules: follow their domain (promoters, genes, DnaA boxes, RNAPs)</li>
        <li>RNAs: full transcripts binomial, partial transcripts follow RNAP domain</li>
        <li>Ribosomes: follow their mRNA, degraded-mRNA ribosomes split binomially</li>
      </ul>
    </li>
    <li><strong>Daughter cell construction</strong> — <code>build_document_from_configs()</code> builds complete
    cell states with fresh process instances from the divided initial state + cached configs</li>
    <li><strong>Structural update</strong> — returns <code>{{'agents': {{'_remove': [mother], '_add': [(d1, state), (d2, state)]}}}}</code>
    which the Composite processes to remove the mother and add two daughter agents</li>
  </ol>

  <h3>Pre-Division Caching</h3>
  <p>To avoid re-running the full simulation to reach division (~30 min simulated time,
  ~4 min wall time), use <code>run_and_cache()</code> to save periodic checkpoints:</p>
  <pre style="background: #f1f5f9; padding: 8px; border-radius: 4px; font-size: 0.85em;">from v2ecoli.composite import run_and_cache
composite = run_and_cache(intervals=[500, 1000, 1500, 1800, 2000])</pre>
  <p>Resume from a checkpoint: <code>composite = load_state('out/checkpoints/t1800.dill')</code></p>
</div>

<h3>Division Test Results</h3>
<div class="section">
  <p>Tests run on initial state (t=0). At actual division time (~1857s), the cell has 2+
  chromosomes and proper domain trees for biologically correct partitioning.</p>
</div>
<div class="metrics">
  <div class="metric"><div class="label">Bulk Conserved</div><div class="value {'green' if div['bulk_conserved'] else 'red'}">{'Yes' if div['bulk_conserved'] else 'No'}</div></div>
  <div class="metric"><div class="label">Mother Bulk</div><div class="value">{div['mother_bulk_count']:,}</div></div>
  <div class="metric"><div class="label">D1 Bulk</div><div class="value">{div['d1_bulk_count']:,}</div></div>
  <div class="metric"><div class="label">D2 Bulk</div><div class="value">{div['d2_bulk_count']:,}</div></div>
  <div class="metric"><div class="label">State Split</div><div class="value blue">{div['split_time']*1000:.0f} ms</div></div>
  <div class="metric"><div class="label">Daughter Build</div><div class="value blue">{div['daughter_build_time']:.1f}s</div></div>
  <div class="metric"><div class="label">Daughter Viable</div><div class="value {'green' if div['daughter_viable'] else 'red'}">{'Yes' if div['daughter_viable'] else 'No'}</div></div>
</div>

<details open>
<summary>Unique Molecule Conservation</summary>
<div class="section" style="overflow-x: auto;">
  <table>
    <thead><tr><th>Molecule</th><th>Mother (active)</th><th>Daughter 1</th><th>Daughter 2</th><th>Conserved</th></tr></thead>
    <tbody>{div_unique_rows}</tbody>
  </table>
</div>
</details>

<!-- ===== Benchmark 5: Long Sim ===== -->
<h2 id="sec-long">5. Long Simulation ({LONG_DURATION/60:.0f} min)</h2>
<div class="metrics">
  <div class="metric"><div class="label">Sim Duration</div><div class="value">{l['duration']:.0f}s</div></div>
  <div class="metric"><div class="label">Wall Time</div><div class="value blue">{l['wall_time']:.1f}s</div></div>
  <div class="metric"><div class="label">Sim/Wall</div><div class="value green">{l['rate']:.1f}x</div></div>
  <div class="metric"><div class="label">Bulk Changed</div><div class="value purple">{l['bulk_changed']}</div></div>
  <div class="metric"><div class="label">Dry Mass</div><div class="value">{l['final_dry_mass']:.1f} fg</div></div>
</div>
<div class="plot"><img src="data:image/png;base64,{mass_long}" alt="Long Mass"></div>
<div class="plot"><img src="data:image/png;base64,{growth_long}" alt="Long Growth"></div>

<!-- ===== Benchmark 5: Step Diagnostics ===== -->
<h2 id="sec-steps">6. Step Diagnostics ({len(diag)} steps)</h2>
<details open>
<summary>Execution Flow</summary>
<div class="section" style="overflow-x: auto;">
  <table>
    <thead><tr><th>Step</th><th>Class</th><th>Config Keys</th><th>Ports</th><th>Priority</th></tr></thead>
    <tbody>{step_rows}</tbody>
  </table>
</div>
</details>

<!-- ===== Bigraph ===== -->
<h2 id="sec-bigraph">7. Process-Bigraph Network Visualization</h2>
<div class="section">
  <p>Visualization of the biological process network. Steps (colored) read from and write to
  shared stores (bulk, unique, listeners). Scroll horizontally to see the full network.</p>
</div>
<div class="bigraph">{bigraph_svg}</div>

<!-- ===== Timing Summary ===== -->
<h2 id="sec-timing">8. Timing Summary</h2>
<div class="section">
  <table>
    <tr><th>Benchmark</th><th>Wall Time</th><th>Sim Time</th><th>Sim/Wall Ratio</th></tr>
    <tr><td>Cache generation</td><td>{b['cache_generation']:.1f}s</td><td>—</td><td>—</td></tr>
    <tr><td>Document build</td><td>{b['document_build']:.2f}s</td><td>—</td><td>—</td></tr>
    <tr><td>Short simulation</td><td>{s['wall_time']:.1f}s</td><td>{s['duration']:.0f}s</td><td>{s['rate']:.1f}x</td></tr>
    <tr><td>v1 comparison (v1)</td><td>{c['v1_time']:.2f}s</td><td>{c['duration']:.0f}s</td><td>{c['duration']/c['v1_time']:.1f}x</td></tr>
    <tr><td>v1 comparison (v2)</td><td>{c['v2_time']:.2f}s</td><td>{c['duration']:.0f}s</td><td>{c['duration']/c['v2_time']:.1f}x</td></tr>
    <tr><td>Long simulation</td><td>{l['wall_time']:.1f}s</td><td>{l['duration']:.0f}s</td><td>{l['rate']:.1f}x</td></tr>
    <tr><td><strong>Total</strong></td><td><strong>{b['cache_generation']+b['document_build']+s['wall_time']+c['v1_time']+c['v2_time']+l['wall_time']:.0f}s</strong></td><td>—</td><td>—</td></tr>
  </table>
</div>

<footer>
  v2ecoli &middot; <a href="https://github.com/vivarium-collective/v2ecoli">github.com/vivarium-collective/v2ecoli</a>
  &middot; All steps run through process-bigraph Composite.run()
</footer>
</body>
</html>""")

    print(f"\nBenchmark report saved to {report_path}")
    return report_path


if __name__ == '__main__':
    run_benchmarks()
