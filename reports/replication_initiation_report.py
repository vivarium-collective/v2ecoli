"""Replication-initiation workflow report.

Layout: a left side-menu navigates an overview, the nine integration phases
(Phase 0 — Region-tag DnaA boxes through Phase 8 — dnaA promoter
autoregulation), the molecular-reference panels sourced from
``docs/references/replication_initiation.md`` (and the underlying PDF),
the trajectory plots under the new ``replication_initiation`` architecture,
and the references list.

Per-phase status is **auto-detected from the codebase**: each phase
declares a check that inspects schema files, molecule_ids, process modules,
sim_data attributes, etc. As phases land, the sidebar pills flip from
``pending`` to ``in progress`` to ``done`` and the gap items in each
section get crossed off — no manual report-side bookkeeping.

Usage:
    python reports/replication_initiation_report.py
        # writes out/reports/replication_initiation_report.html
"""

from __future__ import annotations

import argparse
import ast
import base64
import html as html_lib
import io
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from v2ecoli.data.replication_initiation import (
    DARS1, DARS2, DATA, DNAA_BOX_CONSENSUS, DNAA_BOX_HIGHEST_AFFINITY,
    DNAA_PROMOTER, ORIC, RIDA, SEQA,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = 'out/cache'
WORKFLOW_DIR = 'out/workflow'
OUT_DIR = 'out/reports'
DEFAULT_DURATION = 1500.0
SNAPSHOT_INTERVAL = 50.0


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _img(b64, alt=''):
    if not b64:
        return ''
    return f'<img alt="{html_lib.escape(alt)}" src="data:image/png;base64,{b64}"/>'


def _read_file(rel_path):
    p = os.path.join(REPO_ROOT, rel_path)
    if not os.path.isfile(p):
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()


def _file_contains(rel_path, needle):
    text = _read_file(rel_path)
    return text is not None and needle in text


def _file_exists(rel_path):
    return os.path.isfile(os.path.join(REPO_ROOT, rel_path))


# ---------------------------------------------------------------------------
# Trajectory acquisition
# ---------------------------------------------------------------------------

def _load_cached_trajectory():
    path = os.path.join(WORKFLOW_DIR, 'single_cell.dill')
    if not os.path.exists(path):
        return None
    try:
        import dill
        with open(path, 'rb') as f:
            data = dill.load(f)
    except Exception as exc:
        print(f'  cached trajectory load failed: {exc}')
        return None
    snaps = data.get('snapshots') if isinstance(data, dict) else None
    return snaps or None


def _bool_entry_count(structured):
    if structured is None or not hasattr(structured, 'dtype'):
        return 0
    if '_entryState' not in structured.dtype.names:
        return 0
    return int(structured['_entryState'].view(np.bool_).sum())


def _extract_replication_signals(history):
    snaps = []
    for h in history:
        t = float(h.get('global_time', 0))
        if int(t) % int(SNAPSHOT_INTERVAL) != 0 and t > 1:
            continue
        mass = h.get('listeners', {}).get('mass', {}) if isinstance(h.get('listeners'), dict) else {}
        rep_listener = h.get('listeners', {}).get('replication_data', {}) if isinstance(h.get('listeners'), dict) else {}

        n_oric = _bool_entry_count(h.get('oriC'))
        n_chrom = _bool_entry_count(h.get('full_chromosome'))
        n_rep = _bool_entry_count(h.get('active_replisome'))

        rep = h.get('active_replisome')
        fork_coords = []
        if rep is not None and hasattr(rep, 'dtype') and '_entryState' in rep.dtype.names:
            active = rep[rep['_entryState'].view(np.bool_)]
            if len(active) > 0 and 'coordinates' in rep.dtype.names:
                fork_coords = active['coordinates'].tolist()

        dnaA_total = 0
        dnaA_bound = 0
        boxes = h.get('DnaA_box')
        if boxes is not None and hasattr(boxes, 'dtype') and '_entryState' in boxes.dtype.names:
            entries = boxes[boxes['_entryState'].view(np.bool_)]
            dnaA_total = len(entries)
            if 'DnaA_bound' in boxes.dtype.names:
                dnaA_bound = int(entries['DnaA_bound'].sum())
        free_listener = rep_listener.get('free_DnaA_boxes')
        total_listener = rep_listener.get('total_DnaA_boxes')
        if total_listener is not None:
            dnaA_total = int(total_listener)
        if free_listener is not None and total_listener is not None:
            dnaA_bound = int(total_listener) - int(free_listener)

        snaps.append({
            'time': t,
            'n_oriC': n_oric,
            'n_chromosomes': n_chrom,
            'n_replisomes': n_rep,
            'fork_coords': fork_coords,
            'dnaA_box_total': dnaA_total,
            'dnaA_box_bound': dnaA_bound,
            'dry_mass': float(mass.get('dry_mass', 0)),
            'cell_mass': float(mass.get('cell_mass', 0)),
            'dna_mass': float(mass.get('dna_mass', 0)),
        })
    return snaps


def _run_replication_initiation_sim(duration):
    if not os.path.isdir(CACHE_DIR):
        print(f'  cache dir {CACHE_DIR!r} not present — skipping sim')
        return []
    from v2ecoli.composite_replication_initiation import (
        make_replication_initiation_composite,
    )
    print(f'  building replication_initiation composite from {CACHE_DIR}')
    composite = make_replication_initiation_composite(cache_dir=CACHE_DIR, seed=0)
    cell = composite.state['agents']['0']
    em_edge = cell.get('emitter', {})
    emitter = em_edge.get('instance') if isinstance(em_edge, dict) else None
    t0 = time.time()
    elapsed = 0.0
    while elapsed < duration:
        chunk = min(SNAPSHOT_INTERVAL, duration - elapsed)
        try:
            composite.run(chunk)
        except Exception as exc:
            print(f'  sim aborted at t={elapsed}s: {type(exc).__name__}: {exc}')
            break
        elapsed += chunk
    print(f'  ran {elapsed:.0f}s sim in {time.time() - t0:.1f}s wall time')
    history = emitter.history if emitter is not None else []
    return _extract_replication_signals(history)


# ---------------------------------------------------------------------------
# Plot helpers (trajectory)
# ---------------------------------------------------------------------------

def plot_initiation_signals(snaps):
    if not snaps:
        return ''
    times = np.array([s['time'] / 60 for s in snaps])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Replication-initiation signals (current model)', fontsize=13)

    ax = axes[0, 0]
    ax.step(times, [s['n_oriC'] for s in snaps], where='post',
            color='#10b981', lw=1.8, label='oriC')
    ax.step(times, [s['n_chromosomes'] for s in snaps], where='post',
            color='#7c3aed', lw=1.4, ls='--', label='chromosomes')
    ax.set_ylabel('Count'); ax.set_xlabel('Time (min)')
    ax.set_title('oriC and chromosome counts')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax = axes[0, 1]
    ax.step(times, [s['n_replisomes'] for s in snaps], where='post',
            color='#f59e0b', lw=1.8)
    ax.set_ylabel('Replisome count'); ax.set_xlabel('Time (min)')
    ax.set_title('Active replisomes (forks / 2)')
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax = axes[1, 0]
    bound = np.array([s['dnaA_box_bound'] for s in snaps])
    total = np.array([s['dnaA_box_total'] for s in snaps])
    free = total - bound
    ax.plot(times, total, color='#1e293b', lw=1.4, label='total DnaA boxes')
    ax.plot(times, bound, color='#dc2626', lw=1.4, label='bound')
    ax.plot(times, free, color='#2563eb', lw=1.4, label='free')
    ax.set_ylabel('Count'); ax.set_xlabel('Time (min)')
    ax.set_title('DnaA-box occupancy on the chromosome')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    cell_mass = np.array([s['cell_mass'] for s in snaps])
    dry_mass = np.array([s['dry_mass'] for s in snaps])
    n_oric = np.array([max(1, s['n_oriC']) for s in snaps])
    ax.plot(times, cell_mass / n_oric, color='#0891b2', lw=1.5,
            label='cell mass / oriC')
    ax.plot(times, dry_mass, color='#f59e0b', lw=1.0, alpha=0.6, ls='--',
            label='dry mass')
    ax.set_ylabel('Mass (fg)'); ax.set_xlabel('Time (min)')
    ax.set_title('Mass and mass-per-oriC')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig_to_b64(fig)


def plot_fork_positions(snaps):
    if not snaps:
        return ''
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle('Replication-fork positions over time', fontsize=12)
    has_data = False
    for s in snaps:
        for c in s.get('fork_coords') or []:
            has_data = True
            ax.scatter(s['time'] / 60, c, c='#f59e0b', s=12,
                       alpha=0.7, edgecolors='black', linewidths=0.3)
    if not has_data:
        ax.text(0.5, 0.5, 'No active replisomes during this sim window.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='#475569')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Fork coordinate (bp; oriC=0)')
    ax.axhline(0, color='#10b981', ls='--', lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Phase status detection
# ---------------------------------------------------------------------------

def _check_phase0():
    """Region classifier — coordinate-based, no schema change."""
    text = _read_file('v2ecoli/data/replication_initiation/molecular_reference.py')
    has_classifier = bool(text and 'def region_for_coord' in text)
    has_boundaries = bool(text and 'REGION_BOUNDARIES' in text)
    if has_classifier and has_boundaries:
        return 'done', 'region_for_coord and REGION_BOUNDARIES in molecular_reference'
    if has_boundaries or has_classifier:
        return 'in_progress', (
            'partial: ' + ('boundaries' if has_boundaries else 'classifier')
            + ' present')
    if _file_exists('tests/test_dnaA_box_regions.py'):
        return 'in_progress', 'phase test file exists; classifier not yet added'
    return 'pending', 'no region_for_coord / REGION_BOUNDARIES in molecular_reference.py'


def _check_phase1():
    """DnaA-ADP species."""
    p = 'v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/molecule_ids.py'
    text = _read_file(p)
    if text and ('DnaA-ADP' in text or 'DnaA_ADP' in text):
        return 'done', 'DnaA-ADP molecule ID registered'
    return 'pending', 'no DnaA-ADP molecule ID found in molecule_ids.py'


def _check_phase2():
    """DnaA box binding process."""
    if _file_exists('v2ecoli/processes/dnaA_box_binding.py'):
        return 'done', 'v2ecoli/processes/dnaA_box_binding.py present'
    if _file_exists('tests/test_dnaA_binding.py'):
        return 'in_progress', 'phase test file exists; process not yet added'
    return 'pending', 'no dnaA_box_binding process module'


def _check_phase3():
    """Replace mass-threshold initiation gate."""
    text = _read_file('v2ecoli/processes/chromosome_replication.py')
    if text and 'criticalMassPerOriC' not in text:
        return 'done', 'mass threshold removed from chromosome_replication.py'
    if text and 'dnaA_filament' in text:
        return 'in_progress', 'DnaA-filament gating logic present alongside mass threshold'
    return 'pending', 'mass threshold (`criticalMassPerOriC`) still drives initiation'


def _check_phase4():
    """SeqA sequestration."""
    if _file_exists('v2ecoli/processes/seqA_sequestration.py'):
        return 'done', 'SeqASequestration process present'
    text = _read_file('v2ecoli/library/schema_types.py')
    if text and 'hemimethylated' in text.lower():
        return 'in_progress', 'hemimethylation field added to schema; process pending'
    return 'pending', 'no SeqA process; no hemimethylation tracking'


def _check_phase5():
    """RIDA — Hda + β-clamp."""
    if _file_exists('v2ecoli/processes/rida.py'):
        return 'done', 'RIDA process present'
    p = 'v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/molecule_ids.py'
    text = _read_file(p)
    if text and 'Hda' in text:
        return 'in_progress', 'Hda registered; RIDA process pending'
    return 'pending', 'no Hda molecule; no RIDA process'


def _check_phase6():
    """DDAH — datA-mediated DnaA-ATP hydrolysis."""
    if _file_exists('v2ecoli/processes/ddah.py'):
        return 'done', 'DDAH process present'
    text = _read_file('v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/process/replication.py')
    if text and 'datA' in text:
        return 'in_progress', 'datA region loaded into motif_coordinates; process pending'
    return 'pending', 'no datA region in motif_coordinates; no DDAH process'


def _check_phase7():
    """DARS1/2 reactivation."""
    if _file_exists('v2ecoli/processes/dars_reactivation.py'):
        return 'done', 'DARSReactivation process present'
    text = _read_file('v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/process/replication.py')
    if text and ('DARS1' in text or 'DARS2' in text):
        return 'in_progress', 'DARS region(s) loaded; process pending'
    return 'pending', 'no DARS regions in motif_coordinates; no reactivation process'


def _check_phase8():
    """dnaA promoter autoregulation."""
    text = _read_file('v2ecoli/processes/transcript_initiation.py')
    if text and ('dnaA_autoregulation' in text or 'dnaA_box_occupancy' in text):
        return 'done', 'dnaA-promoter autoregulation hook present in transcript_initiation'
    return 'pending', 'no DnaA-occupancy hook in transcript_initiation'


# ---------------------------------------------------------------------------
# Per-phase analysis renderers
# ---------------------------------------------------------------------------

def _placeholder(text):
    return (f'<div class="placeholder">{html_lib.escape(text)}</div>')


# ---------------------------------------------------------------------------
# Test-file introspection — pull docstrings from the AST so hover tooltips
# describe what each test asserts. For not-yet-existing test files we fall
# back to the planned description carried on the TestSpec.
# ---------------------------------------------------------------------------

def _extract_test_summary(rel_path):
    """Return (n_tests, [(name, summary), ...]) by parsing the test file's AST.

    ``summary`` is the first line of the docstring if present, else a short
    synthesized phrase from the function name. Returns (0, []) if the file
    is missing or fails to parse."""
    text = _read_file(rel_path)
    if text is None:
        return 0, []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0, []
    items = []

    def _docline(node):
        d = ast.get_docstring(node)
        if not d:
            return ''
        return d.strip().split('\n', 1)[0]

    def _humanize(name):
        return name.removeprefix('test_').replace('_', ' ')

    for top in tree.body:
        if isinstance(top, ast.ClassDef) and top.name.startswith('Test'):
            cls_doc = _docline(top)
            for sub in top.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and sub.name.startswith('test_'):
                    items.append((
                        f'{top.name}.{sub.name}',
                        _docline(sub) or cls_doc or _humanize(sub.name),
                    ))
        elif isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and top.name.startswith('test_'):
            items.append((top.name, _docline(top) or _humanize(top.name)))
    return len(items), items


@dataclass
class TestSpec:
    path: str
    description: str  # planned summary; shown when the file does not exist


def _render_test_li(spec):
    """One <li> per test file with a hover tooltip showing per-test summaries."""
    n_tests, items = _extract_test_summary(spec.path)
    present = _file_exists(spec.path)
    mark = '✓' if present else '○'
    cls = 'present' if present else 'missing'

    if present:
        header = f'<strong>{n_tests} test{"" if n_tests == 1 else "s"}</strong>'
        rows = ''.join(
            f'<div class="t-row"><code>{html_lib.escape(name)}</code>'
            f' — {html_lib.escape(summary)}</div>'
            for name, summary in items
        ) or f'<div class="t-row note">{html_lib.escape(spec.description)}</div>'
        body = header + rows
    else:
        body = (f'<strong class="planned">Planned</strong>'
                f'<div class="t-row note">{html_lib.escape(spec.description)}</div>')

    return (f'<li class="test {cls}"><span class="mark">{mark}</span> '
            f'<code>{html_lib.escape(spec.path)}</code>'
            f'<span class="tooltip">{body}</span></li>')


# ---------------------------------------------------------------------------
# "Before / after" analysis blocks — generic renderer driven by per-phase data
# ---------------------------------------------------------------------------

def _expected_region_counts_table():
    rows = [
        ('oriC', len(ORIC.dnaA_boxes)),
        ('dnaA promoter', len(DNAA_PROMOTER.dnaA_boxes)),
        ('datA', DATA.n_dnaA_boxes),
        ('DARS1', len(DARS1.core_box_names) + len(DARS1.extra_box_names)),
        ('DARS2', len(DARS2.core_box_names) + len(DARS2.extra_box_names)),
    ]
    body = ''.join(f'<tr><td>{r[0]}</td><td>{r[1]}</td></tr>' for r in rows)
    return ('<table class="ref"><thead><tr><th>Region</th>'
            '<th>Expected DnaA-box count</th></tr></thead>'
            f'<tbody>{body}</tbody></table>')


def _no_data_msg():
    return _placeholder('No simulation data — re-run with --duration N.')


# Per-phase "before" plots use only data already in the trajectory (so they
# render today even with the model in its current state). When the phase
# lands, the matching ``after_plot`` slot in the Phase dataclass swaps the
# placeholder for a real plot.

def _before_phase0(snaps):
    return ('<p class="note">Today the chromosome carries DnaA boxes from a '
            'single global motif set, with no per-region breakdown. Below is '
            'the expected per-region count when the classifier is applied to '
            'the existing init-state coordinates.</p>'
            + _expected_region_counts_table())


def _before_phase1(snaps):
    return _placeholder(
        'Single DnaA monomer pool with no ATP/ADP distinction. After Phase 1 '
        'this panel becomes two traces (DnaA-ATP, DnaA-ADP) plus a ratio panel.')


def _before_phase2(snaps):
    if not snaps:
        return _no_data_msg()
    times = np.array([s['time'] / 60 for s in snaps])
    bound = np.array([s['dnaA_box_bound'] for s in snaps])
    total = np.array([s['dnaA_box_total'] for s in snaps])
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.plot(times, total, color='#1e293b', lw=1.4, label='total')
    ax.plot(times, bound, color='#dc2626', lw=1.4, label='bound (always 0)')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('DnaA boxes')
    ax.set_title('Current: DnaA_bound is write-only — never set to True')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _img(fig_to_b64(fig), 'DnaA-box occupancy (current)')


def _before_phase3(snaps):
    if not snaps:
        return _no_data_msg()
    times = np.array([s['time'] / 60 for s in snaps])
    cell_mass = np.array([s['cell_mass'] for s in snaps])
    n_oric = np.array([max(1, s['n_oriC']) for s in snaps])
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.plot(times, cell_mass / n_oric, color='#0891b2', lw=1.6,
            label='cell mass / oriC')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Mass (fg)')
    ax.set_title('Current trigger: mass-per-oriC vs M_critical')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _img(fig_to_b64(fig), 'mass-per-oriC trigger (current)')


def _before_phase4(snaps):
    if not snaps:
        return _no_data_msg()
    times = np.array([s['time'] / 60 for s in snaps])
    n_oric = np.array([s['n_oriC'] for s in snaps])
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.step(times, n_oric, where='post', color='#10b981', lw=1.6)
    ax.set_xlabel('Time (min)'); ax.set_ylabel('oriC count')
    ax.set_title('Current: oriC count — no sequestration window after fork passage')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _img(fig_to_b64(fig), 'oriC count (current)')


def _before_phase5(snaps):
    if not snaps:
        return _no_data_msg()
    times = np.array([s['time'] / 60 for s in snaps])
    n_rep = np.array([s['n_replisomes'] for s in snaps])
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.step(times, n_rep, where='post', color='#f59e0b', lw=1.6)
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Active replisomes')
    ax.set_title('Current: replisome activity is not coupled to DnaA-ATP hydrolysis')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _img(fig_to_b64(fig), 'replisome activity (current)')


def _before_phase6(snaps):
    return _placeholder(
        'No datA locus in the model today; no DnaA-ATP hydrolysis flux from datA. '
        'After Phase 6, this panel plots datA-mediated flux versus RIDA flux.')


def _before_phase7(snaps):
    return _placeholder(
        'No DARS reactivation today; DnaA-ADP cannot be converted back to '
        'DnaA-ATP. After Phase 7, this panel plots reactivation flux split '
        'across DARS1 and DARS2, with IHF/Fis state indicators.')


def _before_phase8(snaps):
    return _placeholder(
        'dnaA transcription rate is constant (no occupancy feedback). After '
        'Phase 8, this panel plots dnaA mRNA against DnaA-ATP/ADP occupancy '
        'at p1/p2 — the closed feedback loop.')


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    number: int
    title: str
    goal: str
    status_check: Callable
    tests: tuple[TestSpec, ...]
    gap_items: tuple[str, ...]
    viz_plan: str
    before_plot: Callable
    after_description: str
    after_plot: Optional[Callable] = None  # filled in once phase lands

    @property
    def slug(self) -> str:
        return f'phase-{self.number}'


PHASES: list[Phase] = [
    Phase(
        number=0,
        title='Region classifier (coordinate-based)',
        goal=('Add bp boundaries for oriC / dnaA_promoter / datA / DARS1 / '
              'DARS2 to molecular_reference.py and a `region_for_coord(bp)` '
              'classifier. No schema change — region is derived from each '
              'box\'s existing `coordinates` field.'),
        status_check=_check_phase0,
        tests=(
            TestSpec(
                'tests/test_replication_initiation_reference.py',
                'Locks in the curated PDF facts (counts, sequences, affinity '
                'classes, motif compatibility) as data assertions.'),
            TestSpec(
                'tests/test_dnaA_box_regions.py',
                'Apply region_for_coord over the actual init-state DnaA-box '
                'coordinates from sim_data and assert the count per region '
                'matches the curated reference (oriC=11, dnaA promoter=7, '
                'datA=4, DARS1=3, DARS2=5).'),
        ),
        gap_items=(
            'No way to slice DnaA boxes by regulatory locus (oriC vs dnaA '
            'promoter vs datA vs DARS) — the binding process in Phase 2 needs '
            'this to apply per-region affinity classes.',
            'No init-time check that the bioinformatic motif search finds the '
            'expected number of boxes per locus (11 / 7 / 4 / 3 / 5).',
        ),
        viz_plan=(
            'Bar chart of DnaA-box counts per region, with the curated '
            'reference counts overlaid as horizontal markers. Highlights '
            'any gap between what the motif search returns and what the '
            'literature says should be there.'),
        before_plot=_before_phase0,
        after_description=(
            'After Phase 0: live bar chart of DnaA-box counts per region '
            'computed from sim_data.process.replication.motif_coordinates '
            'via region_for_coord — the same lookup the binding process '
            'will use.'),
    ),
    Phase(
        number=1,
        title='DnaA-ADP species',
        goal=('Add the DnaA-ADP molecule ID alongside DnaA-ATP. Split the '
              'initial pool; listener emits both.'),
        status_check=_check_phase1,
        tests=(
            TestSpec(
                'tests/test_dnaA_nucleotide_pool.py',
                'DnaA-ATP and DnaA-ADP are both registered as bulk molecules; '
                'their initial counts sum to the expected total DnaA-monomer '
                'count; the ATP/ADP ratio matches the configured fraction '
                '(default 70/30) within tolerance.'),
        ),
        gap_items=(
            'Only DnaA-ATP (MONOMER0-160) is registered; nucleotide-state is '
            'invisible to the rest of the model.',
            'No conservation test for DnaA-ATP + DnaA-ADP + box-bound count.',
        ),
        viz_plan=(
            'Two-trace plot of DnaA-ATP and DnaA-ADP bulk counts over time, '
            'with their ratio plotted on a secondary axis. Without RIDA/DARS '
            'these traces are flat — the time-varying behavior emerges in '
            'Phases 5 and 7.'),
        before_plot=_before_phase1,
        after_description=(
            'After Phase 1: DnaA-ATP and DnaA-ADP traces with their ratio. '
            'Phases 5 and 7 will animate the ratio over the cell cycle.'),
    ),
    Phase(
        number=2,
        title='DnaA box binding',
        goal=('New DnaABoxBinding process: reads DnaA-ATP/ADP pools and '
              'per-region affinities, writes DnaA_bound on each box.'),
        status_check=_check_phase2,
        tests=(
            TestSpec(
                'tests/test_dnaA_binding.py',
                'At steady-state DnaA-ATP pool: high-affinity oriC boxes '
                '(R1/R2/R4) are ≥95% occupied; low-affinity oriC boxes track '
                'DnaA-ATP concentration; per-region affinity class drives '
                'occupancy via region_for_coord.'),
        ),
        gap_items=(
            'DnaA_bound is write-only — set to False at init and after fork '
            'passage, but no process ever sets it to True.',
            'tf_binding handles promoter sites only; chromosomal DnaA boxes '
            'have no binding logic.',
        ),
        viz_plan=(
            'Per-region bound-fraction traces (oriC high-affinity, oriC '
            'low-affinity, dnaA promoter, datA, DARS) over time. The '
            "high-affinity oriC trace should sit near 1.0; the low-affinity "
            'trace tracks DnaA-ATP concentration.'),
        before_plot=_before_phase2,
        after_description=(
            'After Phase 2: per-region bound-fraction traces. Replaces the '
            'flat-zero "bound" line with real occupancy dynamics.'),
    ),
    Phase(
        number=3,
        title='Replace mass-threshold initiation',
        goal=('Gate ChromosomeReplication on R1/R2/R4 occupancy + low-affinity '
              'DnaA-ATP filament threshold instead of M_cell/n_oriC.'),
        status_check=_check_phase3,
        tests=(
            TestSpec(
                'tests/test_model_behavior.py',
                'Existing whole-cell behavior tests (cell-cycle timing, mass '
                'doubling, replication-fork conservation). Thresholds may '
                'need rebaselining for the DnaA-driven gate.'),
            TestSpec(
                'tests/test_initiation_dnaA_gate.py',
                'Initiation timing under the DnaA-occupancy gate matches the '
                'mass-threshold version within ±10% under nominal growth, '
                'but shifts measurably when the DnaA pool is perturbed '
                '(±20% DnaA monomer count).'),
        ),
        gap_items=(
            'Initiation in `chromosome_replication.py:243, 320` is purely '
            'mass-driven; DnaA is not even in the bulk inputs.',
            'Behavior thresholds in `test_model_behavior.py` will need '
            'rebaselining once the gate is DnaA-driven.',
        ),
        viz_plan=(
            'Side-by-side: cell mass / oriC trace and DnaA-ATP filament '
            'occupancy at oriC, with vertical markers at every initiation '
            'event. Demonstrates that the new gate fires at biologically '
            'consistent times rather than purely mass-based ones.'),
        before_plot=_before_phase3,
        after_description=(
            'After Phase 3: dual panel with mass-per-oriC (deprecated trigger) '
            'and DnaA-filament occupancy (new trigger), with initiation '
            'events marked on both. Shifts visible when the DnaA pool is '
            'perturbed.'),
    ),
    Phase(
        number=4,
        title='SeqA sequestration',
        goal=('Add SeqA protein, GATC site coords at oriC, hemimethylation '
              'timer; SeqASequestration blocks DnaA rebinding for ~10 min.'),
        status_check=_check_phase4,
        tests=(
            TestSpec(
                'tests/test_seqA_sequestration.py',
                'After a forced initiation, no second initiation occurs '
                'within the SeqA sequestration window (~10 min). The window '
                'duration scales with doubling time as expected.'),
        ),
        gap_items=(
            'SeqA listed as a gene but no protein definition; no sequestration '
            'process; no hemimethylation tracking.',
            'No GATC-site coordinates loaded into motif_coordinates.',
        ),
        viz_plan=(
            'oriC count over time with the hemimethylation window shaded '
            'beneath each initiation event, plus a marker showing what would '
            'have been a too-early re-initiation if SeqA were absent.'),
        before_plot=_before_phase4,
        after_description=(
            'After Phase 4: oriC count with shaded sequestration window and '
            'a counterfactual marker showing pre-Phase-4 timing.'),
    ),
    Phase(
        number=5,
        title='RIDA',
        goal=('Add Hda protein; expose loaded β-clamp count from active '
              'replisomes; hydrolyze DnaA-ATP → DnaA-ADP at rate ∝ activity.'),
        status_check=_check_phase5,
        tests=(
            TestSpec(
                'tests/test_rida.py',
                'Active DnaA-ATP pool drops as replisomes elongate; the drop '
                'rate is ∝ active replisome count; the pool recovers after '
                'replication completes (no replisomes → no flux).'),
        ),
        gap_items=(
            'Hda not in molecule_ids; no clamp count exposed; no RIDA process.',
            'DnaN (β-clamp) protein is in the dnaA operon but is not counted '
            'separately for clamp-loading logic.',
        ),
        viz_plan=(
            'Twin-axis plot: DnaA-ATP pool (left axis) and active replisome '
            'count (right axis) over the cell cycle. Phase-5 effect: the '
            'DnaA-ATP trace develops a visible dip while replisomes are '
            'active.'),
        before_plot=_before_phase5,
        after_description=(
            'After Phase 5: twin-axis plot showing DnaA-ATP dipping during '
            'active replication. Pre-Phase-5 trace shown as ghost line for '
            'comparison.'),
    ),
    Phase(
        number=6,
        title='DDAH (datA)',
        goal=('Load datA region coords; IHF binds the datA IBS; DDAH process '
              'hydrolyzes DnaA-ATP via the datA–IHF complex.'),
        status_check=_check_phase6,
        tests=(
            TestSpec(
                'tests/test_ddah.py',
                'Knocking out datA (parameter gate) accelerates the next '
                'initiation by a measurable amount; flux from DDAH is '
                'non-zero only when IHF is present at the datA IBS.'),
        ),
        gap_items=(
            'datA not loaded into `sim_data.process.replication.motif_coordinates`.',
            'IhfA/IhfB defined as proteins but no DNA-binding logic at datA.',
        ),
        viz_plan=(
            'Stacked-area plot of DnaA-ATP hydrolysis flux split into RIDA '
            'and DDAH contributions over the cell cycle. Both should be '
            'non-trivial; DDAH peaks shortly after initiation when IHF '
            'binds datA.'),
        before_plot=_before_phase6,
        after_description=(
            'After Phase 6: stacked-area RIDA + DDAH flux. Knockout overlay '
            'shows the loss of DDAH flux when datA is removed.'),
    ),
    Phase(
        number=7,
        title='DARS1/2 reactivation',
        goal=('Load DARS coords; DARSReactivation converts DnaA-ADP → DnaA-ATP; '
              'DARS2 modulated by IHF + Fis binding.'),
        status_check=_check_phase7,
        tests=(
            TestSpec(
                'tests/test_dars.py',
                'DnaA-ADP → DnaA-ATP regeneration flux is non-zero, '
                'dominated by DARS2 in vivo, and modulated by the IHF/Fis '
                'binding states; DARS1 alone cannot sustain the cycle.'),
        ),
        gap_items=(
            'DARS1 and DARS2 are absent from motif_coordinates and from the '
            'process list. Fis (CPLX0-7705) is defined as a protein but has no '
            'DNA-binding role.',
        ),
        viz_plan=(
            'Twin-axis plot: DARS1 and DARS2 reactivation fluxes over the '
            'cell cycle, with IHF/Fis binding-state indicators marked on the '
            'time axis. DARS2 should dominate; both should peak after RIDA '
            'has driven DnaA-ADP up.'),
        before_plot=_before_phase7,
        after_description=(
            'After Phase 7: dual-flux plot for DARS1 and DARS2 with IHF/Fis '
            'state indicators. The DnaA-ATP trace from Phase 5 visibly '
            'recovers.'),
    ),
    Phase(
        number=8,
        title='dnaA promoter autoregulation',
        goal=('transcript_initiation reads DnaA-ATP/ADP occupancy at the p1/p2 '
              'boxes and modulates dnaA transcription rate.'),
        status_check=_check_phase8,
        tests=(
            TestSpec(
                'tests/test_dnaA_autoregulation.py',
                'dnaA mRNA level decreases when DnaA-ATP occupancy at p1/p2 '
                'boxes is elevated and recovers when the DnaA pool drops; '
                'p2/p1 promoter activity ratio ≈ 3 at low DnaA occupancy.'),
        ),
        gap_items=(
            'dnaA gene is transcribed via generic `transcript_initiation`; '
            'no DnaA-occupancy feedback on its own promoter.',
        ),
        viz_plan=(
            'Twin-axis plot: dnaA mRNA level (left axis) and DnaA-ATP+ADP '
            'occupancy at p1/p2 boxes (right axis). The two should be '
            'anti-correlated, closing the autoregulatory loop.'),
        before_plot=_before_phase8,
        after_description=(
            'After Phase 8: anti-correlated dnaA mRNA and p1/p2 occupancy '
            'traces; p2/p1 ratio panel.'),
    ),
]


# ---------------------------------------------------------------------------
# HTML rendering — sidebar + main content
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    'done':        ('#16a34a', 'done'),
    'in_progress': ('#f59e0b', 'in progress'),
    'pending':     ('#94a3b8', 'pending'),
}


def _status_pill(status):
    color, label = _STATUS_COLORS.get(status, ('#475569', status))
    return (f'<span class="pill" style="background:{color};">'
            f'{html_lib.escape(label)}</span>')


def _render_oric_table():
    rows = []
    for b in ORIC.dnaA_boxes:
        seq = b.sequence or '—'
        rows.append(
            f'<tr><td>{html_lib.escape(b.name)}</td>'
            f'<td>{html_lib.escape(b.affinity_class)}</td>'
            f'<td><code>{html_lib.escape(seq)}</code></td>'
            f'<td>{html_lib.escape(b.nucleotide_preference)}</td></tr>'
        )
    return ('<table class="ref"><thead><tr><th>Box</th><th>Class</th>'
            '<th>Sequence</th><th>Nucleotide preference</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


def _render_promoter_table():
    rows = []
    for b in DNAA_PROMOTER.dnaA_boxes:
        seq = b.sequence or '—'
        rows.append(
            f'<tr><td>{html_lib.escape(b.name)}</td>'
            f'<td>{html_lib.escape(b.affinity_class)}</td>'
            f'<td><code>{html_lib.escape(seq)}</code></td>'
            f'<td>{html_lib.escape(b.nucleotide_preference)}</td></tr>'
        )
    return ('<table class="ref"><thead><tr><th>Box</th><th>Class</th>'
            '<th>Sequence</th><th>Nucleotide preference</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


def _render_dars_table():
    rows = []
    for d in (DARS1, DARS2):
        rows.append(
            f'<tr><td>{html_lib.escape(d.name)}</td>'
            f'<td>{d.length_bp}</td>'
            f'<td>{", ".join(d.core_box_names)}</td>'
            f'<td>{", ".join(d.extra_box_names) or "—"}</td>'
            f'<td>{d.n_ihf_sites}</td>'
            f'<td>{d.n_fis_sites}</td>'
            f'<td>{"yes" if d.is_dominant_in_vivo else "no"}</td></tr>'
        )
    return ('<table class="ref"><thead><tr><th>Locus</th><th>bp</th>'
            '<th>Core boxes</th><th>Extra boxes</th>'
            '<th>IHF sites</th><th>Fis sites</th>'
            '<th>Dominant</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


def _render_phase_section(phase: Phase, statuses, snaps):
    status, evidence = statuses[phase.number]
    test_rows = ''.join(_render_test_li(t) for t in phase.tests)
    gap_rows = ''.join(
        f'<li>{html_lib.escape(g)}</li>' for g in phase.gap_items
    )

    if phase.after_plot is not None:
        after_block = phase.after_plot(snaps)
    else:
        after_block = _placeholder(phase.after_description)

    return f"""
<section id="{phase.slug}">
  <div class="phase-header">
    <h2>Phase {phase.number} — {html_lib.escape(phase.title)}</h2>
    {_status_pill(status)}
  </div>
  <p class="goal">{html_lib.escape(phase.goal)}</p>
  <p class="evidence">Status check: <em>{html_lib.escape(evidence)}</em></p>
  <h3>Tests <span class="hint">(hover for descriptions)</span></h3>
  <ul class="tests">{test_rows}</ul>
  <h3>Gaps closed by this phase</h3>
  <ul class="gaps">{gap_rows}</ul>
  <h3>Visualization plan</h3>
  <p class="viz-plan">{html_lib.escape(phase.viz_plan)}</p>
  <div class="before-after">
    <div class="ba-col">
      <h4>Before — current model</h4>
      {phase.before_plot(snaps)}
    </div>
    <div class="ba-col">
      <h4>After — Phase {phase.number} lands</h4>
      {after_block}
    </div>
  </div>
</section>
"""


def _render_overview_table(statuses):
    rows = []
    for phase in PHASES:
        status, _ = statuses[phase.number]
        rows.append(
            f'<tr><td>{phase.number}</td>'
            f'<td><a href="#{phase.slug}">{html_lib.escape(phase.title)}</a></td>'
            f'<td>{_status_pill(status)}</td>'
            f'<td>{html_lib.escape(phase.goal)}</td></tr>'
        )
    return ('<table class="overview"><thead><tr><th>#</th><th>Phase</th>'
            '<th>Status</th><th>Goal</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


def _render_sidebar(statuses):
    phase_links = []
    for phase in PHASES:
        status, _ = statuses[phase.number]
        color = _STATUS_COLORS[status][0]
        phase_links.append(
            f'<a href="#{phase.slug}" class="phase-link">'
            f'<span class="dot" style="background:{color};"></span>'
            f'<span class="num">P{phase.number}</span> '
            f'<span class="ttl">{html_lib.escape(phase.title)}</span></a>'
        )
    return f"""
<aside class="sidebar">
  <h3>Navigation</h3>
  <a href="#overview" class="nav-link">Overview</a>
  <div class="phase-list">{''.join(phase_links)}</div>
  <h3>Reference</h3>
  <a href="#ref-oriC" class="nav-link">oriC</a>
  <a href="#ref-promoter" class="nav-link">dnaA promoter</a>
  <a href="#ref-datA" class="nav-link">datA</a>
  <a href="#ref-dars" class="nav-link">DARS1 / DARS2</a>
  <a href="#ref-seqA-rida" class="nav-link">SeqA / RIDA</a>
  <a href="#ref-motifs" class="nav-link">Consensus motifs</a>
  <h3>Data</h3>
  <a href="#trajectory" class="nav-link">Trajectory plots</a>
  <a href="#references" class="nav-link">References</a>
</aside>
"""


def _ref_path(out_path, target):
    """Compute a relative path from the report's output dir to a file
    under ``docs/references/``. Works whether the report lives at
    ``out/reports/...`` (local) or at ``docs/...`` (published)."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    target_abs = os.path.join(REPO_ROOT, 'docs', 'references', target)
    return os.path.relpath(target_abs, out_dir)


def render_html(snaps, sim_meta, out_path):
    statuses = {p.number: p.status_check() for p in PHASES}
    n_done = sum(1 for s, _ in statuses.values() if s == 'done')
    n_in_progress = sum(1 for s, _ in statuses.values() if s == 'in_progress')
    n_total = len(PHASES)

    pdf_link = _ref_path(out_path, 'replication_initiation_molecular_info.pdf')
    md_link = _ref_path(out_path, 'replication_initiation.md')

    try:
        from v2ecoli.generate_replication_initiation import ARCHITECTURE_NAME
    except Exception:
        ARCHITECTURE_NAME = '(import failed)'

    sidebar_html = _render_sidebar(statuses)
    overview_table = _render_overview_table(statuses)
    phase_sections = '\n'.join(
        _render_phase_section(p, statuses, snaps) for p in PHASES
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v2ecoli — replication initiation report</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        margin: 0; background: #f8fafc; color: #0f172a; line-height: 1.5; }}
.layout {{ display: flex; min-height: 100vh; }}
.sidebar {{ width: 240px; background: #0f172a; color: #e2e8f0;
            padding: 20px 16px; position: sticky; top: 0;
            height: 100vh; overflow-y: auto; flex-shrink: 0; }}
.sidebar h3 {{ font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.04em;
               color: #94a3b8; margin: 18px 0 6px; }}
.sidebar a {{ display: block; color: #cbd5e1; text-decoration: none;
              font-size: 0.88em; padding: 4px 6px; border-radius: 4px; }}
.sidebar a:hover {{ background: #1e293b; color: #fff; }}
.phase-link {{ display: flex !important; align-items: baseline; gap: 6px; }}
.phase-link .dot {{ display: inline-block; width: 8px; height: 8px;
                    border-radius: 50%; flex-shrink: 0; }}
.phase-link .num {{ font-family: monospace; font-size: 0.78em; color: #64748b;
                    flex-shrink: 0; }}
.phase-link .ttl {{ flex: 1; }}
.main {{ flex: 1; padding: 28px 36px; max-width: 1000px; }}
h1 {{ color: #0f172a; border-bottom: 3px solid #2563eb; padding-bottom: 8px;
      margin-top: 0; }}
h2 {{ color: #1e3a8a; margin-top: 0; font-size: 1.2em; }}
h3 {{ color: #1e293b; font-size: 0.98em; margin-top: 1.2em;
      margin-bottom: 0.4em; }}
.banner {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
           padding: 10px 16px; font-size: 0.9em; }}
.kv {{ font-family: monospace; font-size: 0.9em; }}
section {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px;
           padding: 18px 22px; margin: 16px 0; }}
.phase-header {{ display: flex; align-items: center; gap: 12px; }}
.phase-header h2 {{ margin: 0; }}
.pill {{ display: inline-block; color: white; border-radius: 12px;
         padding: 1px 10px; font-size: 0.78em; font-weight: 600; }}
.goal {{ color: #334155; font-size: 0.92em; margin-top: 8px; }}
.evidence {{ color: #475569; font-size: 0.85em; }}
.tests {{ list-style: none; padding-left: 0; margin: 4px 0 8px; }}
.tests li.test {{ font-family: monospace; font-size: 0.85em; padding: 2px 0;
                   position: relative; cursor: help; }}
.tests .mark {{ display: inline-block; width: 16px;
                color: #16a34a; font-weight: 700; }}
.tests li.missing .mark {{ color: #94a3b8; }}
.tests li.missing code {{ color: #64748b; }}
.tests .tooltip {{ display: none; position: absolute;
                    left: 24px; top: 100%; z-index: 20;
                    background: #0f172a; color: #e2e8f0;
                    border-radius: 6px; padding: 10px 14px;
                    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                    font-size: 0.82em; line-height: 1.5;
                    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
                    width: 520px; max-width: 80vw; }}
.tests li.test:hover .tooltip {{ display: block; }}
.tests .tooltip strong {{ color: #fff; display: inline-block;
                           margin-bottom: 4px; }}
.tests .tooltip strong.planned {{ color: #fbbf24; }}
.tests .tooltip .t-row {{ margin: 3px 0; }}
.tests .tooltip .t-row code {{ background: #1e293b; color: #93c5fd;
                                padding: 1px 5px; border-radius: 3px; }}
.tests .tooltip .t-row.note {{ color: #cbd5e1; font-style: italic; }}
.hint {{ font-size: 0.75em; font-weight: normal; color: #94a3b8;
         font-family: -apple-system, sans-serif; }}
.gaps {{ margin: 4px 0 8px; padding-left: 22px; font-size: 0.9em;
         color: #334155; }}
.gaps li {{ margin-bottom: 3px; }}
.viz-plan {{ font-size: 0.92em; color: #1e293b;
             background: #fef3c7; border-left: 3px solid #f59e0b;
             padding: 8px 14px; margin: 4px 0 10px;
             border-radius: 0 4px 4px 0; }}
.before-after {{ display: grid; grid-template-columns: 1fr 1fr;
                  gap: 14px; margin-top: 6px; }}
.ba-col {{ background: #f8fafc; border: 1px solid #e2e8f0;
            border-radius: 6px; padding: 12px; }}
.ba-col h4 {{ margin: 0 0 8px; font-size: 0.9em;
              color: #1e293b; text-transform: uppercase;
              letter-spacing: 0.05em; }}
.ba-col img {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0;
                border-radius: 4px; background: white; }}
.placeholder {{ background: #f1f5f9; border: 1px dashed #cbd5e1;
                border-radius: 6px; padding: 12px 16px; color: #475569;
                font-size: 0.85em; font-style: italic; }}
@media (max-width: 900px) {{ .before-after {{ grid-template-columns: 1fr; }} }}
table {{ border-collapse: collapse; width: 100%;
         font-size: 0.88em; margin: 6px 0 12px; }}
table th, table td {{ border: 1px solid #cbd5e1; padding: 5px 9px;
                       text-align: left; vertical-align: top; }}
table thead {{ background: #e2e8f0; }}
table.overview td:first-child {{ width: 28px; text-align: center;
                                  font-family: monospace; }}
table code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
.note {{ font-size: 0.85em; color: #475569; }}
.refs {{ font-size: 0.85em; color: #334155; }}
.refs li {{ margin-bottom: 4px; }}
</style>
</head>
<body>
<div class="layout">
{sidebar_html}
<main class="main">

<h1>Replication-initiation report</h1>
<div class="banner">
<strong>Architecture:</strong> <span class="kv">{html_lib.escape(ARCHITECTURE_NAME)}</span>
&nbsp;|&nbsp; <strong>Snapshots:</strong> {len(snaps)}
&nbsp;|&nbsp; <strong>Phase progress:</strong>
{n_done}/{n_total} done, {n_in_progress} in progress
&nbsp;|&nbsp; <strong>Source:</strong>
<a href="{html_lib.escape(pdf_link)}">PDF</a>
&middot; <a href="{html_lib.escape(md_link)}">Markdown summary</a>
</div>

<section id="overview">
<h2>Overview</h2>
<p class="note">
Each row links to the phase's section. Status is auto-detected from the codebase
(schema fields, molecule_ids entries, process modules, sim_data attributes), so
the indicators below cannot drift from reality.
</p>
{overview_table}
</section>

{phase_sections}

<section id="ref-oriC">
<h2>oriC</h2>
<p class="note">
{ORIC.length_bp} bp, {len(ORIC.dnaA_boxes)} DnaA boxes, {len(ORIC.ihf_sites)} IHF sites.
High-affinity boxes bind both DnaA-ATP and DnaA-ADP (Kd ~1 nM). Low-affinity
boxes prefer DnaA-ATP and bind cooperatively (Kd &gt; 100 nM). Ordered DnaA-ATP
loading on the right arm: {' &rarr; '.join(ORIC.ordered_oligomerization_right_arm)}.
</p>
{_render_oric_table()}
</section>

<section id="ref-promoter">
<h2>dnaA promoter</h2>
<p class="note">
{DNAA_PROMOTER.length_bp} bp; p2 ≈ {DNAA_PROMOTER.p2_to_p1_strength_ratio:g}× p1;
~{DNAA_PROMOTER.promoter_separation_bp} bp separation.
</p>
{_render_promoter_table()}
</section>

<section id="ref-datA">
<h2>datA</h2>
<p class="kv">
{DATA.length_bp} bp at {DATA.chromosomal_position_min} min &nbsp;|&nbsp;
DnaA boxes: {DATA.n_dnaA_boxes} &nbsp;|&nbsp;
essential: {', '.join(DATA.essential_dnaA_box_names)} &nbsp;|&nbsp;
stimulatory: {', '.join(DATA.stimulatory_dnaA_box_names)} &nbsp;|&nbsp;
IHF sites: {DATA.n_ihf_sites}
</p>
</section>

<section id="ref-dars">
<h2>DARS1 / DARS2</h2>
{_render_dars_table()}
</section>

<section id="ref-seqA-rida">
<h2>SeqA / RIDA</h2>
<p class="kv">
SeqA sequestration window: {SEQA.sequestration_window_minutes:g} min
(~{SEQA.fraction_of_doubling_time_at_rapid_growth:.2f} of doubling time at rapid growth).
SeqA binds {SEQA.binds_state} GATC sites (&gt;{SEQA.n_gatc_sites_oriC_lower_bound} per origin).
</p>
<p class="kv">
RIDA: {RIDA.clamp_protein} (β-clamp) + {RIDA.hda_nucleotide_state}-{RIDA.catalytic_partner}
(N-terminal clamp-binding motif) catalyzes <code>{html_lib.escape(RIDA.reaction)}</code>.
</p>
</section>

<section id="ref-motifs">
<h2>Consensus motifs</h2>
<p class="kv">
Consensus: <code>{DNAA_BOX_CONSENSUS}</code> (W = A|T, N = any).
Highest-affinity 9-mer: <code>{DNAA_BOX_HIGHEST_AFFINITY}</code> (Kd ~1 nM).
</p>
</section>

<section id="trajectory">
<h2>Trajectory under the new architecture</h2>
{('<p class="note">' + html_lib.escape(sim_meta) + '</p>') if sim_meta else ''}
{(_img(plot_initiation_signals(snaps), 'replication-initiation signals')
   if snaps else _placeholder('No simulation data — re-run with the ParCa cache present.'))}
{_img(plot_fork_positions(snaps), 'fork positions over time') if snaps else ''}
</section>

<section id="references">
<h2>Key references</h2>
<ul class="refs">
<li>Katayama, Kasho, Kawakami. <em>Frontiers in Microbiology</em> 8 (2017): 2496.</li>
<li>Kasho, Ozaki, Katayama. <em>Int. J. Mol. Sci.</em> 24 (2023): 11572.</li>
<li>Speck, Weigel, Messer. <em>EMBO J.</em> 18 (1999): 6169–6176.</li>
<li>Schaper &amp; Messer. <em>J. Biol. Chem.</em> 270 (1995): 17622–17626.</li>
<li>Hansen et al. <em>J. Mol. Biol.</em> 355 (2006): 85–95.</li>
<li>Olivi et al. <em>Nat. Commun.</em> 16 (2025): 7813.</li>
<li>Riber et al. <em>Front. Mol. Biosci.</em> 3 (2016): 29.</li>
<li>Hansen &amp; Atlung. <em>Front. Microbiol.</em> 9 (2018): 319.</li>
<li>Fujimitsu, Senriuchi, Katayama. <em>Genes &amp; Dev.</em> 23 (2009): 1221–1233.</li>
<li>Kasho et al. <em>Nucleic Acids Res.</em> 42 (2014): 13134–13149.</li>
</ul>
</section>

</main>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION)
    parser.add_argument('--no-sim', action='store_true')
    parser.add_argument('--out', default=os.path.join(OUT_DIR, 'replication_initiation_report.html'))
    args = parser.parse_args()

    snaps = []
    sim_meta = ''
    if not args.no_sim:
        cached = _load_cached_trajectory()
        if cached:
            sim_meta = (f'Reusing cached trajectory from '
                        f'{WORKFLOW_DIR}/single_cell.dill ({len(cached)} snapshots).')
            snaps = cached
        else:
            sim_meta = (f'Ran a fresh single-cell simulation under the '
                        f'replication_initiation architecture for up to '
                        f'{args.duration:.0f}s.')
            snaps = _run_replication_initiation_sim(args.duration)
    else:
        sim_meta = 'Reference-only mode (--no-sim).'

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(render_html(snaps, sim_meta, args.out))
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
