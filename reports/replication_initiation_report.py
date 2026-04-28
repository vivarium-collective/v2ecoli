"""Replication-initiation workflow report.

Renders a single-cell trajectory under the new ``replication_initiation``
architecture, focused on signals relevant to the DnaA-cycle work:

  * oriC count and chromosome count over time
  * Replisome (fork) count and fork positions
  * DnaA box bound vs free counts (current model has these as a structural
    artifact; the new biology will read/write them)
  * DnaA bulk-protein abundance
  * Mass-per-oriC vs critical-mass threshold (the trigger the current
    model uses; the new biology will replace it)

Alongside the trajectory, the page surfaces the curated molecular reference
from ``v2ecoli.data.replication_initiation.molecular_reference`` (sourced
from ``docs/references/replication_initiation.md`` / .pdf) and a gap table
listing which mechanisms from the PDF are not yet captured by the model.

Usage:
    python reports/replication_initiation_report.py
        # writes out/reports/replication_initiation_report.html

If ``out/workflow/single_cell.dill`` exists from a prior workflow run it
will be reused. Otherwise this script runs its own short single-cell
simulation; pass ``--duration N`` (seconds) to bound the sim length.
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from v2ecoli.data.replication_initiation import (
    DARS1, DARS2, DATA, DNAA_BOX_CONSENSUS, DNAA_BOX_HIGHEST_AFFINITY,
    DNAA_PROMOTER, ORIC, RIDA, SEQA,
)


CACHE_DIR = 'out/cache'
WORKFLOW_DIR = 'out/workflow'
OUT_DIR = 'out/reports'
DEFAULT_DURATION = 1500.0  # ~25 min — long enough to see the first initiation
SNAPSHOT_INTERVAL = 50.0


# ---------------------------------------------------------------------------
# Plot helpers
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


# ---------------------------------------------------------------------------
# Trajectory acquisition
# ---------------------------------------------------------------------------

def _load_cached_trajectory():
    """Try to reuse a single-cell trajectory from a previous workflow run."""
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
    if not snaps:
        return None
    return snaps


def _run_replication_initiation_sim(duration):
    """Run a short sim under the replication_initiation architecture and
    return a list of per-snapshot dicts."""
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


def _bool_entry_count(structured):
    """Count active entries in a vEcoli structured-array unique molecule store."""
    if structured is None or not hasattr(structured, 'dtype'):
        return 0
    if '_entryState' not in structured.dtype.names:
        return 0
    return int(structured['_entryState'].view(np.bool_).sum())


def _extract_replication_signals(history):
    """Pull per-time replication-relevant signals out of an emitter history."""
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

        # DnaA boxes — counts of bound vs total
        dnaA_total = 0
        dnaA_bound = 0
        boxes = h.get('DnaA_box')
        if boxes is not None and hasattr(boxes, 'dtype') and '_entryState' in boxes.dtype.names:
            entries = boxes[boxes['_entryState'].view(np.bool_)]
            dnaA_total = len(entries)
            if 'DnaA_bound' in boxes.dtype.names:
                dnaA_bound = int(entries['DnaA_bound'].sum())
        # Listener also exposes free / total — prefer listener if present.
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


# ---------------------------------------------------------------------------
# Plots
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
    ax.set_ylabel('Count')
    ax.set_xlabel('Time (min)')
    ax.set_title('oriC and chromosome counts')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax = axes[0, 1]
    ax.step(times, [s['n_replisomes'] for s in snaps], where='post',
            color='#f59e0b', lw=1.8)
    ax.set_ylabel('Replisome count')
    ax.set_xlabel('Time (min)')
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
    ax.set_ylabel('Count')
    ax.set_xlabel('Time (min)')
    ax.set_title('DnaA-box occupancy on the chromosome')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    cell_mass = np.array([s['cell_mass'] for s in snaps])
    dry_mass = np.array([s['dry_mass'] for s in snaps])
    n_oric = np.array([max(1, s['n_oriC']) for s in snaps])
    ax.plot(times, cell_mass / n_oric, color='#0891b2', lw=1.5,
            label='cell mass / oriC')
    ax.plot(times, dry_mass, color='#f59e0b', lw=1.0, alpha=0.6, ls='--',
            label='dry mass')
    ax.set_ylabel('Mass (fg)')
    ax.set_xlabel('Time (min)')
    ax.set_title('Mass and mass-per-oriC')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

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
# HTML rendering
# ---------------------------------------------------------------------------

_GAP_TABLE = [
    ('Mass-threshold initiation (M_cell / n_oriC ≥ M_critical)',
     'YES', 'Implemented in v2ecoli/processes/chromosome_replication.py.'),
    ('Explicit DnaA-ATP and DnaA-ADP species in bulk',
     'NO', 'Step 1 of the phased plan.'),
    ('R1 / R2 / R4 high-affinity DnaA occupancy at oriC',
     'PARTIAL', 'DnaA_box has a `DnaA_bound` boolean but no nucleotide form, '
     'no per-site affinity, and no gating role in initiation.'),
    ('Low-affinity oligomerization (R5M, τ2, I1, I2, C3, C2, I3, C1)',
     'NO', 'Box identities and ordered loading (C1→I3→C2→C3) not modeled.'),
    ('IHF binding at IBS1 / IBS2',
     'NO', 'Not represented.'),
    ('SeqA sequestration of hemimethylated GATC sites (~10 min)',
     'NO', 'Not represented.'),
    ('RIDA — DnaN (β-clamp) + Hda → DnaA-ATP hydrolysis',
     'NO', 'Hda not in process list; clamp loading kinetics not exposed.'),
    ('DDAH — IHF + datA → DnaA-ATP hydrolysis (backup)',
     'NO', 'datA not represented as a regulatory locus.'),
    ('DARS1 / DARS2 — ADP-DnaA → apo-DnaA → ATP-DnaA',
     'NO', 'DARS loci absent; reactivation pathway absent.'),
    ('dnaA promoter autoregulation by DnaA-ATP / DnaA-ADP',
     'NO', 'dnaA gene transcribed via the generic transcript_initiation '
     'process; no DnaA-occupancy feedback.'),
]


def _render_status_pill(status):
    color = {
        'YES':     '#16a34a',
        'PARTIAL': '#f59e0b',
        'NO':      '#dc2626',
    }.get(status, '#475569')
    return (f'<span style="background:{color};color:white;border-radius:12px;'
            f'padding:1px 9px;font-size:0.78em;font-weight:600;">{status}</span>')


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


def render_html(snaps, sim_meta):
    n_snaps = len(snaps)
    init_img = plot_initiation_signals(snaps)
    fork_img = plot_fork_positions(snaps)

    # Architecture identifier (sanity-check that the new module exposes itself)
    try:
        from v2ecoli.generate_replication_initiation import ARCHITECTURE_NAME
    except Exception:
        ARCHITECTURE_NAME = '(import failed)'

    gap_rows = ''.join(
        f'<tr><td>{html_lib.escape(name)}</td>'
        f'<td style="text-align:center;">{_render_status_pill(status)}</td>'
        f'<td>{html_lib.escape(note)}</td></tr>'
        for name, status, note in _GAP_TABLE
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>v2ecoli — replication initiation report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 1100px; margin: 32px auto; padding: 0 24px;
        background: #f8fafc; color: #0f172a; line-height: 1.55; }}
h1 {{ color: #0f172a; border-bottom: 3px solid #2563eb; padding-bottom: 8px; }}
h2 {{ color: #1e3a8a; margin-top: 1.6em; font-size: 1.2em; }}
h3 {{ color: #1e293b; font-size: 1.0em; margin-top: 1.2em; }}
.banner {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
           padding: 10px 16px; font-size: 0.9em; }}
table.ref, table.gap {{ border-collapse: collapse; width: 100%;
                         font-size: 0.9em; margin: 8px 0 18px; }}
table.ref th, table.ref td, table.gap th, table.gap td {{
    border: 1px solid #cbd5e1; padding: 5px 9px; text-align: left;
    vertical-align: top; }}
table.ref thead, table.gap thead {{ background: #e2e8f0; }}
table.ref code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
img {{ max-width: 100%; height: auto; }}
.note {{ font-size: 0.85em; color: #475569; }}
.refs {{ font-size: 0.85em; color: #334155; }}
.refs li {{ margin-bottom: 4px; }}
.kv {{ font-family: monospace; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Replication-initiation report</h1>
<div class="banner">
<strong>Architecture:</strong> <span class="kv">{html_lib.escape(ARCHITECTURE_NAME)}</span>
&nbsp;|&nbsp; <strong>Snapshots:</strong> {n_snaps}
&nbsp;|&nbsp; <strong>Source:</strong>
<a href="../../docs/references/replication_initiation_molecular_info.pdf">PDF</a>
&middot; <a href="../../docs/references/replication_initiation.md">Markdown summary</a>
</div>

<h2>Trajectory under the new architecture</h2>
{('<p class="note">' + html_lib.escape(sim_meta) + '</p>') if sim_meta else ''}
{(_img(init_img, 'replication-initiation signals') if init_img else
   '<p class="note">No simulation data available — re-run with the ParCa cache present, '
   'or run <code>python reports/workflow_report.py</code> first to populate '
   '<code>out/workflow/single_cell.dill</code>.</p>')}
{_img(fork_img, 'fork positions over time')}

<h2>Curated molecular reference (from the PDF)</h2>

<h3>oriC ({ORIC.length_bp} bp; {len(ORIC.dnaA_boxes)} DnaA boxes; {len(ORIC.ihf_sites)} IHF sites)</h3>
<p class="note">High-affinity boxes bind both DnaA-ATP and DnaA-ADP (Kd ~1 nM).
Low-affinity boxes prefer DnaA-ATP and bind cooperatively (Kd &gt; 100 nM).
Ordered DnaA-ATP loading on the right arm of the DOR:
{' &rarr; '.join(ORIC.ordered_oligomerization_right_arm)}.</p>
{_render_oric_table()}

<h3>dnaA promoter ({DNAA_PROMOTER.length_bp} bp; p2 ≈ {DNAA_PROMOTER.p2_to_p1_strength_ratio:g}× p1)</h3>
{_render_promoter_table()}

<h3>datA ({DATA.length_bp} bp at {DATA.chromosomal_position_min} min)</h3>
<p class="kv">DnaA boxes: {DATA.n_dnaA_boxes} &nbsp;|&nbsp;
essential: {', '.join(DATA.essential_dnaA_box_names)} &nbsp;|&nbsp;
stimulatory: {', '.join(DATA.stimulatory_dnaA_box_names)} &nbsp;|&nbsp;
IHF sites: {DATA.n_ihf_sites}</p>

<h3>DARS1 / DARS2</h3>
{_render_dars_table()}

<h3>SeqA / RIDA</h3>
<p class="kv">SeqA sequestration window: {SEQA.sequestration_window_minutes:g} min
(~{SEQA.fraction_of_doubling_time_at_rapid_growth:.2f} of doubling time at rapid growth).
SeqA binds {SEQA.binds_state} GATC sites in newly-replicated oriC
(&gt;{SEQA.n_gatc_sites_oriC_lower_bound} per origin).</p>
<p class="kv">RIDA: {RIDA.clamp_protein} (β-clamp) + {RIDA.hda_nucleotide_state}-{RIDA.catalytic_partner}
(N-terminal clamp-binding motif) catalyzes <code>{html_lib.escape(RIDA.reaction)}</code>.</p>

<h3>Consensus motifs</h3>
<p class="kv">Consensus: <code>{DNAA_BOX_CONSENSUS}</code> (W=A|T, N=any).
Highest-affinity 9-mer: <code>{DNAA_BOX_HIGHEST_AFFINITY}</code> (Kd ~1 nM).</p>

<h2>What the model captures vs the PDF</h2>
<table class="gap">
<thead><tr><th>Mechanism</th><th>In v2ecoli today</th><th>Notes</th></tr></thead>
<tbody>{gap_rows}</tbody>
</table>

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

</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION,
                        help='Sim duration in seconds (when no cache hit).')
    parser.add_argument('--no-sim', action='store_true',
                        help='Skip the simulation; render reference panels only.')
    parser.add_argument('--out', default=os.path.join(OUT_DIR, 'replication_initiation_report.html'))
    args = parser.parse_args()

    snaps = []
    sim_meta = ''

    if not args.no_sim:
        cached = _load_cached_trajectory()
        if cached:
            sim_meta = (f'Reusing cached trajectory from '
                        f'{WORKFLOW_DIR}/single_cell.dill ({len(cached)} snapshots). '
                        f'Note: cached trajectory was generated under the baseline '
                        f'architecture; signals shown are still informative because '
                        f'the replication_initiation architecture is currently a '
                        f'baseline clone.')
            snaps = cached
        else:
            sim_meta = (f'Ran a fresh single-cell simulation under the '
                        f'replication_initiation architecture for up to '
                        f'{args.duration:.0f}s.')
            snaps = _run_replication_initiation_sim(args.duration)
    else:
        sim_meta = 'Reference-only mode (--no-sim).'

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    html = render_html(snaps, sim_meta)
    with open(args.out, 'w') as f:
        f.write(html)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
