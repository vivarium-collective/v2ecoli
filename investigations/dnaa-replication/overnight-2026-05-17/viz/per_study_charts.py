"""Generate per-study chart SVGs that demonstrate each study's key findings.

Writes:
  studies/<name>/charts/*.svg              ← rendered SVGs
  studies/<name>/charts/*.meta.json        ← {title, caption} sidecars

The dashboard's render_study_charts auto-discovers these via the
_load_static_charts hook in vivarium_dashboard/lib/study_charts.py.

Two kinds of charts produced:
  (a) Finding-summary cards: distilled gate-test results with target bands
  (b) Copies of relevant data charts from overnight-2026-05-17/viz/
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

V2 = Path('/Users/eranagmon/code/v2ecoli')
OV = V2 / 'investigations' / 'dnaa-replication' / 'overnight-2026-05-17' / 'viz'


# ── Generic SVG builder ─────────────────────────────────────────────────

def card_svg(title: str, lines: list[dict], width: int = 760, height: int = 420) -> str:
    """Render a 'finding card' SVG: title + a vertical stack of labeled rows,
    where each row has (label, value, badge, optional bar).

    Each line dict supports:
      label:   left-side label
      value:   right-side value text
      badge:   one of 'pass' | 'fail' | 'partial' | 'info' | None
      bar:     optional dict {value: float in [0,1], color: hex} for a progress bar
    """
    PL, PR, PT, PB = 30, 30, 60, 40
    plot_w = width - PL - PR
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="-apple-system,sans-serif" font-size="13">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<rect x="0" y="0" width="{width}" height="42" fill="#0f172a"/>',
        f'<text x="{width/2}" y="27" text-anchor="middle" fill="white" '
        f'font-weight="600" font-size="15">{title}</text>',
    ]
    badge_color = {'pass': '#16a34a', 'fail': '#dc2626', 'partial': '#f59e0b',
                   'info': '#3b82f6', None: '#64748b'}
    badge_label = {'pass': '✓ PASS', 'fail': '✗ FAIL', 'partial': '⚠ PARTIAL',
                   'info': '◆ INFO', None: ''}

    row_y = PT
    for ln in lines:
        # Row background
        parts.append(f'<rect x="{PL}" y="{row_y}" width="{plot_w}" height="44" '
                     f'fill="#f8fafc" rx="3"/>')
        # Label
        parts.append(f'<text x="{PL+12}" y="{row_y+20}" fill="#0f172a" '
                     f'font-weight="600">{ln.get("label", "")}</text>')
        # Sub-text (sublabel)
        if ln.get('sublabel'):
            parts.append(f'<text x="{PL+12}" y="{row_y+36}" fill="#64748b" '
                         f'font-size="11">{ln["sublabel"]}</text>')
        # Value
        if ln.get('value'):
            parts.append(f'<text x="{PL+plot_w-160}" y="{row_y+26}" '
                         f'text-anchor="end" fill="#0f172a" font-weight="600" '
                         f'font-size="14" font-family="ui-monospace,monospace">'
                         f'{ln["value"]}</text>')
        # Badge
        b = ln.get('badge')
        if b:
            bc = badge_color.get(b, '#64748b')
            parts.append(f'<rect x="{PL+plot_w-110}" y="{row_y+10}" width="100" '
                         f'height="24" rx="12" fill="{bc}"/>')
            parts.append(f'<text x="{PL+plot_w-60}" y="{row_y+26}" '
                         f'text-anchor="middle" fill="white" font-weight="700" '
                         f'font-size="11">{badge_label[b]}</text>')
        row_y += 50

    return '\n'.join(parts) + '\n</svg>'


def write_chart(study_dir: Path, slug: str, svg_text: str, title: str, caption: str):
    """Write one chart SVG + its .meta.json sidecar to studies/<name>/charts/."""
    charts_dir = study_dir / 'charts'
    charts_dir.mkdir(exist_ok=True)
    (charts_dir / f'{slug}.svg').write_text(svg_text)
    (charts_dir / f'{slug}.meta.json').write_text(
        json.dumps({'title': title, 'caption': caption}, indent=2)
    )
    print(f'  ✓ {study_dir.name}/charts/{slug}.svg')


def copy_chart(study_dir: Path, slug: str, src_name: str, title: str, caption: str):
    """Copy an existing SVG from overnight-2026-05-17/viz/ into studies/<name>/charts/."""
    src = OV / src_name
    if not src.exists():
        print(f'  ✗ skip {slug}: source {src_name} missing')
        return
    charts_dir = study_dir / 'charts'
    charts_dir.mkdir(exist_ok=True)
    shutil.copy(src, charts_dir / f'{slug}.svg')
    (charts_dir / f'{slug}.meta.json').write_text(
        json.dumps({'title': title, 'caption': caption}, indent=2)
    )
    print(f'  ✓ {study_dir.name}/charts/{slug}.svg (copied)')


# ── Per-study generators ────────────────────────────────────────────────

def gen_dnaa_01():
    sd = V2 / 'studies' / 'dnaa-01-expression-dynamics'
    print(f'\n=== {sd.name} ===')

    write_chart(sd, '00_summary',
        card_svg('dnaa-01 — Expression Dynamics: gate-test scoreboard', [
            {'label': 'Total DnaA per cell',
             'sublabel': 'Schmidt 2016 mass-spec target: [300, 800]',
             'value': '115 (1× TE baseline)', 'badge': 'fail'},
            {'label': 'Autorepression signature',
             'sublabel': 'Pearson r ≤ −0.3 (5-seed pool)',
             'value': '−0.594', 'badge': 'pass'},
            {'label': 'DnaA pool stability',
             'sublabel': 'Rolling CV ≤ 0.10 (homeostasis)',
             'value': '0.008', 'badge': 'pass'},
            {'label': 'Joint-sweep fix found',
             'sublabel': '(TE=20×, fc=0.7) — see dnaa-01g',
             'value': 'DnaA=707, r=−0.521', 'badge': 'pass'},
            {'label': 'TE root cause',
             'sublabel': 'dnaA at 8.6th percentile of proteome (PD03831)',
             'value': '7.23e-5', 'badge': 'info'},
        ]),
        'dnaa-01 gate-test scoreboard',
        'Summary of the 4 primary gate tests for dnaa-01 plus the TE-percentile root cause. '
        '1× baseline fails count but passes autorepression+stability; the joint (TE×fc) sweep resolved both.')

    copy_chart(sd, '01_te_sweep_count', '01_te_sweep_count.svg',
        'DnaA count vs TE multiplier (literature acceptance band)',
        'Median DnaA per cell at each TE multiplier (5 seeds each). Green band: Schmidt 2016 [300, 800] target.')
    copy_chart(sd, '02_te_sweep_pearson', '02_te_sweep_pearson.svg',
        'Autorepression r vs TE multiplier',
        'Pooled Pearson correlation (DnaA-TF binding ↔ dnaA mRNA). Green pass band: r ≤ -0.3. '
        'Note the sign-flip between 20× and 25× — autorepression saturation.')
    copy_chart(sd, '03_te_sweep_combined', '03_te_sweep_combined.svg',
        'Joint gate landscape: count + autorepression',
        'Two gate tests overlaid. Count bar (left axis, green) and autorepression r (right axis, points). '
        'No single TE multiplier passes both gates.')
    copy_chart(sd, '04_te_percentile', '04_dnaa_te_percentile.svg',
        'dnaA at 8.6th percentile of v2ecoli proteome',
        "dnaA's ParCa-cached translation efficiency sits at the bottom 9% of the proteome — "
        "biologically improbable for a master regulator.")
    copy_chart(sd, '05_dnaa_states_t', '05_dnaa_states_timeseries.svg',
        'DnaA states equilibrate within 60 s',
        'apo (PD03831), DnaA-ATP (MONOMER0-160), DnaA-ADP (MONOMER0-4565). '
        'Equilibrium drives all DnaA to ATP-bound form within 60 s; ADP stays at zero (no hydrolysis sink).')


def gen_dnaa_01f_recalibrate():
    sd = V2 / 'studies' / 'dnaa-01f-recalibrate-eg10235-translation-efficiency-in-parca'
    print(f'\n=== {sd.name} ===')

    write_chart(sd, '00_summary',
        card_svg('dnaa-01f-recalibrate — Joint-sweep result scoreboard', [
            {'label': 'Single-knob TE sweep',
             'sublabel': '19 sims · 5 seeds × 7 multipliers',
             'value': 'No TE passes both', 'badge': 'fail'},
            {'label': '15× single-knob compromise',
             'sublabel': 'Autorep PASS r=−0.533 · count short 18%',
             'value': '247 DnaA/cell', 'badge': 'partial'},
            {'label': 'Phase transition discovered',
             'sublabel': 'Pearson r flips +0.78 at TE=25×',
             'value': '20×→25× boundary', 'badge': 'info'},
            {'label': 'Joint sweep WIN (TE=20×, fc=0.7)',
             'sublabel': '5 seeds; both primary tests PASS',
             'value': 'DnaA=707, r=−0.521', 'badge': 'pass'},
        ]),
        'dnaa-01f recalibration scoreboard',
        'Single-knob TE sweep failed; joint sweep (TE × autorepression fold_change) found a clean fix at (20×, 0.7).')

    copy_chart(sd, '01_te_sweep_count', '01_te_sweep_count.svg',
        'TE sweep — DnaA count', 'Per-multiplier median DnaA. Pass band [300, 800] highlighted.')
    copy_chart(sd, '02_te_sweep_pearson', '02_te_sweep_pearson.svg',
        'TE sweep — autorepression Pearson r', 'Pass band r ≤ -0.3 highlighted. Sign-flip at 20-25× = saturation.')
    copy_chart(sd, '03_te_sweep_combined', '03_te_sweep_combined.svg',
        'TE sweep — both gates overlaid', 'No single multiplier passes both gates.')
    copy_chart(sd, '04_fc_grid_count', '09_fc_grid_dnaa_count.svg',
        'Joint (TE × fc) sweep — DnaA count heatmap', 'Cells colored by median DnaA. (TE=20×, fc=0.7) is the sweet spot.')
    copy_chart(sd, '05_fc_grid_pearson', '10_fc_grid_pearson.svg',
        'Joint (TE × fc) sweep — autorepression r heatmap', 'Cells colored by Pearson r. fc<0.6 weakens autorepression too far.')


def gen_dnaa_02():
    sd = V2 / 'studies' / 'dnaa-02-atp-hydrolysis'
    print(f'\n=== {sd.name} ===')

    write_chart(sd, '00_summary',
        card_svg('dnaa-02 — Nucleotide cycle: scoreboard', [
            {'label': 'DnaA-ATP/ADP/apo species in bulk',
             'sublabel': 'PD03831, MONOMER0-160, MONOMER0-4565',
             'value': 'ALREADY EXIST', 'badge': 'pass'},
            {'label': 'ecoli-equilibrium process wired',
             'sublabel': 'Active every timestep in baseline composite',
             'value': 'WORKING', 'badge': 'pass'},
            {'label': 'DnaA-ATP fraction (baseline)',
             'sublabel': 'Boesen 2024 physiological target [0.20, 0.50]',
             'value': '0.99', 'badge': 'fail'},
            {'label': 'DnaA-ADP pool (baseline)',
             'sublabel': 'RXN0-7444 has no kinetic constraint',
             'value': '0', 'badge': 'fail'},
            {'label': 'Intrinsic hydrolysis @ Boesen rate (0.046/min)',
             'sublabel': 'Drafted Step + external probe',
             'value': '0.987 ATP frac (INSUFFICIENT)', 'badge': 'fail'},
            {'label': 'Mechanism PASSES at 100× rate (k=4.6/min)',
             'sublabel': 'Quantitative target for dnaa-05',
             'value': '0.231 ATP frac', 'badge': 'pass'},
        ]),
        'dnaa-02 nucleotide-cycle scoreboard',
        'Three-state DnaA pool already wired in baseline. Intrinsic hydrolysis alone insufficient; needs ~100× rate from dnaa-05 extrinsic pathways.')

    copy_chart(sd, '01_states_baseline', '05_dnaa_states_timeseries.svg',
        'DnaA states over time (1× baseline)',
        'Equilibrium drives all DnaA to ATP-bound within 60 s. apo → 0, DnaA-ATP → 113, DnaA-ADP stays at 0.')
    copy_chart(sd, '02_atp_fraction_baseline', '06_dnaa_atp_fraction.svg',
        'Baseline DnaA-ATP fraction (1× TE) — 2× over physiological band',
        'ATP fraction holds at ~0.99 across 10 min. Boesen 2024 target is [0.20, 0.50].')
    copy_chart(sd, '03_with_calibration_and_hydrolysis', '11_dnaa_with_hydrolysis.svg',
        'After (TE=20×, fc=0.7) + intrinsic hydrolysis: counts in band, but ATP fraction still 0.989',
        'Total DnaA = 707 (in literature band). But intrinsic hydrolysis alone cannot bring ATP fraction down — needs extrinsic boost.')
    copy_chart(sd, '04_rate_sensitivity', '12_hydrolysis_rate_sensitivity.svg',
        'Hydrolysis rate sensitivity — 100× intrinsic PASSES Boesen band',
        'Sensitivity probe across 4 rates. The mechanism is correct (passes at k=4.6/min); intrinsic-only is insufficient. '
        '1000× over-corrects to 0% ATP — operating window is roughly [1, 50]/min total flux.')


def gen_dnaa_03():
    sd = V2 / 'studies' / 'dnaa-03-box-binding'
    print(f'\n=== {sd.name} ===')

    write_chart(sd, '00_summary',
        card_svg('dnaa-03 — DnaA-box binding: scoreboard', [
            {'label': 'DNAA_BOX_ARRAY type exists',
             'sublabel': 'coordinates, domain_index, DnaA_bound attrs',
             'value': 'ALREADY DEFINED', 'badge': 'pass'},
            {'label': 'Boxes in runtime initial state',
             'sublabel': 'Spec said 322 (307 chrom + 11 oriC + 4 dnaAp)',
             'value': '456 active', 'badge': 'partial'},
            {'label': 'oriCs in runtime initial state',
             'sublabel': 'Mid-cell-cycle init — one per daughter',
             'value': '2 active', 'badge': 'info'},
            {'label': 'DnaA_bound modified at runtime',
             'sublabel': 'No process touches it during sim',
             'value': '0 / 491 throughout', 'badge': 'fail'},
            {'label': 'Simple Hill model (n=2) reproduces 2-step?',
             'sublabel': 'oriC fills FIRST, not last',
             'value': 'NO', 'badge': 'fail'},
            {'label': 'Required: cooperative Hill (n=5-10) at oriC',
             'sublabel': 'DnaA-ATP filament model — Hansen 2018',
             'value': 'NEXT STEP', 'badge': 'info'},
        ]),
        'dnaa-03 box-binding scoreboard',
        'Infrastructure for 456 boxes exists but no binding kinetics. Simple Hill model fails to reproduce textbook two-step pattern; cooperativity at oriC is required.')

    copy_chart(sd, '01_box_occupancy_hill', '13_box_occupancy_two_step.svg',
        'Simple Hill model: oriC fills FIRST (wrong)',
        'External Hill-binding probe (n=2; Kd high/med/low = 5/50/300 DnaA-ATP). '
        'oriC-affinity sites saturate at t=10s; chromosomal sites take until t=130s. '
        'Textbook predicts the OPPOSITE: chromosomal first, then oriC. '
        'Cooperativity at oriC (Hill n=5-10) is the missing ingredient.')


def gen_dnaa_04():
    sd = V2 / 'studies' / 'dnaa-04-initiation-mechanism'
    print(f'\n=== {sd.name} ===')

    write_chart(sd, '00_summary',
        card_svg('dnaa-04 — Initiation mechanism: scoreboard', [
            {'label': 'Current trigger: mass-threshold heuristic',
             'sublabel': 'criticalMassPerOriC >= 1.0',
             'value': 'chromosome_replication.py:244', 'badge': 'info'},
            {'label': 'Swap target: DnaA-occupancy-based',
             'sublabel': 'n_DnaA_bound_at_oriC >= filament_threshold',
             'value': '~11 (from dnaa-03)', 'badge': 'info'},
            {'label': 'Stub processes ready to fill in',
             'sublabel': 'chromosome_initiation.DnaABinder + ChromosomePartition',
             'value': 'EMPTY UPDATE()', 'badge': 'info'},
            {'label': 'Sim state at 10 min (validated baseline)',
             'sublabel': '2 oriCs + 2 forks at ±1.33Mb (mid-replication)',
             'value': 'No NEW init in window', 'badge': 'partial'},
            {'label': 'Validation requires longer sims',
             'sublabel': '≥60 min to capture termination + new initiation',
             'value': 'NEXT WORK', 'badge': 'info'},
            {'label': 'Box machinery is replication-aware',
             'sublabel': 'Boxes grew 456 → 491 behind moving forks',
             'value': 'CORRECT', 'badge': 'pass'},
        ]),
        'dnaa-04 initiation-mechanism scoreboard',
        'Swap point (chromosome_replication.py:244) precisely identified. Empty stubs in chromosome_initiation.py ready to fill in. Validation needs longer sims.')


def gen_dnaa_05():
    sd = V2 / 'studies' / 'dnaa-05-rida-ddah-dars'
    print(f'\n=== {sd.name} ===')

    write_chart(sd, '00_summary',
        card_svg('dnaa-05 — Extrinsic conversion (RIDA / DDAH / DARS): scoreboard', [
            {'label': 'Existing infrastructure in v2ecoli',
             'sublabel': 'grep across codebase',
             'value': '0 hits', 'badge': 'fail'},
            {'label': 'Quantitative target (from dnaa-02 F-04)',
             'sublabel': 'Combined extrinsic flux for [0.20, 0.50] ATP frac',
             'value': '~4.6 / min', 'badge': 'info'},
            {'label': 'Intrinsic baseline (Boesen)',
             'sublabel': 'For comparison',
             'value': '0.046 / min', 'badge': 'info'},
            {'label': 'Implied extrinsic boost',
             'sublabel': 'RIDA + DDAH + DARS together',
             'value': '~100×', 'badge': 'info'},
            {'label': 'Operating window (probe-validated)',
             'sublabel': 'Above 50/min → DnaA-ATP collapses to 0',
             'value': '[1, 50] / min', 'badge': 'info'},
            {'label': 'Recommended: audit MONOMER0-160_RXN reverse rate',
             'sublabel': 'May be miscalibrated — would reduce extrinsic burden',
             'value': 'dnaa-05g spawn', 'badge': 'partial'},
        ]),
        'dnaa-05 extrinsic-conversion scoreboard',
        'Green-field implementation; converted from open-ended design to calibrated target (~4.6/min combined extrinsic flux). Recommend equilibrium-rate audit before authoring 3 new Step modules.')

    copy_chart(sd, '01_target_derivation', '12_hydrolysis_rate_sensitivity.svg',
        'Where the 4.6/min target came from (dnaa-02 probe)',
        'Hydrolysis-rate sensitivity probe from dnaa-02. The 100× cell PASSES Boesen band — this is the dnaa-05 target.')


def gen_dnaa_06():
    sd = V2 / 'studies' / 'dnaa-06-seqa-sequestration'
    print(f'\n=== {sd.name} ===')

    write_chart(sd, '00_summary',
        card_svg('dnaa-06 — SeqA sequestration: scoreboard', [
            {'label': 'SeqA protein (EG12197) in proteome',
             'sublabel': '"negative modulator of initiation of replication"',
             'value': 'PRESENT (181 aa)', 'badge': 'pass'},
            {'label': 'Dam methyltransferase (EG10204) in proteome',
             'sublabel': 'GATC adenine methylation',
             'value': 'PRESENT (278 aa)', 'badge': 'pass'},
            {'label': 'HspQ (G6500) in proteome',
             'sublabel': 'Hemimethylated DNA-binding helper',
             'value': 'PRESENT (104 aa)', 'badge': 'pass'},
            {'label': 'DamX (EG11183) in proteome',
             'sublabel': 'Cell-division-coupled Dam regulator',
             'value': 'PRESENT (428 aa)', 'badge': 'pass'},
            {'label': 'SeqASequestration Process exists',
             'sublabel': 'Toggles oriC sequestered-until timestamp',
             'value': 'NOT IMPLEMENTED', 'badge': 'fail'},
            {'label': 'oriC has sequestered_until attribute',
             'sublabel': 'Current oriC unique store: 12 attributes',
             'value': 'MISSING (need +1)', 'badge': 'fail'},
            {'label': 'Estimated implementation effort',
             'sublabel': 'Simplest of the dnaA upstream chain',
             'value': '~1 day after dnaa-04', 'badge': 'info'},
        ]),
        'dnaa-06 SeqA-sequestration scoreboard',
        'All 4 relevant proteins (SeqA, Dam, HspQ, DamX) present in proteome. Just needs one oriC attribute + one Step. Simplest of the upstream chain.')


def main():
    print('Generating per-study chart sets...')
    gen_dnaa_01()
    gen_dnaa_01f_recalibrate()
    gen_dnaa_02()
    gen_dnaa_03()
    gen_dnaa_04()
    gen_dnaa_05()
    gen_dnaa_06()
    print('\nDone. Charts written to studies/<name>/charts/.')


if __name__ == '__main__':
    main()
