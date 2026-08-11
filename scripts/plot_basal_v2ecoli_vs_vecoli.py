"""Two-page basal-media comparison PDF: v2ecoli origin/main (page 1) vs vEcoli parent (page 2).
Each page shows cell_mass + oriC count trajectories over 10 generations, with the M*
reference line overlaid at 975 fg (the ParCa-derived critical initiation mass for basal,
identical between codebases: min(dry/0.3 * 1.2, 975 fg) caps at 975 for basal).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pyarrow.dataset as ds

V2_MAIN = '/Users/rashmidissasekara/Documents/code/v2ecoli-main/out/main_basal_seed0_30gen_reference_parquet/main_basal_seed0_30gen_reference/history/experiment_id=main_basal_seed0_30gen_reference'
VECOLI = '/Users/rashmidissasekara/Documents/code/vEcoli/out/tau_multigen_basal_seed0_10gen/history/experiment_id=tau_multigen_basal_seed0_10gen'
M_STAR_FG = 975.0
DEFAULT_TITLE_V2 = 'v2ecoli origin/main basal 10-gen (unmodified reference)'
DEFAULT_TITLE_VE = 'vEcoli (parent codebase) basal 10-gen  clean reference'

def load(path, time_col, accumulate_time=False):
    d = ds.dataset(path, partitioning='hive')
    cols = [time_col,'listeners__mass__cell_mass','listeners__replication_data__number_of_oric','generation']
    t = d.to_table(columns=cols).sort_by([('generation','ascending'),(time_col,'ascending')])
    gen = t['generation'].to_numpy()
    tt = t['t'.replace('t',time_col)].to_numpy().astype(float)
    if accumulate_time:
        # v2ecoli's global_time resets each generation; stitch into a continuous axis
        offset = 0.0
        prev_end = 0.0
        for gn in np.unique(gen):
            m = gen==gn
            gen_t = tt[m]
            if gn > np.unique(gen)[0]:
                offset = prev_end - gen_t[0]
            tt[m] = gen_t + offset
            prev_end = tt[m][-1]
    return {
        'gen': gen,
        't':   tt,
        'cm':  t['listeners__mass__cell_mass'].to_numpy(),
        'n':   t['listeners__replication_data__number_of_oric'].to_numpy(),
    }

def per_gen_tau(d, ngen):
    """Uses raw per-gen span; not affected by accumulation offsets since we take diff within a gen."""
    out = []
    for gn in range(1, ngen+1):
        m = d['gen']==gn
        if m.sum()==0:
            out.append(None); continue
        ts = d['t'][m]
        out.append((ts[-1]-ts[0])/60.0)
    return out

def inits_per_gen(d, ngen):
    """count oriC-doubling events in each gen"""
    counts = []
    for gn in range(1, ngen+1):
        m = d['gen']==gn
        if m.sum()==0:
            counts.append(0); continue
        n = d['n'][m]
        rises = int(((np.diff(n) > 0)).sum())
        counts.append(rises)
    return counts

def plot_page(pdf, d, title, ngen, time_col):
    fig, (ax_m, ax_o) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                     gridspec_kw={'height_ratios':[3,1]})
    # cell mass with continuous time axis
    m_used = d['gen'] <= ngen
    time_min = d['t'][m_used] / 60.0
    ax_m.plot(time_min, d['cm'][m_used], color='#1f2937', lw=0.8)
    for gn in range(2, ngen+1):
        m = d['gen']==gn
        if m.sum()==0: continue
        boundary = d['t'][m][0]/60.0
        ax_m.axvline(boundary, color='#94a3b8', lw=0.4, alpha=0.5)
    ax_m.axhline(M_STAR_FG, color='#b91c1c', lw=0.9, ls=':', alpha=0.85,
                 label=f'M* = {M_STAR_FG:g} fg (basal)')
    ax_m.axhline(2*M_STAR_FG, color='#dc2626', lw=0.7, ls='--', alpha=0.55,
                 label=f'2xM* = {2*M_STAR_FG:g} fg (expected init at 2 oriC)')
    ax_m.set_ylabel('cell mass (fg)')
    ax_m.set_title(title)
    ax_m.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax_m.grid(True, alpha=0.3)

    ax_o.step(time_min, d['n'][m_used], where='post', color='#0369a1', lw=0.8)
    for gn in range(2, ngen+1):
        m = d['gen']==gn
        if m.sum()==0: continue
        boundary = d['t'][m][0]/60.0
        ax_o.axvline(boundary, color='#94a3b8', lw=0.4, alpha=0.5)
    ax_o.set_ylabel('oriC count')
    ax_o.set_xlabel('time (min)')
    ax_o.set_ylim(-0.3, max(4.5, d['n'][m_used].max()+0.5))
    ax_o.grid(True, alpha=0.3)

    taus = per_gen_tau(d, ngen)
    inits = inits_per_gen(d, ngen)
    tau_txt = '  '.join(f'g{i+1}:{t:.0f}' if t is not None else f'g{i+1}:—' for i,t in enumerate(taus))
    valid = [t for t in taus if t is not None]
    tau_mean = np.mean(valid) if valid else float('nan')
    init_txt = '  '.join(f'g{i+1}:{c}' for i,c in enumerate(inits))
    init_total = sum(inits)
    init_rate = init_total / max(1, len([t for t in taus if t is not None]))
    footer = (f' tau per gen (min):  {tau_txt}     mean {tau_mean:.1f}\n'
              f'inits per gen:    {init_txt}     total {init_total}/{len(valid)} = {init_rate:.2f}/gen')
    fig.text(0.02, 0.01, footer, family='monospace', fontsize=7, va='bottom')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    pdf.savefig(fig); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/plots/basal_10gen_v2ecoli_main_vs_vecoli.pdf')
    ap.add_argument('--ngen', type=int, default=10)
    ap.add_argument('--v2-main-root', default=V2_MAIN,
                    help='parquet history root for v2ecoli main run (page 1)')
    ap.add_argument('--vecoli-root', default=VECOLI,
                    help='parquet history root for vEcoli run (page 2); pass empty string to skip')
    ap.add_argument('--v2-title', default=DEFAULT_TITLE_V2)
    ap.add_argument('--ve-title', default=DEFAULT_TITLE_VE)
    args = ap.parse_args()

    v2 = load(args.v2_main_root, 'global_time', accumulate_time=True)

    with PdfPages(args.out) as pdf:
        plot_page(pdf, v2, args.v2_title, args.ngen, 'global_time')
        if args.vecoli_root:
            ve = load(args.vecoli_root, 'time', accumulate_time=False)
            plot_page(pdf, ve, args.ve_title, args.ngen, 'time')
    print(f'wrote {args.out}')

if __name__ == '__main__':
    main()
