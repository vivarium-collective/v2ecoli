"""Steady-state per-chromosome cooperativity — N consecutive generations.

Same panel layout as plot_pre_post_single_event.py, parameterized over runs
and generation ranges so it can be reused for any lineage. Shows per-chromosome
oriC_low occupancy (one trace per chromosome, parent in green and daughters
in distinct colors) plus bulk DnaA-ATP, with sustained ≥60 s 8/8 events
highlighted as yellow bands.
"""
import argparse
import os
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp-root", required=True,
                   help="path up to and including .../<exp-id>/history")
    p.add_argument("--gens", required=True,
                   help="comma-separated generation indices, e.g. 5,6,7,8")
    p.add_argument("--title",
                   default="Steady-state cooperative events — consecutive generations")
    p.add_argument("--footer", default="")
    p.add_argument("--out", required=True, help="output PDF or PNG path")
    return p.parse_args()


ARGS = _parse_args()
ROOT_PARQUET = ARGS.exp_root
_GEN_LIST = [int(g) for g in ARGS.gens.split(",") if g.strip()]
GENS = [
    {"gen": g, "agent": "0" * g, "label": f"Gen {g}"} for g in _GEN_LIST
]

POOL_ORIC_LOW = 2
FORM_ATP = 1


CELL_DENSITY_GPL = 1100.0
N_AVOGADRO = 6.02214076e23


def fetch(g, agent):
    from collections import defaultdict
    con = duckdb.connect()
    con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{ROOT_PARQUET}/**/*.pq', hive_partitioning=true)")
    rows = con.sql(f"""
    SELECT global_time,
           bulk__count[10823] AS atp,
           listeners__replication_data__dnaa_box_pool_label AS pool,
           listeners__replication_data__dnaa_box_domain_index AS dom,
           listeners__replication_data__dnaa_box_bound_form AS form,
           listeners__replication_data__number_of_oric AS noric,
           listeners__monomer_counts[3862] AS free_dnaa,
           bulk__count[11115] AS bulk_adp,
           bulk__count[11566] AS bulk_apo,
           listeners__replication_data__chromosomal_high_bound_atp AS chrA,
           listeners__replication_data__chromosomal_high_bound_adp AS chrD,
           listeners__replication_data__oriC_high_bound_atp AS ohA,
           listeners__replication_data__oriC_high_bound_adp AS ohD,
           listeners__replication_data__oriC_low_bound_atp AS olA,
           listeners__replication_data__promoter_high_bound_atp AS prA,
           listeners__replication_data__promoter_high_bound_adp AS prD,
           listeners__mass__cell_mass AS cm
    FROM h WHERE generation={g} AND agent_id='{agent}'
    ORDER BY global_time
    """).fetchall()
    con.close()
    if not rows:
        return None
    t0 = rows[0][0]
    t_min, atp, noric = [], [], []
    totA, totA_nM, atp_nM = [], [], []
    per_dom_trace = defaultdict(list)
    domain_states = defaultdict(lambda: {'in_ep': False, 'ep_start': None, 'eps': []})
    domain_first_tick = {}
    for r in rows:
        (t, a, pool, dom, form, n, free_dnaa, bulk_adp, bulk_apo,
         chrA, chrD, ohA, ohD, olA, prA, prD, cm) = r
        tm = (t - t0) / 60
        t_min.append(tm); atp.append(a); noric.append(n)
        tot = (free_dnaa + bulk_adp + bulk_apo
               + chrA + chrD + ohA + ohD + olA + prA + prD)
        totA.append(tot)
        V_L = cm * 1e-15 / CELL_DENSITY_GPL
        if V_L > 0:
            totA_nM.append(tot / (V_L * N_AVOGADRO) * 1e9)
            atp_nM.append(a / (V_L * N_AVOGADRO) * 1e9)
        else:
            totA_nM.append(0.0); atp_nM.append(0.0)
        per_dom_atp = defaultdict(int); per_dom_tot = defaultdict(int)
        for p, d, f in zip(pool, dom, form):
            if p == POOL_ORIC_LOW:
                per_dom_tot[d] += 1
                if f == FORM_ATP:
                    per_dom_atp[d] += 1
        for d in per_dom_tot:
            if d not in domain_first_tick:
                domain_first_tick[d] = tm
            per_dom_trace[d].append((tm, per_dom_atp[d]))
            sat = (per_dom_atp[d] == 8 and per_dom_tot[d] == 8)
            st = domain_states[d]
            if sat and not st['in_ep']:
                st['in_ep'] = True; st['ep_start'] = tm
            elif not sat and st['in_ep']:
                st['eps'].append((st['ep_start'], tm)); st['in_ep'] = False
    for d, st in domain_states.items():
        if st['in_ep']:
            st['eps'].append((st['ep_start'], t_min[-1]))

    init_idx = next((i for i in range(1, len(noric)) if noric[i] > noric[i-1]), None)
    init_t = t_min[init_idx] if init_idx else None

    all_sustained = []
    for d, st in domain_states.items():
        eps = sorted(st['eps'])
        merged = []
        for s, e in eps:
            if merged and s - merged[-1][1] < 5/60:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            if (e - s) * 60 >= 60:
                all_sustained.append((s, e, d))
    return (t_min, atp, per_dom_trace, domain_first_tick, noric, init_t,
            all_sustained, totA, totA_nM, atp_nM)


_NCOLS = len(GENS)
_NROWS = 5
fig, axes = plt.subplots(_NROWS, _NCOLS,
                          figsize=(3.2 * _NCOLS, 11.5), sharex="col",
                          squeeze=False,
                          gridspec_kw={"height_ratios": [1] * _NROWS})

PALETTE = ["#15803d", "#7c3aed", "#dc2626", "#0891b2", "#a16207"]

for col, info in enumerate(GENS):
    data = fetch(info["gen"], info["agent"])
    ax_bulk = axes[0, col]
    ax_tot = axes[1, col]
    ax_tot_nM = axes[2, col]
    ax_oric = axes[3, col]
    ax_chrom = axes[4, col]
    if data is None:
        ax_bulk.set_title(f"{info['label']}\n(no data)", fontsize=10)
        continue
    (t_min, atp, per_dom_trace, domain_first_tick, noric, init_t,
     sustained, totA, totA_nM, atp_nM) = data

    all_axes = (ax_bulk, ax_tot, ax_tot_nM, ax_oric, ax_chrom)
    for s, e, d in sustained:
        for ax in all_axes:
            ax.axvspan(s, e, color="#fde68a", alpha=0.55, zorder=0)
    if init_t is not None:
        for ax in all_axes:
            ax.axvline(init_t, color="#1e40af", lw=1.2, ls="--")

    # Row 0: bulk DnaA-ATP (molecules)
    ax_bulk.plot(t_min, atp, color="#dc2626", lw=0.9)
    if col == 0:
        ax_bulk.set_ylabel("bulk DnaA-ATP\n(molecules)", fontsize=9)
    ax_bulk.set_title(info["label"], fontsize=11, fontweight="bold")
    ax_bulk.set_ylim(0, max(70, max(atp) * 1.1))

    # Row 1: total DnaA (count)
    ax_tot.plot(t_min, totA, color="black", lw=0.9)
    if col == 0:
        ax_tot.set_ylabel("total DnaA\n(count)", fontsize=9)

    # Row 2: total DnaA concentration (nM)
    ax_tot_nM.plot(t_min, totA_nM, color="#7c3aed", lw=0.9)
    if col == 0:
        ax_tot_nM.set_ylabel("total DnaA\n[nM]", fontsize=9)

    # Row 3: oriC count
    ax_oric.step(t_min, noric, color="#0891b2", lw=1.4, where="post")
    if col == 0:
        ax_oric.set_ylabel("oriC count", fontsize=9)
    ax_oric.set_ylim(-0.3, max(int(max(noric)) + 1, 4))
    ax_oric.set_yticks([0, 1, 2, 3, 4])

    # Row 4: per-chromosome oriC_low
    dom_sorted = sorted(per_dom_trace.keys(), key=lambda d: domain_first_tick[d])
    daughter_idx = 0
    for i, d in enumerate(dom_sorted):
        tr_pairs = per_dom_trace[d]
        tax = [p[0] for p in tr_pairs]; vals = [p[1] for p in tr_pairs]
        if i == 0:
            color = "#15803d"; label = "parent" if col == 0 else None
        else:
            daughter_idx += 1
            color = PALETTE[daughter_idx % len(PALETTE)]
            label = f"daughter {daughter_idx}" if col == 0 else None
        ax_chrom.plot(tax, vals, color=color, lw=0.9, label=label)
    ax_chrom.axhline(8, color="#475569", lw=0.6, ls=":", alpha=0.5)
    if col == 0:
        ax_chrom.set_ylabel("per-chrom oriC low\nbound DnaA-ATP (/ 8)", fontsize=9)
    ax_chrom.set_xlabel("time within gen (min)", fontsize=9)
    ax_chrom.set_ylim(0, 9)
    ax_chrom.set_yticks([0, 2, 4, 6, 8])
    if col == 0:
        ax_chrom.legend(loc="upper right", fontsize=8, framealpha=0.95)

    for ax in all_axes:
        ax.tick_params(labelsize=8)
        ax.grid(False)

# Shared legend in the top-left axis
axes[0, 0].plot([], [], color="#1e40af", lw=1.2, ls="--", label="init (noric step)")
axes[0, 0].fill_between([], [], color="#fde68a", alpha=0.55, label="per-chrom 8/8 ≥60 s")
axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.95)

fig.suptitle(ARGS.title, fontsize=12, y=0.998)
if ARGS.footer:
    fig.text(0.5, 0.01, ARGS.footer, ha="center", fontsize=8.5, color="#475569")

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
os.makedirs(os.path.dirname(ARGS.out) or ".", exist_ok=True)
plt.savefig(ARGS.out, bbox_inches="tight")
print(f"wrote {ARGS.out}")
