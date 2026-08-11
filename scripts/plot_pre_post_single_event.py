"""Side-by-side gen 6 (POST-init) vs gen 7 (PRE-init) figure.

V=1.5 hysteresis baseline (K_d_min=1 nM, seed=2). Both gens fire exactly one
sustained 8/8 cooperative event — gen 6's is POST-init (fork-release driven)
and gen 7's is PRE-init (drives init).

Question: does the PRE/POST distinction matter biologically, and does it
argue for replacing the mass-clock initiation gate with a cooperative-state
gate?
"""
import os
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)

ROOT_PARQUET = "out/dnaa5_coop_hyst_v1.5e-3_apoATP_kinetic_linear_s06_succ_gen3_seed2_parquet/dnaa5_coop_hyst_v1.5e-3_apoATP_kinetic_linear_s06_succ_gen3_seed2/history"

GENS = [
    {"gen": 6, "agent": "000000", "label": "Gen 6 — 1 POST-init event"},
    {"gen": 7, "agent": "0000000", "label": "Gen 7 — 1 PRE-init event"},
]


POOL_ORIC_LOW = 2
FORM_ATP = 1


def fetch(g, agent):
    """Pull per-domain oriC-low occupancy + per-domain ≥60s 8/8 events.

    Lumped oriC-low across multiple chromosomes mis-shades post-init (e.g.
    one daughter at 4/8 + other at 4/8 sums to 8 but neither is saturated).
    Detect events per-chromosome from the dnaa_box_* arrays instead.
    """
    from collections import defaultdict
    con = duckdb.connect()
    con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{ROOT_PARQUET}/**/*.pq', hive_partitioning=true)")
    rows = con.sql(f"""
    SELECT global_time,
           bulk__count[10823] AS atp,
           listeners__replication_data__dnaa_box_pool_label AS pool,
           listeners__replication_data__dnaa_box_domain_index AS dom,
           listeners__replication_data__dnaa_box_bound_form AS form,
           listeners__replication_data__number_of_oric AS noric
    FROM h WHERE generation={g} AND agent_id='{agent}'
    ORDER BY global_time
    """).fetchall()
    con.close()
    if not rows:
        return None
    t0 = rows[0][0]
    t_min = []
    atp = []
    per_dom_trace = defaultdict(list)
    noric = []
    domain_states = defaultdict(lambda: {'in_ep': False, 'ep_start': None, 'eps': []})

    # Track when each domain first appears (for parent/daughter labeling)
    domain_first_tick = {}
    for r in rows:
        t, a, pool, dom, form, n = r
        tm = (t - t0) / 60
        t_min.append(tm)
        atp.append(a)
        noric.append(n)
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
                st['eps'].append((st['ep_start'], tm))
                st['in_ep'] = False

    for d, st in domain_states.items():
        if st['in_ep']:
            st['eps'].append((st['ep_start'], t_min[-1]))

    init_idx = next((i for i in range(1, len(noric)) if noric[i] > noric[i-1]), None)
    init_t = t_min[init_idx] if init_idx else None

    # Merge per-domain episodes within 5s, then keep ≥60s
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

    return t_min, atp, per_dom_trace, domain_first_tick, noric, init_t, all_sustained


fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col",
                          gridspec_kw={"height_ratios": [1, 1]})

PALETTE = ["#15803d", "#7c3aed", "#dc2626", "#0891b2", "#a16207"]

for col, info in enumerate(GENS):
    data = fetch(info["gen"], info["agent"])
    if data is None:
        continue
    t_min, atp, per_dom_trace, domain_first_tick, noric, init_t, sustained = data

    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    for s, e, d in sustained:
        for ax in (ax_top, ax_bot):
            ax.axvspan(s, e, color="#fde68a", alpha=0.55, zorder=0,
                       label="per-chrom 8/8 ≥60 s" if ax is ax_top else None)

    if init_t is not None:
        for ax in (ax_top, ax_bot):
            ax.axvline(init_t, color="#1e40af", lw=1.5, ls="--",
                       label=f"init @ {init_t:.1f} min" if ax is ax_top else None)

    ax_top.plot(t_min, atp, color="#dc2626", lw=1.0)
    ax_top.set_ylabel("bulk DnaA-ATP\n(molecules)", fontsize=10)
    ax_top.set_title(info["label"], fontsize=12, fontweight="bold")
    ax_top.set_ylim(0, max(70, max(atp) * 1.1))
    ax_top.grid(False)
    ax_top.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    # Per-chromosome oric_low traces.
    # First-appearing domain = parent; later-appearing = daughters.
    dom_sorted = sorted(per_dom_trace.keys(), key=lambda d: domain_first_tick[d])
    n_daughters = max(0, len(dom_sorted) - 1)
    daughter_idx = 0
    for i, d in enumerate(dom_sorted):
        tr_pairs = per_dom_trace[d]
        tax = [p[0] for p in tr_pairs]
        vals = [p[1] for p in tr_pairs]
        if i == 0:
            color = "#15803d"  # parent in green
            label = "parent chromosome"
        else:
            daughter_idx += 1
            color = PALETTE[daughter_idx % len(PALETTE)]
            label = f"daughter {daughter_idx}" if n_daughters > 1 else "daughter"
        ax_bot.plot(tax, vals, color=color, lw=1.0, label=label)
    ax_bot.axhline(8, color="#475569", lw=0.7, ls=":", alpha=0.5)
    ax_bot.set_ylabel("per-chrom oriC low\nbound DnaA-ATP (/ 8)", fontsize=10)
    ax_bot.set_xlabel("time within generation (min)", fontsize=10)
    ax_bot.set_ylim(0, 9)
    ax_bot.set_yticks([0, 2, 4, 6, 8])
    ax_bot.grid(False)
    ax_bot.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    if sustained:
        s, e, d = sustained[0]
        tag = "PRE-init" if init_t is not None and s < init_t else "POST-init"
        midp = (s + e) / 2
        ax_bot.annotate(tag, xy=(midp, 8.5), ha="center", fontsize=11,
                        fontweight="bold",
                        color="#92400e" if tag == "POST-init" else "#1e3a8a")

fig.suptitle(
    "Both generations fire exactly 1 sustained cooperative event.\n"
    "Should pre-init vs post-init matter? Should the mass-clock gate be replaced?",
    fontsize=12, y=0.998
)
fig.text(0.5, 0.01,
         "V=1.5e-3 + linear autoreg s=0.6 + F-05 + per-chrom hysteresis cooperativity (K_d_max=100 nM, K_d_min=1 nM, entry 6/8, exit 3/8)   ·   seed=2",
         ha="center", fontsize=8.5, color="#475569")

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
out_pdf = "reports/coop_pre_post_single_event.pdf"
out_png = "reports/coop_pre_post_single_event.png"
os.makedirs("reports", exist_ok=True)
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
