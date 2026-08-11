"""5-panel trajectory plot: bulk DnaA-ATP nM, total DnaA, ATP fraction, oriC, oriC_low occupancy."""
import argparse
import glob
import os
import duckdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)

POOL_ORIC_LOW = 2
FORM_ATP = 1

CELL_DENSITY_GPL = 1100.0
N_AVOGADRO = 6.02214076e23


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--lineage-seed", type=int, default=0)
    ap.add_argument("--gens", type=int, required=True)
    ap.add_argument("--start-gen", type=int, default=1,
                    help="first generation number to plot (default 1)")
    ap.add_argument("--title-extra", default="")
    ap.add_argument("--bulk-gate-nM", type=float, default=None,
                    help="optional bulk DnaA-ATP gate threshold to overlay (nM)")
    ap.add_argument("--m-star-fg", type=float, default=None,
                    help="optional M*/oriC reference (fg) to overlay on cell mass panel")
    ap.add_argument("--init-mass-ref-fg", type=float, default=None,
                    help="optional cell-mass-at-initiation reference (fg); e.g. 2×M* for 2-oriC init")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gen_range = range(args.start_gen, args.start_gen + args.gens)
    agent_for_gen = {g: "0" * g for g in gen_range}

    con = duckdb.connect()
    con.sql(
        f"CREATE VIEW h AS SELECT * FROM read_parquet('{args.exp_root}/history/**/*.pq', hive_partitioning=true)"
    )

    from collections import defaultdict
    t_min, total_dnaa, total_dnaa_nM, atp_frac, bulk_nM, noric, olA, cell_mass = [], [], [], [], [], [], [], []
    # Per-chromosome trace: keyed by (gen, dom_id) so domain_index values
    # that get reused across gens stay separate. Roles ("parent" / "daughter1"
    # / "daughter2") are assigned per-gen by first-appearance order.
    dom_traces = defaultdict(list)        # (gen, dom_id) -> [(t_min, n_bound_atp), ...]
    dom_first_tick = {}                   # (gen, dom_id) -> first cumulative t_min
    dom_gen = {}                          # (gen, dom_id) -> gen (redundant but explicit)
    gen_boundaries = []
    cumulative_t = 0.0

    gen_taus = {}
    for g in gen_range:
        aid = agent_for_gen[g]
        rows = con.sql(
            f"""
            SELECT global_time,
                   bulk__count[10823] AS bulk_atp,
                   bulk__count[11115] AS bulk_adp,
                   bulk__count[11566] AS bulk_apo,
                   listeners__replication_data__chromosomal_high_bound_atp AS chrA,
                   listeners__replication_data__chromosomal_high_bound_adp AS chrD,
                   listeners__replication_data__oriC_high_bound_atp AS ohA,
                   listeners__replication_data__oriC_high_bound_adp AS ohD,
                   listeners__replication_data__oriC_low_bound_atp AS olA,
                   listeners__replication_data__promoter_high_bound_atp AS prA,
                   listeners__replication_data__promoter_high_bound_adp AS prD,
                   listeners__mass__cell_mass AS cm,
                   listeners__replication_data__number_of_oric AS noric,
                   listeners__replication_data__dnaa_box_pool_label AS pool,
                   listeners__replication_data__dnaa_box_domain_index AS dom,
                   listeners__replication_data__dnaa_box_bound_form AS form
            FROM h WHERE generation={g} AND agent_id='{aid}' AND lineage_seed={args.lineage_seed}
            ORDER BY global_time
        """
        ).fetchall()
        if not rows:
            continue
        t0_gen = rows[0][0]
        for r in rows:
            t = r[0]
            bulk_atp = r[1]
            bulk_adp = r[2]
            bulk_apo = r[3]
            chrA = r[4]; chrD = r[5]; ohA = r[6]; ohD = r[7]
            olA_v = r[8]
            prA = r[9]; prD = r[10]
            cm = r[11]
            noric_v = r[12]
            pool = r[13]; dom = r[14]; form = r[15]

            total_atp = bulk_atp + chrA + ohA + olA_v + prA
            total_adp = bulk_adp + chrD + ohD + prD
            total = total_atp + total_adp + bulk_apo

            V_L = cm * 1e-15 / CELL_DENSITY_GPL
            bulk_nM_v = bulk_atp / (V_L * N_AVOGADRO) * 1e9 if V_L > 0 else 0.0
            total_nM_v = total / (V_L * N_AVOGADRO) * 1e9 if V_L > 0 else 0.0
            frac = total_atp / total if total > 0 else 0.0

            tm = (t - t0_gen) / 60.0 + cumulative_t
            t_min.append(tm)
            total_dnaa.append(total)
            total_dnaa_nM.append(total_nM_v)
            atp_frac.append(frac)
            bulk_nM.append(bulk_nM_v)
            noric.append(noric_v)
            olA.append(olA_v)
            cell_mass.append(cm)

            # Per-chromosome occupancy at oriC_low (pool 2), ATP-bound (form 1)
            per_dom = defaultdict(int)
            for p, d, f in zip(pool, dom, form):
                if p == POOL_ORIC_LOW:
                    if f == FORM_ATP:
                        per_dom[d] += 1
                    elif d not in per_dom:
                        per_dom[d] = 0  # ensure presence
            for d, n in per_dom.items():
                key = (g, d)
                if key not in dom_first_tick:
                    dom_first_tick[key] = tm
                    dom_gen[key] = g
                dom_traces[key].append((tm, n))
        gen_start = 0.0 if not gen_boundaries else gen_boundaries[-1]
        gen_taus[g] = t_min[-1] - gen_start
        cumulative_t = t_min[-1]
        gen_boundaries.append(cumulative_t)

    # Compute sat_secs per (gen, dom): consecutive seconds cluster has been >= 8
    # sat_ready per tick = any domain has sat_secs >= 10 (sat-init would fire)
    SAT_THRESHOLD = 8
    SAT_SUSTAINED_S = 1.0
    dom_sat_secs = {}          # (g, dom_id) -> [(t_min, sat_secs_here), ...]
    for key, trace in dom_traces.items():
        s = 0.0
        t_prev = trace[0][0]
        traj = []
        for t_here, n in trace:
            dt_s = max(0.0, (t_here - t_prev) * 60.0)
            if n >= SAT_THRESHOLD:
                s += dt_s
            else:
                s = 0.0
            traj.append((t_here, s))
            t_prev = t_here
        dom_sat_secs[key] = traj

    # Aggregate to a per-tick maximum across all domains (any-oriC-armed indicator)
    t_arr = sorted(set(t_min))
    max_sat_by_t = {t: 0.0 for t in t_arr}
    for traj in dom_sat_secs.values():
        for t_here, s in traj:
            if s > max_sat_by_t.get(t_here, 0.0):
                max_sat_by_t[t_here] = s
    sat_secs_agg = [max_sat_by_t[t] for t in t_min]

    fig, axes = plt.subplots(8, 1, figsize=(14, 19), sharex=True)

    def vlines(ax):
        for b in gen_boundaries[:-1]:
            ax.axvline(b, color="#94a3b8", lw=1.0, ls="--", alpha=0.6, zorder=0)

    # Detect initiation events from noric step-ups (used on panels 1 and 6)
    init_pts = []
    init_idx = []
    for i in range(1, len(noric)):
        if noric[i] > noric[i-1]:
            init_pts.append((t_min[i], cell_mass[i]))
            init_idx.append(i)

    # 1. bulk DnaA-ATP in nM
    ax = axes[0]
    ax.plot(t_min, bulk_nM, color="#dc2626", lw=0.9)
    if args.bulk_gate_nM is not None:
        ax.axhline(args.bulk_gate_nM, color="#475569", lw=0.7, ls=":", alpha=0.7,
                   label=f"bulk gate threshold = {args.bulk_gate_nM:g} nM")
    vlines(ax)
    # Mark initiation events
    if init_pts:
        init_ts = [p[0] for p in init_pts]
        for it in init_ts:
            ax.axvline(it, color="#b91c1c", lw=0.6, ls="-", alpha=0.5, zorder=1)
        init_bulk = [bulk_nM[t_min.index(it)] if it in t_min else 0 for it in init_ts]
        ax.scatter(init_ts, init_bulk, s=35, marker="v", edgecolor="#7f1d1d",
                   facecolor="#fecaca", zorder=5,
                   label=f"initiation events (n={len(init_ts)})")
    ax.set_ylabel("bulk DnaA-ATP\n[nM]")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    # 2. DnaA all forms (count)
    ax = axes[1]
    ax.axhspan(300, 800, color="#94a3b8", alpha=0.18, zorder=0, label="target [300, 800]")
    ax.plot(t_min, total_dnaa, color="black", lw=1.0)
    vlines(ax)
    ax.set_ylabel("total DnaA\n(count)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # 3. Total DnaA concentration (nM)
    ax = axes[2]
    ax.plot(t_min, total_dnaa_nM, color="#7c3aed", lw=1.0)
    vlines(ax)
    ax.set_ylabel("total DnaA\n[nM]")

    # 4. ATP fraction
    ax = axes[3]
    ax.axhspan(0.2, 0.5, color="#86efac", alpha=0.3, zorder=0, label="target [0.2, 0.5]")
    ax.plot(t_min, atp_frac, color="#16a34a", lw=1.0)
    vlines(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("DnaA-ATP / total")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # 5. oriC count
    ax = axes[4]
    ax.step(t_min, noric, color="#7c3aed", lw=1.4, where="post")
    vlines(ax)
    ax.set_ylim(-0.3, max(int(max(noric)) + 1, 4))
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel("oriC count")

    # 6. Cell mass
    ax = axes[5]
    ax.plot(t_min, cell_mass, color="#0d9488", lw=1.0)
    vlines(ax)
    # Mark critical initiation mass at each init event (noric step up)
    init_pts = []
    for i in range(1, len(noric)):
        if noric[i] > noric[i-1]:
            init_pts.append((t_min[i], cell_mass[i]))
    if init_pts:
        xs, ys = zip(*init_pts)
        ax.scatter(xs, ys, s=60, marker="o", edgecolor="#b91c1c",
                   facecolor="#fecaca", zorder=5,
                   label=f"initiation mass (mean={sum(ys)/len(ys):.0f} fg)")
    if args.m_star_fg is not None:
        ax.axhline(args.m_star_fg, color="#b91c1c", lw=0.9, ls=":", alpha=0.7,
                   label=f"critical initiation mass = {args.m_star_fg:g} fg")
    if args.init_mass_ref_fg is not None:
        ax.axhline(args.init_mass_ref_fg, color="#b91c1c", lw=0.9, ls="--", alpha=0.7,
                   label=f"expected initiation mass = {args.init_mass_ref_fg:g} fg")
    if init_pts or args.m_star_fg is not None or args.init_mass_ref_fg is not None:
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_ylabel("cell mass\n(fg)")

    # 7. Per-chromosome oriC_low occupancy (parent green, daughters purple+red)
    ax = axes[6]
    ROLE_COLOR = {"parent": "#15803d",
                  "daughter1": "#7c3aed",
                  "daughter2": "#dc2626"}
    legend_added = set()
    for g in gen_range:
        gen_keys = [k for k in dom_traces if k[0] == g]
        if not gen_keys:
            continue
        # First-appearing domain in this gen = parent; later ones = daughters
        gen_keys.sort(key=lambda k: dom_first_tick[k])
        for i, key in enumerate(gen_keys):
            if i == 0:
                role = "parent"
            elif i == 1:
                role = "daughter1"
            else:
                role = "daughter2"
            tax = [p[0] for p in dom_traces[key]]
            vals = [p[1] for p in dom_traces[key]]
            label = role if role not in legend_added else None
            ax.plot(tax, vals, color=ROLE_COLOR[role], lw=0.8, alpha=0.9,
                    label=label)
            legend_added.add(role)
    ax.axhline(8, color="#475569", lw=0.6, ls=":", alpha=0.5)
    vlines(ax)
    ax.set_ylabel("per-chrom oriC low\nbound DnaA-ATP (/ 8)")
    ax.set_ylim(0, 9)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # 8. Sat-init "sustained ≥ 8/oriC" counter (any domain)
    ax = axes[7]
    ax.fill_between(t_min, sat_secs_agg, step="post", color="#f59e0b", alpha=0.4)
    ax.plot(t_min, sat_secs_agg, color="#b45309", lw=0.9, drawstyle="steps-post")
    ax.axhline(SAT_SUSTAINED_S, color="#dc2626", lw=0.8, ls="--", alpha=0.8,
               label=f"fires at {SAT_SUSTAINED_S:.0f}s sustained")
    vlines(ax)
    ax.set_ylabel(f"max sat_secs\n(cluster ≥ {SAT_THRESHOLD}/oriC)")
    ax.set_xlabel("Time (min)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    for ax in axes:
        ax.tick_params(labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle(args.title_extra, fontsize=11, y=0.998)
    if gen_taus:
        gs_sorted = sorted(gen_taus)
        tau_str = "  ".join(f"g{g}:{gen_taus[g]:.0f}" for g in gs_sorted)
        mean_tau = sum(gen_taus.values()) / len(gen_taus)
        footer = f"tau per gen (min):  {tau_str}     mean {mean_tau:.1f}"
        fig.text(0.01, 0.005, footer, family="monospace", fontsize=8, va="bottom")
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
