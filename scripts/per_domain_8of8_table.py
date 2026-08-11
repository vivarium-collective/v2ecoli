"""Per-domain 8/8 saturation table + DnaA budget audit.

For each generation in a parquet output of run_condition_multigen_parquet.py,
emit:
  - τ, peak #oriC
  - per-domain 8/8 saturation episodes (count, avg duration, total time)
  - mean DnaA-ATP / DnaA-ADP / apo across the cycle (bulk + bound)
  - mean DnaA-ATP fraction = (total DnaA-ATP) / (total DnaA)

Usage:
  python scripts/per_domain_8of8_table.py <parquet_root>
"""
import argparse, sys
import duckdb
from collections import defaultdict

DNAA_ATP_BULK_ID = "MONOMER0-160[c]"
DNAA_ADP_BULK_ID = "MONOMER0-4565[c]"
DNAA_APO_BULK_ID = "PD03831[c]"

POOL_ORIC_LOW = 2
FORM_FREE, FORM_ATP, FORM_ADP = 0, 1, 2


def _bulk_index(con, name):
    sample = con.sql("SELECT bulk__id FROM h LIMIT 1").fetchone()[0]
    for i, n in enumerate(sample):
        if n == name:
            return i + 1  # duckdb list indexing is 1-based
    raise KeyError(name)


def _merge_episodes(eps, merge_window_s):
    """Merge episodes whose gap to the next event is < merge_window_s.

    Each ep is (start_time, duration). Returns merged list of
    (start, duration) where consecutive episodes within merge_window
    are bridged into one.
    """
    if not eps or merge_window_s <= 0:
        return eps
    merged = [list(eps[0])]
    for start, dur in eps[1:]:
        prev_start, prev_dur = merged[-1]
        prev_end = prev_start + prev_dur
        gap = start - prev_end
        if gap < merge_window_s:
            merged[-1][1] = (start + dur) - prev_start
        else:
            merged.append([start, dur])
    return [tuple(m) for m in merged]


def _filter_min_duration(eps, min_dur_s):
    """Drop episodes shorter than min_dur_s seconds entirely."""
    if min_dur_s <= 0:
        return eps
    return [(s, d) for (s, d) in eps if d >= min_dur_s]


def analyse(parquet_root, merge_window_s=0, min_dur_s=0):
    history = f"{parquet_root}/history"
    con = duckdb.connect()
    con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{history}/**/*.pq', hive_partitioning=true)")
    atp_idx = _bulk_index(con, DNAA_ATP_BULK_ID)
    adp_idx = _bulk_index(con, DNAA_ADP_BULK_ID)
    apo_idx = _bulk_index(con, DNAA_APO_BULK_ID)

    main_agents = ['0','00','000','0000','00000','000000','0000000',
                   '00000000','000000000','0000000000','00000000000','000000000000']

    print(f"{'gen':>3} {'τ_min':>5} {'noric':>5} {'8/8 ev':>6} {'per_domain':>30} "
          f"{'avg_d(s)':>8} {'tot_min':>7} {'A-ATP':>5} {'A-ADP':>5} {'A-apo':>5} "
          f"{'totA':>5} {'ATPfr':>5}")
    print("-" * 110)
    for g in range(1, 13):
        ag = main_agents[g-1]
        rows = con.sql(f"""
          SELECT global_time,
                 listeners__replication_data__dnaa_box_pool_label AS pool,
                 listeners__replication_data__dnaa_box_domain_index AS dom,
                 listeners__replication_data__dnaa_box_bound_form AS form,
                 listeners__replication_data__number_of_oric AS noric,
                 bulk__count[{atp_idx}] AS bATP,
                 bulk__count[{adp_idx}] AS bADP,
                 bulk__count[{apo_idx}] AS bApo,
                 listeners__replication_data__chromosomal_high_bound_atp AS chATP,
                 listeners__replication_data__chromosomal_high_bound_adp AS chADP,
                 listeners__replication_data__oriC_high_bound_atp AS ohATP,
                 listeners__replication_data__oriC_high_bound_adp AS ohADP,
                 listeners__replication_data__oriC_low_bound_atp AS olATP,
                 listeners__replication_data__promoter_high_bound_atp AS prATP,
                 listeners__replication_data__promoter_high_bound_adp AS prADP
          FROM h WHERE generation={g} AND agent_id='{ag}' ORDER BY global_time
        """).fetchall()
        if not rows:
            continue

        # Per-domain state tracking
        domain_states = defaultdict(lambda: {'in_ep': False, 'ep_start': None, 'eps': []})
        atp_acc = adp_acc = apo_acc = total_acc = atpfr_acc = 0.0
        n_acc = 0
        times = []
        noric_pk = 0
        for r in rows:
            (t, pool, dom, form, noric,
             bATP, bADP, bApo,
             chATP, chADP, ohATP, ohADP, olATP, prATP, prADP) = r
            times.append(t)
            if noric > noric_pk:
                noric_pk = noric

            # Per-domain oric_low ATP counting
            per_dom_atp = defaultdict(int); per_dom_tot = defaultdict(int)
            for p, d, f in zip(pool, dom, form):
                if p == POOL_ORIC_LOW:
                    per_dom_tot[d] += 1
                    if f == FORM_ATP:
                        per_dom_atp[d] += 1
            for d in per_dom_tot:
                sat = (per_dom_atp[d] == 8 and per_dom_tot[d] == 8)
                st = domain_states[d]
                if sat and not st['in_ep']:
                    st['in_ep'] = True; st['ep_start'] = t
                elif not sat and st['in_ep']:
                    st['eps'].append((st['ep_start'], t - st['ep_start']))
                    st['in_ep'] = False

            # DnaA budget per tick
            atp_total = bATP + chATP + ohATP + olATP + prATP
            adp_total = bADP + chADP + ohADP + prADP
            tot = atp_total + adp_total + bApo
            atp_acc += atp_total
            adp_acc += adp_total
            apo_acc += bApo
            total_acc += tot
            atpfr_acc += atp_total / tot if tot > 0 else 0
            n_acc += 1

        for d, st in domain_states.items():
            if st['in_ep']:
                st['eps'].append((st['ep_start'], times[-1] - st['ep_start']))
            # Apply merge window: bridge events separated by < merge_window_s
            st['eps'] = _merge_episodes(st['eps'], merge_window_s)
            # Then filter out short episodes (didn't actually sustain)
            st['eps'] = _filter_min_duration(st['eps'], min_dur_s)

        tau = (times[-1] - times[0]) / 60
        total_eps = sum(len(s['eps']) for s in domain_states.values())
        per_dom_str = ','.join(f"d{d}:{len(s['eps'])}"
                                for d, s in sorted(domain_states.items()))[:30]
        all_durs = [d for s in domain_states.values() for _, d in s['eps']]
        avg_dur = sum(all_durs) / len(all_durs) if all_durs else 0
        tot_sat = sum(all_durs) / 60

        mean_atp = atp_acc / n_acc
        mean_adp = adp_acc / n_acc
        mean_apo = apo_acc / n_acc
        mean_tot = total_acc / n_acc
        mean_fr = atpfr_acc / n_acc

        print(f"{g:>3} {tau:>5.1f} {noric_pk:>5} {total_eps:>6} {per_dom_str:>30} "
              f"{avg_dur:>8.1f} {tot_sat:>7.1f} "
              f"{mean_atp:>5.0f} {mean_adp:>5.0f} {mean_apo:>5.0f} "
              f"{mean_tot:>5.0f} {mean_fr:>5.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_root")
    ap.add_argument("--merge-window-s", type=float, default=0.0,
                    help="Merge events separated by < this many seconds. "
                         "Useful to suppress single-molecule hydrolysis "
                         "flicker. Default 0 (no merge).")
    ap.add_argument("--min-duration-s", type=float, default=0.0,
                    help="Drop episodes shorter than this many seconds "
                         "(applied AFTER merge). Default 0 (no filter).")
    args = ap.parse_args()
    print(f"# merge_window = {args.merge_window_s:.1f} s, "
          f"min_duration = {args.min_duration_s:.1f} s")
    analyse(args.parquet_root, args.merge_window_s, args.min_duration_s)
