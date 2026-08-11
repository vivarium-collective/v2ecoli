"""Per-chromosome saturation event counter (1 per chromosome per gen).

Biologically: an "event" is the cluster reaching saturation at some point
during the cycle, regardless of subsequent dips. Once oriC has committed to
the high-n basin, brief downward fluctuations don't undo it. The merged
episode count (per_domain_8of8_table.py) over-counts because it treats each
return to 8/8 as a new event.

This counter: per chromosome (domain key) per gen, return 1 if the cluster
ever spent ≥min_duration seconds sustained at n=8, else 0. Multiple visits
to 8 still count as 1 event.

Usage:
    python scripts/per_chrom_saturation_event.py <parquet_root> [--min-duration-s 60]
"""
import argparse
import duckdb
from collections import defaultdict

POOL_ORIC_LOW = 2
FORM_ATP = 1


def analyse(parquet_root, min_duration_s=60.0):
    history = f"{parquet_root}/history"
    con = duckdb.connect()
    con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{history}/**/*.pq', hive_partitioning=true)")

    agents = {1: '0', 2: '00', 3: '000', 4: '0000', 5: '00000', 6: '000000',
              7: '0000000', 8: '00000000', 9: '000000000', 10: '0000000000',
              11: '00000000000', 12: '000000000000'}

    print(f"# min_duration = {min_duration_s} s (single event per chromosome per gen)")
    print(f"{'gen':>3} {'τ_min':>5} {'noric':>5} {'chrom_events':>13}  {'per_dom_did_saturate':>30}")
    print("-" * 78)

    for g in range(1, 13):
        ag = agents.get(g)
        if ag is None:
            continue
        rows = con.sql(f"""
            SELECT global_time,
                   listeners__replication_data__dnaa_box_pool_label AS pool,
                   listeners__replication_data__dnaa_box_domain_index AS dom,
                   listeners__replication_data__dnaa_box_bound_form AS form,
                   listeners__replication_data__number_of_oric AS noric
            FROM h WHERE generation={g} AND agent_id='{ag}'
            ORDER BY global_time
        """).fetchall()
        if not rows:
            continue
        t0 = rows[0][0]
        tau_min = (rows[-1][0] - t0) / 60
        max_noric = max(r[4] for r in rows)
        # Collect raw at-8 episodes per domain
        dom_at_8_start = {}
        dom_episodes = defaultdict(list)
        for r in rows:
            t, pool, dom, form, n = r
            per_dom_atp = defaultdict(int)
            per_dom_tot = defaultdict(int)
            for p, d, f in zip(pool, dom, form):
                if p == POOL_ORIC_LOW:
                    per_dom_tot[d] += 1
                    if f == FORM_ATP:
                        per_dom_atp[d] += 1
            for d in per_dom_tot:
                if per_dom_tot[d] == 8 and per_dom_atp[d] == 8:
                    if d not in dom_at_8_start:
                        dom_at_8_start[d] = t
                else:
                    if d in dom_at_8_start:
                        dom_episodes[d].append((dom_at_8_start[d], t))
                        del dom_at_8_start[d]
        for d, t_start in dom_at_8_start.items():
            dom_episodes[d].append((t_start, rows[-1][0]))

        # Merge episodes within 5s window; longest merged run = "did saturate?"
        MERGE_S = 5.0
        events_per_dom = {}
        for d, eps in dom_episodes.items():
            merged = []
            for s, e in sorted(eps):
                if merged and s - merged[-1][1] < MERGE_S:
                    merged[-1] = (merged[-1][0], e)
                else:
                    merged.append([s, e])
            # Cap at 1 event per chromosome — did it ever sustain ≥min_duration?
            ever_sustained = any((e - s) >= min_duration_s for s, e in merged)
            events_per_dom[d] = 1 if ever_sustained else 0
        total_events = sum(events_per_dom.values())
        per_dom_str = ",".join(f"d{d}:{v}" for d, v in sorted(events_per_dom.items()))
        print(f"{g:>3} {tau_min:>5.1f} {max_noric:>5} {total_events:>13}  {per_dom_str[:30]:>30}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_root")
    ap.add_argument("--min-duration-s", type=float, default=60.0)
    args = ap.parse_args()
    analyse(args.parquet_root, args.min_duration_s)
