"""Check whether v8 gen 1 flicker is single-domain or multi-domain alternation.

If a single domain flickers 0→8→0, we'd see the same domain index with
n_bound oscillating. If two domains alternate, n_bound across both stays
8 but distributes between them.
"""
import duckdb
from collections import defaultdict

V8 = "out/dnaa5_adaptive_v8_carry_succ_gen3_seed2_parquet/dnaa5_adaptive_v8_carry_succ_gen3_seed2/history"

con = duckdb.connect()
con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{V8}/**/*.pq', hive_partitioning=true)")
rows = con.sql("""
    SELECT global_time,
           listeners__replication_data__dnaa_box_pool_label AS pool,
           listeners__replication_data__dnaa_box_domain_index AS dom,
           listeners__replication_data__dnaa_box_bound_form AS form
    FROM h WHERE generation=1 AND agent_id='0'
    ORDER BY global_time
    LIMIT 30
""").fetchall()

t0 = rows[0][0]
print(f"{'tick':>4} {'time':>5} | per-domain (oric_low) n_bound_atp")
print("-" * 80)
for i, r in enumerate(rows):
    t, pool, dom, form = r
    per_dom = defaultdict(int)
    for p, d, f in zip(pool, dom, form):
        if p == 2 and f == 1:  # oric_low + bound_atp
            per_dom[d] += 1
    out = ", ".join(f"d{d}={n}" for d, n in sorted(per_dom.items()))
    print(f"{i:>4} {t-t0:>5.0f} | {out if out else '(all 0)'}")
