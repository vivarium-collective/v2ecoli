"""Bulk ATP[c] trajectory + per-tick ATP request, comparing runs.

Probes whether the dnaa_box equilibrium leaks into metabolism by examining:
  (a) bulk ATP[c] mean per gen — proxy for cell ATP pool
  (b) total atp_requested per tick (sum over processes) — gross consumption
  (c) total atp_requested per molecule of dry mass — request density

If v8 burns more ATP via the DnaA-ATP regeneration cycle, we'd see
either lower bulk ATP[c] OR higher per-mass ATP request density.
If neither shifts, the leak is elsewhere.

Usage:
    python scripts/atp_pool_per_gen.py <parquet_root>
"""
import argparse
import duckdb

ATP_IDX = 804  # ATP[c] index in bulk array


def per_gen(parquet_root):
    history = f"{parquet_root}/history"
    con = duckdb.connect()
    con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{history}/**/*.pq', hive_partitioning=true)")

    print(f"{'gen':>3} {'mean_ATP':>10} {'mean_dm':>8} {'ATP/dm':>8} "
          f"{'sum_req':>10} {'req/dm':>10}")
    print("-" * 60)

    for g in range(1, 13):
        # Aggregate per gen
        rows = con.sql(f"""
            SELECT global_time,
                   bulk__count[{ATP_IDX + 1}] AS atp,
                   listeners__mass__dry_mass AS dm,
                   list_sum(listeners__atp__atp_requested) AS req
            FROM h WHERE generation={g}
            ORDER BY global_time
        """).fetchall()
        if not rows:
            continue
        n = len(rows)
        mean_atp = sum(r[1] for r in rows) / n
        mean_dm = sum(r[2] for r in rows) / n
        mean_req = sum(r[3] for r in rows if r[3] is not None) / n
        atp_per_dm = mean_atp / mean_dm if mean_dm > 0 else 0
        req_per_dm = mean_req / mean_dm if mean_dm > 0 else 0
        print(f"{g:>3} {mean_atp:>10.0f} {mean_dm:>8.1f} {atp_per_dm:>8.0f} "
              f"{mean_req:>10.0f} {req_per_dm:>10.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_root")
    args = ap.parse_args()
    per_gen(args.parquet_root)
