"""Per-gen cycle breakdown: t_init, t_div, mass at init, growth rate.

Separates τ shortening into: earlier initiation (mass-clock fires earlier) vs.
faster post-init progression (replication+D-period).

For each gen, report:
  t_init_min : minutes from gen start to first noric increase (2→4)
  t_div_min  : gen total length (division)
  t_post_min : t_div - t_init  (time from init to division)
  mass_at_init_fg : dry mass at t_init (lower → mass-clock fired earlier)
  μ_init   : dry-mass doubling rate before init (1/min)
  μ_post   : dry-mass doubling rate after init

Usage:
    python scripts/cycle_breakdown.py <parquet_root>
"""
import argparse
import duckdb
import math


def per_gen(parquet_root):
    history = f"{parquet_root}/history"
    con = duckdb.connect()
    con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{history}/**/*.pq', hive_partitioning=true)")

    print(f"{'gen':>3} {'t_init':>7} {'t_div':>7} {'t_post':>7} "
          f"{'mass@init':>9} {'μ_pre':>7} {'μ_post':>7}")
    print("-" * 60)

    for g in range(1, 13):
        rows = con.sql(f"""
            SELECT global_time,
                   listeners__replication_data__number_of_oric AS noric,
                   listeners__mass__dry_mass AS dm
            FROM h WHERE generation={g}
            ORDER BY global_time
        """).fetchall()
        if not rows:
            continue
        t0 = rows[0][0]
        n0 = rows[0][1]
        # First noric increase = init
        init_t = None
        init_mass = None
        for i in range(1, len(rows)):
            if rows[i][1] > rows[i-1][1]:
                init_t = (rows[i][0] - t0) / 60
                init_mass = rows[i-1][2]
                break
        div_t = (rows[-1][0] - t0) / 60
        m_start = rows[0][2]
        m_end = rows[-1][2]
        if init_t is not None and init_mass is not None and m_start > 0:
            mu_pre = math.log(init_mass / m_start) / init_t if init_t > 0 else 0
            mu_post = math.log(m_end / init_mass) / (div_t - init_t) if (div_t - init_t) > 0 else 0
            post_t = div_t - init_t
            print(f"{g:>3} {init_t:>7.1f} {div_t:>7.1f} {post_t:>7.1f} "
                  f"{init_mass:>9.1f} {mu_pre:>7.4f} {mu_post:>7.4f}")
        else:
            print(f"{g:>3} {'—':>7} {div_t:>7.1f} {'—':>7} "
                  f"{'—':>9} {'—':>7} {'—':>7}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_root")
    args = ap.parse_args()
    per_gen(args.parquet_root)
