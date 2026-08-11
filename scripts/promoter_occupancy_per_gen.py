"""Per-gen mean DnaA promoter (P_DnaA) occupancy across runs.

The dnaA promoter has 2 high-affinity DnaA boxes (pool_label=3, K_d=1 nM).
Linear autoregulation s=0.6 scales transcription as V × (1 - 0.6 × P_DnaA),
where P_DnaA = (bound_atp + bound_adp) / 2.

Higher promoter occupancy → stronger repression of dnaA transcription.
v8 carry-forward sequesters DnaA-ATP at oriC → suspect lower P_DnaA than
the hysteresis baseline.

Usage:
    python scripts/promoter_occupancy_per_gen.py <parquet_root>
"""
import argparse
import duckdb


def per_gen(parquet_root):
    history = f"{parquet_root}/history"
    con = duckdb.connect()
    con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{history}/**/*.pq', hive_partitioning=true)")

    print(f"{'gen':>3} {'P_DnaA':>7} {'free':>5} {'ATP':>5} {'ADP':>5} {'1-s·P':>7}")
    print("-" * 40)

    for g in range(1, 13):
        rows = con.sql(f"""
          SELECT AVG(listeners__replication_data__promoter_high_free) AS f,
                 AVG(listeners__replication_data__promoter_high_bound_atp) AS a,
                 AVG(listeners__replication_data__promoter_high_bound_adp) AS d,
                 COUNT(*) AS n
          FROM h WHERE generation={g}
        """).fetchone()
        if rows is None or rows[3] == 0:
            continue
        f, a, d, _ = rows
        total = f + a + d
        P = (a + d) / total if total > 0 else 0
        repression_factor = 1 - 0.6 * P
        print(f"{g:>3} {P:>7.3f} {f:>5.2f} {a:>5.2f} {d:>5.2f} {repression_factor:>7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_root")
    args = ap.parse_args()
    per_gen(args.parquet_root)
