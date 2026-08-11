"""Trace DnaA-ATP partition within a cycle.

For a given gen, show how DnaA-ATP is distributed across:
  - free bulk
  - chromosomal_high bound (302 sites at K_d=1 nM, competes with ADP)
  - oriC_high bound (3 sites)
  - oriC_low bound (8 sites per oriC, ATP-only at K_d=100 nM)
  - promoter_high bound (2 sites)
And same for DnaA-ADP (for ATP/ADP competition view).

Plus cell volume (from cell_mass × cell_density).

Helps answer:
  - Where is DnaA-ATP going?
  - How much is "trapped" at chromosomal_high vs available for oriC_low?
  - How does free A_f concentration evolve within the cycle?

Usage:
    python scripts/dnaa_atp_partition_trace.py <parquet_root> --gen N
"""
import argparse
import duckdb


def trace(parquet_root, gen=3):
    history = f"{parquet_root}/history"
    con = duckdb.connect()
    con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{history}/**/*.pq', hive_partitioning=true)")

    agents = {1: '0', 2: '00', 3: '000', 4: '0000', 5: '00000', 6: '000000',
              7: '0000000', 8: '00000000', 9: '000000000', 10: '0000000000',
              11: '00000000000', 12: '000000000000'}
    ag = agents.get(gen)

    rows = con.sql(f"""
        SELECT global_time,
               bulk__count[10823] AS bulk_atp,
               bulk__count[10824] AS bulk_adp,
               listeners__replication_data__chromosomal_high_bound_atp AS chrA,
               listeners__replication_data__chromosomal_high_bound_adp AS chrD,
               listeners__replication_data__oriC_high_bound_atp AS ohA,
               listeners__replication_data__oriC_high_bound_adp AS ohD,
               listeners__replication_data__oriC_low_bound_atp AS olA,
               listeners__replication_data__promoter_high_bound_atp AS prA,
               listeners__replication_data__promoter_high_bound_adp AS prD,
               listeners__mass__cell_mass AS cm,
               listeners__mass__dry_mass AS dm,
               listeners__replication_data__number_of_oric AS noric
        FROM h WHERE generation={gen} AND agent_id='{ag}'
        ORDER BY global_time
        LIMIT 50000
    """).fetchall()
    if not rows:
        print(f"no data for gen {gen}")
        return

    t0 = rows[0][0]
    print(f"# gen {gen}: {len(rows)} ticks, τ={(rows[-1][0]-t0)/60:.1f} min")
    print(f"{'t_min':>6} {'noric':>5} {'cell_mass':>9} {'V_fL':>5} | "
          f"{'bulkA':>5} {'chrA':>5} {'ohA':>4} {'olA':>4} {'prA':>4} | "
          f"{'bulkD':>5} {'chrD':>5} | {'totA':>5} {'totD':>5} {'A_f_nM':>7} {'D_f_nM':>7}")
    print("-" * 130)

    # Sample every 60s (1 min)
    last_t = -100
    for r in rows:
        (t, bA, bD, chrA, chrD, ohA, ohD, olA, prA, prD, cm, dm, noric) = r
        if t - last_t < 60 and t != rows[-1][0]:
            continue
        last_t = t
        tm = (t - t0) / 60
        # Cell volume from cell mass + density (1100 g/L typical)
        # cell_mass is in fg = 1e-15 g. V = mass / density
        V_L = cm * 1e-15 / 1100  # L
        V_fL = V_L * 1e15
        totA = bA + chrA + ohA + olA + prA
        totD = bD + chrD + ohD + prD
        # Free A_f concentration in nM
        A_f_M = bA / (V_L * 6.022e23) if V_L > 0 else 0
        D_f_M = bD / (V_L * 6.022e23) if V_L > 0 else 0
        A_f_nM = A_f_M * 1e9
        D_f_nM = D_f_M * 1e9
        print(f"{tm:>6.1f} {noric:>5} {cm:>9.1f} {V_fL:>5.2f} | "
              f"{bA:>5} {chrA:>5} {ohA:>4} {olA:>4} {prA:>4} | "
              f"{bD:>5} {chrD:>5} | {totA:>5} {totD:>5} "
              f"{A_f_nM:>7.1f} {D_f_nM:>7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_root")
    ap.add_argument("--gen", type=int, default=3)
    args = ap.parse_args()
    trace(args.parquet_root, args.gen)
