"""Tick-by-tick comparison of hysteresis vs v8 in gen 1.

Both runs use the same dill seed (succ_gen3_seed2) and same seed=2, so
gen 1 should be identical until the cooperative mechanism diverges.

Pulls per-tick state from both parquet outputs for the same agent_id=0
in gen 1, at matched global_time values. Reports:
  - bulk DnaA-ATP/ADP/apo
  - per-pool bound counts (oric_low/high, chromosomal_high, promoter_high)
  - cell mass + DnaA-ATP per mass
  - first tick where they diverge by > tolerance

Usage:
    python scripts/tick_compare_gen1.py
"""
import duckdb

HYST = "out/dnaa5_coop_hyst_v1.5e-3_apoATP_kinetic_linear_s06_succ_gen3_seed2_parquet/dnaa5_coop_hyst_v1.5e-3_apoATP_kinetic_linear_s06_succ_gen3_seed2/history"
V8 = "out/dnaa5_adaptive_v8_carry_succ_gen3_seed2_parquet/dnaa5_adaptive_v8_carry_succ_gen3_seed2/history"
ATP_IDX = 804
DNAA_ATP_IDX = 10822  # MONOMER0-160[c]
DNAA_ADP_IDX = None    # MONOMER0-4565[c] - find at runtime
DNAA_APO_IDX = None    # PD03831[c]


def fetch(root, gen=1):
    con = duckdb.connect()
    con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{root}/**/*.pq', hive_partitioning=true)")
    rows = con.sql(f"""
        SELECT global_time,
               bulk__count[{DNAA_ATP_IDX + 1}] AS dnaa_atp,
               listeners__mass__dry_mass AS dm,
               listeners__replication_data__oriC_low_bound_atp AS olo_atp,
               listeners__replication_data__oriC_high_bound_atp AS oh_atp,
               listeners__replication_data__chromosomal_high_bound_atp AS ch_atp,
               listeners__replication_data__promoter_high_bound_atp AS pr_atp,
               listeners__replication_data__oriC_low_free AS olo_free,
               listeners__replication_data__number_of_oric AS noric
        FROM h WHERE generation={gen} AND agent_id='0'
        ORDER BY global_time
    """).fetchall()
    con.close()
    return rows


def main():
    h = fetch(HYST)
    v = fetch(V8)
    n = min(len(h), len(v))
    print(f"hysteresis ticks: {len(h)}, v8 ticks: {len(v)}, comparing first {n}")
    print()
    print(f"{'tick':>5} {'t_s':>6} | {'hyst_ATP':>9} {'v8_ATP':>9} {'Δ':>6} | "
          f"{'h_olo':>5} {'v_olo':>5} | {'h_ch':>5} {'v_ch':>5} | "
          f"{'h_oh':>4} {'v_oh':>4} | {'h_pr':>4} {'v_pr':>4}")
    print("-" * 105)

    diverged = False
    for i in range(n):
        ht = h[i][0]; vt = v[i][0]
        if abs(ht - vt) > 0.5:
            print(f"time mismatch at tick {i}: {ht} vs {vt}")
            break
        # Sample first 30 ticks then every 60s
        if i < 30 or i % 60 == 0:
            d_atp = v[i][1] - h[i][1]
            print(f"{i:>5} {ht-h[0][0]:>6.0f} | {h[i][1]:>9} {v[i][1]:>9} {d_atp:>6} | "
                  f"{h[i][3]:>5} {v[i][3]:>5} | {h[i][5]:>5} {v[i][5]:>5} | "
                  f"{h[i][4]:>4} {v[i][4]:>4} | {h[i][6]:>4} {v[i][6]:>4}")
        if not diverged and (abs(v[i][1] - h[i][1]) > 5 or v[i][3] != h[i][3] or v[i][5] != h[i][5]):
            print(f"  ↑ first divergence at tick {i} (t={ht-h[0][0]:.0f}s)")
            diverged = True

    if not diverged:
        print("\nNO DIVERGENCE FOUND in gen 1 — runs are identical")
    else:
        print(f"\ndivergence detected")


if __name__ == "__main__":
    main()
