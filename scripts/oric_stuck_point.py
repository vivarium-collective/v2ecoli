"""Per-gen, per-domain oriC low occupancy histogram + 'stuck point'.

For each generation of a parquet output, identify where each chromosome's
oriC low-affinity sites tend to linger before either crossing to full
saturation or sliding back. Stuck point = modal occupancy when the domain
is NOT fully saturated (i.e., excludes n=8 from the mode).

Output columns per row (one per chromosome domain per gen):
  gen, dom, n_ticks, mode (0..7), mean, %0..%8, init_min

Usage:
  python scripts/oric_stuck_point.py <parquet_root>
"""
import argparse
import duckdb
from collections import Counter, defaultdict


POOL_ORIC_LOW = 2
FORM_ATP = 1


def analyse(parquet_root):
    history = f"{parquet_root}/history"
    con = duckdb.connect()
    con.sql(f"CREATE VIEW h AS SELECT * FROM read_parquet('{history}/**/*.pq', hive_partitioning=true)")

    agents = {1:'0',2:'00',3:'000',4:'0000',5:'00000',6:'000000',7:'0000000',
              8:'00000000',9:'000000000',10:'0000000000',11:'00000000000',12:'000000000000'}

    print(f"{'gen':>3} {'dom':>3} {'ticks':>6} {'mode':>4} {'mean':>5} | "
          f"{'%0':>4} {'%1':>4} {'%2':>4} {'%3':>4} {'%4':>4} "
          f"{'%5':>4} {'%6':>4} {'%7':>4} {'%8':>4} | {'init':>5}")
    print("-" * 100)

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
          FROM h WHERE generation={g} AND agent_id='{ag}' ORDER BY global_time
        """).fetchall()
        if not rows:
            continue

        dom_series = defaultdict(list)
        t0 = rows[0][0]
        init_idx = None
        prev_n = rows[0][4]
        for i, r in enumerate(rows):
            t, pool, dom, form, n = r
            if init_idx is None and n > prev_n:
                init_idx = i
            prev_n = n
            per_dom_atp = defaultdict(int)
            per_dom_tot = defaultdict(int)
            for p, d, f in zip(pool, dom, form):
                if p == POOL_ORIC_LOW:
                    per_dom_tot[d] += 1
                    if f == FORM_ATP:
                        per_dom_atp[d] += 1
            for d in per_dom_tot:
                if per_dom_tot[d] == 8:
                    dom_series[d].append(per_dom_atp[d])

        init_min = (rows[init_idx][0] - t0) / 60 if init_idx is not None else None

        for d in sorted(dom_series.keys()):
            series = dom_series[d]
            if not series:
                continue
            counts = Counter(series)
            total = len(series)
            below_full = {k: v for k, v in counts.items() if k < 8}
            mode = max(below_full.items(), key=lambda kv: kv[1])[0] if below_full else 8
            mean = sum(series) / total
            pct = [counts.get(n, 0) * 100.0 / total for n in range(9)]
            init_str = f"{init_min:.1f}" if init_min is not None else "—"
            print(f"{g:>3} {d:>3} {total:>6} {mode:>4} {mean:>5.2f} | "
                  f"{pct[0]:>4.0f} {pct[1]:>4.0f} {pct[2]:>4.0f} {pct[3]:>4.0f} {pct[4]:>4.0f} "
                  f"{pct[5]:>4.0f} {pct[6]:>4.0f} {pct[7]:>4.0f} {pct[8]:>4.0f} | {init_str:>5}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_root")
    args = ap.parse_args()
    analyse(args.parquet_root)
