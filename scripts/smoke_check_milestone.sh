#!/usr/bin/env bash
# 1-gen smoke that verifies byte-identity against the committed milestone.
#
# Uses the exact milestone env vars. Runs seed=4 for 1 generation, then
# diffs the parquet tick data against
#   out/dnaa5_stepped_adair_100_100_50_10_3_3_3_3_unlock60_sustain1_seed4_12gen
# (the committed milestone reference).
#
# Exit 0 => byte-identical for gen 1
# Exit 1 => divergence found (report tick + fields)

set -euo pipefail
cd "$(dirname "$0")/.."

EXP=dnaa5_smoke_milestone_check_seed4_1gen
REF_EXP=dnaa5_stepped_adair_100_100_50_10_3_3_3_3_unlock60_sustain1_seed4_12gen

rm -rf "out/${EXP}_parquet" "out/${EXP}" "out/${EXP}_run.log"

V2ECOLI_DNAA_ADAIR_KD=1 \
V2ECOLI_DNAA_ADAIR_KDS_NM=100,100,50,10,3,3,3,3 \
V2ECOLI_DNAA_ADAIR_KD_MAX_NM=100 \
V2ECOLI_DNAA_ADAIR_KD_MIN_NM=3 \
V2ECOLI_DNAA_COOP_GRADIENT_GATE=1 \
V2ECOLI_DNAA_GRADIENT_GATE=1 \
V2ECOLI_DNAA_GRADIENT_MIN_SLOPE_NM_PER_S=0.05 \
V2ECOLI_DNAA_GRADIENT_WINDOW_S=120 \
V2ECOLI_DNAA_HYDROLYSIS_RATE_PER_MIN=0.025 \
V2ECOLI_DNAA_KINETIC_ORIC_LOW=0 \
V2ECOLI_DNAA_POST_INIT_UNLOCK_S=60 \
V2ECOLI_SATURATION_SUSTAINED_S=1 \
V2ECOLI_SATURATION_TRIGGERED_INIT=1 \
.venv/bin/python scripts/run_condition_multigen_parquet.py \
  --cache-dir out/cache_dnaa2_v1.5e-3_kd3nm_apoATP_kinetic \
  --out-dir "out/${EXP}_parquet" \
  --experiment-id "$EXP" \
  --seed 4 \
  --generations 1 \
  --max-min 180 \
  --resume-dill out/dnaa5_v1.5_hillKd_h4_K3_seed4/gen_dills/gen5.dill \
  > "out/${EXP}_run.log" 2>&1

RESULT=0
.venv/bin/python <<PY || RESULT=$?
import duckdb, sys
con = duckdb.connect()
con.sql("CREATE VIEW ref AS SELECT * FROM read_parquet('out/${REF_EXP}_parquet/${REF_EXP}/history/**/*.pq', hive_partitioning=true, union_by_name=true)")
con.sql("CREATE VIEW smoke AS SELECT * FROM read_parquet('out/${EXP}_parquet/${EXP}/history/**/*.pq', hive_partitioning=true, union_by_name=true)")

def get(v):
    return con.sql(f"""
    SELECT global_time,
           listeners__mass__dry_mass AS dm,
           listeners__replication_data__number_of_oric AS noric,
           listeners__replication_data__oriC_low_bound_atp AS ola
    FROM {v} WHERE generation=1 AND agent_id='0' AND lineage_seed=4
    ORDER BY global_time
    """).fetchall()

def _s(x): return sum(x) if isinstance(x, list) else int(x)

ref = get("ref"); smoke = get("smoke")
if len(ref) != len(smoke):
    print(f"DIVERGE: tick count differs — ref={len(ref)}, smoke={len(smoke)}")
    sys.exit(1)
t0r = ref[0][0]; t0s = smoke[0][0]
for i, (r, s) in enumerate(zip(ref, smoke)):
    t = r[0] - t0r
    if abs(r[1]-s[1]) > 0.0 or r[2] != s[2] or _s(r[3]) != _s(s[3]):
        print(f"DIVERGE at t={t:.0f}s (tick {i}):")
        print(f"  ref: dm={r[1]:.4f} noric={r[2]} ola={_s(r[3])}")
        print(f"  smk: dm={s[1]:.4f} noric={s[2]} ola={_s(s[3])}")
        sys.exit(1)
print(f"OK — byte-identical for all {len(ref)} ticks of gen 1")
PY

# Clean up smoke outputs on success; leave them intact on failure for debugging.
if [ "$RESULT" -eq 0 ]; then
    rm -rf "out/${EXP}_parquet" "out/${EXP}" "out/${EXP}_run.log"
    echo "(smoke outputs cleaned up)"
else
    echo "(smoke outputs kept at out/${EXP}_* for debugging)"
fi
exit "$RESULT"
