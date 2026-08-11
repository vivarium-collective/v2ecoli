#!/usr/bin/env bash
# Extension driver: seeds 5-25 for both 50% and 75% chromblock,
# alternating configs pair-by-pair so both accumulate in parallel.
# Re-aggregates synchrony_summary.json after each pair (fixes the
# extract-overwrites bug + emits full pair schema).
#
# Usage:
#   nohup bash scripts/run_matched_chromblock_5075_extend.sh \
#     > out/matched_chromblock_5075_extend_driver.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Seeds to add (0-4 already done in sanity driver)
NEW_SEEDS=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)

extract_and_aggregate() {
    local frac=$1
    local seed=$2
    local frac_pct
    frac_pct=$(python3 -c "print(int(float('$frac')*100))")
    local subst="basal_v1.5_satinit_softgrad_chromblock${frac_pct}_fromBurninGen10v1.0"
    local outdir="out/synchrony_softgrad_chromblock${frac_pct}_v2"
    mkdir -p "$outdir"
    local pq_root="out/${subst}_seed${seed}_12gen_parquet/${subst}_seed${seed}_12gen/history/experiment_id=${subst}_seed${seed}_12gen"
    if [ -d "$pq_root" ]; then
        log "extract ${frac_pct}% seed $seed"
        .venv/bin/python scripts/extract_initiation_lineage.py "$pq_root" --out-dir "$outdir" 2>&1 | tail -1
    else
        log "SKIP extract ${frac_pct}% seed $seed — no parquet"
    fi
    # Always re-aggregate so synchrony_summary.json holds the full accumulated set
    .venv/bin/python scripts/aggregate_lineage_to_synchrony.py "$outdir" 2>&1 | tail -1
}

cleanup_seed() {
    local frac=$1
    local seed=$2
    local frac_pct
    frac_pct=$(python3 -c "print(int(float('$frac')*100))")
    local subst="basal_v1.5_satinit_softgrad_chromblock${frac_pct}_fromBurninGen10v1.0"
    log "delete ${frac_pct}% seed $seed parquet"
    rm -rf "out/${subst}_seed${seed}_12gen_parquet"
    rm -f "out/${subst}_seed${seed}_12gen_run.log"
}

run_batch() {
    local frac=$1
    shift
    local seeds=("$@")
    local frac_pct
    frac_pct=$(python3 -c "print(int(float('$frac')*100))")
    log "==== FRAC=$frac seeds ${seeds[*]} ===="
    bash scripts/run_basal_softgrad_chromblock.sh "$frac" "${seeds[@]}" \
        > "out/matched_chromblock${frac_pct}_extend_seeds$(IFS=_; echo "${seeds[*]}")_launcher.log" 2>&1 || true

    # Wait for all sims in this batch to finish
    local pattern="chromblock${frac_pct}_fromBurninGen10v1.0_seed($(IFS='|'; echo "${seeds[*]}"))_12gen"
    while pgrep -f "$pattern" > /dev/null; do
        sleep 60
    done

    for s in "${seeds[@]}"; do
        extract_and_aggregate "$frac" "$s"
        cleanup_seed "$frac" "$s"
    done
    log "batch ${frac_pct}% seeds ${seeds[*]} done. Disk: $(df -h . | awk 'NR==2{print $4}') free."
}

# Wait for any prior sims
while pgrep -f 'run_condition_multigen_parquet' > /dev/null; do
    log "waiting for previous sims to finish..."
    sleep 60
done

# Alternate 50% and 75% pair-by-pair
n=${#NEW_SEEDS[@]}
i=0
while [ $i -lt $n ]; do
    if [ $((i+1)) -lt $n ]; then
        A=${NEW_SEEDS[$i]}
        B=${NEW_SEEDS[$((i+1))]}
        run_batch 0.5  "$A" "$B"
        run_batch 0.75 "$A" "$B"
        i=$((i+2))
    else
        # Last odd seed
        A=${NEW_SEEDS[$i]}
        run_batch 0.5  "$A"
        run_batch 0.75 "$A"
        i=$((i+1))
    fi
done

log "==== EXTEND DRIVER DONE ===="
