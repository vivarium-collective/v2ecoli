#!/usr/bin/env bash
# Sanity driver: 5 matched seeds (0-4) each for 50% and 75% chromosomal
# DnaA-box blocking. Runs sequentially in 2-seed batches, extracts each
# seed's synchrony_summary.json, and deletes parquets after extract to
# save disk. Feeds a 4-way (0% / 25% / 50% / 75%) comparison PDF.
#
# Usage:
#   nohup bash scripts/run_matched_chromblock_5075_sanity.sh \
#     > out/matched_chromblock_5075_sanity_driver.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

extract_config() {
    local frac=$1  # 0.5 or 0.75
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
}

cleanup_config() {
    local frac=$1
    local seed=$2
    local frac_pct
    frac_pct=$(python3 -c "print(int(float('$frac')*100))")
    local subst="basal_v1.5_satinit_softgrad_chromblock${frac_pct}_fromBurninGen10v1.0"
    log "delete ${frac_pct}% seed $seed parquet"
    rm -rf "out/${subst}_seed${seed}_12gen_parquet"
    rm -f "out/${subst}_seed${seed}_12gen_run.log"
}

run_pair() {
    local frac=$1
    local a=$2
    local b=$3
    local frac_pct
    frac_pct=$(python3 -c "print(int(float('$frac')*100))")
    log "==== batch: FRAC=$frac seeds $a,$b ===="
    bash scripts/run_basal_softgrad_chromblock.sh "$frac" "$a" "$b" \
        > "out/matched_chromblock${frac_pct}_sanity_seeds${a}and${b}_launcher.log" 2>&1 || true

    # Wait for the two launched sims to finish (nohup + backgrounded).
    while pgrep -f "chromblock${frac_pct}_fromBurninGen10v1.0_seed(${a}|${b})_12gen" > /dev/null; do
        sleep 60
    done

    extract_config "$frac" "$a"
    extract_config "$frac" "$b"
    cleanup_config "$frac" "$a"
    cleanup_config "$frac" "$b"
    log "batch ${frac_pct}% seeds $a,$b done. Disk: $(df -h . | awk 'NR==2{print $4}') free."
}

run_single() {
    local frac=$1
    local a=$2
    local frac_pct
    frac_pct=$(python3 -c "print(int(float('$frac')*100))")
    log "==== batch: FRAC=$frac seed $a (solo) ===="
    bash scripts/run_basal_softgrad_chromblock.sh "$frac" "$a" \
        > "out/matched_chromblock${frac_pct}_sanity_seed${a}_launcher.log" 2>&1 || true
    while pgrep -f "chromblock${frac_pct}_fromBurninGen10v1.0_seed${a}_12gen" > /dev/null; do
        sleep 60
    done
    extract_config "$frac" "$a"
    cleanup_config "$frac" "$a"
    log "batch ${frac_pct}% seed $a done. Disk: $(df -h . | awk 'NR==2{print $4}') free."
}

# Wait for any previously running sim
while pgrep -f 'run_condition_multigen_parquet' > /dev/null; do
    log "waiting for previous sims to finish..."
    sleep 60
done

# ---- 50% chromblock: seeds 0,1,2,3 (seed 4 already extracted) ----
run_pair 0.5 0 1
run_pair 0.5 2 3

# ---- 75% chromblock: seeds 0,1,2,3,4 ----
run_pair 0.75 0 1
run_pair 0.75 2 3
run_single 0.75 4

log "==== SANITY DRIVER DONE ===="
