#!/usr/bin/env bash
# Autonomous driver: runs matched no-block + chromblock25 seed pairs with
# volume extraction, deleting parquets between batches to conserve disk.
#
# Assumes chromblock25 0+1 is already running (via manual launch).
# Picks up from seed pair (2, 3) onwards.
#
# Usage: nohup bash scripts/run_matched_volume_batches.sh > out/matched_volume_driver.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

RAMP="100,80,60,40,25,15,8,3"
KDMIN=3
FRAC=0.25
INTERSECTION_PAIRS="2,3 6,7 10,11 13,14 16,17"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_for_pid() {
    local pid=$1
    log "waiting for PID $pid..."
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "PID $pid done."
}

extract_config() {
    local config=$1  # "noblock" or "chromblock25"
    local seed_a=$2
    local seed_b=$3
    local outdir subst
    if [ "$config" = "noblock" ]; then
        subst="basal_v1.5_satinit_softgrad_fromBurninGen10v1.0"
        outdir="out/synchrony_softgrad_v2"
    else
        subst="basal_v1.5_satinit_softgrad_chromblock25_fromBurninGen10v1.0"
        outdir="out/synchrony_softgrad_chromblock25_v2"
    fi
    mkdir -p "$outdir"
    for S in "$seed_a" "$seed_b"; do
        local pq_root="out/${subst}_seed${S}_12gen_parquet/${subst}_seed${S}_12gen/history/experiment_id=${subst}_seed${S}_12gen"
        if [ -d "$pq_root" ]; then
            log "extracting $config seed $S..."
            .venv/bin/python scripts/extract_initiation_lineage.py "$pq_root" --out-dir "$outdir" 2>&1 | tail -1
        else
            log "SKIP extract $config seed $S — no parquet found"
        fi
    done
}

cleanup_config() {
    local config=$1  # "noblock" or "chromblock25"
    local seed_a=$2
    local seed_b=$3
    local subst
    if [ "$config" = "noblock" ]; then
        subst="basal_v1.5_satinit_softgrad_fromBurninGen10v1.0"
    else
        subst="basal_v1.5_satinit_softgrad_chromblock25_fromBurninGen10v1.0"
    fi
    for S in "$seed_a" "$seed_b"; do
        log "deleting $config seed $S parquet + log..."
        rm -rf "out/${subst}_seed${S}_12gen_parquet"
        rm -f "out/${subst}_seed${S}_12gen_run.log"
    done
}

# Wait for currently-running chromblock25 0+1
log "checking if chromblock25 seed 0/1 still running..."
if pgrep -f 'run_condition_multigen_parquet.*chromblock25.*seed[01]_' > /dev/null; then
    log "chromblock25 seeds 0+1 still running — waiting..."
    while pgrep -f 'run_condition_multigen_parquet.*chromblock25.*seed[01]_' > /dev/null; do
        sleep 60
    done
    log "chromblock25 seeds 0+1 done."
    extract_config chromblock25 0 1
    cleanup_config chromblock25 0 1
else
    log "no active chromblock25 0+1 sim — assuming already done."
fi

for pair in $INTERSECTION_PAIRS; do
    A=${pair%,*}; B=${pair#*,}
    log "==== batch: seed pair $A, $B ===="

    # No-block batch
    log "launching no-block seeds $A, $B..."
    bash scripts/run_basal_soft_kd.sh softgrad "$RAMP" $KDMIN "$A" "$B" > "out/matched_noblock_seeds${A}and${B}_launcher.log" 2>&1
    log "no-block seeds $A, $B done."
    extract_config noblock "$A" "$B"
    cleanup_config noblock "$A" "$B"

    # Chromblock25 batch
    log "launching chromblock25 seeds $A, $B..."
    bash scripts/run_basal_softgrad_chromblock.sh $FRAC "$A" "$B" > "out/matched_chromblock25_seeds${A}and${B}_launcher.log" 2>&1
    log "chromblock25 seeds $A, $B done."
    extract_config chromblock25 "$A" "$B"
    cleanup_config chromblock25 "$A" "$B"

    log "batch $A, $B complete. Disk: $(df -h . | awk 'NR==2{print $4}') free."
done

log "==== ALL BATCHES DONE ===="
