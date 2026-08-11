#!/usr/bin/env bash
# Final push: seeds 24, 25 for both configs to get no-block over 100 pairs.

set -uo pipefail
cd "$(dirname "$0")/.."

RAMP="100,80,60,40,25,15,8,3"
KDMIN=3
FRAC=0.25
INTERSECTION_PAIRS="24,25"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

extract_config() {
    local config=$1
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
            log "SKIP extract $config seed $S — no parquet"
        fi
    done
}

cleanup_config() {
    local config=$1
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

while pgrep -f 'run_condition_multigen_parquet' > /dev/null; do
    log "waiting for previous sims to finish..."
    sleep 60
done

for pair in $INTERSECTION_PAIRS; do
    A=${pair%,*}; B=${pair#*,}
    log "==== batch: seed pair $A, $B ===="
    log "launching no-block seeds $A, $B..."
    bash scripts/run_basal_soft_kd.sh softgrad "$RAMP" $KDMIN "$A" "$B" > "out/matched_noblock_v4_seeds${A}and${B}_launcher.log" 2>&1
    log "no-block seeds $A, $B done."
    extract_config noblock "$A" "$B"
    cleanup_config noblock "$A" "$B"

    log "launching chromblock25 seeds $A, $B..."
    bash scripts/run_basal_softgrad_chromblock.sh $FRAC "$A" "$B" > "out/matched_chromblock25_v4_seeds${A}and${B}_launcher.log" 2>&1
    log "chromblock25 seeds $A, $B done."
    extract_config chromblock25 "$A" "$B"
    cleanup_config chromblock25 "$A" "$B"
    log "batch $A, $B complete."
done

log "==== V4 DONE ===="
