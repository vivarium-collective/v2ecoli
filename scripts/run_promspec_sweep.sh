#!/usr/bin/env bash
# promoter-specific-regulation phase C: 2 arms x 3 seeds, 12 generations.
#
# Arms differ ONLY in which ParCa cache they load; both caches were built from
# the same code with V2ECOLI_PROMOTER_SPECIFIC_TF as the only difference, so
# the comparison is not confounded by anything else.
#
# Every run goes through scripts/simlock.py, which caps concurrency across ALL
# worktrees on this machine. Launch all six at once and they queue themselves;
# they will also queue behind sims started from any other checkout.
set -uo pipefail
cd "$(dirname "$0")/.."
STUDY=promoter-specific-regulation
GENS=12
JOBS=6                     # simlock caps real concurrency machine-wide; no local throttle needed
LOG=out/promspec_sweep.log
: > "$LOG"

run_one () {
  local arm="$1" cache="$2" seed="$3"
  local eid="${STUDY}__${arm}__seed${seed}"
  local out="out/${eid}"
  mkdir -p "$out"
  .venv/bin/python3 scripts/simlock.py run --label "${eid}" -- \
      .venv/bin/python3 scripts/run_condition_multigen_parquet.py \
      --cache-dir "$cache" --out-dir "$out" --experiment-id "$eid" \
      --generations "$GENS" --seed "$seed" \
      --config-override 'ecoli-chromosome-replication.mechanistic_replisome=true' \
      --study-dir "workspace/studies/${STUDY}" --spec-id "$STUDY" \
      > "${out}/run.log" 2>&1
  echo "$( [ $? -eq 0 ] && echo ok || echo FAIL )  ${eid}" >> "$LOG"
}

i=0
for seed in 0 1 2; do
  for pair in "promoter-specific:out/cache_promspec" "attribution-control:out/cache_tudedup_full"; do
    arm="${pair%%:*}"; cache="${pair##*:}"
    run_one "$arm" "$cache" "$seed" &
    i=$((i+1))
    if [ $((i % JOBS)) -eq 0 ]; then wait; fi
  done
done
wait
echo "sweep complete" >> "$LOG"
