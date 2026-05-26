#!/usr/bin/env bash
# Run v2ecoli 1-gen sims for all 5 canonical conditions from the apr24 fixture.
# Writes a separate cache + SQLite DB per condition under out/<cond>_apr24_*
set -e

FIXTURE=out/parca_state_apr24.pkl.gz

get_max_dur() {
    case "$1" in
        basal)     echo 5400 ;;
        with_aa)   echo 3000 ;;
        acetate)   echo 12000 ;;
        succinate) echo 9000 ;;
        no_oxygen) echo 12000 ;;
    esac
}

for cond in basal with_aa acetate succinate no_oxygen; do
    cache=out/cache_${cond}_apr24
    db=out/${cond}_apr24_run.db
    dur=$(get_max_dur "$cond")

    echo "=== ${cond} (apr24 fixture, max_duration=${dur}s) ==="

    if [ ! -f "${cache}/sim_data_cache.dill" ]; then
        echo "  building ${cache} ..."
        python scripts/build_cache.py \
            --fixture "${FIXTURE}" \
            --cache   "${cache}" \
            --condition "${cond}"
    else
        echo "  cache ${cache} exists, skipping build"
    fi

    if [ -f "${db}" ]; then
        rows=$(sqlite3 "${db}" 'SELECT COUNT(*) FROM history' 2>/dev/null || echo 0)
        if [ "${rows}" -gt 100 ]; then
            echo "  ${db} already has ${rows} rows, skipping sim"
            continue
        fi
    fi
    rm -f "${db}" "${db}-shm" "${db}-wal"

    echo "  running 1-gen sim ..."
    python studies/stage1_sanity_check/run_sim.py \
        --cache-dir   "${cache}" \
        --generations 1 \
        --seed        0 \
        --max-duration "${dur}" \
        --db-path     "${db}"
done

echo
echo "=== summary ==="
for cond in basal with_aa acetate succinate no_oxygen; do
    db=out/${cond}_apr24_run.db
    if [ -f "${db}" ]; then
        max_step=$(sqlite3 "${db}" 'SELECT MAX(step) FROM history' 2>/dev/null || echo 0)
        echo "  ${cond}: ${max_step} steps in ${db}"
    fi
done
