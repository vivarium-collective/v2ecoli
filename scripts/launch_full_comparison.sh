#!/usr/bin/env bash
# launch_full_comparison.sh — fire the full two-engine v2ecoli↔vEcoli comparison
# on GovCloud via sms-api: both composites × 5 conditions × N seeds × N gens.
#
# Each submit is one POST /api/v1/simulations (query params), routed by sms-api
# to Ray Batch. Output = compact-emitter zarr per seed at
#   s3://<bucket>/vecoli-output/<experiment_id>/<engine>_seed<NN>.zarr
# Aggregate afterwards with scripts/s3_compare_report.py.
#
# Usage:  bash scripts/launch_full_comparison.sh <simulator_id> [n_seeds] [n_gens] [tag]
#   defaults: n_seeds=16  n_gens=16  tag=full
# Requires the SSM tunnel up on localhost:8080 (ptools-proxy.sh -s smsvpctest).
set -u
SIM_ID="${1:?simulator_id required (e.g. 50)}"
N_SEEDS="${2:-16}"
N_GENS="${3:-16}"
TAG="${4:-full}"
BASE="${SMS_API_BASE:-http://localhost:8080}"
CONF="api_simulation_default_aws_cdk.json"
ENGINES=(v2ecoli vecoli)
CONDITIONS=(basal with_aa succinate no_oxygen acetate)

curl -s -m 8 "$BASE/version" >/dev/null 2>&1 || { echo "tunnel down at $BASE — start ptools-proxy first"; exit 1; }
echo "sms-api $(curl -s -m 8 "$BASE/version") | sim=$SIM_ID seeds=$N_SEEDS gens=$N_GENS tag=$TAG"
echo "engine    condition    HTTP  experiment_id / job_id"
echo "------------------------------------------------------------"
for eng in "${ENGINES[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    exp="${TAG}-cmp-${eng}-${cond}"
    url="$BASE/api/v1/simulations?simulator_id=${SIM_ID}&experiment_id=${exp}&composite=${eng}&condition=${cond}&num_seeds=${N_SEEDS}&num_generations=${N_GENS}&max_generations=${N_GENS}&simulation_config_filename=${CONF}"
    resp=$(curl -s -m 60 -w "\n%{http_code}" -X POST "$url" 2>&1)
    code=$(echo "$resp" | tail -1)
    body=$(echo "$resp" | sed '$d')
    ids=$(echo "$body" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('experiment_id',''),d.get('job_id','')[:8])" 2>/dev/null)
    printf "%-9s %-12s %-5s %s\n" "$eng" "$cond" "$code" "${ids:-$(echo "$body" | tail -c 80)}"
    sleep 2
  done
done
echo "------------------------------------------------------------"
echo "Submitted ${#ENGINES[@]}×${#CONDITIONS[@]} = $(( ${#ENGINES[@]} * ${#CONDITIONS[@]} )) workflows."
echo "Aggregate per condition when done, e.g.:"
echo "  .venv/bin/python scripts/s3_compare_report.py --experiment ${TAG}-cmp-vecoli-basal  # (see note)"
echo "NOTE: s3_compare_report pairs ONE experiment holding BOTH engines' zarrs; the"
echo "  launcher writes a separate experiment per engine, so either point --engines at the"
echo "  two per-engine experiment prefixes or sync both into one --local-dir before aggregating."
