#!/usr/bin/env bash
# comparison_harness.sh — ONE reproducible entry point for the v2ecoli <-> vEcoli
# faithful-port comparison. Wraps the three steps so launch -> report is a single
# deterministic chain:
#
#   register   build/register simulators (once per code change) -> v2ecoli sim id
#   launch     submit 5 conditions x 2 engines to sms-api (Ray + Nextflow)
#   report     export AWS creds + render the standardized report card
#   all        register (unless --v2-sim given) -> launch -> report
#
# Runs go through GovCloud sms-api over an SSM tunnel; outputs land at
#   s3://<bucket>/vecoli-output/<experiment_id>/...   (v2ecoli zarr; vEcoli parquet)
# launch writes the per-condition experiment ids to out/full_compare/experiments.json
# so `report` reads exactly the runs `launch` created (no hand-editing CONDITIONS).
#
# Prereqs (see README "v2ecoli <-> vEcoli comparison harness"):
#   aws sso login --profile stanford-sso
#   nohup bash ~/code/sms-cdk/scripts/ptools-proxy.sh -s smsvpctest >/tmp/ptools.log 2>&1 &
#
# Usage:
#   bash scripts/comparison_harness.sh register [--commit H] [--branch B] [--repo URL]
#   bash scripts/comparison_harness.sh launch  --v2-sim ID [--ve-sim 47] \
#         [--seeds 4] [--gens 2] [--conditions "basal with_aa succinate no_oxygen acetate"]
#   bash scripts/comparison_harness.sh report  [--only all]
#   bash scripts/comparison_harness.sh all     [--v2-sim ID] [--seeds 4] [--gens 2] ...
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${SMS_API_BASE:-http://localhost:8080}"
PROFILE="${AWS_PROFILE:-stanford-sso}"
REGION="${AWS_DEFAULT_REGION:-us-gov-west-1}"
PY="$REPO_ROOT/.venv/bin/python"
OUT="$REPO_ROOT/out/full_compare"
EXP_JSON="$OUT/experiments.json"
CONF="api_simulation_default_aws_cdk.json"
DEFAULT_CONDITIONS="basal with_aa succinate no_oxygen acetate"
DEFAULT_REPO="https://github.com/vivarium-collective/v2ecoli"   # NO .git suffix

# defaults (sensible: 4 seeds x 2 gens)
V2_SIM="" ; VE_SIM="47" ; SEEDS="4" ; GENS="2" ; CONDITIONS="$DEFAULT_CONDITIONS"
ONLY="all" ; COMMIT="" ; BRANCH="" ; REPO="$DEFAULT_REPO" ; TAG="cmp"

die() { echo "ERROR: $*" >&2 ; exit 1 ; }
need_tunnel() {
  curl -s -m 8 "$BASE/version" >/dev/null 2>&1 \
    || die "sms-api tunnel down at $BASE — start ptools-proxy.sh -s smsvpctest first"
  echo "sms-api $(curl -s -m 8 "$BASE/version") @ $BASE"
}
export_creds() {
  export AWS_PROFILE="$PROFILE" AWS_DEFAULT_REGION="$REGION"
  eval "$(aws configure export-credentials --format env)" 2>/dev/null \
    || die "aws creds export failed — run: aws sso login --profile $PROFILE"
  echo "AWS creds exported (profile=$PROFILE region=$REGION)"
}
# parse the experiment_id sms-api returns from a submit response body.
_exp_id() { "$PY" -c "import sys,json;d=json.load(sys.stdin);print(d.get('experiment_id',''))" 2>/dev/null; }

cmd_register() {
  need_tunnel
  COMMIT="${COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
  BRANCH="${BRANCH:-$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)}"
  echo "Registering v2ecoli: repo=$REPO commit=${COMMIT:0:8} branch=$BRANCH"
  local body
  body=$(curl -s -m 60 -X POST "$BASE/core/v1/simulator/upload" \
    -H 'Content-Type: application/json' \
    -d "{\"git_repo_url\":\"$REPO\",\"git_commit_hash\":\"$COMMIT\",\"git_branch\":\"$BRANCH\"}")
  V2_SIM=$("$PY" -c "import sys,json;print(json.load(sys.stdin).get('simulator_id',''))" <<<"$body" 2>/dev/null)
  [ -n "$V2_SIM" ] || die "no simulator_id in upload response: $body"
  echo "v2ecoli simulator_id=$V2_SIM (vEcoli simulator_id=$VE_SIM, vEcoli@62924758 Nextflow)"
  echo "Poll build:  curl -s '$BASE/core/v1/simulator/status?simulator_id=$V2_SIM'"
}

cmd_launch() {
  need_tunnel
  [ -n "$V2_SIM" ] || die "launch needs --v2-sim ID (run 'register' first)"
  mkdir -p "$OUT"
  echo "launch: v2-sim=$V2_SIM ve-sim=$VE_SIM seeds=$SEEDS gens=$GENS"
  echo "conditions: $CONDITIONS"
  : >"$OUT/.exp.tsv"
  for cond in $CONDITIONS; do
    # --- v2ecoli (Ray) ---
    local v2_exp="${TAG}-v2-${cond}"
    local v2_url="$BASE/api/v1/simulations?simulator_id=${V2_SIM}&experiment_id=${v2_exp}&composite=v2ecoli&condition=${cond}&num_seeds=${SEEDS}&num_generations=${GENS}&max_generations=${GENS}&simulation_config_filename=${CONF}"
    local v2_id ; v2_id=$(curl -s -m 60 -X POST "$v2_url" | _exp_id)
    # --- vEcoli (Nextflow): clear stale K8s job/configmap that collide by config name ---
    if command -v kubectl >/dev/null 2>&1; then
      kubectl delete job "nf-cond-${cond}" --ignore-not-found >/dev/null 2>&1 || true
      kubectl delete configmap "nf-cond-${cond}" --ignore-not-found >/dev/null 2>&1 || true
    fi
    local ve_exp="${TAG}-ve-${cond}"
    local ve_url="$BASE/api/v1/simulations?simulator_id=${VE_SIM}&experiment_id=${ve_exp}&num_seeds=${SEEDS}&num_generations=${GENS}&simulation_config_filename=cond_${cond}.json"
    local ve_id ; ve_id=$(curl -s -m 60 -X POST "$ve_url" | _exp_id)
    printf '%s\t%s\t%s\n' "$cond" "${v2_id:-$v2_exp}" "${ve_id:-$ve_exp}" >>"$OUT/.exp.tsv"
    printf '  %-10s v2=%-32s ve=%-32s\n' "$cond" "${v2_id:-FAILED}" "${ve_id:-FAILED}"
    sleep 2
  done
  V2_SIM="$V2_SIM" VE_SIM="$VE_SIM" SEEDS="$SEEDS" GENS="$GENS" "$PY" - "$OUT/.exp.tsv" "$EXP_JSON" <<'PY'
import json, os, sys, datetime
tsv, outp = sys.argv[1], sys.argv[2]
conds = {}
for line in open(tsv):
    c, v2, ve = line.rstrip("\n").split("\t")
    conds[c] = [v2, ve]
doc = {"generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "v2_simulator_id": os.environ["V2_SIM"], "ve_simulator_id": os.environ["VE_SIM"],
       "seeds": int(os.environ["SEEDS"]), "gens": int(os.environ["GENS"]),
       "conditions": conds}
json.dump(doc, open(outp, "w"), indent=2)
print("wrote", outp)
PY
  rm -f "$OUT/.exp.tsv"
  echo "Runs submitted. Poll/wait for completion on S3, then: bash $0 report"
}

cmd_report() {
  export_creds
  [ -f "$EXP_JSON" ] && echo "reading launched experiment ids from $EXP_JSON" \
    || echo "no $EXP_JSON — comparison_report_card.py falls back to its CONDITIONS default"
  "$PY" "$REPO_ROOT/scripts/comparison_report_card.py" --only "$ONLY" --out "$OUT"
}

# ---- arg parsing ----
SUB="${1:-}" ; shift || true
[ -n "$SUB" ] || die "usage: $0 {register|launch|report|all} [flags] (see --help)"
case "$SUB" in -h|--help|help)
  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ; exit 0 ;; esac
while [ $# -gt 0 ]; do
  case "$1" in
    --v2-sim) V2_SIM="$2"; shift 2;;
    --ve-sim) VE_SIM="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --gens) GENS="$2"; shift 2;;
    --conditions) CONDITIONS="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --commit) COMMIT="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0;;
    *) die "unknown flag: $1";;
  esac
done

case "$SUB" in
  register) cmd_register ;;
  launch)   cmd_launch ;;
  report)   cmd_report ;;
  all)      [ -n "$V2_SIM" ] || cmd_register ; cmd_launch ; cmd_report ;;
  *) die "unknown subcommand: $SUB (register|launch|report|all)" ;;
esac
