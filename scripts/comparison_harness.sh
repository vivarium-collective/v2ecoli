#!/usr/bin/env bash
# comparison_harness.sh — ONE deterministic, self-sufficient entry point for the
# v2ecoli <-> vEcoli faithful-port comparison. Every workaround that used to be
# done by hand each run is baked in here; the ONLY manual step is the SSO browser
# login (which the harness detects and tells you to run, then exits cleanly).
#
# Subcommands:
#   register   build/register the v2ecoli simulator from the local HEAD, poll the
#              build to completed/failed -> prints + persists the v2ecoli sim id.
#   launch     for each condition: clear stale K8s job/configmap, submit v2ecoli
#              (Ray) + vEcoli (Nextflow) to sms-api with transient-retry, capture
#              the wrapped experiment_ids, write out/full_compare/experiments.json
#              as {"conditions": {cond: [v2_exp, ve_exp]}} (v2 first).
#   wait       poll S3 until every launched run has output (v2 zarr chunks for all
#              seeds; vEcoli parquet history). Per-condition matrix each poll.
#   report     export AWS env creds, render the standardized report card.
#   all        register -> launch -> wait -> report (stops on first hard failure).
#
# Networked subcommands run _preflight first: verify SSO token, (re)start the SSM
# tunnel if down, export AWS env creds for s3fs/duckdb.
#
# Usage:
#   bash scripts/comparison_harness.sh register [--commit H] [--branch B] [--repo URL]
#   bash scripts/comparison_harness.sh launch  --v2-sim ID [--ve-sim 47] \
#         [--seeds 4] [--gens 2] [--conditions "basal with_aa succinate no_oxygen acetate"]
#   bash scripts/comparison_harness.sh wait    [--timeout 7200]
#   bash scripts/comparison_harness.sh report  [--only all]
#   bash scripts/comparison_harness.sh all     [--v2-sim ID] [--seeds 4] [--gens 2] ...
#
# Flags: --seeds N --gens N --conditions "<space list>" --v2-sim ID --ve-sim ID
#        --timeout SECONDS --only all --commit H --branch B --repo URL --tag STR
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${SMS_API_BASE:-http://localhost:8080}"
PROFILE="${AWS_PROFILE:-stanford-sso}"
REGION="${AWS_DEFAULT_REGION:-us-gov-west-1}"
BUCKET="${SMS_BUCKET:-smsvpctest-shared-sharedbucket60d199d6-abfvwv0day91}"
PREFIX="${SMS_PREFIX:-vecoli-output}"
K8S_NS="${SMS_K8S_NS:-sms-api-stanford-test}"
TUNNEL_CMD="${SMS_TUNNEL_CMD:-bash $HOME/code/sms-cdk/scripts/ptools-proxy.sh -s smsvpctest}"
PY="$REPO_ROOT/.venv/bin/python" ; [ -x "$PY" ] || PY="python3"
PYJ="python3"   # stdlib-only JSON parsing (no venv deps needed)
OUT="$REPO_ROOT/out/full_compare"
EXP_JSON="$OUT/experiments.json"
CONF="api_simulation_default_aws_cdk.json"
DEFAULT_CONDITIONS="basal with_aa succinate no_oxygen acetate"
DEFAULT_REPO="https://github.com/vivarium-collective/v2ecoli"   # NO .git suffix

V2_SIM="" ; VE_SIM="47" ; SEEDS="4" ; GENS="2" ; CONDITIONS="$DEFAULT_CONDITIONS"
ONLY="all" ; COMMIT="" ; BRANCH="" ; REPO="$DEFAULT_REPO" ; TAG="cmp" ; TIMEOUT="7200"

log() { echo "[harness] $*" ; }
die() { echo "[harness] ERROR: $*" >&2 ; exit 1 ; }
_json_get() { "$PYJ" -c "import sys,json
try: d=json.load(sys.stdin)
except Exception: d={}
print(d.get('$1','') if isinstance(d,dict) else '')" 2>/dev/null ; }

# POST with transient-retry. $1=url, remaining args are passed to curl (headers,
# body, etc). Echoes the response body; returns 0 on HTTP 2xx, non-zero (still
# echoing the last body) once retries on 000/502/5xx are exhausted.
_post() {
  local url="$1" ; shift
  local tries=0 max=5 resp code body
  while [ "$tries" -lt "$max" ]; do
    resp="$(curl -s -m 120 -w $'\n%{http_code}' -X POST "$@" "$url" 2>/dev/null)"
    code="${resp##*$'\n'}" ; body="${resp%$'\n'*}"
    case "$code" in
      2*) printf '%s' "$body" ; return 0 ;;
      000|5*) ;;                              # transient -> back off + retry
      *)  printf '%s' "$body" ; return 1 ;;   # 4xx etc -> surface immediately
    esac
    tries=$((tries+1)) ; sleep $((tries*3))
  done
  printf '%s' "${body:-}" ; return 1
}

# ---- preflight: SSO / tunnel / creds (the one manual step is surfaced here) ----
_check_sso() {
  export AWS_PROFILE="$PROFILE" AWS_DEFAULT_REGION="$REGION"
  if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "[harness] SSO token missing/expired. Run the ONE manual step, then re-run:" >&2
    echo "             aws sso login --profile $PROFILE" >&2
    exit 3
  fi
  log "SSO token valid (profile=$PROFILE region=$REGION)"
}
_ensure_tunnel() {
  if curl -s -m 5 "$BASE/version" >/dev/null 2>&1; then
    log "sms-api tunnel up: $(curl -s -m 5 "$BASE/version") @ $BASE" ; return 0
  fi
  log "tunnel down at $BASE — starting ptools-proxy in background"
  nohup $TUNNEL_CMD >/tmp/ptools-proxy.log 2>&1 &
  local tries=0
  while [ "$tries" -lt 20 ]; do
    sleep 3
    if curl -s -m 5 "$BASE/version" >/dev/null 2>&1; then
      log "tunnel up after restart: $(curl -s -m 5 "$BASE/version")" ; return 0
    fi
    tries=$((tries+1))
  done
  die "tunnel never came up at $BASE after restart (see /tmp/ptools-proxy.log)"
}
_export_creds() {
  eval "$(aws configure export-credentials --format env 2>/dev/null)" \
    || die "aws creds export failed — run: aws sso login --profile $PROFILE"
  log "AWS env creds exported for s3fs/duckdb (profile=$PROFILE)"
}
_preflight() { _check_sso ; _ensure_tunnel ; _export_creds ; }
# creds-only preflight (report step: needs env creds + S3, tunnel optional)
_preflight_creds() { _check_sso ; _export_creds ; }

cmd_register() {
  _preflight
  COMMIT="${COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)}"
  BRANCH="${BRANCH:-$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)}"
  [ -n "$COMMIT" ] || die "could not resolve a commit (pass --commit)"
  log "registering v2ecoli: repo=$REPO commit=${COMMIT:0:8} branch=${BRANCH:-?}"
  local body
  body="$(_post "$BASE/core/v1/simulator/upload" -H 'Content-Type: application/json' \
        -d "{\"git_repo_url\":\"$REPO\",\"git_commit_hash\":\"$COMMIT\",\"git_branch\":\"$BRANCH\"}")" \
    || log "upload POST returned non-2xx (body below); continuing to parse"
  V2_SIM="$(_json_get simulator_id <<<"$body")"
  [ -n "$V2_SIM" ] || die "no simulator_id in upload response: $body"
  mkdir -p "$OUT" ; echo "$V2_SIM" >"$OUT/.v2_sim_id"
  log "v2ecoli simulator_id=$V2_SIM (persisted to $OUT/.v2_sim_id)"
  local deadline=$(( $(date +%s) + TIMEOUT )) st
  while :; do
    st="$(curl -s -m 30 "$BASE/core/v1/simulator/status?simulator_id=$V2_SIM" | _json_get status)"
    case "$st" in
      completed|complete|succeeded|ready) log "build status=$st — done" ; break ;;
      failed|error)  die "simulator build FAILED (status=$st, sim=$V2_SIM)" ;;
      *) log "build status=${st:-pending}… (sim=$V2_SIM)" ;;
    esac
    [ "$(date +%s)" -ge "$deadline" ] && die "build did not finish within ${TIMEOUT}s (sim=$V2_SIM)"
    sleep 15
  done
}

cmd_launch() {
  _preflight
  [ -n "$V2_SIM" ] || { [ -f "$OUT/.v2_sim_id" ] && V2_SIM="$(cat "$OUT/.v2_sim_id")"; }
  [ -n "$V2_SIM" ] || die "launch needs --v2-sim ID (run 'register' first)"
  mkdir -p "$OUT"
  log "launch: v2-sim=$V2_SIM ve-sim=$VE_SIM seeds=$SEEDS gens=$GENS"
  log "conditions: $CONDITIONS"
  : >"$OUT/.exp.tsv"
  local cond v2_url ve_url v2_body ve_body v2_id ve_id
  for cond in $CONDITIONS; do
    # clear stale K8s resources that collide by Nextflow config name (idempotent)
    if command -v kubectl >/dev/null 2>&1; then
      kubectl delete job,configmap -n "$K8S_NS" \
        "nf-cond-${cond}" "nf-cond-${cond}-config" --ignore-not-found >/dev/null 2>&1 || true
    fi
    v2_url="$BASE/api/v1/simulations?simulator_id=${V2_SIM}&experiment_id=${TAG}-v2-${cond}&composite=v2ecoli&condition=${cond}&num_seeds=${SEEDS}&num_generations=${GENS}&max_generations=${GENS}&simulation_config_filename=${CONF}"
    ve_url="$BASE/api/v1/simulations?simulator_id=${VE_SIM}&experiment_id=${TAG}-ve-${cond}&num_seeds=${SEEDS}&num_generations=${GENS}&simulation_config_filename=cond_${cond}.json"
    v2_body="$(_post "$v2_url")" ; v2_id="$(_json_get experiment_id <<<"$v2_body")"
    ve_body="$(_post "$ve_url")" ; ve_id="$(_json_get experiment_id <<<"$ve_body")"
    [ -n "$v2_id" ] || die "v2ecoli submit for '$cond' returned no experiment_id: $v2_body"
    [ -n "$ve_id" ] || die "vEcoli submit for '$cond' returned no experiment_id: $ve_body"
    printf '%s\t%s\t%s\n' "$cond" "$v2_id" "$ve_id" >>"$OUT/.exp.tsv"
    printf '[harness]   %-10s v2=%-34s ve=%-34s\n' "$cond" "$v2_id" "$ve_id"
    sleep 2
  done
  V2_SIM="$V2_SIM" VE_SIM="$VE_SIM" SEEDS="$SEEDS" GENS="$GENS" \
    "$PYJ" "$REPO_ROOT/scripts/_write_experiments.py" "$OUT/.exp.tsv" "$EXP_JSON" \
    || die "failed to write $EXP_JSON"
  rm -f "$OUT/.exp.tsv"
  log "wrote $EXP_JSON — next: bash $0 wait  (then report)"
}

cmd_wait() {
  _preflight_creds
  [ -f "$EXP_JSON" ] || die "no $EXP_JSON — run 'launch' first"
  # emit: cond<TAB>v2_exp<TAB>ve_exp  (and pull seeds back from the json)
  local rows seedn
  rows="$("$PYJ" -c "import json,sys;d=json.load(open('$EXP_JSON'));c=d.get('conditions',d);
[print('\t'.join([k,v[0],v[1]])) for k,v in c.items()]")" || die "could not parse $EXP_JSON"
  seedn="$("$PYJ" -c "import json;d=json.load(open('$EXP_JSON'));print(int(d.get('seeds',$SEEDS)))" 2>/dev/null)"
  [ -n "$seedn" ] && SEEDS="$seedn"
  log "wait: polling S3 for ${SEEDS}-seed completion (timeout ${TIMEOUT}s)"
  local deadline=$(( $(date +%s) + TIMEOUT ))
  while :; do
    local all_done=1 missing="" cond v2_exp ve_exp ls v2done vedone
    while IFS=$'\t' read -r cond v2_exp ve_exp; do
      [ -n "$cond" ] || continue
      # v2ecoli: distinct seed zarr stores that actually have chunk objects (.../c/)
      ls="$(aws s3 ls --recursive "s3://$BUCKET/$PREFIX/$v2_exp/" 2>/dev/null)"
      v2done="$(printf '%s\n' "$ls" | grep -E '_seed[0-9]+\.zarr/.*/c/' \
                | grep -oE '_seed[0-9]+\.zarr' | sort -u | wc -l | tr -d ' ')"
      # vEcoli: distinct lineage_seed under parquet history (fallback: any history pq)
      ls="$(aws s3 ls --recursive "s3://$BUCKET/$PREFIX/$ve_exp/" 2>/dev/null)"
      vedone="$(printf '%s\n' "$ls" | grep -E "cond_${cond}/.*\.pq" \
                | grep -oE 'lineage_seed=[0-9]+' | sort -u | wc -l | tr -d ' ')"
      if [ "${vedone:-0}" -eq 0 ] && printf '%s\n' "$ls" | grep -qE "cond_${cond}/.*history/.*\.pq"; then
        vedone="$SEEDS"
      fi
      printf '[harness]   %-10s v2 %s/%s  ve %s/%s\n' "$cond" "${v2done:-0}" "$SEEDS" "${vedone:-0}" "$SEEDS"
      if [ "${v2done:-0}" -lt "$SEEDS" ] || [ "${vedone:-0}" -lt "$SEEDS" ]; then
        all_done=0 ; missing="$missing $cond(v2:${v2done:-0},ve:${vedone:-0})"
      fi
    done <<<"$rows"
    if [ "$all_done" -eq 1 ]; then log "all conditions complete on S3" ; return 0 ; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      die "timeout after ${TIMEOUT}s — still missing:$missing"
    fi
    log "incomplete:$missing — re-poll in 60s"
    sleep 60
  done
}

cmd_report() {
  _preflight_creds
  [ -f "$EXP_JSON" ] && log "reading launched experiment ids from $EXP_JSON" \
    || log "no $EXP_JSON — report card falls back to its CONDITIONS default"
  "$PY" "$REPO_ROOT/scripts/comparison_report_card.py" --only "$ONLY" --out "$OUT"
}

# ---- arg parsing ----
SUB="${1:-}" ; shift || true
[ -n "$SUB" ] || die "usage: $0 {register|launch|wait|report|all} [flags] (see --help)"
case "$SUB" in -h|--help|help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ; exit 0 ;; esac
while [ $# -gt 0 ]; do
  case "$1" in
    --v2-sim) V2_SIM="$2"; shift 2;;
    --ve-sim) VE_SIM="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --gens) GENS="$2"; shift 2;;
    --conditions) CONDITIONS="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --commit) COMMIT="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0;;
    *) die "unknown flag: $1";;
  esac
done

case "$SUB" in
  register) cmd_register ;;
  launch)   cmd_launch ;;
  wait)     cmd_wait ;;
  report)   cmd_report ;;
  all)      [ -n "$V2_SIM" ] || cmd_register ; cmd_launch ; cmd_wait ; cmd_report ;;
  *) die "unknown subcommand: $SUB (register|launch|wait|report|all)" ;;
esac
