#!/usr/bin/env bash
# comparison_harness.sh — ONE deterministic, self-sufficient entry point for the
# v2ecoli <-> vEcoli faithful-port comparison. Every workaround that used to be
# done by hand each run is baked in here; the ONLY manual step is the SSO browser
# login (which the harness detects and tells you to run, then exits cleanly).
#
# REPRODUCIBILITY — the manifest (comparison_spec.json):
#   A single committed source-of-truth manifest pins BOTH engines (repo+commit+
#   branch, NOT just a sim id) and lets seeds/generations/config vary PER
#   CONDITION. Re-run with the committed spec to reproduce; edit it to vary.
#   With --spec, `register` registers BOTH engines from their pinned refs and
#   `launch` runs each condition at ITS seeds/gens/config. Without a spec the
#   legacy CLI behaviour is unchanged (v2ecoli from --commit/--branch/--repo,
#   vEcoli from --ve-sim, global --seeds/--gens).
#
#   comparison_spec.json schema:
#     {"v2ecoli": {"repo","commit","branch"},
#      "vecoli":  {"repo","commit","branch"},   # the upstream vEcoli FORK to WRAP
#      "vecoli_engine": "upstream-wrapper"|"nextflow",   # default upstream-wrapper
#      "from_vecoli_config": "configs/default.json",     # optional; drives BOTH engines
#      "defaults": {"seeds": N, "gens": M},
#      "conditions": [{"name","config","seeds","gens"}, ...]}
#   Field resolution: per-condition seeds/gens  >  defaults  >  CLI --seeds/--gens.
#   `config` is the vEcoli config filename; v2ecoli gets --condition <name> + the
#   standard api_simulation_default_aws_cdk.json.
#
# vecoli_engine routes (the headline one-command contract):
#   upstream-wrapper (DEFAULT) — register ONE v2ecoli image (it bundles the spec's
#     vEcoli fork, cloned at build via the vecoli.{repo,commit} build-args). Per
#     condition, launch composite=v2ecoli AND composite=vecoli on Ray, BOTH on that
#     one simulator id; BOTH emit v2ecoli-format zarr; report reads both via zarr
#     (--pbg-vs-pbg). So `comparison_harness.sh all --spec <fork>.json` is the whole
#     pbg-vs-pbg pipeline. To compare a NEW fork: edit vecoli.{repo,commit} +
#     from_vecoli_config, rebuild the image, rerun. No code change, no agent.
#   nextflow (legacy) — register vEcoli separately (Nextflow, parquet output) and
#     run it via its own sim id; report reads vEcoli parquet.
#
# --dry-run (launch): print the POSTs that WOULD be made for BOTH composites,
#   send nothing, need no creds — to verify the route/URLs offline.
#
# Subcommands:
#   register   register BOTH engines from the spec (v2ecoli->Ray, vEcoli->Nextflow,
#              routed by repo) at their pinned commits; poll each build to
#              completed/failed; persist sim ids (out/full_compare/.v2_sim_id,
#              .ve_sim_id). --v2-sim/--ve-sim skip the respective registration.
#              No spec -> legacy: register only v2ecoli from --commit/--branch.
#   launch     for each condition IN THE SPEC: clear stale K8s job/configmap,
#              submit v2ecoli + vEcoli with THAT condition's seeds/gens/config,
#              capture experiment_ids -> out/full_compare/experiments.json as
#              {"conditions": {cond: [v2_exp, ve_exp]}, "meta": {cond:{seeds,gens}}}.
#   wait       poll S3 until each run has output, using PER-CONDITION expected seed
#              counts from experiments.json meta (16-seed waits for 16, 4 for 4).
#   report     export AWS env creds, render the standardized report card.
#   all        register -> launch -> wait -> report (stops on first hard failure).
#
# Networked subcommands run _preflight first: verify SSO token, (re)start the SSM
# tunnel if down, export AWS env creds for s3fs/duckdb.
#
# Usage:
#   bash scripts/comparison_harness.sh all     [--spec comparison_spec.json]
#   bash scripts/comparison_harness.sh register [--spec FILE] [--v2-sim ID] [--ve-sim ID]
#   bash scripts/comparison_harness.sh launch   [--spec FILE] [--v2-sim ID] [--ve-sim ID]
#   bash scripts/comparison_harness.sh wait     [--timeout 7200]
#   bash scripts/comparison_harness.sh report   [--only all]
#   (legacy, no spec): register --commit H --branch B --repo URL ;
#                      launch --v2-sim ID --ve-sim 47 --seeds 4 --gens 2 --conditions "..."
#
# Flags: --spec FILE  (default comparison_spec.json if present; per-condition
#          seeds/gens/config + both engine refs come from it — --seeds/--gens
#          become fallback DEFAULTS only)
#        --seeds N --gens N --conditions "<space list>" --v2-sim ID --ve-sim ID
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
DEFAULT_SPEC="$REPO_ROOT/comparison_spec.json"

V2_SIM="" ; VE_SIM="" ; VE_SIM_GIVEN=0 ; SEEDS="4" ; GENS="2" ; CONDITIONS="$DEFAULT_CONDITIONS"
ONLY="all" ; COMMIT="" ; BRANCH="" ; REPO="$DEFAULT_REPO" ; TAG="cmp" ; TIMEOUT="7200"
SPEC="" ; SPEC_GIVEN=0 ; DRY=0
# engine refs (populated from the spec by _load_spec_engines)
V2_REPO="" ; V2_COMMIT="" ; V2_BRANCH="" ; VE_REPO="" ; VE_COMMIT="" ; VE_BRANCH=""
# vecoli engine route: "upstream-wrapper" (default — vEcoli run as a Ray
# composite=vecoli job on the SAME v2ecoli image/sim id, both engines emit
# v2ecoli-format zarr) or "nextflow" (legacy — separate Nextflow registration +
# parquet output). Read from the spec; legacy/no-spec stays nextflow.
VE_ENGINE="nextflow"

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

# Load both engine refs from the spec into V2_REPO/V2_COMMIT/V2_BRANCH/VE_* .
_load_spec_engines() {
  [ -f "$SPEC" ] || die "spec file not found: $SPEC"
  local line
  line="$("$PYJ" "$REPO_ROOT/scripts/_read_spec.py" "$SPEC" engine v2ecoli)" \
    || die "could not parse v2ecoli engine from $SPEC"
  IFS=$'\t' read -r V2_REPO V2_COMMIT V2_BRANCH <<<"$line"
  line="$("$PYJ" "$REPO_ROOT/scripts/_read_spec.py" "$SPEC" engine vecoli)" \
    || die "could not parse vecoli engine from $SPEC"
  IFS=$'\t' read -r VE_REPO VE_COMMIT VE_BRANCH <<<"$line"
  VE_ENGINE="$("$PYJ" "$REPO_ROOT/scripts/_read_spec.py" "$SPEC" vecoli-engine 2>/dev/null)"
  [ -n "$VE_ENGINE" ] || VE_ENGINE="upstream-wrapper"
}

# Poll a registered build to completion/failure. $1=label $2=sim_id
_poll_build() {
  local label="$1" sim="$2" deadline=$(( $(date +%s) + TIMEOUT )) st
  while :; do
    st="$(curl -s -m 30 "$BASE/core/v1/simulator/status?simulator_id=$sim" | _json_get status)"
    case "$st" in
      completed|complete|succeeded|ready) log "$label build status=$st — done" ; break ;;
      failed|error)  die "$label build FAILED (status=$st, sim=$sim)" ;;
      *) log "$label build status=${st:-pending}… (sim=$sim)" ;;
    esac
    [ "$(date +%s)" -ge "$deadline" ] && die "$label build did not finish within ${TIMEOUT}s (sim=$sim)"
    sleep 15
  done
}

# Register one engine via /core/v1/simulator/upload {repo,commit,branch}; poll the
# build; persist the sim id. $1=label $2=repo $3=commit $4=branch $5=outfile.
# Sets global REG_SIM_ID. sms-api routes by repo (v2ecoli->Ray, vEcoli->Nextflow).
_register_engine() {
  local label="$1" repo="$2" commit="$3" branch="$4" outfile="$5"
  [ -n "$commit" ] || die "$label: could not resolve a commit (check the spec)"
  log "registering $label: repo=$repo commit=${commit:0:8} branch=${branch:-?}"
  local body
  body="$(_post "$BASE/core/v1/simulator/upload" -H 'Content-Type: application/json' \
        -d "{\"git_repo_url\":\"$repo\",\"git_commit_hash\":\"$commit\",\"git_branch\":\"$branch\"}")" \
    || log "$label upload POST returned non-2xx (body below); continuing to parse"
  REG_SIM_ID="$(_json_get simulator_id <<<"$body")"
  [ -n "$REG_SIM_ID" ] || die "no simulator_id in $label upload response: $body"
  mkdir -p "$OUT" ; echo "$REG_SIM_ID" >"$outfile"
  log "$label simulator_id=$REG_SIM_ID (persisted to $outfile)"
  _poll_build "$label" "$REG_SIM_ID"
}

cmd_register() {
  _preflight
  if [ -n "$SPEC" ]; then
    _load_spec_engines
    if [ -n "$V2_SIM" ]; then
      log "v2ecoli: using --v2-sim $V2_SIM (skipping registration)"
    else
      _register_engine v2ecoli "$V2_REPO" "$V2_COMMIT" "$V2_BRANCH" "$OUT/.v2_sim_id"
      V2_SIM="$REG_SIM_ID"
    fi
    if [ "$VE_SIM_GIVEN" -eq 1 ]; then
      log "vEcoli: using --ve-sim $VE_SIM (skipping registration)"
    elif [ "$VE_ENGINE" = "upstream-wrapper" ]; then
      # The vEcoli side is the PRISTINE upstream fork WRAPPED INSIDE the v2ecoli
      # image (cloned at build via the spec's vecoli.{repo,commit} build-args) and
      # run as a Ray composite=vecoli job — so it shares the v2ecoli simulator id.
      # ONE image registration covers both engines; no Nextflow registration.
      VE_SIM="$V2_SIM"
      mkdir -p "$OUT" ; echo "$VE_SIM" >"$OUT/.ve_sim_id"
      log "vEcoli (upstream-wrapper): reusing v2ecoli sim id $VE_SIM (composite=vecoli on the same image)"
    else
      _register_engine vEcoli "$VE_REPO" "$VE_COMMIT" "$VE_BRANCH" "$OUT/.ve_sim_id"
      VE_SIM="$REG_SIM_ID"
    fi
    return 0
  fi
  # legacy (no spec): register only v2ecoli from CLI / local HEAD
  COMMIT="${COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)}"
  BRANCH="${BRANCH:-$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)}"
  [ -n "$COMMIT" ] || die "could not resolve a commit (pass --commit)"
  _register_engine v2ecoli "$REPO" "$COMMIT" "$BRANCH" "$OUT/.v2_sim_id"
  V2_SIM="$REG_SIM_ID"
}

cmd_launch() {
  [ "$DRY" -eq 1 ] || _preflight
  [ -n "$SPEC" ] && [ "$VE_ENGINE" = "nextflow" ] && [ -z "$V2_REPO" ] && _load_spec_engines
  # resolve sim ids: explicit flag > persisted register output
  [ -n "$V2_SIM" ] || { [ -f "$OUT/.v2_sim_id" ] && V2_SIM="$(cat "$OUT/.v2_sim_id")"; }
  [ -n "$V2_SIM" ] || { [ "$DRY" -eq 1 ] && V2_SIM="<V2_SIM>"; }
  [ -n "$V2_SIM" ] || die "launch needs --v2-sim ID (run 'register' first)"
  if [ "$VE_SIM_GIVEN" -eq 0 ] && [ -z "$VE_SIM" ] && [ -f "$OUT/.ve_sim_id" ]; then
    VE_SIM="$(cat "$OUT/.ve_sim_id")"
  fi
  # upstream-wrapper: the vEcoli side runs as composite=vecoli on the SAME image,
  # so it shares the v2ecoli simulator id (no separate registration).
  if [ "$VE_ENGINE" = "upstream-wrapper" ] && [ "$VE_SIM_GIVEN" -eq 0 ]; then
    VE_SIM="$V2_SIM"
  fi
  [ -n "$VE_SIM" ] || VE_SIM="47"   # legacy nextflow fallback when no spec/registration
  mkdir -p "$OUT"
  # Build the launch rows: name<TAB>config<TAB>seeds<TAB>gens .
  local rows
  if [ -n "$SPEC" ]; then
    [ -n "$V2_REPO" ] || _load_spec_engines   # for provenance in experiments.json
    rows="$("$PYJ" "$REPO_ROOT/scripts/_read_spec.py" "$SPEC" conditions "$SEEDS" "$GENS")" \
      || die "could not parse conditions from $SPEC"
    log "launch (spec=$SPEC): v2-sim=$V2_SIM ve-sim=$VE_SIM"
  else
    rows="$(local c; for c in $CONDITIONS; do printf '%s\tcond_%s.json\t%s\t%s\n' "$c" "$c" "$SEEDS" "$GENS"; done)"
    log "launch (legacy): v2-sim=$V2_SIM ve-sim=$VE_SIM seeds=$SEEDS gens=$GENS conditions: $CONDITIONS"
  fi
  log "vecoli engine route: $VE_ENGINE"
  [ "$DRY" -eq 1 ] && log "DRY-RUN: printing the POSTs that WOULD be made (no requests sent)"
  : >"$OUT/.exp.tsv"
  local cond cfg cseeds cgens v2_url ve_url v2_body ve_body v2_id ve_id
  while IFS=$'\t' read -r cond cfg cseeds cgens; do
    [ -n "$cond" ] || continue
    # clear stale K8s resources that collide by Nextflow config name (idempotent);
    # only the legacy Nextflow route creates them. Skip on dry-run.
    if [ "$DRY" -eq 0 ] && [ "$VE_ENGINE" = "nextflow" ] && command -v kubectl >/dev/null 2>&1; then
      kubectl delete job,configmap -n "$K8S_NS" \
        "nf-cond-${cond}" "nf-cond-${cond}-config" --ignore-not-found >/dev/null 2>&1 || true
    fi
    v2_url="$BASE/api/v1/simulations?simulator_id=${V2_SIM}&experiment_id=${TAG}-v2-${cond}&composite=v2ecoli&condition=${cond}&num_seeds=${cseeds}&num_generations=${cgens}&max_generations=${cgens}&simulation_config_filename=${CONF}"
    if [ "$VE_ENGINE" = "upstream-wrapper" ]; then
      # vEcoli as a Ray composite=vecoli job on the SAME v2ecoli image/sim id —
      # emits v2ecoli-format zarr (vecoli_seed*.zarr), NOT Nextflow parquet.
      ve_url="$BASE/api/v1/simulations?simulator_id=${VE_SIM}&experiment_id=${TAG}-ve-${cond}&composite=vecoli&condition=${cond}&num_seeds=${cseeds}&num_generations=${cgens}&max_generations=${cgens}&simulation_config_filename=${CONF}"
    else
      ve_url="$BASE/api/v1/simulations?simulator_id=${VE_SIM}&experiment_id=${TAG}-ve-${cond}&num_seeds=${cseeds}&num_generations=${cgens}&simulation_config_filename=${cfg}"
    fi
    if [ "$DRY" -eq 1 ]; then
      printf '[dry-run] %-10s POST (v2ecoli) %s\n' "$cond" "$v2_url"
      printf '[dry-run] %-10s POST (vecoli ) %s\n' "$cond" "$ve_url"
      continue
    fi
    v2_body="$(_post "$v2_url")" ; v2_id="$(_json_get experiment_id <<<"$v2_body")"
    ve_body="$(_post "$ve_url")" ; ve_id="$(_json_get experiment_id <<<"$ve_body")"
    [ -n "$v2_id" ] || die "v2ecoli submit for '$cond' returned no experiment_id: $v2_body"
    [ -n "$ve_id" ] || die "vEcoli submit for '$cond' returned no experiment_id: $ve_body"
    printf '%s\t%s\t%s\t%s\t%s\n' "$cond" "$v2_id" "$ve_id" "$cseeds" "$cgens" >>"$OUT/.exp.tsv"
    printf '[harness]   %-10s seeds=%-2s gens=%-2s cfg=%-18s v2=%-30s ve=%-30s\n' \
      "$cond" "$cseeds" "$cgens" "$cfg" "$v2_id" "$ve_id"
    sleep 2
  done <<<"$rows"
  if [ "$DRY" -eq 1 ]; then
    log "DRY-RUN complete (no $EXP_JSON written)"
    return 0
  fi
  V2_SIM="$V2_SIM" VE_SIM="$VE_SIM" SEEDS="$SEEDS" GENS="$GENS" \
    V2_REPO="$V2_REPO" V2_COMMIT="$V2_COMMIT" V2_BRANCH="$V2_BRANCH" \
    VE_REPO="$VE_REPO" VE_COMMIT="$VE_COMMIT" VE_BRANCH="$VE_BRANCH" \
    "$PYJ" "$REPO_ROOT/scripts/_write_experiments.py" "$OUT/.exp.tsv" "$EXP_JSON" \
    || die "failed to write $EXP_JSON"
  rm -f "$OUT/.exp.tsv"
  log "wrote $EXP_JSON — next: bash $0 wait  (then report)"
}

cmd_wait() {
  _preflight_creds
  [ -f "$EXP_JSON" ] || die "no $EXP_JSON — run 'launch' first"
  # emit: cond<TAB>v2_exp<TAB>ve_exp<TAB>expected_seeds  (per-condition from meta;
  # falls back to the top-level/default seeds, then the global SEEDS).
  local rows
  rows="$("$PYJ" -c "import json;d=json.load(open('$EXP_JSON'));c=d.get('conditions',d);m=d.get('meta',{});
df=int(d.get('seeds',$SEEDS) or $SEEDS)
[print('\t'.join([k,v[0],v[1],str(int(m.get(k,{}).get('seeds',df) or df))])) for k,v in c.items()]")" \
    || die "could not parse $EXP_JSON"
  log "wait: polling S3 for per-condition seed completion (timeout ${TIMEOUT}s)"
  local deadline=$(( $(date +%s) + TIMEOUT ))
  while :; do
    local all_done=1 missing="" cond v2_exp ve_exp need ls v2done vedone
    while IFS=$'\t' read -r cond v2_exp ve_exp need; do
      [ -n "$cond" ] || continue
      need="${need:-$SEEDS}"
      # v2ecoli: distinct seed zarr stores that actually have chunk objects (.../c/)
      ls="$(aws s3 ls --recursive "s3://$BUCKET/$PREFIX/$v2_exp/" 2>/dev/null)"
      v2done="$(printf '%s\n' "$ls" | grep -E '_seed[0-9]+\.zarr/.*/c/' \
                | grep -oE '_seed[0-9]+\.zarr' | sort -u | wc -l | tr -d ' ')"
      ls="$(aws s3 ls --recursive "s3://$BUCKET/$PREFIX/$ve_exp/" 2>/dev/null)"
      if [ "$VE_ENGINE" = "upstream-wrapper" ]; then
        # vEcoli (composite=vecoli): distinct vecoli_seedNN.zarr stores with chunks.
        vedone="$(printf '%s\n' "$ls" | grep -E 'vecoli_seed[0-9]+\.zarr/.*/c/' \
                  | grep -oE 'vecoli_seed[0-9]+\.zarr' | sort -u | wc -l | tr -d ' ')"
      else
        # legacy Nextflow: distinct lineage_seed under parquet history (fallback: any history pq)
        vedone="$(printf '%s\n' "$ls" | grep -E "cond_${cond}/.*\.pq" \
                  | grep -oE 'lineage_seed=[0-9]+' | sort -u | wc -l | tr -d ' ')"
        if [ "${vedone:-0}" -eq 0 ] && printf '%s\n' "$ls" | grep -qE "cond_${cond}/.*history/.*\.pq"; then
          vedone="$need"
        fi
      fi
      printf '[harness]   %-10s v2 %s/%s  ve %s/%s\n' "$cond" "${v2done:-0}" "$need" "${vedone:-0}" "$need"
      if [ "${v2done:-0}" -lt "$need" ] || [ "${vedone:-0}" -lt "$need" ]; then
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
  # upstream-wrapper: BOTH engines emit v2ecoli-format zarr to S3, so the report
  # reads both via the zarr reader (--pbg-vs-pbg). Legacy nextflow stays default.
  local extra=""
  [ "$VE_ENGINE" = "upstream-wrapper" ] && extra="--pbg-vs-pbg"
  "$PY" "$REPO_ROOT/scripts/comparison_report_card.py" --only "$ONLY" --out "$OUT" $extra
}

# ---- arg parsing ----
# print the whole leading comment block (everything after the shebang up to the
# first non-comment line) so --help stays in sync however long the header grows.
_print_help() { awk 'NR>1{ if(/^#/){sub(/^# ?/,"");print} else exit }' "${BASH_SOURCE[0]}"; }
SUB="${1:-}" ; shift || true
[ -n "$SUB" ] || die "usage: $0 {register|launch|wait|report|all} [flags] (see --help)"
case "$SUB" in -h|--help|help) _print_help ; exit 0 ;; esac
while [ $# -gt 0 ]; do
  case "$1" in
    --spec) SPEC="$2"; SPEC_GIVEN=1; shift 2;;
    --v2-sim) V2_SIM="$2"; shift 2;;
    --ve-sim) VE_SIM="$2"; VE_SIM_GIVEN=1; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --gens) GENS="$2"; shift 2;;
    --conditions) CONDITIONS="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --commit) COMMIT="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) _print_help; exit 0;;
    *) die "unknown flag: $1";;
  esac
done
# spec resolution: explicit --spec wins; otherwise auto-use comparison_spec.json
# if it exists (committed manifest -> reproducible by default). Empty = legacy CLI.
if [ "$SPEC_GIVEN" -eq 0 ] && [ -z "$SPEC" ] && [ -f "$DEFAULT_SPEC" ]; then
  SPEC="$DEFAULT_SPEC"
fi
# Resolve the vecoli engine route once so every subcommand (register/launch/wait/
# report) agrees on it. With a spec it comes from vecoli_engine (default
# upstream-wrapper); without one, the legacy Nextflow route stands.
if [ -n "$SPEC" ] && [ -f "$SPEC" ]; then
  VE_ENGINE="$("$PYJ" "$REPO_ROOT/scripts/_read_spec.py" "$SPEC" vecoli-engine 2>/dev/null)"
  [ -n "$VE_ENGINE" ] || VE_ENGINE="upstream-wrapper"
fi
[ -z "$SPEC" ] || log "using manifest: $SPEC (vecoli engine: $VE_ENGINE)"

case "$SUB" in
  register) cmd_register ;;
  launch)   cmd_launch ;;
  wait)     cmd_wait ;;
  report)   cmd_report ;;
  all)      # with a spec, register handles BOTH engines (skipping any whose sim id
            # was supplied); without one, only register when no v2 sim is given.
            if [ -n "$SPEC" ]; then
              { [ -n "$V2_SIM" ] && [ "$VE_SIM_GIVEN" -eq 1 ]; } || cmd_register
            else
              [ -n "$V2_SIM" ] || cmd_register
            fi
            cmd_launch ; cmd_wait ; cmd_report ;;
  *) die "unknown subcommand: $SUB (register|launch|wait|report|all)" ;;
esac
