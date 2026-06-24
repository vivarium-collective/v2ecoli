#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ray-batch-entrypoint.sh
#
# Ray head/worker bootstrap for an AWS Batch MULTI-NODE PARALLEL (MNP) job.
# Batch gang-schedules N nodes for one job and sets these env vars in every node's
# container; this script uses them to form a transient Ray cluster, run the job on
# the main node, and let Batch tear the whole allocation down when main exits.
#
#   AWS_BATCH_JOB_NODE_INDEX                       this node's index (0..N-1)
#   AWS_BATCH_JOB_MAIN_NODE_INDEX                  the main node's index (usually 0)
#   AWS_BATCH_JOB_NUM_NODES                        total nodes in the job
#   AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS   main node IP (child nodes only)
#
# NETWORKING: AWS Batch MNP jobs run in the ECS `awsvpc` network mode, so each node's
# container gets its OWN elastic network interface with a routable VPC private IP in
# the host's subnet — exactly what Ray's node-to-node raylets need. A node's own
# address is its ENI IP (`hostname -I`), NOT the instance metadata IP.
#
# OBSERVABILITY:
#   - High-signal lines go to stdout → CloudWatch (keep this stream small).
#   - The full Ray session logs (/tmp/ray/session_latest/logs) are tar'd to S3 on exit
#     (RAY_LOG_S3_PREFIX), which is ~25x cheaper than CloudWatch ingestion for bulk.
#   - RAY_DEBUG_HOLD keeps the head alive after the job so you can SSM in and inspect a
#     live cluster; it prints the exact SSM commands.
#
# Env knobs (all optional, injected by lib/ray-batch-stack.ts):
#   RAY_JOB_CMD              workload to run on the head (required at run time)
#   RAY_LOG_S3_PREFIX        e.g. s3://bucket/ray-logs/  (empty disables S3 upload)
#   RAY_REPORT_PATH          if this file exists after the job, it's uploaded as report.json
#   RAY_STAGE_S3             S3 URI of an input dataset to fetch on EVERY node before the
#                            job (e.g. a precomputed ParCa cache the workers read). Paired
#                            with RAY_STAGE_DIR; the image stays data-free.
#   RAY_STAGE_DIR            local dir to sync RAY_STAGE_S3 into (e.g. the v2ecoli out/cache).
#                            For `--composite vecoli` (the pristine upstream wrapper) this is
#                            the dir holding the UPSTREAM-built simData.cPickle that the run's
#                            --cache-dir points at (built by scripts/build_upstream_parca.py).
#   V2E_BUILD_UPSTREAM_PARCA when '1', build the upstream ParCa locally on EVERY node BEFORE
#                            the job (scripts/build_upstream_parca.py --outdir $V2E_VECOLI_DIR/out
#                            --copy-to $RAY_STAGE_DIR) instead of staging it from S3. Use for a
#                            one-shot/dev image; the normal flow is to build it ONCE offline,
#                            upload to RAY_STAGE_S3, and let stage_inputs fetch it (cheaper than
#                            re-fitting ParCa on each node). Heavy (minutes). Honors PARCA_CPUS.
#   RAY_OUT_DIR              local dir the workload writes results to (e.g. the v2ecoli
#                            .pbg/runs/... tree). Synced to RAY_OUT_S3 from EVERY node —
#                            tasks run on the workers, so each node ships its own outputs —
#                            periodically during the job and once at teardown, so results
#                            survive cluster teardown. Paired with RAY_OUT_S3.
#   RAY_OUT_S3               S3 URI to sync RAY_OUT_DIR to (per node; keys don't collide)
#   RAY_OUT_SYNC_INTERVAL    seconds between periodic output syncs (default 30)
#   RAY_DEBUG_HOLD           '0' (off) | '1' (hold on failure) | 'always'
#   RAY_DEBUG_HOLD_SECONDS   hold duration (default 1800)
#   RAY_BACKEND_LOG_LEVEL    Ray C++ backend log level (default 'warning' — quieter)
#   RAY_GCS_PORT             GCS port (default 6379)
#
# On the head, RAY_ADDRESS is exported to this cluster's GCS before RAY_JOB_CMD runs, so a
# workload that calls a plain `ray.init()` (e.g. v2ecoli's run_seeds_parallel) attaches to
# THIS multi-node cluster instead of starting a local single-node one.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NODE_INDEX="${AWS_BATCH_JOB_NODE_INDEX:?must run as an AWS Batch multi-node job}"
MAIN_INDEX="${AWS_BATCH_JOB_MAIN_NODE_INDEX:-0}"
NUM_NODES="${AWS_BATCH_JOB_NUM_NODES:?}"
GCS_PORT="${RAY_GCS_PORT:-6379}"

# Quiet the chatty backend logs by default (keeps the CloudWatch stream small).
export RAY_BACKEND_LOG_LEVEL="${RAY_BACKEND_LOG_LEVEL:-warning}"

# This node's own routable awsvpc ENI IP. In AWS Batch awsvpc mode the container also
# has an ECS-metadata link-local address (169.254.x.x) which `hostname -I` may list
# first — Ray must advertise the ROUTABLE VPC IP, so exclude link-local and loopback.
SELF_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' \
  | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' \
  | grep -vE '^(169\.254\.|127\.)' | head -n1)"
if [[ -z "${SELF_IP}" ]]; then
  echo "[ray-batch] FATAL: could not determine a routable IP from 'hostname -I'" >&2
  exit 1
fi

RAY_LOG_S3_PREFIX="${RAY_LOG_S3_PREFIX:-}"
BASE_JOB_ID="${AWS_BATCH_JOB_ID%%#*}"          # strip the '#<node>' suffix on child nodes

# ── Helpers ──────────────────────────────────────────────────────────────────
imds_instance_id() {
  local t
  t="$(curl -sS -X PUT 'http://169.254.169.254/latest/api/token' \
        -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)"
  curl -sS -H "X-aws-ec2-metadata-token: ${t}" \
       'http://169.254.169.254/latest/meta-data/instance-id' 2>/dev/null || echo 'unknown'
}

# Best-effort: never let log shipping fail the job.
upload_logs() {
  [[ -z "${RAY_LOG_S3_PREFIX}" ]] && return 0
  local archive="/tmp/ray-logs-${NODE_INDEX}.tgz"
  local dest="${RAY_LOG_S3_PREFIX%/}/${BASE_JOB_ID:-unknown}/node-${NODE_INDEX}.tgz"
  # -h dereferences the session_latest symlink.
  tar czhf "${archive}" -C /tmp/ray session_latest/logs 2>/dev/null || return 0
  if command -v aws >/dev/null 2>&1; then
    if aws s3 cp "${archive}" "${dest}" >/dev/null 2>&1; then
      echo "[ray-batch] node ${NODE_INDEX} logs -> ${dest}"
    else
      echo "[ray-batch] WARN: log upload to ${dest} failed" >&2
    fi
  else
    echo "[ray-batch] WARN: aws CLI not in image; skipping S3 log upload" >&2
  fi
}

# Best-effort: upload the workload's metrics report next to the logs, if it wrote one.
upload_report() {
  local path="${RAY_REPORT_PATH:-}"
  [[ -z "${RAY_LOG_S3_PREFIX}" || -z "${path}" || ! -f "${path}" ]] && return 0
  local dest="${RAY_LOG_S3_PREFIX%/}/${BASE_JOB_ID:-unknown}/report.json"
  if command -v aws >/dev/null 2>&1 && aws s3 cp "${path}" "${dest}" >/dev/null 2>&1; then
    echo "[ray-batch] report -> ${dest}"
  else
    echo "[ray-batch] WARN: report upload to ${dest} skipped" >&2
  fi
}

# Stage an input dataset from S3 to a local dir on EVERY node before the job runs (e.g. a
# precomputed ParCa cache that all Ray workers read). Keeps model data out of the image.
# Fails hard — if it's configured, the job needs it.
stage_inputs() {
  [[ -z "${RAY_STAGE_S3:-}" || -z "${RAY_STAGE_DIR:-}" ]] && return 0
  command -v aws >/dev/null 2>&1 || { echo "[ray-batch] FATAL: aws CLI needed for input staging" >&2; exit 1; }
  echo "[ray-batch] node ${NODE_INDEX}: staging ${RAY_STAGE_S3} -> ${RAY_STAGE_DIR}"
  mkdir -p "${RAY_STAGE_DIR}"
  if ! aws s3 sync "${RAY_STAGE_S3}" "${RAY_STAGE_DIR}" --only-show-errors; then
    echo "[ray-batch] FATAL: input staging from ${RAY_STAGE_S3} failed" >&2
    exit 1
  fi
}

# Output capture (RAY_OUT_DIR -> RAY_OUT_S3), symmetric to stage_inputs().
#
# WHY PER-NODE: the workload's tasks run on the WORKERS (the head is --num-cpus=0),
# so each worker writes its share of the results (e.g. v2ecoli seed_NN/store.zarr)
# to its OWN local RAY_OUT_DIR. A head-only sync would miss all of that. Every node
# therefore syncs its own outputs to the shared RAY_OUT_S3 prefix — the keys don't
# collide across nodes (different seeds), so the union reconstructs the full result.
#
# WHY PERIODIC + FINAL: AWS Batch SIGTERMs the child nodes when the head exits, then
# SIGKILLs after a short grace window. A from-scratch sync of a large zarr tree may
# not finish in that window. start_output_sync() uploads incrementally while the job
# runs, so the final stage_outputs() on teardown is a small delta that fits the grace.
OUTPUT_SYNC_PID=""

# Begin periodic background sync of RAY_OUT_DIR -> RAY_OUT_S3. No-op unless both are
# set and the aws CLI is present. Best-effort: never fails the job on its own.
start_output_sync() {
  [[ -z "${RAY_OUT_S3:-}" || -z "${RAY_OUT_DIR:-}" ]] && return 0
  if ! command -v aws >/dev/null 2>&1; then
    echo "[ray-batch] WARN: aws CLI not in image; no S3 output sync" >&2
    return 0
  fi
  local interval="${RAY_OUT_SYNC_INTERVAL:-30}"
  mkdir -p "${RAY_OUT_DIR}"
  ( while true; do
      aws s3 sync "${RAY_OUT_DIR}" "${RAY_OUT_S3}" --only-show-errors >/dev/null 2>&1 || true
      sleep "${interval}"
    done ) &
  OUTPUT_SYNC_PID=$!
  echo "[ray-batch] node ${NODE_INDEX}: periodic output sync ${RAY_OUT_DIR} -> ${RAY_OUT_S3} (every ${interval}s)"
}

# Final authoritative sync: stop the periodic loop, then do one last aws s3 sync.
# Returns non-zero on a real upload failure so the head can fail an otherwise-
# successful job (the results are the point). Workers call it best-effort.
stage_outputs() {
  if [[ -n "${OUTPUT_SYNC_PID}" ]]; then
    kill "${OUTPUT_SYNC_PID}" 2>/dev/null || true
    wait "${OUTPUT_SYNC_PID}" 2>/dev/null || true
    OUTPUT_SYNC_PID=""
  fi
  [[ -z "${RAY_OUT_S3:-}" || -z "${RAY_OUT_DIR:-}" ]] && return 0
  command -v aws >/dev/null 2>&1 || return 0
  if [[ ! -d "${RAY_OUT_DIR}" ]]; then
    echo "[ray-batch] node ${NODE_INDEX}: no ${RAY_OUT_DIR}; nothing to upload" >&2
    return 0
  fi
  echo "[ray-batch] node ${NODE_INDEX}: final output sync ${RAY_OUT_DIR} -> ${RAY_OUT_S3}"
  if ! aws s3 sync "${RAY_OUT_DIR}" "${RAY_OUT_S3}" --only-show-errors; then
    echo "[ray-batch] FATAL: output staging to ${RAY_OUT_S3} failed (node ${NODE_INDEX})" >&2
    return 1
  fi
}

# Worker teardown: capture this node's outputs to S3, then ship its logs — exactly
# once. Batch SIGTERMs the worker (TERM trap runs, then exits, which fires the EXIT
# trap); the guard keeps the work from running twice. Best-effort: a worker's exit
# code doesn't affect the MNP job result (only the main node's does).
_WORKER_FINALIZED=0
_finalize_worker() {
  [[ "${_WORKER_FINALIZED}" == 1 ]] && return 0
  _WORKER_FINALIZED=1
  stage_outputs || true
  upload_logs
}

# Keep the head alive for interactive debugging, printing copy-paste SSM commands.
debug_hold() {
  local rc="$1"
  local mode="${RAY_DEBUG_HOLD:-0}"
  local secs="${RAY_DEBUG_HOLD_SECONDS:-1800}"
  local do_hold=0
  case "${mode}" in
    always)      do_hold=1 ;;
    1|true|yes)  [[ "${rc}" -ne 0 ]] && do_hold=1 ;;
  esac
  [[ "${do_hold}" -eq 1 ]] || return 0

  local iid; iid="$(imds_instance_id)"
  cat >&2 <<MSG
=========================== RAY DEBUG HOLD ===========================
 job rc=${rc}; holding the HEAD container for ${secs}s for debugging.
 instance: ${iid}    head ENI IP: ${SELF_IP}

 shell into the Ray container:
   aws ssm start-session --target ${iid}
   sudo docker ps                      # find the ray task container
   sudo docker exec -it <container-id> bash
     # then: ray status / ray list tasks / ray logs

 forward the Ray dashboard to localhost:8265 (awsvpc: target the ENI IP):
   aws ssm start-session --target ${iid} \\
     --document-name AWS-StartPortForwardingSessionToRemoteHost \\
     --parameters host=${SELF_IP},portNumber=8265,localPortNumber=8265
======================================================================
MSG
  sleep "${secs}"
}

# Optional: build the upstream-vEcoli ParCa simData on this node (instead of, or
# in addition to, staging a precomputed one from S3). For `--composite vecoli`
# the external wrapper needs an UPSTREAM-MASTER-built simData.cPickle; the normal
# flow stages it via stage_inputs, but a self-building image can set
# V2E_BUILD_UPSTREAM_PARCA=1 to fit it here. Heavy (minutes). Fails hard.
build_upstream_parca() {
  [[ "${V2E_BUILD_UPSTREAM_PARCA:-0}" == "1" ]] || return 0
  local vd="${V2E_VECOLI_DIR:-/app/vEcoli}"
  local dest="${RAY_STAGE_DIR:-/app/v2ecoli/out/cache}"
  echo "[ray-batch] node ${NODE_INDEX}: building upstream ParCa -> ${dest}/simData.cPickle"
  if ! python /app/v2ecoli/scripts/build_upstream_parca.py \
        --outdir "${vd}/out" --cpus "${PARCA_CPUS:-8}" --copy-to "${dest}"; then
    echo "[ray-batch] FATAL: upstream ParCa build failed (node ${NODE_INDEX})" >&2
    exit 1
  fi
}

# ── Stage inputs on every node, then head vs. worker ─────────────────────────
stage_inputs           # all nodes (head + workers) need the data before Ray scheduling
build_upstream_parca   # optional self-build of the upstream simData (gated; see above)

if [[ "${NODE_INDEX}" == "${MAIN_INDEX}" ]]; then
  trap upload_logs EXIT                       # ship session logs no matter how we exit
  echo "[ray-batch] HEAD on node ${NODE_INDEX} (${SELF_IP}), expecting ${NUM_NODES} node(s)"
  ray start --head \
    --node-ip-address="${SELF_IP}" \
    --port="${GCS_PORT}" \
    --num-cpus=0 \
    --dashboard-host=0.0.0.0

  registered=0
  for _ in $(seq 1 60); do
    registered="$(ray status 2>/dev/null | grep -Eco 'node_[0-9a-f]{16,}' || true)"
    [[ "${registered:-0}" -ge "${NUM_NODES}" ]] && break
    sleep 5
  done
  echo "[ray-batch] cluster formed (${registered:-0}/${NUM_NODES}); running job"

  # Attach a plain `ray.init()` in the workload to THIS cluster (not a new local one).
  export RAY_ADDRESS="${SELF_IP}:${GCS_PORT}"

  set +e
  bash -lc "${RAY_JOB_CMD:?RAY_JOB_CMD not set}"
  rc=$?
  set -e

  # Capture results to S3 before teardown. On an otherwise-successful run, a failed
  # upload fails the job; if the job already failed, keep its rc but still try.
  if ! stage_outputs && [[ "${rc}" -eq 0 ]]; then
    rc=1
  fi

  upload_report                               # ship report.json if the job wrote one
  ray stop || true
  debug_hold "${rc}"                          # optional interactive window
  echo "[ray-batch] job exited rc=${rc}"
  exit "${rc}"                                 # main exit ends the MNP job; EXIT trap uploads
else
  MAIN_IP="${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS:?}"
  # On Batch stopping this node (SIGTERM), ship this worker's OUTPUTS then its logs.
  trap '_finalize_worker; exit 0' TERM INT
  trap _finalize_worker EXIT
  echo "[ray-batch] WORKER ${NODE_INDEX} (${SELF_IP}) -> head ${MAIN_IP}:${GCS_PORT}"
  # Start Ray as a daemon (not --block) so we can trap termination and upload logs.
  ray start --address="${MAIN_IP}:${GCS_PORT}" --node-ip-address="${SELF_IP}"
  # Tasks (e.g. v2ecoli seeds) run HERE and write results to this node's RAY_OUT_DIR;
  # upload them incrementally so the teardown sync is a small delta within the grace window.
  start_output_sync
  echo "[ray-batch] worker ${NODE_INDEX} joined; waiting for the job to finish"
  while true; do sleep 30; done               # Batch SIGTERMs us when the head exits
fi
