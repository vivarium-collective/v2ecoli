#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# batch-array-entrypoint.sh
#
# Entrypoint for one child of an AWS Batch ARRAY job. Unlike the multi-node
# Ray cluster (ray-batch-entrypoint.sh), each array child is a fully
# independent, single-container task — no cluster formation, no head/worker
# roles, no inter-task networking. AWS Batch's OWN managed compute environment
# handles fleet sizing per queue depth (no fixed node-count guess).
#
#   AWS_BATCH_JOB_ARRAY_INDEX   this child's index (0..arrayProperties.size-1)
#
# ARRAY_JOB_CMD is a shell string built by the submitter (viva-api). For
# per-child work (e.g. one seed per child), the submitter embeds a raw
# `$AWS_BATCH_JOB_ARRAY_INDEX` (or an arithmetic expansion of it) directly in
# that string — it is exported below before ARRAY_JOB_CMD runs, so the shell
# resolves it at container-start time, once this child actually knows its
# index. Nothing in this script is workload-aware; it only stages data in,
# runs the command it's given, and ships data/logs back out.
#
# OBSERVABILITY:
#   - High-signal lines go to stdout → CloudWatch (keep this stream small).
#   - Full session-adjacent logs, if any, are not collected here — a single
#     array child has no equivalent of Ray's per-session log tree; workload
#     stdout/stderr already goes to CloudWatch via the container's own logs.
#   - ARRAY_DEBUG_HOLD keeps a FAILED child alive so you can SSM in and
#     inspect it; it prints the exact SSM command.
#
# Env knobs (all optional except ARRAY_JOB_CMD, injected by
# lib/ray-batch-stack.ts's RayArrayJobDef and overridden per-submission):
#   ARRAY_JOB_CMD            workload to run for this child (required at run time)
#   ARRAY_STAGE_S3           S3 URI of an input dataset to fetch before the job
#                            (e.g. a precomputed ParCa cache). Paired with
#                            ARRAY_STAGE_DIR; the image stays data-free.
#   ARRAY_STAGE_DIR          local dir to sync ARRAY_STAGE_S3 into
#   ARRAY_OUT_DIR            local dir the workload writes results to. Synced to
#                            ARRAY_OUT_S3 periodically during the job and once at
#                            teardown, so results survive a Spot interruption's
#                            SIGTERM grace window. Paired with ARRAY_OUT_S3.
#   ARRAY_OUT_S3             S3 URI to sync ARRAY_OUT_DIR to (per child; each
#                            child owns a disjoint slice of the output, e.g. one
#                            seed's zarr/parquet files, so children's uploads
#                            never collide)
#   ARRAY_OUT_SYNC_INTERVAL  seconds between periodic output syncs (default 30)
#   ARRAY_REPORT_PATH        if this file exists after the job, it's uploaded
#   ARRAY_LOG_S3_PREFIX      s3:// prefix for this child's report.json (empty disables)
#   ARRAY_DEBUG_HOLD         '0' (off) | '1' (hold on failure) | 'always'
#   ARRAY_DEBUG_HOLD_SECONDS hold duration (default 1800)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ARRAY_INDEX="${AWS_BATCH_JOB_ARRAY_INDEX:?must run as an AWS Batch array job}"
export AWS_BATCH_JOB_ARRAY_INDEX="${ARRAY_INDEX}"

BASE_JOB_ID="${AWS_BATCH_JOB_ID%%#*}"          # strip the '#<array-index>' suffix

# ── Helpers ──────────────────────────────────────────────────────────────────
imds_instance_id() {
  local t
  t="$(curl -sS -X PUT 'http://169.254.169.254/latest/api/token' \
        -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)"
  curl -sS -H "X-aws-ec2-metadata-token: ${t}" \
       'http://169.254.169.254/latest/meta-data/instance-id' 2>/dev/null || echo 'unknown'
}

# Best-effort: upload the workload's metrics report next to the logs, if it wrote one.
upload_report() {
  local path="${ARRAY_REPORT_PATH:-}"
  [[ -z "${ARRAY_LOG_S3_PREFIX:-}" || -z "${path}" || ! -f "${path}" ]] && return 0
  local dest="${ARRAY_LOG_S3_PREFIX%/}/${BASE_JOB_ID:-unknown}/${ARRAY_INDEX}/report.json"
  if command -v aws >/dev/null 2>&1 && aws s3 cp "${path}" "${dest}" >/dev/null 2>&1; then
    echo "[batch-array] child ${ARRAY_INDEX}: report -> ${dest}"
  else
    echo "[batch-array] WARN: report upload to ${dest} skipped" >&2
  fi
}

# Stage an input dataset from S3 to a local dir before the job runs (e.g. a
# precomputed ParCa cache). Fails hard -- if it's configured, the job needs it.
stage_inputs() {
  [[ -z "${ARRAY_STAGE_S3:-}" || -z "${ARRAY_STAGE_DIR:-}" ]] && return 0
  command -v aws >/dev/null 2>&1 || { echo "[batch-array] FATAL: aws CLI needed for input staging" >&2; exit 1; }
  echo "[batch-array] child ${ARRAY_INDEX}: staging ${ARRAY_STAGE_S3} -> ${ARRAY_STAGE_DIR}"
  mkdir -p "${ARRAY_STAGE_DIR}"
  if ! aws s3 sync "${ARRAY_STAGE_S3}" "${ARRAY_STAGE_DIR}" --only-show-errors; then
    echo "[batch-array] FATAL: input staging from ${ARRAY_STAGE_S3} failed" >&2
    exit 1
  fi
}

# Output capture (ARRAY_OUT_DIR -> ARRAY_OUT_S3), symmetric to stage_inputs().
#
# WHY PERIODIC + FINAL: AWS Batch SIGTERMs a child on Spot interruption or
# array-job cancellation, then SIGKILLs after a short grace window. A
# from-scratch sync of a full generation sweep may not finish in that window.
# start_output_sync() uploads incrementally while the job runs, so the final
# stage_outputs() on teardown is a small delta that fits the grace window.
OUTPUT_SYNC_PID=""

start_output_sync() {
  [[ -z "${ARRAY_OUT_S3:-}" || -z "${ARRAY_OUT_DIR:-}" ]] && return 0
  if ! command -v aws >/dev/null 2>&1; then
    echo "[batch-array] WARN: aws CLI not in image; no S3 output sync" >&2
    return 0
  fi
  local interval="${ARRAY_OUT_SYNC_INTERVAL:-30}"
  mkdir -p "${ARRAY_OUT_DIR}"
  ( while true; do
      aws s3 sync "${ARRAY_OUT_DIR}" "${ARRAY_OUT_S3}" --only-show-errors >/dev/null 2>&1 || true
      sleep "${interval}"
    done ) &
  OUTPUT_SYNC_PID=$!
  echo "[batch-array] child ${ARRAY_INDEX}: periodic output sync ${ARRAY_OUT_DIR} -> ${ARRAY_OUT_S3} (every ${interval}s)"
}

# Final authoritative sync: stop the periodic loop, then do one last aws s3 sync.
# Returns non-zero on a real upload failure so a failed upload fails the task
# even if the workload itself exited 0 (the results are the point).
stage_outputs() {
  if [[ -n "${OUTPUT_SYNC_PID}" ]]; then
    kill "${OUTPUT_SYNC_PID}" 2>/dev/null || true
    wait "${OUTPUT_SYNC_PID}" 2>/dev/null || true
    OUTPUT_SYNC_PID=""
  fi
  [[ -z "${ARRAY_OUT_S3:-}" || -z "${ARRAY_OUT_DIR:-}" ]] && return 0
  command -v aws >/dev/null 2>&1 || return 0
  if [[ ! -d "${ARRAY_OUT_DIR}" ]]; then
    echo "[batch-array] child ${ARRAY_INDEX}: no ${ARRAY_OUT_DIR}; nothing to upload" >&2
    return 0
  fi
  echo "[batch-array] child ${ARRAY_INDEX}: final output sync ${ARRAY_OUT_DIR} -> ${ARRAY_OUT_S3}"
  if ! aws s3 sync "${ARRAY_OUT_DIR}" "${ARRAY_OUT_S3}" --only-show-errors; then
    echo "[batch-array] FATAL: output staging to ${ARRAY_OUT_S3} failed (child ${ARRAY_INDEX})" >&2
    return 1
  fi
}

# Keep the container alive for interactive debugging, printing copy-paste SSM commands.
debug_hold() {
  local rc="$1"
  local mode="${ARRAY_DEBUG_HOLD:-0}"
  local secs="${ARRAY_DEBUG_HOLD_SECONDS:-1800}"
  local do_hold=0
  case "${mode}" in
    always)      do_hold=1 ;;
    1|true|yes)  [[ "${rc}" -ne 0 ]] && do_hold=1 ;;
  esac
  [[ "${do_hold}" -eq 1 ]] || return 0

  local iid; iid="$(imds_instance_id)"
  cat >&2 <<MSG
========================= BATCH ARRAY DEBUG HOLD =========================
 child ${ARRAY_INDEX} rc=${rc}; holding this container for ${secs}s for debugging.
 instance: ${iid}

 shell into the container:
   aws ssm start-session --target ${iid}
   sudo docker ps                      # find this array child's container
   sudo docker exec -it <container-id> bash
============================================================================
MSG
  sleep "${secs}"
}

# ── Stage inputs, run the workload, capture outputs ───────────────────────────
_FINALIZED=0
_finalize() {
  [[ "${_FINALIZED}" == 1 ]] && return 0
  _FINALIZED=1
  stage_outputs || true
}
trap _finalize EXIT

echo "[batch-array] child ${ARRAY_INDEX} starting (job ${BASE_JOB_ID})"
stage_inputs
start_output_sync

set +e
bash -lc "${ARRAY_JOB_CMD:?ARRAY_JOB_CMD not set}"
rc=$?
set -e

if ! stage_outputs && [[ "${rc}" -eq 0 ]]; then
  rc=1
fi
_FINALIZED=1   # stage_outputs already ran above; the EXIT trap must not repeat it

upload_report
debug_hold "${rc}"
echo "[batch-array] child ${ARRAY_INDEX} exited rc=${rc}"
exit "${rc}"
