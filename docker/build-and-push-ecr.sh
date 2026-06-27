#!/usr/bin/env bash
# Build the self-contained v2ecoli Ray-on-Batch image (root Dockerfile) and push it to
# ECR. Mirrors vEcoli's runscripts/container/build-and-push-ecr.sh so sms-api can drive
# it the same way it builds vecoli:{commit} — workload-owned image, app-orchestrated.
#
# Pushes the requested IMAGE_TAG (the v2ecoli commit sha) — runs reference the TRUE commit
# image, not a floating tag. sms-api derives a per-commit Ray-MNP job-def revision pointing
# at v2ecoli:<sha>, so no moving `:latest` is needed. Pass -t to ALSO push a second tag.
#
# The upstream vEcoli fork the image WRAPS (cloned into /app/vEcoli) is SPEC-DRIVEN:
# this script reads comparison_spec.json's vecoli.{repo,commit} via scripts/_read_spec.py
# (vecoli-fork mode) and passes them as --build-arg VECOLI_UPSTREAM_REPO/REF. So
# comparing a NEW fork = edit the spec's vecoli block + rerun this — no code change.
# If the spec is absent/unreadable the Dockerfile defaults (CovertLab/vEcoli@master) win.
#
# Usage: docker/build-and-push-ecr.sh [-i IMAGE_TAG] [-r REPO_NAME] [-R REGION] [-t EXTRA_TAG] [-s SPEC]
#   -i: image tag (default: short git sha, or "<user>-image" outside git).
#   -r: ECR repository name (default: v2ecoli).
#   -R: AWS region (default: $AWS_DEFAULT_REGION / $AWS_REGION / us-gov-west-1).
#   -t: OPTIONAL extra tag to also push (e.g. a deploy alias). Default: none.
#   -s: comparison spec (default: $ROOT/comparison_spec.json) — supplies the fork build-args.
#   -h: help.
#
# Build it on (or for) linux/amd64 — the MNP nodes are m5 (x86_64).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DEFAULT_TAG="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo "${USER:-build}-image")"
IMAGE_TAG="$DEFAULT_TAG"
REPO_NAME="v2ecoli"
AWS_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-gov-west-1}}"
EXTRA_TAG=""
PLATFORM="${V2ECOLI_BUILD_PLATFORM:-linux/amd64}"
SPEC="${COMPARISON_SPEC:-$ROOT/comparison_spec.json}"

while getopts 'i:r:R:t:s:h' flag; do
  case "${flag}" in
    i) IMAGE_TAG="${OPTARG}" ;;
    r) REPO_NAME="${OPTARG}" ;;
    R) AWS_REGION="${OPTARG}" ;;
    t) EXTRA_TAG="${OPTARG}" ;;
    s) SPEC="${OPTARG}" ;;
    h) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown flag" >&2; exit 2 ;;
  esac
done

# Resolve the fork (repo<TAB>ref) from the spec; pass as build-args when available.
BUILD_ARGS=()
if [[ -f "$SPEC" ]] && command -v python3 >/dev/null 2>&1; then
  if fork_line="$(python3 "$ROOT/scripts/_read_spec.py" "$SPEC" vecoli-fork 2>/dev/null)"; then
    ve_repo="${fork_line%%$'\t'*}"; ve_ref="${fork_line##*$'\t'}"
    if [[ -n "$ve_repo" && -n "$ve_ref" ]]; then
      BUILD_ARGS+=(--build-arg "VECOLI_UPSTREAM_REPO=${ve_repo}" --build-arg "VECOLI_UPSTREAM_REF=${ve_ref}")
      echo "[v2ecoli-build] vEcoli fork (from $SPEC): ${ve_repo} @ ${ve_ref}"
    fi
  fi
fi
[[ ${#BUILD_ARGS[@]} -eq 0 ]] && echo "[v2ecoli-build] vEcoli fork: Dockerfile defaults (no spec build-args)"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "ERROR: aws CLI not found" >&2; exit 1; }
export DOCKER_BUILDKIT=1

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REPO_URI="${REGISTRY}/${REPO_NAME}"
IMAGE="${REPO_URI}:${IMAGE_TAG}"

echo "[v2ecoli-build] image:    ${IMAGE}"
[[ -n "${EXTRA_TAG}" ]] && echo "[v2ecoli-build] extra:    ${REPO_URI}:${EXTRA_TAG}"
echo "[v2ecoli-build] platform: ${PLATFORM}"

aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${REPO_NAME}" --region "${AWS_REGION}" >/dev/null
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"

docker build --platform "${PLATFORM}" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} -t "${IMAGE}" "${ROOT}"
docker push "${IMAGE}"

if [[ -n "${EXTRA_TAG}" ]]; then
  docker tag "${IMAGE}" "${REPO_URI}:${EXTRA_TAG}"
  docker push "${REPO_URI}:${EXTRA_TAG}"
fi

echo "[v2ecoli-build] pushed ${IMAGE}${EXTRA_TAG:+ and ${REPO_URI}:${EXTRA_TAG}}"
