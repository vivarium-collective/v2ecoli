#!/usr/bin/env bash
# Manage the local sms-ptools Pathway Tools server that backs the
# `ptools_overview` analysis (the EcoCyc Cellular Overview / Omics Viewer).
#
#   scripts/ptools_server.sh up        # start it (idempotent; no-op if healthy)
#   scripts/ptools_server.sh restart   # force a fresh container
#   scripts/ptools_server.sh status    # is it up?
#   scripts/ptools_server.sh down       # remove the container
#
# Hard-won robustness notes baked in here:
#   * Tile generation is memory-hungry: under the default 2 GiB colima VM the
#     server is OOM-killed (exit 137) the moment you browse the overview. Needs
#     >= ~8 GiB.  -> colima stop && colima start --cpu 4 --memory 8
#   * `docker start` on a stopped container reuses its dirty filesystem, whose
#     stale Xvfb /tmp/.X1-lock makes Pathway Tools fail with
#     "cannot open the display: :1" and never start the web server. ALWAYS
#     recreate from scratch (docker rm -f + docker run), which this script does.
#   * --restart unless-stopped lets it recover after a daemon/VM restart.
#   * Image is amd64-only; on Apple Silicon it runs under qemu (slow first boot).
set -euo pipefail

IMAGE="ghcr.io/vivarium-collective/sms-ptools:0.8.2"
NAME="sms-ptools"
URL="http://localhost:1555/overviewsWeb/celOv.shtml?orgid=ECOLI"
MIN_VM_GIB=6

ensure_runtime() {
  if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon not reachable. Start it, e.g.:  colima start --cpu 4 --memory 8" >&2
    exit 1
  fi
  if command -v colima >/dev/null 2>&1; then
    local mem
    mem=$(colima list 2>/dev/null | awk 'NR>1 && $1=="default"{print $5}' | tr -dc '0-9')
    if [ -n "${mem:-}" ] && [ "$mem" -lt "$MIN_VM_GIB" ]; then
      echo "WARNING: colima VM has ${mem} GiB RAM; the overview can OOM-kill ptools (<${MIN_VM_GIB} GiB)." >&2
      echo "         Recommend: colima stop && colima start --cpu 4 --memory 8" >&2
    fi
  fi
}

run_fresh() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --platform linux/amd64 --name "$NAME" --restart unless-stopped \
    -e SERVER_HOST_NAME=localhost -p 1555:1555 -p 5008:5008 "$IMAGE" >/dev/null
}

wait_ready() {
  printf 'waiting for ptools to load the EcoCyc KB'
  for _ in $(seq 1 30); do
    sleep 5
    if curl -s --max-time 8 -o /dev/null "$URL"; then
      printf ' ready.\n%s\n' "$URL"; return 0
    fi
    printf '.'
  done
  printf '\nnot ready after 150s; check:  docker logs %s\n' "$NAME" >&2
  return 1
}

case "${1:-up}" in
  up)
    ensure_runtime
    if curl -s --max-time 5 -o /dev/null "$URL"; then echo "already up: $URL"; exit 0; fi
    run_fresh; wait_ready ;;
  restart) ensure_runtime; run_fresh; wait_ready ;;
  down) docker rm -f "$NAME" >/dev/null 2>&1 && echo "removed" || echo "not running" ;;
  status)
    docker ps -a --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' || true
    code=$(curl -s --max-time 8 -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null || echo 000)
    echo "overview -> HTTP ${code}  (${URL})" ;;
  *) echo "usage: $0 [up|restart|down|status]" >&2; exit 2 ;;
esac
