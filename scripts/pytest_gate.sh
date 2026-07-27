#!/usr/bin/env bash
# Run the fast-test gate, EXCLUDING the quarantined tests in
# tests/known_failures.txt. See that file for why quarantine exists.
#
#   scripts/pytest_gate.sh              # the blocking gate (fast suite minus quarantine)
#   scripts/pytest_gate.sh --only-known # quarantine watch (never fails; reports)
#
# Why quarantine: a single genuinely-broken-but-known test (e.g. one pinned to a
# third-party lib's internal error text, or a coverage guard waiting on a
# committed fixture) otherwise hard-blocks the gate on EVERY PR. That trains
# people to merge past red, and red then accumulates on main. Parking such a
# test here — with a reason and an owner — keeps the gate GREEN and meaningful,
# so branch protection can actually require it. The watch mode never fails; it
# only flags a quarantined test that has started PASSING, so the line can be
# deleted.
#
# The base selection MUST mirror .github/workflows/ci.yml `fast-tests` exactly.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
known="${repo_root}/tests/known_failures.txt"

# In-process (no xdist): the Linux runner's execnet worker intermittently dies
# mid-suite; pytest-timeout catches a genuine hang instead. Mirrors ci.yml.
base=( -m "not slow and not sim"
       --ignore=tests/test_initialize.py
       --ignore=tests/test_kinetics_units.py
       --ignore=tests/test_unit_bridge.py
       -p no:xdist )

ids=()
if [[ -f "$known" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"                      # strip trailing comments
    line="$(echo "$line" | xargs || true)"  # trim whitespace
    [[ -n "$line" ]] && ids+=("$line")
  done < "$known"
fi

summary() {
  [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && echo -e "$1" >> "$GITHUB_STEP_SUMMARY"
  echo -e "$1"
}

if [[ "${1:-}" == "--only-known" ]]; then
  if [[ ${#ids[@]} -eq 0 ]]; then
    summary "No quarantined tests — tests/known_failures.txt is empty. The quarantine machinery can go."
    exit 0
  fi
  report="$(mktemp)"
  # `|| true`: a non-zero exit is the EXPECTED outcome for a quarantined test.
  uv run pytest -q --no-header -rA -p no:xdist "${ids[@]}" > "$report" 2>&1 || true
  passed=()
  for id in "${ids[@]}"; do
    grep -qxF "PASSED ${id}" "$report" && passed+=("$id")
  done
  total=${#ids[@]}
  if [[ ${#passed[@]} -eq 0 ]]; then
    summary "### Quarantine watch\n\nAll **${total}** quarantined tests still failing, as expected — nothing to do."
  else
    summary "### Quarantine watch — ${#passed[@]} of ${total} now PASSING\n"
    summary "These now pass — remove them from \`tests/known_failures.txt\`:\n"
    for id in "${passed[@]}"; do summary "- \`${id}\`"; done
  fi
  exit 0  # always succeed: this job is a report, not a gate.
fi

args=()
for id in "${ids[@]}"; do args+=(--deselect "$id"); done
echo "running fast tests IN-PROCESS, excluding ${#ids[@]} quarantined test(s)"
exec uv run pytest "${base[@]}" "${args[@]}" -v --tb=short
