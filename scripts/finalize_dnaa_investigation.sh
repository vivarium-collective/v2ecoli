#!/usr/bin/env bash
# Finalize the DnaA-oriC mechanistic-initiation investigation: run the three
# analyses on the full fleet output, copy figures into the study chart dirs,
# and print the meta.json headline numbers for the study verdicts.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=out/analysis

echo "=== 1/3 succinate asynchrony (sat-init vs mass-clock control) ==="
$PY scripts/analyze_initiation_asynchrony.py \
  --mech 'out/dnaa5_succ_mech_seed*_parquet/*/history' \
  --control 'out/dnaa5_succ_ctrl_seed*_parquet/*/history' \
  --out $A/asynchrony_succinate.svg --title "succinate" 2>&1 | grep -vE "Setting|skipping|numcodecs|Warning"

echo "=== 2/3 succinate division distributions ==="
$PY scripts/analyze_division_distributions.py \
  --log 'out/dnaa5_succ_mech_seed*.log' \
  --out $A/division_succinate.svg --title "succinate (sat-init)" 2>&1 | grep -vE "Setting|skipping|numcodecs|Warning"

echo "=== 3/3 basal division distributions ==="
$PY scripts/analyze_division_distributions.py \
  --log 'out/dnaa5_basal_mech_seed*.log' \
  --out $A/division_basal.svg --title "basal (sat-init)" 2>&1 | grep -vE "Setting|skipping|numcodecs|Warning"

echo "=== copy figures into study charts/ ==="
cp $A/asynchrony_succinate.png workspace/studies/dnaa-10-async-quantification/charts/
cp $A/division_succinate.png   workspace/studies/dnaa-10-async-quantification/charts/
cp $A/division_basal.png       workspace/studies/dnaa-11-basal-mechanistic/charts/
echo "  copied."

echo "=== headline numbers (for the study verdicts) ==="
for m in $A/asynchrony_succinate.meta.json $A/division_succinate.meta.json $A/division_basal.meta.json; do
  echo "--- $m ---"; cat "$m"; echo
done
