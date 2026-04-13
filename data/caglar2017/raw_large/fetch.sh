#!/usr/bin/env bash
# Fetch the large Caglar et al. 2017 supplementary files that are gitignored.
# Source: https://www.nature.com/articles/srep45303
set -euo pipefail
cd "$(dirname "$0")"

BASE="https://static-content.springer.com/esm/art%3A10.1038%2Fsrep45303/MediaObjects"

files=(
  "41598_2017_BFsrep45303_MOESM53_ESM.csv"   # mRNA abundances (normalized)
  "41598_2017_BFsrep45303_MOESM59_ESM.csv"   # RNA-seq raw counts (wide)
  "41598_2017_BFsrep45303_MOESM60_ESM.csv"   # protein abundances (normalized)
  "41598_2017_BFsrep45303_MOESM61_ESM.csv"   # protein mass-spec raw
)

for f in "${files[@]}"; do
  if [[ -f "$f" ]]; then
    echo "have $f"
  else
    echo "fetch $f"
    curl -L --fail -o "$f" "$BASE/$f"
  fi
done

# Main paper PDF
if [[ ! -f srep45303.pdf ]]; then
  curl -L --fail -o srep45303.pdf "https://www.nature.com/articles/srep45303.pdf"
fi
