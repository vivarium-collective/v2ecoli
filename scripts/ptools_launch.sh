#!/usr/bin/env bash
# Paint a ptools omics TSV onto the EcoCyc Cellular Overview of the local
# sms-ptools server and open it auto-loaded (no manual upload).
#
#   scripts/ptools_launch.sh <file.tsv> [class] [orgid]
#     class: gene | reaction | protein | compound
#            (default: inferred from the filename — ptools_rxns→reaction,
#             ptools_proteins→protein, ptools_rna/other→gene)
#     orgid: PGDB organism id (default: ECOLI)
#
# Why it stages the file INTO the container: this sms-ptools image's Omics
# Viewer only loads data files that are LOCAL to the Pathway Tools server
# (verified — a remote url=http://… is not fetched), so we docker-cp the TSV
# into the server's htdocs and point url=file://… at it. The viewer auto-loads
# via celOv.shtml?omics=t&url=…&class=…&column1=1-N, animating across timepoints.
#
# Companion to scripts/ptools_server.sh (which starts the server).
set -euo pipefail

NAME="sms-ptools"
SERVER="${PTOOLS_SERVER_URL:-http://localhost:1555}"
HTDOCS="/app/ptools/aic-export/htdocs"

tsv="${1:?usage: scripts/ptools_launch.sh <file.tsv> [class] [orgid]}"
[ -f "$tsv" ] || { echo "file not found: $tsv" >&2; exit 1; }

infer_class() {
  case "$(basename "$1" | tr '[:upper:]' '[:lower:]')" in
    *rxn*|*reaction*)        echo reaction ;;
    *protein*)               echo protein ;;
    *metabolite*|*compound*) echo compound ;;
    *)                       echo gene ;;
  esac
}
cls="${2:-$(infer_class "$tsv")}"
orgid="${3:-ECOLI}"

docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -qx "$NAME" || {
  echo "ptools server '$NAME' is not running. Start it:  scripts/ptools_server.sh up" >&2
  exit 1
}

# Timepoint columns = fields after the ID on the first non-comment line.
# column1=1-N animates the overlay across all timepoints; 1 if single-column.
# Single awk (no pipe) avoids a SIGPIPE that would trip `set -o pipefail`.
ncol=$(awk -F'\t' '!/^[#;]/ && NF { print NF - 1; exit }' "$tsv")
if [ -n "${ncol:-}" ] && [ "$ncol" -gt 1 ] 2>/dev/null; then
  cols="1-$ncol"
else
  cols="1"
fi

dest="ptools_launch_$(basename "$tsv")"
docker cp "$tsv" "$NAME:$HTDOCS/$dest" >/dev/null
url="$SERVER/overviewsWeb/celOv.shtml?omics=t&url=file://$dest&orgid=$orgid&class=$cls&column1=$cols"

echo "staged $(basename "$tsv") -> $NAME:$HTDOCS/$dest"
echo "class=$cls  column1=$cols"
echo "$url"
if command -v open >/dev/null 2>&1; then
  open "$url"
else
  echo "Open the URL above in a browser."
fi
