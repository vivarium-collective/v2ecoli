"""How many of the model's TFs actually have positioned binding sites in the export?

The closed study recorded "only 9 of the model's 23 TFs have positioned sites",
and used that to explain why the attribution rule stripped 89.5% of the network.
That number does not reproduce. It is an artefact of parsing the export's Site
column as "<TF> DNA-binding-site ...": the entry for CRP is
"CRP-cyclic-AMP DNA binding transcriptional dual regulator" and for ArcA it is
"ArcA-PAsp54", neither of which that parse recovers -- yet they carry 309 and 191
positioned sites respectively.

Matching the TF gene name with a word boundary instead gives 18 of 23.

Run:  .venv/bin/python3 <this> [cache_dir]
"""
from __future__ import annotations
import csv, re, sys
import dill

CACHE = sys.argv[1] if len(sys.argv) > 1 else "out/cache"
TF_TSV = ".venv/lib/python3.12/site-packages/ecoli_sources/data/flat/transcription_factors.tsv"
# byte-identical to reconstruction/ecoli/flat/transcription_factors.tsv (verified 2026-09-05);
# there is no transcription_factors_added.tsv in either tree.
EXPORT = "references/all-transcription-factor-binding-sites.txt"

# The export names some TFs by their assembled complex, not the subunit gene the
# model keys on. Without these, ihfA and hns look like coverage gaps when they
# carry 105 and 67 positioned sites.
ALIASES = {"ihfa": ["IHF"], "ihfb": ["IHF"], "hns": ["H-NS"]}

clean = lambda s: (s or "").strip('"').strip()  # noqa: E731

id2name: dict[str, str] = {}
for r in csv.DictReader(open(TF_TSV), delimiter="\t"):
    nm = clean(r["TF"])
    for col in ("oneComponentId", "twoComponentId", "nonMetaboliteBindingId", "activeId"):
        # These cells hold COMMA-SEPARATED id lists ("CPLX-172, PC00003" for araC),
        # so splitting matters: treating the cell as one id loses araC, argP, tyrR.
        for v in (clean(x) for x in clean(r.get(col)).split(",")):
            if v:
                id2name.setdefault(v, nm)

sd = dill.load(open(f"{CACHE}/sim_data_cache.dill", "rb"))
tf_ids = [str(x) for x in sd["configs"]["ecoli-tf-binding"]["tf_ids"]]

rows = list(csv.DictReader(open(EXPORT, errors="replace"), delimiter="\t"))
sites = [(r["Site"], bool((r.get("Left") or "").strip())) for r in rows]

print(f"{'model TF':20s} {'gene':8s} {'rows':>5s} {'positioned':>11s}")
covered, unmapped = 0, []
for t in tf_ids:
    nm = id2name.get(t)
    if not nm:
        unmapped.append(t)
        print(f"  {t:18s} {'?':8s}  (unmapped in transcription_factors.tsv)")
        continue
    names = [nm] + ALIASES.get(nm.lower(), [])
    pat = re.compile("|".join(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])"
                              for n in names), re.I)
    hits = [p for s, p in sites if pat.search(s)]
    covered += sum(hits) > 0
    print(f"  {t:18s} {nm:8s} {len(hits):5d} {sum(hits):11d}")

print(f"\nmodel TFs with >=1 positioned site: {covered}/{len(tf_ids)}"
      f"   (study.yaml limitation claims 9)")
print(f"unmapped model TF ids: {unmapped}")
