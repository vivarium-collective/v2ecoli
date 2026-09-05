"""Derive the TF->TU re-attribution table from the EcoCyc PGDB flatfiles.

The model assigns a transcription factor to every TU containing a regulated
cistron (relation.py::_build_RNA_to_tf_mapping). Where EcoCyc records that TF
at the same operon but on a DIFFERENT transcription unit, that assignment is a
misattribution -- the rpsU-dnaG-rpoD case, where LexA's operator belongs to
TU00435 but the model applies it to TU00352.

This writes the correction as a small table so the ParCa build does not depend
on the licensed PGDB export. Regenerate with the flatfiles present:

    .venv/bin/python3 scripts/build_tf_tu_reattribution.py

Input :  references/ecocyc-30.0/{regulation,dnabindsites,transunits}.dat  (gitignored)
Output:  references/tf_tu_reattribution.tsv                               (committed)
"""
from __future__ import annotations
import collections, os, sys

R = "references/ecocyc-30.0/"
OUT = "references/tf_tu_reattribution.tsv"
TF_ALIAS = {"FNR-4FE-4S-CPLX": "CPLX0-7797", "PHOSPHO-DCUR": "CPLX0-7721",
            "PUTA-CPLX": "PUTA-MONOMER"}

def records(path):
    rec = collections.defaultdict(list)
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("//"):
            if rec:
                yield dict(rec)
            rec = collections.defaultdict(list)
            continue
        if line.startswith("#") or " - " not in line:
            continue
        k, v = line.split(" - ", 1)
        rec[k].append(v)
    if rec:
        yield dict(rec)

if not os.path.isdir(R):
    sys.exit(f"{R} not found. Generate it from a local Pathway Tools install:\n"
             "  pathway-tools -lisp  ->  (so 'ECOLI) (create-flat-files-for-current-kb)")

tu_components = {r["UNIQUE-ID"][0]: r.get("COMPONENTS", []) for r in records(R + "transunits.dat")}
tu_genes = {tu: {c for c in comps if not c.startswith(("PM", "TERM", "BS"))}
            for tu, comps in tu_components.items()}
prom_to_tu = collections.defaultdict(set)
for tu, comps in tu_components.items():
    for c in comps:
        if c.startswith("PM"):
            prom_to_tu[c].add(tu)
site_to_tu = collections.defaultdict(set)
for r in records(R + "dnabindsites.dat"):
    for c in r.get("COMPONENT-OF", []):
        if c.startswith("TU"):
            site_to_tu[r["UNIQUE-ID"][0]].add(c)

tf_to_tus = collections.defaultdict(set)
for r in records(R + "regulation.dat"):
    if "Transcription-Factor-Binding" not in r.get("TYPES", []):
        continue
    tf = (r.get("REGULATOR") or [None])[0]
    if not tf:
        continue
    ent = (r.get("REGULATED-ENTITY") or [None])[0]
    bs = (r.get("ASSOCIATED-BINDING-SITE") or [None])[0]
    tus = set()
    if ent and ent.startswith("TU"):
        tus.add(ent)
    if ent and ent.startswith("PM"):
        tus |= prom_to_tu.get(ent, set())
    if bs:
        tus |= site_to_tu.get(bs, set())
    tf_to_tus[tf] |= tus

gene_to_tus = collections.defaultdict(set)
for tu, genes in tu_genes.items():
    for g in genes:
        gene_to_tus[g].add(tu)

rows = []
for tf_model, eco_tf in [(t, TF_ALIAS.get(t, t)) for t in sorted(
        set(TF_ALIAS) | {t for t in tf_to_tus})]:
    eco_tus = tf_to_tus.get(eco_tf, set())
    if not eco_tus:
        continue
    for tu in sorted(eco_tus):
        for g in sorted(tu_genes.get(tu, ())):
            rows.append((tf_model, eco_tf, tu, g))

with open(OUT, "w") as f:
    f.write("# TF -> transcription-unit attribution, derived from EcoCyc 30.0\n")
    f.write("# regulation.dat (Transcription-Factor-Binding) + dnabindsites.dat + transunits.dat.\n")
    f.write("# Regenerate: scripts/build_tf_tu_reattribution.py\n")
    f.write("# A TF listed here acts on THIS transcription unit; the model's cistron-content\n")
    f.write("# rule must not place it on a sibling TU of the same operon that is absent here.\n")
    f.write("tf_id\tecocyc_regulator\ttranscription_unit\tgene\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print(f"wrote {OUT}: {len(rows)} (tf, tu, gene) rows over "
      f"{len({r[0] for r in rows})} regulators, {len({r[2] for r in rows})} TUs")
