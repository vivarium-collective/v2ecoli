"""Split the model's declared regulatory edges by what EcoCyc actually records.

The disruption number that failed this study conflates two very different cases.
Using EcoCyc's own flatfiles (references/ecocyc-30.0/, generated from the local
Pathway Tools 30.0 install) each model edge is classified:

  CONFIRMED    EcoCyc records this TF regulating THIS transcription unit.
  CONTRADICTED EcoCyc records this TF at this operon but on a DIFFERENT TU --
               a real misattribution, the dnaG case.
  UNRECORDED   EcoCyc records nothing for this TF at this operon. Absence of
               evidence; the model's fold-change data may still be right.

Only CONTRADICTED justifies changing the model.

The join needs no name matching: regulation.dat REGULATOR values are the model's
TF ids and dnabindsites/transunits resolve to the model's TU ids.

Run:  .venv/bin/python3 <this> [cache_dir]
"""
from __future__ import annotations
import collections, json, sys
import numpy as np, dill, scipy.sparse as sp

CACHE = sys.argv[1] if len(sys.argv) > 1 else "out/cache"
R = "references/ecocyc-30.0/"

# Model TF ids that are not the frame EcoCyc uses as a REGULATOR. Each verified by
# name in proteins.dat, and for FNR/DcuR by tracing a site from the text export
# through dnabindsites.dat into its regulation record.
TF_ALIAS = {
    "FNR-4FE-4S-CPLX": "CPLX0-7797",   # DNA-binding transcriptional dual regulator FNR (204 records)
    "PHOSPHO-DCUR":    "CPLX0-7721",   # DcuR-phosphorylated (11 records)
    "PUTA-CPLX":       "PUTA-MONOMER", # PutA (5 records)
}

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

one = lambda r, k, d=None: r.get(k, [d])[0]  # noqa: E731

# --- EcoCyc structure -------------------------------------------------------
tu_components, tu_genes = {}, {}
for r in records(R + "transunits.dat"):
    tu = one(r, "UNIQUE-ID")
    comps = r.get("COMPONENTS", [])
    tu_components[tu] = comps
    tu_genes[tu] = {c for c in comps if not c.startswith(("PM", "TERM", "BS"))}

prom_to_tu = collections.defaultdict(set)
for tu, comps in tu_components.items():
    for c in comps:
        if c.startswith("PM"):
            prom_to_tu[c].add(tu)

site_to_tu = collections.defaultdict(set)
for r in records(R + "dnabindsites.dat"):
    bs = one(r, "UNIQUE-ID")
    for c in r.get("COMPONENT-OF", []):
        if c.startswith("TU"):
            site_to_tu[bs].add(c)

# TF -> the EcoCyc TUs it regulates
tf_to_tus = collections.defaultdict(set)
n_reg = 0
for r in records(R + "regulation.dat"):
    if "Transcription-Factor-Binding" not in r.get("TYPES", []):
        continue
    tf = one(r, "REGULATOR")
    if not tf:
        continue
    n_reg += 1
    ent = one(r, "REGULATED-ENTITY")
    bs = one(r, "ASSOCIATED-BINDING-SITE")
    tus = set()
    if ent and ent.startswith("TU"):
        tus.add(ent)
    if ent and ent.startswith("PM"):
        tus |= prom_to_tu.get(ent, set())
    if bs:
        tus |= site_to_tu.get(bs, set())
    tf_to_tus[tf] |= tus

# gene -> EcoCyc TUs containing it
gene_to_tus = collections.defaultdict(set)
for tu, genes in tu_genes.items():
    for g in genes:
        gene_to_tus[g].add(tu)

# --- model edges ------------------------------------------------------------
sd = dill.load(open(f"{CACHE}/sim_data_cache.dill", "rb"))
ti = sd["configs"]["ecoli-transcript-initiation"]
ids = [str(x) for x in ti["rna_data"]["id"]]
tf_ids = [str(x) for x in sd["configs"]["ecoli-tf-binding"]["tf_ids"]]
dp = ti["delta_prob"]
edges = list(zip(dp["deltaI"].tolist(), dp["deltaJ"].tolist()))

# model TU -> its gene set, via the cistron mapping
import pickle
st = pickle.load(open("out/parca_tudedup_full/parca_state.pkl", "rb"))
tr = st["process"]["transcription"]
M = sp.csc_matrix(tr.cistron_tu_mapping_matrix)
cid = [str(x) for x in tr.cistron_data["id"]]
full_ids = [str(x) for x in tr.rna_data["id"]]
row_of = {r: i for i, r in enumerate(full_ids)}

def model_genes(rna_id):
    i = row_of.get(rna_id)
    if i is None:
        return set()
    return {cid[r].replace("_RNA", "") for r in M[:, i].nonzero()[0]}

def ecocyc_tu(rna_id):
    base = rna_id[:-3] if rna_id.endswith("[c]") else rna_id
    return base if base.startswith("TU") else None

print(f"EcoCyc: {n_reg} TF-binding records; {len(tf_to_tus)} distinct regulators; "
      f"{len(tu_genes)} TUs")
print(f"model : {len(edges)} declared edges over {len(ids)} TUs, {len(tf_ids)} TFs\n")

counts = collections.Counter()
contradicted = []
for i, j in edges:
    rna_id, tf = ids[i], tf_ids[j]
    eco_tus = tf_to_tus.get(TF_ALIAS.get(tf, tf), set())
    if not eco_tus:
        counts["no_ecocyc_record_for_this_TF"] += 1
        continue
    mtu = ecocyc_tu(rna_id)
    genes = model_genes(rna_id)
    # every EcoCyc TU sharing a gene with this model TU = the operon neighbourhood
    neighbourhood = set().union(*(gene_to_tus.get(g, set()) for g in genes)) if genes else set()
    hits = eco_tus & neighbourhood
    if mtu and mtu in eco_tus:
        counts["CONFIRMED"] += 1
    elif hits:
        counts["CONTRADICTED"] += 1
        contradicted.append((rna_id, tf, sorted(hits)[:4]))
    else:
        counts["UNRECORDED"] += 1

total = len(edges)
print("--- edge classification")
for k in ("CONFIRMED", "CONTRADICTED", "UNRECORDED", "no_ecocyc_record_for_this_TF"):
    print(f"  {k:32s} {counts[k]:5d}   {counts[k]/total:6.2%}")

print(f"\n--- contradicted edges (first 15 of {len(contradicted)})")
for rna_id, tf, hits in contradicted[:15]:
    print(f"  model puts {tf:16s} on {rna_id:16s};  EcoCyc records it on {hits}")

dnag = [c for c in contradicted if c[0].startswith("TU00352")]
print(f"\n--- the dnaG edge: {dnag if dnag else 'NOT in the contradicted set'}")

json.dump({"counts": dict(counts), "total": total,
           "contradicted": [[a, b, c] for a, b, c in contradicted]},
          open("workspace/studies/promoter-specific-regulation/analyses/edge_evidence_split.json", "w"),
          indent=2)
