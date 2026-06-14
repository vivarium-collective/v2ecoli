#!/usr/bin/env python
"""Inject gene-level metadata + outlier-table parameters into the omics criteria
of the report-card reference fixtures.

The transcriptome / proteome axes grade a positional ensemble-mean vector (one
value per mRNA cistron / monomer). To render the outlier-gene tables (genes that
disagree most between the measured and reference models) the renderer needs, per
axis, the ordered gene ids + symbol + descriptive name aligned to the ref_vector,
plus the table thresholds. These come from sim_data (the id ORDER) + the
reconstruction flat files (symbol/name) — see ``v2ecoli.library.gene_meta``.

This is a migration that patches the *existing* committed references in place
(no sweep re-read). Going forward the pin scripts bake the same fields, so a
re-pin reproduces this state; run this only to upgrade a reference pinned before
the outlier tables existed.

Usage:
    PYTHONPATH=$PWD .venv/bin/python scripts/add_omics_gene_metadata.py \
        --sim-data out/sim_data_full/parca_state.pkl \
        --reference tests/fixtures/population_phenotype_basal_reference.json \
        --reference docs/report_cards/population_phenotype_basal/vs_vecoli/vecoli_reference.json
"""
import argparse
import json
import pickle

from v2ecoli.library.gene_meta import omics_labels

# card axis path -> key in omics_labels()
_AXES = {"omics.transcriptome": "transcriptome", "omics.proteome": "proteome"}
# default outlier-table parameters (Chris: min_count adjustable, default 10)
_PARAMS = {"outlier_log2fc": 2.0, "min_count": 10, "outlier_top_n": 20}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-data", required=True,
                   help="ParCa sim_data pickle (parca_state.pkl) for gene order")
    p.add_argument("--reference", action="append", required=True,
                   help="Reference fixture to patch (repeatable)")
    args = p.parse_args()

    print(f"[sim_data] {args.sim_data}")
    with open(args.sim_data, "rb") as f:
        sd = pickle.load(f)
    labels = omics_labels(sd)

    for ref_path in args.reference:
        with open(ref_path, encoding="utf-8") as f:
            ref = json.load(f)
        axes = ref.get("axes", {})
        patched = []
        for path, key in _AXES.items():
            ax = axes.get(path)
            if not ax:
                continue
            crit = ax.setdefault("criterion", {})
            meta = labels[key]
            n_ref = len(crit.get("ref_vector") or [])
            if n_ref and n_ref != len(meta["ids"]):
                raise SystemExit(
                    f"{ref_path}:{path} ref_vector len {n_ref} != "
                    f"metadata len {len(meta['ids'])} — order mismatch, aborting")
            crit["ids"] = meta["ids"]
            crit["symbols"] = meta["symbols"]
            crit["names"] = meta["names"]
            for k, v in _PARAMS.items():
                crit.setdefault(k, v)
            patched.append(f"{path} ({len(meta['ids'])} genes)")
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(ref, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[patched] {ref_path}: {', '.join(patched) or 'no omics axes'}")


if __name__ == "__main__":
    main()
