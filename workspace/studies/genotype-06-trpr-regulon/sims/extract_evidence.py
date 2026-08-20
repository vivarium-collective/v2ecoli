#!/usr/bin/env python
"""Evidence extractor for genotype-06-trpr-regulon.

Every number this study cites publicly is re-derived here from v2ecoli's OWN
raw_data flat files, so each is independently checkable by anyone with the
repo -- no simulation, no ParCa, no measured dataset. That is the point of the
study: the diagnosis rests on the model's shipped regulatory tables plus
published literature, both of which a reader can inspect.

    E-1  TRN density        -- how much of the genome carries ANY TF edge.
    E-2  the trpR regulon   -- the model's 7 fold_changes rows, as fold changes.
    E-3  trp attenuation    -- the tRNA-Trp attenuation term on the same operon.
    E-4  repression vs attenuation -- the per-target ordering of E-2 against E-3.
    E-5  regulon gaps       -- genes present in the model with NO regulatory edge.

Reads only; writes data/evidence.json.

Run from the workspace root (the canonical_runs contract):
    python workspace/studies/genotype-06-trpr-regulon/sims/extract_evidence.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
EVIDENCE = STUDY_DIR / "data" / "evidence.json"

# The tables ParCa itself loads. FLAT_DIR is read off the installed
# reconstruction package rather than hardcoded, so this script measures the
# same files the build does instead of a copy that may have drifted.
def flat_dir() -> Path:
    import reconstruction.ecoli.knowledge_base_raw as kb
    return Path(kb.FLAT_DIR)


# EcoCyc's TrpR regulon: the five operator-bearing regions TrpR is documented to
# control. LITERATURE, not model data -- it is the reference E-5 measures the
# model against, and it is what makes "2 of 5" a comparison rather than a count.
ECOCYC_TRPR_REGIONS = {
    "trpLEDCBA": ["trpL", "trpE", "trpD", "trpC", "trpB", "trpA"],
    "aroH": ["aroH"],
    "aroL": ["aroL"],
    "mtr": ["mtr"],
    "trpR (autoregulation)": ["trpR"],
}


def read_tsv(path: Path) -> list[dict]:
    """Flat files carry '#'-prefixed provenance headers above the real header."""
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", quotechar='"'))


def gene_name_index(genes: list[dict]) -> dict[str, str]:
    """Map every name a gene answers to -> its frame id.

    Synonyms matter and are easy to skip: matching on `symbol` alone leaves 15
    fold_changes targets unresolved and invites the claim that the TRN silently
    loses 15 names. Honouring `synonyms` drops that to 5. The 5 are real.
    """
    index: dict[str, str] = {}
    for gene in genes:
        index.setdefault(gene["symbol"], gene["id"])
        try:
            synonyms = json.loads(gene["synonyms"])
        except (json.JSONDecodeError, KeyError):
            synonyms = []
        for name in synonyms:
            index.setdefault(name, gene["id"])
    return index


def kept_fold_changes(flat: Path) -> list[dict]:
    """fold_changes.tsv minus the (TF, Target) pairs fold_changes_removed.tsv drops.

    knowledge_base_raw.py maps "fold_changes" -> "fold_changes_removed", so the
    removed list is a real filter, not documentation. Skipping it inflates the
    target count from 675 to 682.
    """
    removed = {(r["TF"], r["Target"]) for r in read_tsv(flat / "fold_changes_removed.tsv")}
    return [r for r in read_tsv(flat / "fold_changes.tsv")
            if (r["TF"], r["Target"]) not in removed]


def fold(log2_fc: str) -> float:
    """Report magnitude, direction-free: a -2.46 log2 repression is a 5.50x term."""
    return round(2.0 ** abs(float(log2_fc)), 2)


def main() -> None:
    flat = flat_dir()
    genes = read_tsv(flat / "genes.tsv")
    rnas = read_tsv(flat / "rnas.tsv")
    edges = kept_fold_changes(flat)
    name_to_id = gene_name_index(genes)
    gene_to_type = {r["gene_id"]: r["type"] for r in rnas}
    mrna_ids = {g["id"] for g in genes if gene_to_type.get(g["id"]) == "mRNA"}

    # --- E-1  TRN density ------------------------------------------------
    targets = {r["Target"] for r in edges}
    resolved = {t: name_to_id[t] for t in targets if t in name_to_id}
    covered_mrna = {gid for gid in resolved.values() if gid in mrna_ids}
    e1 = {
        "fold_changes_rows_total": len(read_tsv(flat / "fold_changes.tsv")),
        "fold_changes_rows_kept": len(edges),
        "distinct_tfs": len({r["TF"] for r in edges}),
        "distinct_targets": len(targets),
        "targets_resolving_to_a_gene": len(resolved),
        "targets_not_resolving": sorted(targets - set(resolved)),
        "covered_mrna_cistrons": len(covered_mrna),
        "total_mrna_cistrons": len(mrna_ids),
        "total_cistrons": len(genes),
        "coverage_fraction_of_mrna_cistrons": round(len(covered_mrna) / len(mrna_ids), 4),
    }

    # --- E-2  the model's trpR regulon -----------------------------------
    trpr = [r for r in edges if r["TF"] == "trpR"]
    e2 = {
        "n_rows": len(trpr),
        "targets": {r["Target"]: {"log2_fc": float(r["log2 FC mean"]),
                                  "fold_change": fold(r["log2 FC mean"])}
                    for r in trpr},
        "fold_change_range": [min(fold(r["log2 FC mean"]) for r in trpr),
                              max(fold(r["log2 FC mean"]) for r in trpr)] if trpr else None,
    }

    # --- E-3  trp attenuation --------------------------------------------
    atten = [r for r in read_tsv(flat / "transcriptional_attenuation.tsv")
             if r["tRNA"] == "tRNA-Trp"]
    e3 = {
        "n_rows": len(atten),
        "targets": {r["Target"]: {"log2_fc": float(r["log2 FC"]),
                                  "fold_change": fold(r["log2 FC"])}
                    for r in atten},
        "fold_change_range": [min(fold(r["log2 FC"]) for r in atten),
                              max(fold(r["log2 FC"]) for r in atten)] if atten else None,
    }

    # --- E-4  repression vs attenuation, per shared target ----------------
    # Literature has TrpR repression (70-300x) dominating attenuation (8-10x).
    # This asks whether the model preserves that ordering on the genes it covers.
    shared = sorted(set(e2["targets"]) & set(e3["targets"]))
    comparison = {
        t: {
            "trpR_repression_fold": e2["targets"][t]["fold_change"],
            "attenuation_fold": e3["targets"][t]["fold_change"],
            "repression_exceeds_attenuation": (
                e2["targets"][t]["fold_change"] > e3["targets"][t]["fold_change"]),
        }
        for t in shared
    }
    e4 = {
        "shared_targets": shared,
        "per_target": comparison,
        "n_inverted_vs_literature": sum(
            1 for v in comparison.values() if not v["repression_exceeds_attenuation"]),
    }

    # --- E-5  regulon gaps ------------------------------------------------
    # A gene can be absent from the regulon two ways: absent from the model
    # entirely, or present with no regulatory edge. The second is the
    # interesting one -- the parts are there, only the edge is missing.
    with_any_incoming = {r["Target"] for r in edges}
    gaps = {}
    for region, members in ECOCYC_TRPR_REGIONS.items():
        gaps[region] = {
            "genes": {
                g: {
                    "in_model": g in name_to_id,
                    "gene_id": name_to_id.get(g),
                    "has_any_tf_edge": g in with_any_incoming,
                    "has_trpR_edge": g in e2["targets"],
                }
                for g in members
            },
            "represented_in_model_regulon": any(g in e2["targets"] for g in members),
        }
    e5 = {
        "ecocyc_regions": len(ECOCYC_TRPR_REGIONS),
        "regions_represented_in_model": sum(
            1 for v in gaps.values() if v["represented_in_model_regulon"]),
        "detail": gaps,
    }

    evidence = {
        "provenance": {
            "flat_dir": str(flat),
            "note": "All values derived from v2ecoli raw_data + EcoCyc region "
                    "membership. No simulation, no ParCa, no measured dataset.",
        },
        "E-1_trn_density": e1,
        "E-2_model_trpR_regulon": e2,
        "E-3_trp_attenuation": e3,
        "E-4_repression_vs_attenuation": e4,
        "E-5_regulon_gaps": e5,
    }

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"wrote {EVIDENCE.relative_to(Path.cwd()) if EVIDENCE.is_relative_to(Path.cwd()) else EVIDENCE}")
    print(f"  E-1  {e1['covered_mrna_cistrons']} / {e1['total_mrna_cistrons']} mRNA cistrons "
          f"carry a TF edge ({100 * e1['coverage_fraction_of_mrna_cistrons']:.1f}%)")
    print(f"  E-2  trpR regulon: {e2['n_rows']} rows, {e2['fold_change_range'][0]}x-{e2['fold_change_range'][1]}x")
    print(f"  E-3  attenuation:  {e3['n_rows']} rows, {e3['fold_change_range'][0]}x-{e3['fold_change_range'][1]}x")
    print(f"  E-4  ordering inverted vs literature on {e4['n_inverted_vs_literature']}/{len(shared)} shared targets")
    print(f"  E-5  {e5['regions_represented_in_model']} of {e5['ecocyc_regions']} EcoCyc TrpR regions represented")


if __name__ == "__main__":
    main()
