"""Gene-level metadata for the omics (transcriptome / proteome) card axes.

The omics vectors are *positional* — index i is the i-th mRNA cistron
(transcriptome) or the i-th monomer (proteome) in sim_data's ordering, with no
ids carried in the parquet. To label the high-disagreement genes in the outlier
tables we need, per axis, the ordered id list plus human-readable metadata
(gene symbol + EcoCyc descriptive name). The id *order* comes from sim_data; the
symbol/name join comes from the reconstruction flat files (always installed).

`omics_labels(sim_data)` returns, for each axis, parallel arrays aligned to the
card's ref_vector order, which the pin scripts bake into the omics criterion so
the renderer stays reference-self-contained (no sim_data at render time).

Two id vocabularies matter and are not interchangeable: the transcriptome's
``ids`` are mRNA *cistron* ids (``EG10001_RNA``) while its ``gene_ids`` are
EcoCyc *gene* ids (``EG10001``); the proteome's ``ids`` are EcoCyc monomer ids.
Join on an id, never on ``symbols`` — symbol spaces differ between sources and
are not injective across them.
"""
from __future__ import annotations

import os
import re


def _flat_dir() -> str:
    """Directory of the reconstruction flat TSVs (installed package)."""
    import reconstruction.ecoli.flat as flat
    if getattr(flat, "__file__", None):
        return os.path.dirname(flat.__file__)
    return list(flat.__path__)[0]  # namespace package (no __init__.py)


def _load_tsv(path: str) -> list[dict]:
    """Minimal TSV reader for the reconstruction flat files (quoted cells)."""
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    hdr = [h.strip().strip('"') for h in lines[0].rstrip("\n").split("\t")]
    rows = []
    for ln in lines[1:]:
        cells = [c.strip().strip('"') for c in ln.rstrip("\n").split("\t")]
        rows.append(dict(zip(hdr, cells)))
    return rows


def _strip_html(s: str) -> str:
    """EcoCyc common_names carry inline markup (<i>, &beta;, …)."""
    s = re.sub(r"<[^>]+>", "", s)
    return (s.replace("&beta;", "β").replace("&alpha;", "α")
             .replace("&gamma;", "γ").replace("&delta;", "δ").strip())


def omics_labels(sim_data) -> dict:
    """Return ``{"transcriptome": {ids, symbols, names},
    "proteome": {ids, symbols, names}}`` — parallel arrays in the same order
    the card's omics ref_vectors use (mRNA cistrons, then monomers).

    ``sim_data`` is the ParCa sim_data (a dict with ``["process"]["transcription"]``
    / ``["translation"]`` — the parca_state serialization the pin scripts load).
    """
    tx = sim_data["process"]["transcription"]
    tl = sim_data["process"]["translation"]
    flat = _flat_dir()
    genes = _load_tsv(os.path.join(flat, "genes.tsv"))
    rnas = _load_tsv(os.path.join(flat, "rnas.tsv"))
    prots = _load_tsv(os.path.join(flat, "proteins.tsv"))
    sym_by_gene = {r["id"]: r["symbol"] for r in genes}
    rna_meta = {r["id"]: (_strip_html(r["common_name"]), r["gene_id"]) for r in rnas}
    prot_name = {r["id"]: _strip_html(r["common_name"]) for r in prots}

    # transcriptome: mRNA cistrons (listener emits mRNA_cistron_counts in this order)
    cd = tx.cistron_data
    cistron_ids = [str(c) for c in cd["id"][cd["is_mRNA"]]]
    # EcoCyc GENE ids, read from the field sim_data builds off genes.tsv rather
    # than by stripping the "_RNA" suffix off the cistron id — string surgery on
    # identifiers is how id-space bugs start, and it would break silently if the
    # cistron-id convention ever changed.
    tx_gene_ids = [str(g) for g in cd["gene_id"][cd["is_mRNA"]]]
    tx_sym, tx_name = [], []
    for cid in cistron_ids:
        name, gid = rna_meta.get(cid, ("", ""))
        tx_sym.append(sym_by_gene.get(gid, ""))
        tx_name.append(name)

    # proteome: monomers (listener emits monomer_counts in this order). Symbol
    # comes via the monomer's cistron -> gene; the descriptive name from proteins.
    mon_ids_full = [str(m) for m in tl.monomer_data["id"]]
    mon_cistron = [str(c) for c in tl.monomer_data["cistron_id"]]
    pr_sym, pr_name = [], []
    for mid_full, cid in zip(mon_ids_full, mon_cistron):
        mid = mid_full.split("[")[0]
        _, gid = rna_meta.get(cid, ("", ""))
        pr_sym.append(sym_by_gene.get(gid, ""))
        pr_name.append(prot_name.get(mid, ""))

    return {
        "transcriptome": {"ids": cistron_ids, "gene_ids": tx_gene_ids,
                          "symbols": tx_sym, "names": tx_name},
        "proteome": {"ids": [m.split("[")[0] for m in mon_ids_full],
                     "symbols": pr_sym, "names": pr_name},
    }
