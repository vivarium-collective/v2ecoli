"""Minimal protein-validation-data loader for v2ecoli.

Exposes an object with the same attribute shape that
``ecoli/analysis/multiseed/protein_counts_validation.py`` expects:

    validation_data.protein.schmidt2015Data["monomerId"]
    validation_data.protein.schmidt2015Data["glucoseCounts"]
    validation_data.protein.wisniewski2014Data["monomerId"]
    validation_data.protein.wisniewski2014Data["avgCounts"]

Construction faithfully replicates vEcoli's
``validation/ecoli/validation_data.py::Protein._loadSchmidt2015Counts`` and
``_loadWisniewski2014Counts``, with one deliberate adaptation: where vEcoli
resolves gene-ID→monomer-ID via a full KnowledgeBaseEcoli, we build the same
mapping from ``sim_data.process.translation.monomer_data`` (cistron_id strips
the ``_RNA`` suffix to recover the EcoCycID gene ID).  Genes absent from the
mapping are logged and skipped rather than raising a KeyError (matching
vEcoli's wisniewski loader and gracefully handling the handful of genes that
are in the published proteomics tables but not in v2ecoli's ParCa model).

Usage::

    from v2ecoli.library.validation_data import build_validation_data
    vd = build_validation_data(sim_data)
    # vd.protein.schmidt2015Data["monomerId"], ["glucoseCounts"]
    # vd.protein.wisniewski2014Data["monomerId"], ["avgCounts"]
"""

from __future__ import annotations

import csv
import importlib.resources
import logging
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flat-file location
# ---------------------------------------------------------------------------

def _flat_file(name: str) -> Path:
    """Locate a flat validation TSV by name (package-relative)."""
    # importlib.resources is the robust way to locate package data files.
    pkg = "v2ecoli.validation.ecoli.flat"
    try:
        ref = importlib.resources.files(pkg).joinpath(name)
        return Path(str(ref))
    except (AttributeError, ModuleNotFoundError, FileNotFoundError):
        # Fallback: relative to this file (development layout).
        here = Path(__file__).parent
        return here.parent / "validation" / "ecoli" / "flat" / name


def _read_tsv(name: str) -> list[dict[str, str]]:
    """Read a flat-file TSV into a list of dicts; strip outer quotes from values."""
    path = _flat_file(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Validation flat file not found: {path}. "
            "Run the copy step described in the porting spec."
        )
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append({k.strip('"'): v.strip('"') for k, v in row.items()})
    return rows


# ---------------------------------------------------------------------------
# Gene ID → monomer ID mapping (built from sim_data)
# ---------------------------------------------------------------------------

def _build_gene_to_monomer(sim_data) -> dict[str, str]:
    """Build EcoCycID (gene ID) → monomer ID+compartment mapping from sim_data.

    vEcoli builds this via KnowledgeBaseEcoli (genes + rnas + compartments).
    v2ecoli stores the equivalent information in
    ``sim_data.process.translation.monomer_data``:

    * ``monomer_data["id"]`` — monomer ID with compartment, e.g.
      ``"THIOREDOXIN-MONOMER[c]"``
    * ``monomer_data["cistron_id"]`` — cistron (RNA) ID, e.g.
      ``"EG10339_RNA"``

    The EcoCycID gene ID is recovered by stripping the ``_RNA`` suffix.  This
    is identical to vEcoli's ``rna_id_to_gene_id`` (from ``genes``) composed
    with the RNA→monomer lookup (from ``rnas``).
    """
    gene_to_monomer: dict[str, str] = {}
    md = sim_data.process.translation.monomer_data
    for row in md.struct_array:
        cistron_id = str(row["cistron_id"])
        monomer_id = str(row["id"])
        if cistron_id.endswith("_RNA"):
            gene_id = cistron_id[:-4]  # strip _RNA → EcoCycID gene ID
            gene_to_monomer[gene_id] = monomer_id
    if not gene_to_monomer:
        raise ValueError(
            "gene_to_monomer is empty — monomer_data may be missing or malformed"
        )
    return gene_to_monomer


# ---------------------------------------------------------------------------
# Protein sub-object
# ---------------------------------------------------------------------------

class _Protein:
    """Exposes schmidt2015Data and wisniewski2014Data as structured numpy arrays."""

    def __init__(self, sim_data):
        gene_to_monomer = _build_gene_to_monomer(sim_data)
        self._load_schmidt(gene_to_monomer)
        self._load_wisniewski(gene_to_monomer)

    def _load_schmidt(self, gene_to_monomer: dict[str, str]) -> None:
        """Replicate vEcoli's ``_loadSchmidt2015Counts``.

        Source columns used:
            EcoCycID  → gene ID → monomer ID (via gene_to_monomer)
            Glucose   → glucoseCounts
            LB        → LB_counts

        Genes absent from gene_to_monomer are skipped with a warning (vEcoli
        would KeyError, but our sim_data may exclude a small number of genes).
        """
        rows = _read_tsv("schmidt2015_javier_table.tsv")
        if not rows:
            raise ValueError("schmidt2015_javier_table.tsv is empty")
        required = {"EcoCycID", "Glucose", "LB"}
        missing_cols = required - set(rows[0].keys())
        if missing_cols:
            raise KeyError(f"schmidt2015 TSV missing columns: {missing_cols}")

        monomer_ids: list[str] = []
        glucose_counts: list[float] = []
        lb_counts: list[float] = []
        skipped: list[str] = []

        for row in rows:
            gene_id = row["EcoCycID"]
            if gene_id not in gene_to_monomer:
                skipped.append(gene_id)
                continue
            monomer_ids.append(gene_to_monomer[gene_id])
            glucose_counts.append(float(row["Glucose"]))
            lb_counts.append(float(row["LB"]))

        if skipped:
            warnings.warn(
                f"schmidt2015: {len(skipped)} gene IDs not in sim_data monomer "
                f"table (skipped): {skipped[:10]}{'...' if len(skipped) > 10 else ''}",
                stacklevel=3,
            )
        if not monomer_ids:
            raise ValueError(
                "schmidt2015: zero monomers resolved — gene_to_monomer mismatch"
            )

        n = len(monomer_ids)
        arr = np.zeros(n, dtype=[("monomerId", "U50"), ("glucoseCounts", "f8"),
                                  ("LB_counts", "f8")])
        arr["monomerId"] = monomer_ids
        arr["glucoseCounts"] = glucose_counts
        arr["LB_counts"] = lb_counts
        self.schmidt2015Data = arr

    def _load_wisniewski(self, gene_to_monomer: dict[str, str]) -> None:
        """Replicate vEcoli's ``_loadWisniewski2014Counts``.

        Source columns used:
            EcoCycID  → gene ID → monomer ID (via gene_to_monomer)
            rep1, rep2, rep3 → avgCounts = mean(rep1, rep2, rep3)

        Genes absent from gene_to_monomer are skipped (matching vEcoli exactly).
        """
        rows = _read_tsv("wisniewski2014_supp2.tsv")
        if not rows:
            raise ValueError("wisniewski2014_supp2.tsv is empty")
        required = {"EcoCycID", "rep1", "rep2", "rep3"}
        missing_cols = required - set(rows[0].keys())
        if missing_cols:
            raise KeyError(f"wisniewski2014 TSV missing columns: {missing_cols}")

        monomer_ids: list[str] = []
        avg_counts: list[float] = []

        for row in rows:
            gene_id = row["EcoCycID"]
            if gene_id not in gene_to_monomer:
                continue
            r1 = float(row["rep1"])
            r2 = float(row["rep2"])
            r3 = float(row["rep3"])
            monomer_ids.append(gene_to_monomer[gene_id])
            avg_counts.append((r1 + r2 + r3) / 3.0)

        if not monomer_ids:
            raise ValueError(
                "wisniewski2014: zero monomers resolved — gene_to_monomer mismatch"
            )

        n = len(monomer_ids)
        arr = np.zeros(n, dtype=[("monomerId", "U50"), ("avgCounts", "f8")])
        arr["monomerId"] = monomer_ids
        arr["avgCounts"] = avg_counts
        self.wisniewski2014Data = arr


# ---------------------------------------------------------------------------
# Top-level validation object
# ---------------------------------------------------------------------------

class _ValidationData:
    """Minimal validation data container for v2ecoli analysis porting."""

    def __init__(self, sim_data):
        self.protein = _Protein(sim_data)


def build_validation_data(sim_data) -> _ValidationData:
    """Build minimal validation data from sim_data + copied flat files.

    Parameters
    ----------
    sim_data:
        Loaded ParCa sim_data object (provides gene-ID→monomer-ID mapping).

    Returns
    -------
    _ValidationData
        Object with ``.protein.schmidt2015Data`` and
        ``.protein.wisniewski2014Data`` structured arrays.

    Raises
    ------
    FileNotFoundError
        If the flat TSV files have not been copied into
        ``v2ecoli/validation/ecoli/flat/``.
    ValueError
        If zero monomers are resolved (likely a sim_data mismatch).
    """
    return _ValidationData(sim_data)
