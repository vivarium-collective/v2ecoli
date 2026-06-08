"""Re-pin the basal-phenotype report-card reference from a blessed ensemble.

The reference fixture (``tests/fixtures/population_phenotype_basal_reference.json``) is a
*generated* pin: it bakes the blessed ensemble's measured values as the grading
reference so the card self-pins green and gates on drift. This script
regenerates it, preserving the human-authored presentation/criterion structure
and re-baking the numbers.

It adds the Layer-3 axes on top of the existing Growth + Composition axes:

  - **Exchange fluxes** (group, ordered before Composition):
      * ``fluxes.exchange`` — whole-vector scatter (R² on matched + appeared/
        disappeared count). Bakes the 87-element ``ref_vector`` and the
        ``flux_ids`` order (= ``sorted(sim_data.external_state
        .all_external_exchange_molecules)``, the order the FBA model emits).
      * named KPI sub-axes (glucose/O2/N/CO2/acetate) — per-cell ttest+violin,
        sliced from the flux matrix by ``flux_id``.
  - **Omics**: ``omics.transcriptome`` / ``omics.proteome`` — log-log R² vs the
    blessed ensemble-mean vector (self-pin => R²=1; PDMP lights it up).

Run (full-mode cache ensemble + its sim_data):

    python scripts/pin_population_phenotype_basal_reference.py \\
        --sweep-dir out/population_phenotype_basal \\
        --sim-data out/sim_data_full/parca_state.pkl \\
        --gen-lb 3

Heavy (~minute): reads the sweep parquet and loads sim_data.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from v2ecoli.library.card_vectors import extract_vectors
from v2ecoli.library.gene_meta import omics_labels
from v2ecoli.library.report_card import card_from_analysis

# outlier-table parameters baked into the omics criteria (Chris: min_count
# adjustable, default 10). The renderer reads these + the gene metadata.
_OMICS_OUTLIER = {"outlier_log2fc": 2.0, "min_count": 10, "outlier_top_n": 20}

# Physiology (cell-cycle) axes: (card path, label, units, display-scale,
# how-it's-measured). ref_values are baked from the blessed measured card.
_PHYSIO_AXES = [
    ("physiology.doubling_time", "Doubling time", "h", 1 / 3600,
     "Cell-cycle time over confirmed divisions (one value per divided cell)."),
    ("physiology.cell_mass", "Cell mass", "fg", 1.0,
     "Time-mean total cell mass (listeners__mass__cell_mass), one value/cell."),
    ("physiology.cell_volume", "Cell volume", "fL", 1.0,
     "Time-mean cell volume (listeners__mass__volume), one value/cell."),
    ("physiology.oric", "Replication origins (oriC)", "origins", 1.0,
     "Time-mean number_of_oric — >1 indicates overlapping replication rounds."),
    ("physiology.replication_initiation", "Replication initiation", "min", 1 / 60,
     "Per-cell time since birth of the first oriC step-up (a new round fires). "
     "Absent for cells born between rounds that don't re-initiate in-cycle."),
    ("physiology.replication_completion", "Replication completion", "min", 1 / 60,
     "Per-cell time since birth when replication forks first clear (a round "
     "terminates). NOTE: under overlapping rounds this tracks a DIFFERENT round "
     "than initiation above — they are not one round's start/end."),
]
_PHYSIO_SOURCE = " Source: v2ecoli/workflow/analysis_runner.py::build_cell_records."

# Ribosomes group (per-cell time-means). Mirrors vEcoli ribosome_usage.py:
# total = active + min(free 30S, 50S); usage = active/total + elongation rate.
_RIBO_AXES = [
    ("ribosomes.total", "Total ribosomes", "ribosomes", 1.0,
     "Time-mean total ribosomes = active (translating) + min(free 30S, 50S). "
     "Active is a clean listener; inactive subunits are pulled from the bulk "
     "arrays by molecule id. ~20k is the canonical E. coli content."),
    ("ribosomes.active_fraction", "Active fraction", "fraction", 1.0,
     "Fraction of ribosomes actively translating (active / total) — the "
     "translating-vs-idle reading of ribosome usage. ~0.8 at balanced growth."),
    ("ribosomes.elongation_rate", "Elongation rate", "aa/s", 1.0,
     "Time-mean effective peptide elongation rate per ribosome — the "
     "efficiency reading of ribosome usage. ~15-20 aa/s in E. coli."),
    ("ribosomes.production", "Ribosome production (rRNA init)", "rRNA/s", 1.0,
     "Time-mean total rRNA operons initiated per timestep — a ribosome-"
     "biogenesis-rate proxy (ribosome_data__did_initialize isn't emitted)."),
]
_RIBO_SOURCE = " Source: v2ecoli/workflow/analysis_runner.py::build_cell_records."

# card path -> (sim_data exchange-molecule id, presentation label)
_FLUX_KPIS = [
    ("fluxes.glucose", "GLC[p]", "Glucose exchange"),
    ("fluxes.o2", "OXYGEN-MOLECULE[p]", "O₂ exchange"),
    ("fluxes.nitrogen", "AMMONIUM[c]", "Ammonium (N source) exchange"),
    ("fluxes.co2", "CARBON-DIOXIDE[p]", "CO₂ exchange"),
    ("fluxes.acetate", "ACET[p]", "Acetate exchange (overflow sentinel)"),
]

_FLUX_HOW = ("Cell-level mean exchange flux (negative = uptake, positive = "
             "secretion). Time-mean within cell over gens≥3, then per-cell "
             "values across the ensemble. Order = sorted(sim_data.external_state"
             ".all_external_exchange_molecules) = the FBA emit order. "
             "Source: v2ecoli/library/card_vectors.py::extract_vectors.")

_OMICS = [
    ("omics.transcriptome", "Transcriptome (mRNA cistron counts)",
     "listeners__rna_counts__mRNA_cistron_counts"),
    ("omics.proteome", "Proteome (monomer counts)",
     "listeners__monomer_counts"),
]

_OMICS_HOW = ("Cell-level ensemble-mean count vector (time-mean within cell over "
              "gens≥3, then mean across cells). Graded by log-log R² vs "
              "the pinned blessed vector — self-pin => R²=1; a reformulation "
              "(e.g. PDMP) is the first run that moves it. "
              "Source: v2ecoli/library/card_vectors.py::extract_vectors.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", default="out/population_phenotype_basal")
    p.add_argument("--sim-data", default="out/sim_data_full/parca_state.pkl")
    p.add_argument("--analysis", default="out/population_phenotype_basal/analysis.json")
    p.add_argument("--gen-lb", type=int, default=3)
    p.add_argument("--reference", default=os.path.join(
        "tests", "fixtures", "population_phenotype_basal_reference.json"))
    args = p.parse_args()

    with open(args.reference, encoding="utf-8") as f:
        ref = json.load(f)

    print(f"[analysis] loading measured physiology card from {args.analysis} …")
    with open(args.analysis, encoding="utf-8") as f:
        measured = card_from_analysis(json.load(f))

    print(f"[sim_data] loading {args.sim_data} for the exchange-molecule order …")
    with open(args.sim_data, "rb") as f:
        sd = pickle.load(f)
    flux_ids = sorted(sd["external_state"].all_external_exchange_molecules)
    print(f"[sim_data] {len(flux_ids)} exchange molecules (FBA order)")

    print(f"[vectors] extracting from {args.sweep_dir} (gen >= {args.gen_lb}) …")
    vec = extract_vectors(args.sweep_dir, args.gen_lb)
    omics = vec.get("omics", {})
    exch = (vec.get("fluxes") or {}).get("exchange") or {}
    per_cell = exch.get("per_cell") or []
    n_cells = exch.get("n_cells", 0)
    if len(exch.get("vector", [])) != len(flux_ids):
        raise SystemExit(f"flux vector len {len(exch.get('vector', []))} != "
                         f"{len(flux_ids)} ids — order mismatch, aborting")

    axes = ref["axes"]
    composition = {k: v for k, v in axes.items() if v.get("group") == "Composition"}

    # ttest/violin axes whose ref_values are baked from the blessed measured
    # card's per-cell value lists (Physiology + Ribosomes share this shape).
    def _build_ttest_group(specs, group, source):
        out: dict[str, dict] = {}
        for path, label, units, scale, how in specs:
            node = measured
            for part in path.split("."):
                node = (node or {}).get(part)
            vals = (node or {}).get("values") or []
            # per-lineage [[seed, gen, value], ...] for the gen-trend companion
            # plot's reference (grey) lineages; self-pin => equals measured.
            bycell = (node or {}).get("variance", {}).get("by_cell") or []
            out[path] = {
                "group": group, "label": label, "units": units,
                "how": how + source, "plot": "violin", "y_from_zero": True,
                "scale": scale,
                "criterion": {"type": "ttest", "p_min": 0.05, "within_pct": 0.05,
                              "mismatch_pct": 0.10,
                              "ref_values": [round(v, 8) for v in vals],
                              "ref_by_cell": bycell},
            }
            print(f"[{group.lower()}] {path}: n={len(vals)} ref values")
        return out

    physiology = _build_ttest_group(_PHYSIO_AXES, "Physiology", _PHYSIO_SOURCE)
    ribosomes = _build_ttest_group(_RIBO_AXES, "Ribosomes", _RIBO_SOURCE)

    # per-flux cell-to-cell std (for scatter error bars + the companion table)
    ref_std = np.asarray(per_cell, float).std(axis=0) if per_cell else np.zeros(len(flux_ids))

    # --- Exchange-fluxes group ----------------------------------------------
    flux_axes: dict[str, dict] = {}
    flux_axes["fluxes.exchange"] = {
        "group": "Exchange fluxes",
        "label": "Exchange-flux fingerprint",
        "units": "mmol/gDCW/h",
        "how": ("All 87 external exchange fluxes (ensemble-mean) vs the pinned "
                "reference. A flux that appears (0→active) or disappears "
                "(active→0) is a qualitative regression = mismatch, flagged on "
                "the plot. " + _FLUX_HOW),
        "plot": "flux_scatter",
        "criterion": {
            "type": "flux_scatter",
            "ref_vector": [round(x, 8) for x in exch["vector"]],
            "ref_std": [round(float(x), 8) for x in ref_std],
            "flux_ids": flux_ids,
            "active_eps": 1e-6,
            "r2_min": 0.99,
            "r2_drift": 0.95,
        },
    }
    idx = {m: i for i, m in enumerate(flux_ids)}
    for path, fid, label in _FLUX_KPIS:
        j = idx[fid]
        ref_values = [round(row[j], 8) for row in per_cell]
        flux_axes[path] = {
            "group": "Exchange fluxes",
            "label": label,
            "units": "mmol/gDCW/h",
            "how": _FLUX_HOW,
            "plot": "violin",
            "y_from_zero": True,
            "criterion": {
                "type": "ttest",
                "p_min": 0.05,
                "within_pct": 0.05,
                "mismatch_pct": 0.10,
                "inactive_eps": 1e-6,
                "flux_id": fid,
                "ref_values": ref_values,
            },
        }

    # --- Gene expression group (transcriptome + proteome) -------------------
    # Gene metadata (ordered ids + symbol + descriptive name) for the outlier
    # tables — order from sim_data, symbol/name from the reconstruction flats.
    gene_labels = omics_labels(sd)
    geneexp_axes: dict[str, dict] = {}
    for path, label, _col in _OMICS:
        name = path.split(".")[1]
        node = omics.get(name) or {}
        meta = gene_labels[name]
        geneexp_axes[path] = {
            "group": "Gene expression",
            "label": label,
            "units": "counts",
            "how": _OMICS_HOW,
            "plot": "loglog",
            "criterion": {
                "type": "r2",
                "r2_min": 0.99,
                "r2_drift": 0.95,
                "ref_vector": [round(x, 6) for x in node.get("vector", [])],
                "ids": meta["ids"],
                "symbols": meta["symbols"],
                "names": meta["names"],
                **_OMICS_OUTLIER,
            },
        }

    # Order: Physiology -> Composition -> Ribosomes -> Exchange fluxes ->
    # Gene expression.
    ref["axes"] = {**physiology, **composition, **ribosomes, **flux_axes,
                   **geneexp_axes}
    ref.setdefault("stimulus", {})["generation_lower_bound"] = args.gen_lb
    ref["stimulus"]["n_cells_vectors"] = n_cells

    with open(args.reference, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {args.reference}: "
          f"{len(physiology)} physiology + {len(composition)} composition + "
          f"{len(ribosomes)} ribosomes + {len(flux_axes)} flux + "
          f"{len(geneexp_axes)} gene-expression axes (flux n_cells={n_cells})")


if __name__ == "__main__":
    main()
