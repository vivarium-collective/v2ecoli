"""Bake the model-side fixtures for the basal card's Metabolism + Proteome sections.

Mirrors ``render_basal_vs_literature.physiology_from_sweep``: reads the blessed
baseline sweep parquet, aggregates per cell (time-average within each cell,
gen >= LB), and writes committed JSON fixtures so the card + tests stay
independent of the (gitignored) sweep.

Outputs (under docs/report_cards/population_phenotype_basal/vs_literature/):
  * model_metabolism.json — the G6P/glycolysis branch-point composition
    (EMP/oxPPP/ED + closure residual) and the boundary exchanges
    (O2/CO2/acetate/glucose, absolute + C-mol balance + per-glucose normalized + RQ).
  * model_proteome.json — ensemble-mean protein copies/cell per gene symbol.

Aggregation (cell-level discipline): for every per-cell quantity we time-average
*within* a cell first, then take the ensemble mean across cells; per-cell values
are retained for the card's distribution views. Internal-flux compositions use
flux ratios (unit-free); exchanges and the closure residual use specific fluxes
in mmol/gDW/h (base_reaction_fluxes are raw mmol/gDW/s -> *3600/coefficient;
external_exchange_fluxes are already specific). Run:

    python scripts/bake_model_metabolism.py --from-sweep out/population_phenotype_basal
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import duckdb
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/report_cards/population_phenotype_basal/vs_literature"
_SWEEP = REPO / "out/population_phenotype_basal"
_PARCA_STATE = REPO / "out/sim_data_full/parca_state.pkl.gz"
GEN_LB = 3

# Model base-reaction ids (EcoCyc) for the validated G6P/glycolysis node.
# EMP = phosphoglucose isomerase; oxPPP = 6-phosphogluconate dehydrogenase;
# ED = KDPG aldolase. (AcCoA/isocitrate nodes are held pending mapping
# validation — their single-id maps did not reproduce a physical flux ordering.)
G6P_RX = {"EMP": "PGLUCISOM-RXN", "oxPPP": "RXN-9952", "ED": "KDPGALDOL-RXN"}
# external_exchange_fluxes is emitted in sorted(all_external_exchange_molecules)
# order; 1-indexed positions verified against the sweep.
EX_IDX = {"glucose": 37, "o2": 66, "co2": 11, "acetate": 3}
M_C = 12.011  # g/mol carbon


def _load():
    """Return (parca_state dict, hydrated SimulationDataEcoli). The dict feeds
    ``gene_meta.omics_labels`` (which indexes ``state["process"][...]``); the
    object feeds the metabolism extraction (``.process.metabolism`` etc.)."""
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state)
    state = load_parca_state(str(_PARCA_STATE))
    return state, hydrate_sim_data_from_state(state)


def _parquet_glob(sweep_dir: Path) -> str:
    files = glob.glob(os.path.join(str(sweep_dir), "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        raise SystemExit(f"no parquet under {sweep_dir} (need a blessed-baseline sweep)")
    return "[" + ",".join(f"'{f}'" for f in files) + "]"


def metabolism_from_sweep(sweep_dir: Path, sim_data) -> dict:
    """Per-cell G6P composition + boundary exchanges from the sweep."""
    from wholecell.utils import units
    brids = list(sim_data.process.metabolism.base_reaction_ids)
    bidx = {r: i for i, r in enumerate(brids)}
    dens = sim_data.constants.cell_density.asNumber(units.g / units.L)
    flist = _parquet_glob(sweep_dir)
    con = duckdb.connect()
    selb = ", ".join(
        f"list_extract(listeners__fba_results__base_reaction_fluxes,{bidx[v] + 1}) {k}"
        for k, v in G6P_RX.items())
    selx = ", ".join(
        f"list_extract(listeners__fba_results__external_exchange_fluxes,{j}) e_{k}"
        for k, j in EX_IDX.items())
    df = con.sql(f"""
        SELECT lineage_seed s, generation g, agent_id a,
               listeners__mass__cell_mass cm, listeners__mass__dry_mass dm,
               {selb}, {selx}
        FROM read_parquet({flist}, hive_partitioning=true)
        WHERE generation >= {GEN_LB}
          AND len(listeners__fba_results__base_reaction_fluxes) > 0
    """).df()
    coef = df.dm / df.cm * dens                       # g/L
    for k in G6P_RX:                                  # raw mmol/gDW/s -> mmol/gDW/h
        df[k] = df[k] / coef * 3600.0
    for k in EX_IDX:
        df[f"e_{k}"] = df[f"e_{k}"].abs() if k != "co2" else df[f"e_{k}"]
    pc = df.groupby(["s", "g", "a"]).mean(numeric_only=True)   # per-cell time means
    n = len(pc)

    # G6P composition: per-cell ternary fractions + closure residual vs glucose influx.
    branches = ["EMP", "oxPPP", "ED"]
    bsum = pc[branches].sum(axis=1)
    fr = pc[branches].div(bsum, axis=0)               # ternary (renormalized)
    resid = 1.0 - bsum / pc["e_glucose"]              # unaccounted G6P -> biomass
    M = pc.mean()
    g6p = {
        "branches": branches,
        "model_flux": {k: float(M[k]) for k in branches},      # mmol/gDW/h
        "influx": float(M["e_glucose"]),
        "fractions": {k: float(fr[k].mean()) for k in branches},
        "residual": float(resid.mean()),
        "per_cell_fractions": fr[branches].round(4).values.tolist(),
    }
    # Exchanges: absolute, C-mol balance, per-glucose normalized, RQ.
    glc, o2, co2, ac = (float(M["e_glucose"]), float(M["e_o2"]),
                        float(M["e_co2"]), float(M["e_acetate"]))
    gC = glc * 6.0
    exch = {
        "absolute": {"glucose": glc, "o2": o2, "co2": co2, "acetate": ac},
        "per_cell": {k: pc[f"e_{k}"].round(4).tolist()
                     for k in EX_IDX},
        "cmol_pct": {"co2": 100 * co2 / gC, "acetate": 100 * ac * 2 / gC,
                     "biomass": 100 * (gC - co2 - ac * 2) / gC},
        "per_glucose": {"o2": o2 / glc, "co2": co2 / glc, "acetate": ac / glc},
        "rq": co2 / o2 if o2 else None,
    }
    return {"n_cells": n, "gen_lb": GEN_LB,
            "method": "blessed baseline sweep; per-cell time-mean, gen>=%d" % GEN_LB,
            "nodes": {"g6p": g6p}, "exchanges": exch}


def proteome_from_sweep(sweep_dir: Path, parca_state) -> dict:
    """Ensemble-mean protein copies/cell per gene symbol from monomer_counts."""
    from v2ecoli.library.gene_meta import omics_labels
    labels = omics_labels(parca_state)["proteome"]
    symbols = [str(s) for s in labels["symbols"]]
    flist = _parquet_glob(sweep_dir)
    con = duckdb.connect()
    cells = con.sql(f"""
        SELECT DISTINCT lineage_seed s, generation g, agent_id a
        FROM read_parquet({flist}, hive_partitioning=true)
        WHERE generation >= {GEN_LB} AND len(listeners__monomer_counts) > 0
    """).df()
    per_cell = []
    for _, c in cells.iterrows():
        rows = con.sql(f"""
            SELECT listeners__monomer_counts mc
            FROM read_parquet({flist}, hive_partitioning=true)
            WHERE lineage_seed={int(c.s)} AND generation={int(c.g)}
              AND agent_id='{c.a}' AND len(listeners__monomer_counts) > 0
        """).fetchall()
        per_cell.append(np.stack([r[0] for r in rows]).mean(axis=0))  # time-mean
    ensemble = np.stack(per_cell).mean(axis=0)                        # across cells
    if len(symbols) != len(ensemble):
        raise SystemExit(f"symbol/count width mismatch: {len(symbols)} vs {len(ensemble)}")
    # collapse to gene symbol (sum monomers sharing a symbol is overkill here; ids are 1:1)
    by_symbol = {}
    for sym, val in zip(symbols, ensemble):
        if sym and sym != "None":
            by_symbol[sym] = by_symbol.get(sym, 0.0) + float(val)
    return {"n_cells": len(per_cell), "gen_lb": GEN_LB,
            "method": "blessed baseline sweep; per-cell time-mean monomer_counts, gen>=%d" % GEN_LB,
            "units": "copies/cell", "by_symbol": by_symbol}


def main(from_sweep: str | None) -> None:
    sweep = Path(from_sweep) if from_sweep else _SWEEP
    state, sim_data = _load()
    OUT.mkdir(parents=True, exist_ok=True)
    met = metabolism_from_sweep(sweep, sim_data)
    (OUT / "model_metabolism.json").write_text(json.dumps(met, indent=2))
    pro = proteome_from_sweep(sweep, state)
    (OUT / "model_proteome.json").write_text(json.dumps(pro, indent=2))

    g = met["nodes"]["g6p"]["fractions"]
    e = met["exchanges"]
    print(f"model_metabolism.json: {met['n_cells']} cells")
    print(f"  G6P ternary: EMP {100*g['EMP']:.1f}% oxPPP {100*g['oxPPP']:.1f}% "
          f"ED {100*g['ED']:.1f}% | residual {100*met['nodes']['g6p']['residual']:.1f}%")
    print(f"  exchanges abs: {e['absolute']} | RQ {e['rq']:.2f}")
    print(f"  C-mol %: {e['cmol_pct']}")
    print(f"model_proteome.json: {pro['n_cells']} cells, {len(pro['by_symbol'])} symbols")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-sweep", metavar="DIR", default=None)
    a = ap.parse_args()
    main(from_sweep=a.from_sweep)
