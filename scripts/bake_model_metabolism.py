"""Bake the model-side fixtures for the basal card's Metabolism + Proteome sections.

Mirrors ``render_basal_vs_literature.physiology_from_sweep``: reads the blessed
baseline sweep parquet, aggregates per cell (time-average within each cell,
gen >= LB), and writes committed JSON fixtures so the card + tests stay
independent of the (gitignored) sweep.

Outputs (committed golden fixtures under tests/fixtures/population_phenotype_basal/,
each carrying a `provenance` block — see scripts/_provenance.py):
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ for _provenance
from _provenance import provenance_stamp  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# Golden model fixtures (committed, sim-derived) -> tests/fixtures; see _provenance.py.
FIXTURES = REPO / "tests/fixtures/population_phenotype_basal"
_SWEEP = REPO / "out/population_phenotype_basal"
_GEN_LB = 3  # generation_lower_bound of the blessed sweep (must match the config)


def _parca_inputs_hash():
    try:
        return json.load(open(REPO / "out/cache/cache_version.json"))["inputs_hash"]
    except Exception:
        return None


def _stamp(n_cells):
    return provenance_stamp(
        REPO, config="v2ecoli/configs/population_phenotype_basal.json",
        sweep={"sweep_dir": str(_SWEEP), "parca_inputs_hash": _parca_inputs_hash(),
               "n_cells": n_cells, "gen_lb": _GEN_LB},
        bake_script="scripts/bake_model_metabolism.py --from-sweep <dir>")
_PARCA_STATE = REPO / "out/sim_data_full/parca_state.pkl.gz"
GEN_LB = 3

# Crown↔model metabolite ids (cytoplasm), for sign-aligning model fluxes to the
# reference's substrate->product direction. The model carries three G6P synonyms
# (GLC-6-P / ALPHA-GLC-6-P / D-glucopyranose-6-phosphate) joined by epimerases:
# glycolytic Pgi acts on ALPHA-GLC-6-P, oxidative-PPP Zwf on D-glucopyranose-6-P.
CC_MET = {
    "G6P": "ALPHA-GLC-6-P[c]", "dG6P": "D-glucopyranose-6-phosphate[c]",
    "F6P": "FRUCTOSE-6P[c]", "FBP": "FRUCTOSE-16-DIPHOSPHATE[c]",
    "DHAP": "DIHYDROXY-ACETONE-PHOSPHATE[c]", "GAP": "GAP[c]", "DPG": "DPG[c]",
    "2PG": "2-PG[c]", "PEP": "PHOSPHO-ENOL-PYRUVATE[c]", "Pyr": "PYRUVATE[c]",
    "6PGL": "D-6-P-GLUCONO-DELTA-LACTONE[c]", "6PG": "CPD-2961[c]",
    "Ru5P": "RIBULOSE-5P[c]", "X5P": "XYLULOSE-5-PHOSPHATE[c]", "R5P": "RIBOSE-5P[c]",
    "KDPG": "2-KETO-3-DEOXY-6-P-GLUCONATE[c]", "AcCoA": "ACETYL-COA[c]",
    "AcP": "ACETYL-P[c]", "Cit": "CIT[c]", "ICit": "THREO-DS-ISO-CITRATE[c]",
    "AKG": "2-KETOGLUTARATE[c]", "SucCoA": "SUC-COA[c]", "Suc": "SUC[c]",
    "Fum": "FUM[c]", "Mal": "MAL[c]", "OAC": "OXALACETIC_ACID[c]",
    "Glyox": "GLYOX[c]", "Ac": "ACET[c]",
}
# Central-carbon reaction set, paired to Crown 2015 COMPLETE-MFA. Each row:
# (label, crown_fid, base-reaction id, substrate, product, group, flag). The
# model's signed flux is aligned to the (substrate->product) direction Crown
# writes, so a genuine reverse-runner (Fum/MDH) plots negative against Crown's
# positive. GAPDH/Eno are series-lumped in Crown (through 1,3-BPG / 2-PG) — we
# take the representative member's flux (equal at steady state). flags:
#   pts_coupled       — PEP->Pyr partition is non-identifiable vs the model's
#                       glucokinase-heavy glucose entry (PTS vs GLK+PYK give the
#                       same ¹³C labeling); not an independent claim.
#   aldolase_bypass   — the model routes most hexose->triose through fructose-6-P
#                       aldolase + sedoheptulose-1,7-bisP aldolase + DHA kinase,
#                       so Pfk/Fba carry only ~⅓ of the throughput. A known FBA
#                       carbon-rearrangement degeneracy; node balances unaffected.
#   reductive_reverse — the model runs the lower TCA reductively (OAC->Mal->Fum).
CC_SPEC = [
    ("Pgi", "crown_f2", "PGLUCISOM-RXN", "G6P", "F6P", "glycolysis", None),
    ("Pfk", "crown_f3", "6PFRUCTPHOS-RXN", "F6P", "FBP", "glycolysis", "aldolase_bypass"),
    ("Fba", "crown_f4", "F16ALDOLASE-RXN", "FBP", "GAP", "glycolysis", "aldolase_bypass"),
    ("Tpi", "crown_f5", "TRIOSEPISOMERIZATION-RXN", "DHAP", "GAP", "glycolysis", None),
    ("GAPDH", "crown_f6", "GAPOXNPHOSPHN-RXN", "GAP", "DPG", "glycolysis", None),
    ("Eno", "crown_f7", "2PGADEHYDRAT-RXN", "2PG", "PEP", "glycolysis", None),
    ("Pyk", "crown_f8", "PEPDEPHOS-RXN", "PEP", "Pyr", "glycolysis", "pts_coupled"),
    ("Zwf", "crown_f9", "GLU6PDEHYDROG-RXN", "dG6P", "6PGL", "oxPPP", None),
    ("6PGDH", "crown_f10", "RXN-9952", "6PG", "Ru5P", "oxPPP", None),
    ("Rpe", "crown_f11", "RIBULP3EPIM-RXN", "Ru5P", "X5P", "PPP", None),
    ("Rpi", "crown_f12", "RIB5PISOM-RXN", "Ru5P", "R5P", "PPP", None),
    ("EDD / EDA", "crown_f18", "PGLUCONDEHYDRAT-RXN", "6PG", "KDPG", "ED", None),
    ("PDH", "crown_f20", "PYRUVDEH-RXN", "Pyr", "AcCoA", "TCA", None),
    ("CS", "crown_f21", "CITSYN-RXN", "AcCoA", "Cit", "TCA", None),
    ("ICDH", "crown_f23", "ISOCITDEH-RXN", "ICit", "AKG", "TCA", None),
    ("aKGDH", "crown_f24", "2OXOGLUTARATEDEH-RXN", "AKG", "SucCoA", "TCA", None),
    ("SDH", "crown_f26", "SUCCINATE-DEHYDROGENASE-UBIQUINONE-RXN", "Suc", "Fum", "TCA", None),
    ("Fum", "crown_f27", "FUMHYDR-RXN", "Fum", "Mal", "TCA", "reductive_reverse"),
    ("MDH", "crown_f28", "MALATE-DEH-RXN", "Mal", "OAC", "TCA", "reductive_reverse"),
    ("Icl", "crown_f29", "ISOCIT-CLEAV-RXN", "ICit", "Glyox", "glyoxylate", None),
    ("MS", "crown_f30", "MALSYN-RXN", "Glyox", "Mal", "glyoxylate", None),
    ("Ppc", "crown_f33", "PEPCARBOX-RXN", "PEP", "OAC", "anaplerotic", None),
    ("Ack", "crown_f35", "ACETATEKIN-RXN", "AcP", "Ac", "overflow", None),
]
_LABEL2BASE = {row[0]: row[2] for row in CC_SPEC}
# G6P-fate composition branches -> their label in CC_SPEC.
G6P_BRANCH = {"EMP": "Pgi", "oxPPP": "6PGDH", "ED": "EDD / EDA"}
ACCOA_MET = "ACETYL-COA[c]"   # acetyl-CoA, the AcCoA-fate node metabolite
# external_exchange_fluxes is emitted in sorted(all_external_exchange_molecules)
# order; 1-indexed positions verified against the sweep.
EX_IDX = {"glucose": 37, "o2": 66, "co2": 11, "acetate": 3}
M_C = 12.011  # g/mol carbon


def _phys_stoich(metabolism) -> dict:
    """Net stoichiometry of each base reaction at POSITIVE ``base_reaction_flux``
    (its physical-forward direction). ``reaction_stoich`` stores some base
    reactions only as a ``(reverse)`` entry — there the stored 'forward' is the
    gluconeogenic/reverse convention, so positive flux runs the negated reaction.
    Negating those is what makes pyruvate kinase / PEP carboxylase / triose-P
    isomerase read in their true physiological direction rather than flipped."""
    rs = metabolism.reaction_stoich
    by_base: dict = {}
    for d, b in metabolism.reaction_id_to_base_reaction_id.items():
        by_base.setdefault(b, []).append(d)
    out = {}
    for b in metabolism.base_reaction_ids:
        if b in rs:
            out[b] = rs[b]
        elif f"{b} (reverse)" in rs:
            out[b] = {k: -v for k, v in rs[f"{b} (reverse)"].items()}
        else:
            cands = [d for d in by_base.get(b, []) if d in rs]
            if cands:
                out[b] = ({k: -v for k, v in rs[cands[0]].items()}
                          if "(reverse)" in cands[0] else rs[cands[0]])
    return out


def _sign_mult(phys: dict, base: str, sub: str, prod: str) -> float:
    """+1 if positive base flux carries ``sub``->``prod``, else -1 — aligns the
    model's signed flux to the reference reaction's substrate->product direction.
    Falls back to +1 when stoichiometry is unavailable (only the near-zero ED
    branch, where the sign is immaterial)."""
    st = phys.get(base)
    if not st or sub not in st or prod not in st:
        return 1.0
    return 1.0 if st[sub] < 0 else -1.0


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
    """Per-cell central-carbon fluxes (signed, reference-direction-aligned), the
    G6P branch-point composition, and boundary exchanges from the sweep."""
    from wholecell.utils import units
    metabolism = sim_data.process.metabolism
    brids = list(metabolism.base_reaction_ids)
    bidx = {r: i for i, r in enumerate(brids)}
    phys = _phys_stoich(metabolism)
    dens = sim_data.constants.cell_density.asNumber(units.g / units.L)

    # Extract the CC_SPEC reactions + every base reaction touching acetyl-CoA
    # (needed for the AcCoA-fate node's total-consumption denominator).
    accoa_bases = [b for b in brids if ACCOA_MET in (phys.get(b) or {})]
    bases = sorted({row[2] for row in CC_SPEC} | set(accoa_bases))
    col = {b: f"r{i}" for i, b in enumerate(bases)}    # SQL-safe aliases
    flist = _parquet_glob(sweep_dir)
    con = duckdb.connect()
    selb = ", ".join(
        f"list_extract(listeners__fba_results__base_reaction_fluxes,{bidx[b] + 1}) {col[b]}"
        for b in bases)
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
    coef = df.dm / df.cm * dens                        # g/L
    for b in bases:                                    # raw mmol/gDW/s -> mmol/gDW/h
        df[col[b]] = df[col[b]] / coef * 3600.0
    for k in EX_IDX:
        df[f"e_{k}"] = df[f"e_{k}"].abs() if k != "co2" else df[f"e_{k}"]
    pc = df.groupby(["s", "g", "a"]).mean(numeric_only=True)   # per-cell time means
    n = len(pc)
    M = pc.mean()

    def signed(label):
        """Per-cell signed mmol/gDW/h for a CC_SPEC reaction, aligned to the
        reference (substrate->product) direction (reverse-runners go negative)."""
        _, _, base, sub, prod, _, _ = next(r for r in CC_SPEC if r[0] == label)
        return _sign_mult(phys, base, CC_MET[sub], CC_MET[prod]) * pc[col[base]]

    # G6P composition: per-cell ternary fractions + closure residual vs glucose influx.
    bran = list(G6P_BRANCH)                            # [EMP, oxPPP, ED]
    bser = {b: signed(G6P_BRANCH[b]) for b in bran}    # branch -> per-cell flux
    bsum = sum(bser[b] for b in bran)
    fr = {b: bser[b] / bsum for b in bran}             # ternary (renormalized)
    resid = 1.0 - bsum / pc["e_glucose"]               # unaccounted G6P -> biomass
    g6p = {
        "branches": bran,
        "model_flux": {b: float(bser[b].mean()) for b in bran},   # mmol/gDW/h
        "influx": float(M["e_glucose"]),
        "fractions": {b: float(fr[b].mean()) for b in bran},
        "residual": float(resid.mean()),
        "per_cell_fractions": [[round(float(fr[b].iloc[i]), 4) for b in bran]
                               for i in range(n)],
    }

    # Central-carbon flux vector: per reaction, signed and normalized to glucose
    # uptake = 100 (per cell, then ensemble) — the model-vs-Crown scatter. Each
    # row carries its annotation flag (pts_coupled / aldolase_bypass /
    # reductive_reverse); the Crown pairing is resolved card-side via crown_fid.
    reactions = []
    for label, fid, base, sub, prod, group, flag in CC_SPEC:
        sg = signed(label)
        norm = (sg / pc["e_glucose"]) * 100.0           # per-cell normalized flux
        reactions.append({
            "label": label, "crown_fid": fid, "group": group, "flag": flag,
            "model": float(norm.mean()),
            "model_std": float(norm.std()),              # cell-to-cell spread
            "model_abs": float(sg.mean()),
        })
    central_carbon = {"normalized_to_glucose_100": True, "reactions": reactions}

    # Isocitrate fate: oxidative TCA (ICDH) vs glyoxylate shunt (ICL) — the only
    # two consumers of isocitrate, so influx = their sum (no residual).
    icdh, icl = float(signed("ICDH").mean()), float(signed("Icl").mean())
    iso_in = icdh + icl                                 # ratio-of-means (robust)
    isocitrate = {
        "branches": ["oxidative_TCA", "glyoxylate"],
        "model_flux": {"oxidative_TCA": icdh, "glyoxylate": icl},
        "influx": iso_in,
        "fractions": {"oxidative_TCA": icdh / iso_in, "glyoxylate": icl / iso_in},
    }

    # AcCoA fate: TCA (citrate synthase) / acetate overflow / biosynthesis (the
    # rest — fatty acids + amino acids). influx = total per-cell AcCoA consumption
    # (Σ over every consuming reaction); biosynthesis is the balance.
    cs = float(signed("CS").mean())                     # AcCoA -> citrate
    consume = float(sum(((-(phys[b][ACCOA_MET])) * pc[col[b]]).clip(lower=0)
                        for b in accoa_bases).mean())
    ac_fate = float(pc["e_acetate"].mean())             # acetate carbon leaving (~0)
    bios = consume - cs - ac_fate
    accoa = {
        "branches": ["TCA", "acetate", "biosynthesis"],
        "model_flux": {"TCA": cs, "acetate": ac_fate, "biosynthesis": bios},
        "influx": consume,                              # ratio-of-means (robust)
        "fractions": {"TCA": cs / consume, "acetate": ac_fate / consume,
                      "biosynthesis": bios / consume},
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
            "nodes": {"g6p": g6p, "isocitrate": isocitrate, "accoa": accoa},
            "exchanges": exch, "central_carbon": central_carbon}


def proteome_from_sweep(sweep_dir: Path, parca_state) -> dict:
    """Ensemble-mean protein copies/cell, keyed by EcoCyc monomer id AND by gene
    symbol, from monomer_counts.

    ``by_id`` is the join key for any cross-source comparison; ``by_symbol`` is
    kept because the literature reference this card grades against
    (``data/basal/proteome.tsv``) is itself symbol-keyed. Symbol is not a safe
    join key *across* sources — symbol spaces differ between databases and are
    not injective across them — so a consumer joining to anything but that
    reference should use ``by_id``.
    """
    from v2ecoli.library.gene_meta import omics_labels
    labels = omics_labels(parca_state)["proteome"]
    symbols = [str(s) for s in labels["symbols"]]
    monomer_ids = [str(m) for m in labels["ids"]]
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
    by_symbol, n_symbol_collisions, n_symbol_dropped = {}, 0, 0
    for sym, val in zip(symbols, ensemble):
        if sym and sym != "None":
            if sym in by_symbol:
                n_symbol_collisions += 1
            by_symbol[sym] = by_symbol.get(sym, 0.0) + float(val)
        else:
            n_symbol_dropped += 1
    by_id, n_id_dropped = {}, 0
    for mid, val in zip(monomer_ids, ensemble):
        if mid and mid != "None":
            by_id[mid] = float(val)
        else:
            n_id_dropped += 1
    return {"n_cells": len(per_cell), "gen_lb": GEN_LB,
            "method": "blessed baseline sweep; per-cell time-mean monomer_counts, gen>=%d" % GEN_LB,
            "units": "copies/cell", "by_id": by_id, "by_symbol": by_symbol,
            "id_key": "EcoCyc monomer id",
            "keying": {"n_monomers": len(ensemble),
                       "n_by_id": len(by_id), "n_id_unmapped": n_id_dropped,
                       "n_by_symbol": len(by_symbol),
                       "n_symbol_unmapped": n_symbol_dropped,
                       "n_symbol_collisions": n_symbol_collisions}}


def composition_from_sweep(sweep_dir: Path) -> dict:
    """Per-cell macromolecular dry-mass fractions (protein / RNA / DNA / other)
    + the model's doubling time, from the sweep. RNA is total RNA (rRNA+tRNA+
    mRNA); 'other' is the dry-weight remainder (the model's small-molecule pool).
    Graded vs the Bremer & Dennis growth-rate composition at the matched td."""
    flist = _parquet_glob(sweep_dir)
    con = duckdb.connect()
    df = con.sql(f"""
        SELECT lineage_seed s, generation g, agent_id a, global_time t,
               listeners__mass__dry_mass dm, listeners__mass__protein_mass pm,
               listeners__mass__rna_mass rm, listeners__mass__dna_mass dn
        FROM read_parquet({flist}, hive_partitioning=true)
        WHERE generation >= {GEN_LB} AND listeners__mass__dry_mass > 0
    """).df()
    df["protein"] = df.pm / df.dm
    df["rna"] = df.rm / df.dm
    df["dna"] = df.dn / df.dm
    pc = df.groupby(["s", "g", "a"])[["protein", "rna", "dna"]].mean()
    span = df.groupby(["s", "g", "a"]).t.agg(lambda x: x.max() - x.min())
    n = len(pc)
    fr = {c: float(pc[c].mean()) for c in ("protein", "rna", "dna")}
    fr["other"] = 1.0 - fr["protein"] - fr["rna"] - fr["dna"]
    branches = ["protein", "rna", "dna", "other"]
    per_cell = [[round(float(pc[c].iloc[i]), 4) for c in ("protein", "rna", "dna")]
                + [round(1.0 - sum(pc[c].iloc[i] for c in ("protein", "rna", "dna")), 4)]
                for i in range(n)]
    return {"n_cells": n, "gen_lb": GEN_LB,
            "method": "blessed baseline sweep; per-cell time-mean mass fractions, gen>=%d" % GEN_LB,
            "doubling_time_min": float(span.mean() / 60.0),
            "branches": branches, "fractions": fr,
            "per_cell_fractions": per_cell}


def metabolite_pools_from_sweep(sweep_dir: Path) -> dict:
    """Per-metabolite realized intracellular concentrations (mol/L) from the
    sweep — bulk molecule count / (cell volume · Nₐ), per-cell time-mean then
    ensemble — over the metabolites carrying a Bennett-derived concentration
    target (the model's `metabolite_concentrations.tsv` Bennett column gives the
    model-EcoCyc-id ↔ Bennett-value mapping). Reports the aggregate total pool
    vs the Bennett total over the same set, plus the per-metabolite pairs (the
    model side of the per-metabolite scatter; the name→id map to the validation
    slot is a follow-up)."""
    import csv as _csv
    import ecoli_sources
    NA = 6.022e23
    es = os.path.dirname(ecoli_sources.__file__)
    recon = {}
    with open(os.path.join(es, "data/flat/metabolite_concentrations.tsv")) as f:
        for r in _csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
            mcol = next(k for k in r if k.startswith("Metabolite"))
            vcol = next(k for k in r if "Bennett" in k)
            if r[vcol] and r[vcol] != "NaN":
                recon[r[mcol]] = float(r[vcol])

    flist = _parquet_glob(sweep_dir)
    con = duckdb.connect()
    bid = con.sql(f"SELECT bulk__id FROM read_parquet({flist}) "
                  f"WHERE len(bulk__id)>0 LIMIT 1").fetchone()[0]
    idpos = {b: i for i, b in enumerate(bid)}
    matched = {}                                       # ecocyc id -> bulk index
    for mid in recon:
        for cand in (f"{mid}[c]", f"{mid}[p]", mid):
            if cand in idpos:
                matched[mid] = idpos[cand]
                break
    idxs = sorted(set(matched.values()))
    sel = ", ".join(f"list_extract(bulk__count,{i + 1}) c{i}" for i in idxs)
    df = con.sql(f"""
        SELECT lineage_seed s, generation g, agent_id a,
               listeners__mass__volume vol, {sel}
        FROM read_parquet({flist}, hive_partitioning=true)
        WHERE generation >= {GEN_LB} AND len(bulk__count) > 0
          AND listeners__mass__volume > 0
    """).df()
    for i in idxs:                                     # count / (vol[fL]·1e-15·Nₐ) -> mol/L
        df[f"m{i}"] = df[f"c{i}"] / (df.vol * 1e-15 * NA)
    pc = df.groupby(["s", "g", "a"])[[f"m{i}" for i in idxs]].mean()
    M = pc.mean()
    per = {mid: {"model": float(M[f"m{matched[mid]}"]), "bennett": recon[mid]}
           for mid in matched}
    model_total = sum(p["model"] for p in per.values())
    bennett_total = sum(p["bennett"] for p in per.values())
    return {"n_cells": len(pc), "gen_lb": GEN_LB,
            "method": "blessed baseline sweep; bulk count / (volume·Nₐ), per-cell time-mean, gen>=%d" % GEN_LB,
            "units": "mol/L", "n_matched": len(matched),
            "model_total": model_total, "bennett_total": bennett_total,
            "ratio": model_total / bennett_total if bennett_total else None,
            "per_metabolite": per}


def main(from_sweep: str | None, out: str | None = None,
         gen_lb: int | None = None) -> None:
    global GEN_LB, _GEN_LB, FIXTURES
    sweep = Path(from_sweep) if from_sweep else _SWEEP
    if gen_lb is not None:
        # Every aggregation reads the module-level GEN_LB (it appears in each
        # SQL WHERE), so the flag sets it once here rather than threading an
        # argument through nine call sites.
        GEN_LB = _GEN_LB = int(gen_lb)
    if out is not None:
        FIXTURES = Path(out)
    state, sim_data = _load()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    met = metabolism_from_sweep(sweep, sim_data)
    met["provenance"] = _stamp(met.get("n_cells"))
    (FIXTURES / "model_metabolism.json").write_text(json.dumps(met, indent=2))
    pro = proteome_from_sweep(sweep, state)
    pro["provenance"] = _stamp(pro.get("n_cells"))
    (FIXTURES / "model_proteome.json").write_text(json.dumps(pro, indent=2))
    comp = composition_from_sweep(sweep)
    comp["provenance"] = _stamp(comp.get("n_cells"))
    (FIXTURES / "model_composition.json").write_text(json.dumps(comp, indent=2))
    print(f"model_composition.json: {comp['n_cells']} cells, td {comp['doubling_time_min']:.1f} min, "
          f"protein {comp['fractions']['protein']:.3f} RNA {comp['fractions']['rna']:.3f} "
          f"DNA {comp['fractions']['dna']:.4f} other {comp['fractions']['other']:.3f}")
    pools = metabolite_pools_from_sweep(sweep)
    pools["provenance"] = _stamp(pools.get("n_cells"))
    (FIXTURES / "model_metabolite_pools.json").write_text(json.dumps(pools, indent=2))
    print(f"model_metabolite_pools.json: {pools['n_matched']} metabolites, "
          f"model total {pools['model_total']*1000:.0f} mM vs Bennett {pools['bennett_total']*1000:.0f} mM "
          f"(ratio {pools['ratio']:.2f})")

    g = met["nodes"]["g6p"]["fractions"]
    e = met["exchanges"]
    print(f"model_metabolism.json: {met['n_cells']} cells")
    print(f"  G6P ternary: EMP {100*g['EMP']:.1f}% oxPPP {100*g['oxPPP']:.1f}% "
          f"ED {100*g['ED']:.1f}% | residual {100*met['nodes']['g6p']['residual']:.1f}%")
    print(f"  exchanges abs: {e['absolute']} | RQ {e['rq']:.2f}")
    print(f"  C-mol %: {e['cmol_pct']}")
    k = pro["keying"]
    print(f"model_proteome.json: {pro['n_cells']} cells, {k['n_by_id']} ids / "
          f"{k['n_by_symbol']} symbols over {k['n_monomers']} monomers "
          f"({k['n_symbol_collisions']} symbol collisions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-sweep", metavar="DIR", default=None)
    ap.add_argument("--out", metavar="DIR", default=None,
                    help=f"write fixtures here (default: {FIXTURES.relative_to(REPO)})")
    ap.add_argument("--gen-lb", type=int, default=None,
                    help=f"generation lower bound / burn-in (default: {GEN_LB}); "
                         "must match the sweep's config")
    a = ap.parse_args()
    main(from_sweep=a.from_sweep, out=a.out, gen_lb=a.gen_lb)
