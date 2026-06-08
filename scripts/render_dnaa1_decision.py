"""dnaa-1 expression-tuning DECISION figure (generation-aligned, 4 panels).

Connects the intervention (raised dnaA transcription, V=1.7e-3) to its
molecular consequence and the preserved cell-cycle phenotype:

  (a) dnaA mRNA initiation events per minute   [early-cycle SAMPLE — see note]
  (b) total DnaA monomer count                 (all 3 bulk forms)
  (c) DnaA concentration (nM)
  (d) oriC count

All four panels share the per-generation x-axis (gen 1 .. gen N).

DATA PROVENANCE / LIMITATION
----------------------------
The only dnaa-1 canonical run retained on this branch
(parquet-runs/dnaa1-mechA-1.7e-3-7gen) emits per-generation HISTORY parquet
truncated to roughly the first minute of each cycle (24-58 ticks at 1 s
cadence), plus one end-of-generation dill snapshot per generation. There is NO
full within-cycle trajectory on this branch.

  - Panels (b), (c), (d) use the genuine end-of-generation dill snapshots
    (one real point per generation = the pre-division peak state).
  - Panel (a) uses the dnaA-cistron initiation events found in the available
    early-cycle parquet ticks, divided by the SAMPLED minutes. Each generation
    only samples 0.0-0.9 min, so this is a sparse early-cycle ESTIMATE, NOT a
    per-cycle rate. It is drawn with open markers and labelled as such; the
    study's own canonical rate is "inferred ~0.85/min" for the same reason.

Re-running the full run is intentionally NOT done here (read-only, plot from
existing data only).

    python scripts/render_dnaa1_decision.py \
        --run studies/dnaa-1-expression/parquet-runs/dnaa1-mechA-1.7e-3-7gen \
        --out studies/dnaa-1-expression/charts
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import polars as pl

DNAA_FORMS = ["PD03831[c]", "MONOMER0-160[c]", "MONOMER0-4565[c]"]
DNAA_CISTRON_GENE = "EG10235"
AVOGADRO = 6.022e23
BAND_LO, BAND_HI = 300, 800          # accepted total-DnaA band (counts)
EXP = "dnaa1_mechA_1p7e-3_7gen"


def _cistron_idx() -> int:
    from v2ecoli.processes.parca.data_loader import (
        hydrate_sim_data_from_state, load_parca_state)
    sd = hydrate_sim_data_from_state(
        load_parca_state("models/parca/parca_state.pkl.gz"))
    gene_ids = np.asarray(sd.process.transcription.cistron_data["gene_id"])
    return int(np.where(gene_ids == DNAA_CISTRON_GENE)[0][0])


def _load(run_dir: str):
    """Return per-generation arrays from dills (b/c/d) + early-cycle init (a)."""
    import dill
    hist = os.path.join(run_dir, EXP, "history")
    dills = os.path.join(run_dir, "gen_dills")
    cis = _cistron_idx()

    gens = sorted(int(os.path.basename(p)[3:-5])
                  for p in glob.glob(os.path.join(dills, "gen*.dill")))
    if not gens:
        raise SystemExit(f"no gen*.dill under {dills}")

    rows = []
    for g in gens:
        # --- end-of-generation dill snapshot (real, one point/gen) ---
        with open(os.path.join(dills, f"gen{g}.dill"), "rb") as f:
            snap = dill.load(f)
        bulk = snap["bulk"]
        ids = bulk["id"]
        cnt = bulk["count"]
        idx = {m: int(np.where(ids == m)[0][0]) for m in DNAA_FORMS}
        dnaa_total = int(sum(cnt[idx[m]] for m in DNAA_FORMS))
        vol = snap["listeners"]["mass"]["volume"]
        vol_fL = float(getattr(vol, "magnitude", vol))
        conc_nM = dnaa_total / (vol_fL * AVOGADRO * 1e-15) * 1e9
        oric = int(snap["listeners"]["replication_data"]["number_of_oric"])

        # --- early-cycle parquet sample for the init rate (sparse) ---
        rate, sampled_min = float("nan"), 0.0
        gfiles = glob.glob(
            os.path.join(hist, f"**/generation={g}/**/*.pq"), recursive=True)
        if gfiles:
            df = (pl.read_parquet(gfiles[0])
                  .filter(pl.col(
                      "listeners__rnap_data__rna_init_event_per_cistron")
                      .list.len() > cis)
                  .sort("global_time"))
            if df.height > 1:
                t = df["global_time"].to_numpy()
                sampled_min = (t.max() - t.min()) / 60.0
                ev = (df["listeners__rnap_data__rna_init_event_per_cistron"]
                      .list.get(cis).fill_null(0).to_numpy())
                if sampled_min > 0:
                    rate = float(ev.sum()) / sampled_min
        rows.append(dict(gen=g, dnaa_total=dnaa_total, conc_nM=conc_nM,
                         oric=oric, init_rate=rate, sampled_min=sampled_min))
    return rows


def render(run_dir: str, out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load(run_dir)
    g = [r["gen"] for r in rows]
    total = [r["dnaa_total"] for r in rows]
    conc = [r["conc_nM"] for r in rows]
    oric = [r["oric"] for r in rows]
    rate = [r["init_rate"] for r in rows]

    fig, ax = plt.subplots(4, 1, figsize=(9.5, 11), sharex=True)

    # (a) dnaA mRNA initiation rate — early-cycle SAMPLE (open markers)
    valid = [(gi, ri) for gi, ri in zip(g, rate) if ri == ri]
    if valid:
        ax[0].plot([v[0] for v in valid], [v[1] for v in valid],
                   "o--", color="#8c564b", mfc="white", mew=1.5, ms=8)
    ax[0].set_ylabel("dnaA mRNA init\n(events/min)")
    ax[0].set_ylim(bottom=0)
    ax[0].text(0.5, 0.92,
               "early-cycle sample only (≤1 min/gen) — sparse estimate, "
               "not a per-cycle rate",
               transform=ax[0].transAxes, ha="center", va="top", fontsize=7.5,
               color="#8c564b", style="italic")

    # (b) total DnaA monomer count (end-of-gen, with accepted band)
    ax[1].axhspan(BAND_LO, BAND_HI, color="0.5", alpha=0.12, lw=0,
                  label=f"accepted band [{BAND_LO}, {BAND_HI}]")
    ax[1].plot(g, total, "o-", color="#222222", ms=7)
    ax[1].set_ylabel("total DnaA\n(counts)")
    ax[1].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax[1].text(0.5, 0.04,
               "end-of-generation (pre-division peak) snapshot",
               transform=ax[1].transAxes, ha="center", va="bottom",
               fontsize=7.5, color="0.4", style="italic")

    # (c) DnaA concentration
    ax[2].plot(g, conc, "o-", color="#ff7f0e", ms=7)
    ax[2].set_ylabel("DnaA conc\n(nM)")

    # (d) oriC count
    ax[3].plot(g, oric, "o-", color="#9467bd", ms=7)
    ax[3].set_ylabel("oriC\ncount")
    ax[3].set_ylim(0, max(4, max(oric)) + 0.4)
    ax[3].set_yticks(range(0, max(4, max(oric)) + 1))
    ax[3].set_xlabel("generation")
    ax[3].set_xticks(g)

    for a in ax:
        a.grid(True, alpha=0.15)

    fig.suptitle(
        "dnaa-1 expression tuning — DECISION figure (V=1.7e-3, seed 0)\n"
        "intervention → molecular consequence → preserved cell-cycle phenotype\n"
        "(generation-aligned; panels b–d = end-of-gen dill snapshots; "
        "panel a = sparse early-cycle sample)",
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "dnaa1_decision")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)

    meta = {
        "source_run_id": "dnaa1-mechA-1.7e-3-7gen",
        "generation_id": None,
        "rendered_at": time.time(),
        "command": (f"python scripts/render_dnaa1_decision.py --run {run_dir} "
                    f"--out {out_dir}"),
        "note": ("Panels b/c/d from genuine end-of-generation dill snapshots "
                 "(one point/gen, pre-division peak). Panel a from the dnaA "
                 "cistron init events in the available early-cycle parquet "
                 "(≤1 min/gen sampled) — a sparse estimate, NOT a per-cycle "
                 "rate. The branch retains no full within-cycle dnaa-1 "
                 "trajectory; not re-simulated (read-only)."),
        "per_generation": rows,
    }
    for ext in (".png", ".svg"):
        with open(base + ext + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    print(f"wrote {base}.svg / .png")
    for r in rows:
        print(f"  gen{r['gen']}: DnaA={r['dnaa_total']} conc={r['conc_nM']:.0f}nM "
              f"oriC={r['oric']} init~{r['init_rate']:.2f}/min "
              f"(sampled {r['sampled_min']:.2f}min)")
    return base + ".svg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",
                    default="studies/dnaa-1-expression/parquet-runs/"
                            "dnaa1-mechA-1.7e-3-7gen")
    ap.add_argument("--out", default="studies/dnaa-1-expression/charts")
    a = ap.parse_args()
    render(a.run, a.out)


if __name__ == "__main__":
    main()
