"""Render / bake the basal report card's ``vs_literature`` reference mode (CLI).

Thin CLI wrapper around ``v2ecoli.library.vs_literature`` (the importable core —
reference-building + grading inputs, used by the Gen 2
``VsLiteratureCard(ReportCardStep)`` and the tests). This script keeps the two
things that need the (gitignored) blessed sweep + local provenance and so do NOT
belong in the import-safe library:

  * ``--from-sweep``: recompute the baked per-cell ``model_physiology.json`` by
    DIRECT mass balance from the sweep parquet (``physiology_from_sweep``), and
  * ``main``: re-render the standalone HTML card + verdict json into ``docs/``.

The card + reference logic (``build``, ``literature``, ``build_reference``,
``build_metabolism``, …) lives in the library and is re-exported here so existing
callers (``tests/test_basal_vs_literature.py``) keep importing them from this
module.

Run:
    python scripts/render_basal_vs_literature.py                 # re-render from baked json
    python scripts/render_basal_vs_literature.py --from-sweep out/population_phenotype_basal
    # -> docs/report_cards/population_phenotype_basal/vs_literature/
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ for _provenance
from v2ecoli.library.report_card import grade_card, render_html, verdict_json  # noqa: E402
# Re-export the importable core so existing callers keep importing these names
# from this module (tests/test_basal_vs_literature.py). The library is the source
# of truth; the Gen 2 card imports the same functions.
from v2ecoli.library.vs_literature import (  # noqa: E402,F401
    M_C, M_GLC, _AXES, _CC_FLAG_GLYPH, _COMP_JSON, _EXCH_AXES, _G6P_CROWN,
    _INCLUDE_CC_SCATTER, _MET_JSON, _MODEL_JSON, _POOLS_JSON, _PRO_JSON,
    BASAL_SPEC, FIXTURES, bennett_pools, build, build_card, build_composition,
    build_metabolism, build_metabolite_pools, build_proteome, build_reference,
    composition_reference, crown_fate_nodes, crown_g6p_composition, literature,
    model_physiology, proteome_reference)
from _provenance import provenance_stamp  # noqa: E402

_SWEEP_GEN_LB = 3  # generation_lower_bound of the blessed sweep (must match the config)


def _parca_inputs_hash():
    """The ParCa cache identity (out/cache/cache_version.json inputs_hash), or None."""
    try:
        return json.load(open(REPO / "out/cache/cache_version.json"))["inputs_hash"]
    except Exception:
        return None

OUT = REPO / "docs/report_cards/population_phenotype_basal/vs_literature"  # render OUTPUTS (gitignored)
_SWEEP = REPO / "out/population_phenotype_basal"   # blessed ensemble (gitignored; --from-sweep)

_GLC_IDX, _CO2_IDX, _ACET_IDX = 37, 11, 3  # 1-indexed positions in the 87-flux array


def physiology_from_sweep(sweep_dir: Path = _SWEEP, gen_lb: int = 3) -> dict:
    """Per-cell physiology by DIRECT mass balance from the sweep parquet.

    For each post-burn-in cell, integrated over its cycle from the per-timestep
    dry mass + specific exchange fluxes:
      * growth rate    μ      = ln 2 / cell-cycle time
      * glucose uptake q_glc  = time-mean |GLC[p] exchange|
      * biomass yield  Yxs    = ΔDW / (∫ q_glc·DW dt · M_glc)  — a direct mass
        ratio (g DW made / g glucose eaten), NOT the steady-state μ/(q·M) shortcut.
    Also reports the carbon implied into biomass per gDW
    (= (glucose-C − CO₂-C − acetate-C) / ΔDW) — a conservation sanity check;
    a physically plausible ~0.45–0.50 gC/gDW means carbon is conserved."""
    import duckdb
    files = glob.glob(os.path.join(str(sweep_dir), "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        raise SystemExit(f"no parquet under {sweep_dir} (need a blessed-baseline sweep)")
    flist = "[" + ",".join("'" + f + "'" for f in files) + "]"
    rows = duckdb.sql(f"""
      SELECT variant, lineage_seed, generation, agent_id, global_time,
        listeners__mass__dry_mass AS dw,
        list_extract(listeners__fba_results__external_exchange_fluxes, {_GLC_IDX})  AS glc,
        list_extract(listeners__fba_results__external_exchange_fluxes, {_CO2_IDX})  AS co2,
        list_extract(listeners__fba_results__external_exchange_fluxes, {_ACET_IDX}) AS acet
      FROM read_parquet({flist}, hive_partitioning=true)
      WHERE generation >= {gen_lb}
      ORDER BY variant, lineage_seed, generation, agent_id, global_time""").fetchall()
    cells: dict = {}
    for r in rows:
        cells.setdefault(r[:4], []).append(r[4:])

    mu_c, q_c, y_c, impliedC = [], [], [], []
    for rs in cells.values():
        if len(rs) < 3:
            continue
        t = [x[0] / 3600.0 for x in rs]            # h
        dw = [x[1] * 1e-15 for x in rs]            # g (dry_mass is fg)
        glc = [abs(float(x[1 + 1] or 0)) for x in rs]   # |GLC| specific (mmol/gDW/h)
        co2 = [float(x[1 + 2] or 0) for x in rs]
        acet = [float(x[1 + 3] or 0) for x in rs]
        dts = [t[i + 1] - t[i] for i in range(len(t) - 1)]

        def integ(spec):  # trapezoid of (specific flux · DW) -> absolute mmol over the cycle
            rate = [spec[i] * dw[i] for i in range(len(spec))]
            return sum(0.5 * (rate[i] + rate[i + 1]) * dts[i] for i in range(len(dts)))

        glc_mmol, ddw = integ(glc), dw[-1] - dw[0]
        if glc_mmol <= 0 or ddw <= 0:
            continue
        mu_c.append(math.log(2) / (t[-1] - t[0]))
        q_c.append(sum(glc) / len(glc))
        y_c.append(ddw / (glc_mmol * M_GLC))
        c_into_biomass_mmol = glc_mmol * 6 - integ(co2) - 2 * integ(acet)  # mmol C
        impliedC.append(c_into_biomass_mmol * M_C / 1000.0 / ddw)          # gC/gDW

    n = len(y_c)
    return {
        "biomass_yield": sum(y_c) / n, "growth_rate": sum(mu_c) / n,
        "glucose_uptake": sum(q_c) / n,
        "biomass_yield_cells": y_c, "growth_rate_cells": mu_c, "glucose_uptake_cells": q_c,
        "implied_biomass_C_gC_per_gDW": sum(impliedC) / n,
        "method": "direct mass balance (ΔDW / ∫q_glc·DW dt)",
        "ensemble": f"blessed baseline, gen≥{gen_lb}", "n_cells": n,
    }


def main(from_sweep: str | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if from_sweep:
        model = physiology_from_sweep(Path(from_sweep))
        model["provenance"] = provenance_stamp(
            REPO, config="v2ecoli/configs/population_phenotype_basal.json",
            sweep={"sweep_dir": str(Path(from_sweep)),
                   "parca_inputs_hash": _parca_inputs_hash(),
                   "n_cells": model.get("n_cells"), "gen_lb": _SWEEP_GEN_LB},
            bake_script=f"scripts/render_basal_vs_literature.py --from-sweep {from_sweep}")
        FIXTURES.mkdir(parents=True, exist_ok=True)
        with open(_MODEL_JSON, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)
        print(f"baked {_MODEL_JSON.name}: {model['n_cells']} cells · "
              f"Yxs {model['biomass_yield']:.3f} · μ {model['growth_rate']:.3f} · "
              f"q_glc {model['glucose_uptake']:.3f} · implied biomass C "
              f"{model['implied_biomass_C_gC_per_gDW']:.3f} gC/gDW")
    card, reference, model = build()
    generated = time.strftime("%Y-%m-%d %H:%M")
    model_ref = f"v2ecoli baseline ({model['ensemble']})"
    report = grade_card(card, reference)

    (OUT / "report_card.html").write_text(
        render_html(card, reference, model_ref=model_ref, generated=generated))
    with open(OUT / "literature_reference.json", "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)
    with open(OUT / "report_card_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_json(report, model_ref=model_ref,
                               reference_model=reference["stimulus"]["reference_model"],
                               generated=generated), f, indent=2, ensure_ascii=False)

    print(f"overall: {report['overall']}")
    for path, a in report["axes"].items():
        fp = " [first-principles violation]" if a.get("detail", {}).get(
            "first_principles_violation") else ""
        print(f"  {path:28} {a['verdict']:11} {a.get('meter','')}{fp}")
    print(f"\nwrote {OUT}/ (report_card.html + literature_reference.json + "
          "report_card_verdict.json)")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-sweep", metavar="DIR", default=None,
                    help="recompute the baked model_physiology.json from a sweep parquet dir")
    a = ap.parse_args()
    main(from_sweep=a.from_sweep)
