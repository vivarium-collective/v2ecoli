"""Hardened dnaa-2 multiseed sweep: one workflow subprocess PER SEED.

The in-process meta-composite OOM-kills with 4 full-cell branches in a single
process (one cell's FBA GLP_NOFEAS retry storm tips it over). This driver runs
each lineage_seed as its OWN `v2ecoli-workflow` subprocess (n_init_sims=1), at
most CONCURRENCY at a time, all writing to the SAME hive-partitioned out_dir
(partitions keyed by lineage_seed, so the four runs compose into one dataset).
Then it reconstructs summary.json from the complete history and runs the
multiseed + multigeneration analyses.

    python scripts/run_dnaa2_multiseed_isolated.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Env-parameterized so the same isolated-per-seed workflow driver serves both
# the Step-2 canonical run (defaults below) and the Step-3 mechanism run
# (DnaA-ADP non-equilibrium): pass DNAA2_CACHE / DNAA2_OUT_ROOT / DNAA2_SEEDS /
# DNAA2_GENS, plus DNAA_ADP_RELEASE_RATE (propagated to each workflow subprocess).
SEEDS = [int(s) for s in os.environ.get("DNAA2_SEEDS", "0,1,2,3").split(",")]
CONCURRENCY = int(os.environ.get("DNAA2_CONCURRENCY", "2"))  # composites max in RAM
GENERATIONS = int(os.environ.get("DNAA2_GENS", "4"))
OUT_ROOT = os.environ.get(
    "DNAA2_OUT_ROOT",
    "studies/dnaa-2-atp-hydrolysis/parquet-runs/dnaa2-multiseed-v2")
OUT_DIR = os.path.join(OUT_ROOT, "parquet")
EXPERIMENT_ID = "dnaa2-hydrolysis-multiseed"  # stable -> partitions compose
PY = sys.executable

BASE = {
    "experiment_id": EXPERIMENT_ID,
    "cache_dir": os.environ.get("DNAA2_CACHE", "out/cache-succinate-mechA-2e-3"),
    "out_dir": OUT_DIR,
    "generations": GENERATIONS,
    "n_init_sims": 1,
    "single_daughters": True,
    "time_step": 1.0,
    "max_duration_per_gen": 7200.0,
    "emitter": "parquet",
    "features": ["ppgpp_regulation", "dnaa_nucleotide"],
    "resume_dill": "out/steady_state_inputs/succinate_default_gen3_start.dill",
}


def _write_seed_config(seed: int) -> str:
    cfg = dict(BASE, lineage_seed=seed)
    path = os.path.join(OUT_ROOT, f"_config_seed{seed}.json")
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def _launch(seed: int) -> subprocess.Popen:
    cfg = _write_seed_config(seed)
    log = open(os.path.join("out/logs", f"dnaa2-multiseed-v2-s{seed}.log"), "w")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] launching seed {seed} -> {cfg}")
    return subprocess.Popen(
        [PY, "-m", "v2ecoli.workflow.run", "--config", cfg,
         "--out", os.path.join(OUT_ROOT, f"run_seed{seed}")],
        stdout=log, stderr=subprocess.STDOUT, env=env)


def main() -> None:
    os.makedirs("out/logs", exist_ok=True)
    t0 = time.time()
    pending = list(SEEDS)
    running: dict[int, subprocess.Popen] = {}
    failed: list[int] = []
    while pending or running:
        while pending and len(running) < CONCURRENCY:
            s = pending.pop(0)
            running[s] = _launch(s)
        time.sleep(15)
        for s, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[s]
                status = "ok" if rc == 0 else f"FAILED rc={rc}"
                print(f"[{time.strftime('%H:%M:%S')}] seed {s} done: {status}")
                if rc != 0:
                    failed.append(s)
    print(f"all seeds finished in {(time.time()-t0)/60:.1f} min "
          f"(failed: {failed or 'none'})")

    # --- reconstruct summary.json from complete history -----------------
    import polars as pl
    files = glob.glob(os.path.join(OUT_DIR, "**", "history", "**", "*.pq"),
                      recursive=True)
    if not files:
        print("NO history parquet found — nothing to analyze.")
        return
    df = (pl.scan_parquet(files, hive_partitioning=True)
          .select(["lineage_seed", "generation", "global_time",
                   "listeners__mass__dry_mass"]).collect())
    g = (df.group_by(["lineage_seed", "generation"])
         .agg([pl.col("global_time").max().alias("dur"),
               pl.col("listeners__mass__dry_mass").last().alias("dry")])
         .sort(["lineage_seed", "generation"]))
    present = {(int(r["lineage_seed"]), int(r["generation"]))
               for r in g.iter_rows(named=True)}
    seeds: dict[str, dict] = {}
    for r in g.iter_rows(named=True):
        s, gen = int(r["lineage_seed"]), int(r["generation"])
        key = f"variant=0/seed={s}"
        seeds.setdefault(key, {"generations": []})
        seeds[key]["generations"].append({
            "generation": gen, "agent_id": "0" * (gen + 1),
            "duration": float(r["dur"]), "dry_mass": float(r["dry"]),
            "divided": (s, gen + 1) in present})
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(seeds, f, indent=2)

    # --- per-(seed,gen) DnaA-ATP fraction table -------------------------
    fr = (pl.scan_parquet(files, hive_partitioning=True)
          .select(["lineage_seed", "generation",
                   "listeners__dnaA_cycle__atp_fraction",
                   "listeners__dnaA_cycle__total"]).collect())
    fg = (fr.group_by(["lineage_seed", "generation"]).agg([
        pl.col("listeners__dnaA_cycle__atp_fraction").mean().alias("af"),
        pl.col("listeners__dnaA_cycle__total").mean().alias("tot")])
        .sort(["lineage_seed", "generation"]))
    print("\n=== dnaa-2-v2 per-(seed,gen) DnaA-ATP fraction ===")
    for r in fg.iter_rows(named=True):
        print(f"  seed {int(r['lineage_seed'])} gen {int(r['generation'])}: "
              f"%ATP={r['af']:.4f}  total={r['tot']:.0f}")

    # --- official analyses ---------------------------------------------
    from v2ecoli.workflow.analysis_runner import run_analyses
    res = run_analyses(OUT_DIR, {"multiseed": {"doubling_time_distribution": {}},
                                 "multigeneration": {"mass_growth_across_generations": {}}})
    md = res.get("multiseed", {}).get("doubling_time_distribution", {}).get("variant=0", {})
    print("\n=== multiseed doubling-time ===")
    print(f"  n_cells={md.get('n_cells')}  n_divided={md.get('n_divided')}  "
          f"doubling={md.get('doubling_time_mean', 0)/60:.1f}min "
          f"+/- {md.get('doubling_time_std', 0)/60:.1f}min")

    # --- AUTO-REFRESH the canonical figure FROM THIS RUN -----------------
    # Prevents the recurring "verdict updated but plots stale" problem: the
    # figure is re-rendered from the run that just finished, so it can never lag
    # the data. Renders the canonical seed only (compact, per reviewer). The
    # bespoke sixpanel renderer is used because this viz reads parquet directly
    # (no inputs_map), so the generic render_study_visualizations can't.
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "render_dnaa2_sixpanel", os.path.join(HERE, "render_dnaa2_sixpanel.py"))
        _r = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_r)
        canonical = SEEDS[0]
        step = int(os.environ.get("DNAA2_STEP", "3"))
        figdir = "reports/figures/dnaa-2-atp-hydrolysis"
        os.makedirs(figdir, exist_ok=True)
        _r.render(OUT_DIR, canonical, step, "studies/dnaa-2-atp-hydrolysis/charts")
        _r.render_html(OUT_DIR, canonical, step,
                       f"{figdir}/dnaa2_step{step}_sixpanel_seed{canonical}.html")
        print(f"  auto-refreshed figure: dnaa2_step{step}_sixpanel_seed{canonical} "
              f"(from this run, canonical seed {canonical})")
    except Exception as e:  # noqa: BLE001 — viz refresh must never fail the run
        print(f"  WARNING: figure auto-refresh failed: {type(e).__name__}: {e}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
