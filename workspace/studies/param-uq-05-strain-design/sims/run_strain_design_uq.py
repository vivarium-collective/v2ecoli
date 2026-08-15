#!/usr/bin/env python
"""param-uq-05-strain-design — GFP new-gene forward-UQ ensemble driver.

Adapted from pbg-uq/examples/v2ecoli_demo.py (physiology sweep) to a
STRAIN-DESIGN sweep of the two public new_gene_internal_shift knobs
(``exp``, ``trl_eff``) on a GFP-new-gene sim_data, plus an adversarial
INERT decoy knob that is drawn but never applied (must return ~0 Sobol).

Per (sample, seed) it:
  1. loads the GFP sim_data, patches the DnaA_box unique-molecule schema
     (adds v2ecoli's pool_label / DnaA_bound_form fields), and applies the
     new-gene expression/translation-efficiency modification via
     ``modify_new_gene_exp_trl`` (induction baked at generation 0 — see NOTE),
  2. writes a v2ecoli sim-input bundle (save_sim_input) and builds the
     ecoli_baseline composite from it,
  3. runs it with the XArray emitter for max_steps over up to max_generations,
  4. reads the GFP monomer count (reader.select monomer_id NG-GFP-MONOMER[c])
     and instantaneous_growth_rate, time-averaged PER GENERATION.

The per-(sample, seed, generation) table is saved to results/raw_records.json
and Sobol indices are computed under the three aggregation strategies:
  strategy 1 (population)   : average over seeds+gens+time  -> one Sobol set
  strategy 2 (by-generation): average over seeds+time per gen -> Sobol per gen
  strategy 3 (by-seed)      : average over gens+time per seed  -> Sobol per seed
via pbg_uq.uqpc.fit_pce_and_sobol (order-2 PCE, analytic Sobol).

NOTE on induction: v2ecoli's baseline() bakes sim_data into a precomputed
cache bundle (configs + initial_state); the vEcoli generational
internal_shift_dict is not applied by the v2ecoli composite runner. So this
driver applies the new-gene modification IMMEDIATELY (generation 0) rather than
via induction_gen=1. The knob semantics (exp, trl_eff factors) are identical;
only the induction timing differs. This is recorded honestly in the study.

Usage (from the v2ecoli worktree, with PYTHONPATH=<worktree>):
    python run_strain_design_uq.py --mode smoke
    python run_strain_design_uq.py --n 24 --seeds 3 --gens 2 --steps 3600 \
        --out results_production
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("V2ECOLI_SKIP_CACHE_VERIFY", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
import pint

from v2ecoli.types.quantity import ureg
pint.set_application_registry(ureg)

import dill

# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DILL = "/Users/eranagmon/code/v2ecoli/out/paramuq05/gfp_sim_data.dill"
GFP_MONOMER_ID = "NG-GFP-MONOMER[c]"

PARAM_NAMES = ["exp", "trl_eff", "inert_decoy"]
# exp bounds set by a screen (sims screen log): the new-gene synth-prob baseline
# is ~1.7e-9 (genes near-knocked-out by default), so a factor of O(1) yields ~0
# GFP. GFP becomes measurable only for factors O(1e3-1e4): screen gave
# exp=100->3, 1000->30, 10000->409 GFP over 1000 steps. So the productive
# expression-factor range is [1e3, 1e4] (study.yaml anticipated this screen).
BOUNDS = np.array([
    [1000.0, 10000.0],  # exp        — induced new-gene expression factor
    [0.5, 2.0],         # trl_eff    — induced new-gene translation efficiency
    [0.0, 1.0],         # inert_decoy — adversarial null: drawn, NEVER applied
])
OBS_NAMES = ["gfp_product_level", "instantaneous_growth_rate"]

# lazy heavy imports (populated in _setup)
_core = None
_baseline = None
_run_multigen_xarray = None
_view = None
_RunReader = None
_save_sim_input = None
_modify = None
_set_null = None
_load_bundle = None
_Composite = None


def _setup():
    global _core, _baseline, _run_multigen_xarray, _view, _RunReader
    global _save_sim_input, _modify, _set_null, _load_bundle, _Composite
    from v2ecoli.core import save_sim_input, build_core
    from v2ecoli.composites.ecoli_baseline import (
        baseline, set_null_emitter_override, load_cache_bundle)
    from v2ecoli.library.xarray_run import view_from_emit_paths, run_multigen_xarray
    from ecoli.variants.new_gene_internal_shift import modify_new_gene_exp_trl
    from pbg_emitters.run_reader import RunReader
    from process_bigraph import Composite
    _save_sim_input = save_sim_input
    _baseline = baseline
    _set_null = set_null_emitter_override
    _load_bundle = load_cache_bundle
    _run_multigen_xarray = run_multigen_xarray
    _modify = modify_new_gene_exp_trl
    _RunReader = RunReader
    _Composite = Composite
    _core = build_core()
    _view = view_from_emit_paths([
        "listeners.mass.instantaneous_growth_rate",
        "listeners.mass.cell_mass",
        "listeners.monomer_counts",
    ])


def _load_gfp_simdata(dill_path):
    sd = dill.load(open(dill_path, "rb"))
    d = sd.internal_state.unique_molecule.unique_molecule_definitions["DnaA_box"]
    for k, v in [("pool_label", "i1"), ("DnaA_bound_form", "i1")]:
        d.setdefault(k, v)
    return sd


def _read_vector_element(reader, obs, idx):
    """{generation: [values]} for column ``idx`` of a vector observable, using
    the reader's internal xarray helpers (the same ones series() uses, which
    resolve time coords correctly — unlike select's element path)."""
    dt = reader._xarray_open()
    node_path = "/" + obs.replace(".", "/")
    node_ds = dt[node_path].ds
    time_ds = reader._xarray_time_ds(dt)
    gen_info = reader._xarray_gen_info(dt)
    out = {}
    for gen, _coo, time_var in gen_info:
        var = f"generation={gen}"
        if var not in node_ds.data_vars or time_var not in time_ds.data_vars:
            continue
        arr = node_ds[var].values
        col = arr[:, idx] if arr.ndim == 2 else arr.ravel()
        out.setdefault(int(gen), []).extend(float(v) for v in col)
    return out


def _run_one(dill_path, exp, trl_eff, seed, steps, gens, chunk):
    """One (sample, seed) run -> list of per-generation records."""
    sd = _load_gfp_simdata(dill_path)
    _modify(sd, expression=float(exp), translation_efficiency=float(trl_eff))
    tmp = tempfile.mkdtemp(prefix="gfp_uq_")
    try:
        cache_dir = os.path.join(tmp, "cache")
        _save_sim_input(sd, cache_dir)
        bundle = _load_bundle(cache_dir)
        meta = {"experiment_id": "gfp_uq", "variant": 0, "lineage_seed": int(seed),
                "time_step": 1.0, "max_duration": float(steps)}
        _set_null(True)
        try:
            doc = _baseline(core=_core, seed=int(seed), bundle=bundle, emitter="xarray")
        finally:
            _set_null(False)
        comp = _Composite(doc, core=_core)
        store = os.path.join(tmp, "run.zarr")
        _run_multigen_xarray(comp, store_path=store, view=_view, metadata_base=meta,
                             max_steps=int(steps), max_generations=int(gens),
                             chunk=int(chunk), single_daughters=True)
        reader = _RunReader.open(store, kind="xarray")
        mobs = [o for o in reader.observables() if o.endswith("monomer_counts")][0]
        cat = reader.catalog(mobs)
        gidx = cat.index(GFP_MONOMER_ID)
        # GFP monomer count per (generation, time). RunReader.select's element
        # path has a time-coord lookup bug for this store layout, so read the
        # column directly via the same internal helpers series() uses.
        gfp_by_gen = _read_vector_element(reader, mobs, gidx)
        gobs = [o for o in reader.observables()
                if o.endswith("instantaneous_growth_rate")][0]
        gr = reader.series(gobs)
        recs = []
        for g in sorted(gfp_by_gen):
            gfp_g = np.asarray(gfp_by_gen[g], dtype=float)
            gr_g = gr.filter(gr["generation"] == g)["value"].to_numpy()
            if len(gfp_g) == 0:
                continue
            recs.append({
                "generation": int(g),
                "gfp_product_level": float(np.mean(gfp_g)),
                "instantaneous_growth_rate": float(np.mean(gr_g)) if len(gr_g) else 0.0,
                "n_timepoints": int(len(gfp_g)),
            })
        return recs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_ensemble(dill_path, n, n_test, seeds, gens, steps, chunk, out_dir, seed0):
    os.makedirs(out_dir, exist_ok=True)
    from pbg_uq.sampling import draw_samples
    X_train, X_test = draw_samples(BOUNDS, n_samples=n, n_test=n_test, seed=42)
    X_all = X_train if X_test is None else np.vstack([X_train, X_test])
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    if X_test is not None:
        np.save(os.path.join(out_dir, "X_test.npy"), X_test)

    records = []
    raw_path = os.path.join(out_dir, "raw_records.json")
    t_start = time.time()
    n_all = X_all.shape[0]
    for i, x in enumerate(X_all):
        split = "train" if i < n else "test"
        for s in range(seeds):
            seed = seed0 + s
            t0 = time.time()
            try:
                recs = _run_one(dill_path, x[0], x[1], seed, steps, gens, chunk)
                for r in recs:
                    r.update({"sample": i, "split": split, "seed": seed,
                              "exp": float(x[0]), "trl_eff": float(x[1]),
                              "inert_decoy": float(x[2])})
                    records.append(r)
                gens_seen = sorted({r["generation"] for r in recs})
                gfp_last = recs[-1]["gfp_product_level"] if recs else float("nan")
                print(f"[uq] sample {i+1}/{n_all} ({split}) seed {seed} "
                      f"exp={x[0]:.2f} trl={x[1]:.2f} gens={gens_seen} "
                      f"gfp~{gfp_last:.0f} in {time.time()-t0:.1f}s", flush=True)
            except Exception as exc:
                print(f"[uq] sample {i+1}/{n_all} seed {seed} FAILED: {exc!r}",
                      flush=True)
            # incremental save so a killed run keeps partial data
            with open(raw_path, "w") as f:
                json.dump(records, f, indent=2)
    print(f"[uq] ensemble done: {len(records)} records in "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)
    return X_train, X_test, records


# ---------------------------------------------------------------------------
# Aggregation -> Sobol
# ---------------------------------------------------------------------------
def _sobol_for(X, Y, X_test=None, Y_test=None):
    """Fit an order-2 PCE + analytic Sobol PER OUTPUT (fit_pce_and_sobol returns
    a single-output (n_params,) Sobol vector, so we call it once per column)."""
    from pbg_uq.uqpc import fit_pce_and_sobol
    out = {"total_order": {}, "first_order": {}, "rel_test_error": {}}
    have_test = (X_test is not None and Y_test is not None and len(X_test) > 0)
    for oi, oname in enumerate(OBS_NAMES):
        kw = {}
        if have_test:
            kw = {"X_test": X_test, "Y_test": Y_test[:, oi:oi + 1]}
        res = fit_pce_and_sobol(X, Y[:, oi:oi + 1], BOUNDS, polynomial_order=2,
                                parameter_names=PARAM_NAMES, **kw)
        tot = np.asarray(res.sobol.total_order).ravel()
        fir = np.asarray(res.sobol.first_order).ravel()
        out["total_order"][oname] = {PARAM_NAMES[pi]: float(tot[pi])
                                     for pi in range(len(PARAM_NAMES))}
        out["first_order"][oname] = {PARAM_NAMES[pi]: float(fir[pi])
                                     for pi in range(len(PARAM_NAMES))}
        relerr = res.relerr_test if getattr(res, "relerr_test", None) is not None \
            and np.asarray(res.relerr_test).size else res.relerr_cv
        if relerr is not None and np.asarray(relerr).size:
            out["rel_test_error"][oname] = float(np.asarray(relerr).ravel()[0])
    return out


def _agg_matrix(records, X, split_filter, group_keys, offset=0):
    """Build (X, Y) for the samples that have data, averaging observables over
    the records matching each sample after applying group_keys filter.

    ``offset`` maps X's local row index to the global sample id stored in the
    records (test samples are numbered n_train..n_train+n_test-1)."""
    # group_keys: dict of field->value to filter (e.g. {"generation": 2} or
    # {"seed": 0}); None value means don't filter that field.
    n = X.shape[0]
    Y = np.full((n, len(OBS_NAMES)), np.nan)
    for i in range(n):
        rows = [r for r in records
                if r["sample"] == offset + i and r["split"] == split_filter
                and all(gk is None or r.get(k) == gk for k, gk in group_keys.items())]
        if not rows:
            continue
        for oi, oname in enumerate(OBS_NAMES):
            Y[i, oi] = float(np.mean([r[oname] for r in rows]))
    mask = ~np.isnan(Y).any(axis=1)
    return X[mask], Y[mask]


def compute_all_strategies(records, X_train, X_test, seeds, seed0):
    all_gens = sorted({r["generation"] for r in records})
    all_seeds = sorted({r["seed"] for r in records})
    n_train = X_train.shape[0]
    result = {"meta": {"generations_present": all_gens, "seeds_present": all_seeds,
                       "n_records": len(records)}}

    def _fit(split_group_train, split_group_test):
        Xtr, Ytr = _agg_matrix(records, X_train, "train", split_group_train)
        Xte = Yte = None
        if X_test is not None and X_test.shape[0] > 0:
            Xte, Yte = _agg_matrix(records, X_test, "test", split_group_test,
                                   offset=n_train)
        if Xtr.shape[0] <= len(PARAM_NAMES):
            return {"error": f"insufficient samples ({Xtr.shape[0]})"}
        return _sobol_for(Xtr, Ytr, Xte, Yte)

    # Strategy 1: population (all gens, all seeds)
    result["strategy_1_population"] = _fit(
        {"generation": None, "seed": None}, {"generation": None, "seed": None})

    # Strategy 2: by-generation
    s2 = {}
    for g in all_gens:
        s2[f"generation_{g}"] = _fit({"generation": g, "seed": None},
                                     {"generation": g, "seed": None})
    result["strategy_2_by_generation"] = s2

    # Strategy 3: by-seed
    s3 = {}
    for s in all_seeds:
        s3[f"seed_{s}"] = _fit({"generation": None, "seed": s},
                               {"generation": None, "seed": s})
    result["strategy_3_by_seed"] = s3
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["smoke", "production", "custom"],
                    default="custom")
    ap.add_argument("--dill", default=DEFAULT_DILL)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--gens", type=int, default=1)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    args = ap.parse_args()

    if args.mode == "smoke":
        args.n, args.n_test, args.seeds, args.gens, args.steps = 12, 3, 3, 1, 700
        args.chunk = 100
    elif args.mode == "production":
        args.n, args.n_test, args.seeds, args.gens, args.steps = 24, 4, 3, 3, 4000
        args.chunk = 100

    out_dir = os.path.abspath(args.out)
    print(f"[uq] mode={args.mode} n={args.n}+{args.n_test}test seeds={args.seeds} "
          f"gens={args.gens} steps={args.steps} out={out_dir}", flush=True)
    t = time.time()
    _setup()
    print(f"[uq] setup done in {time.time()-t:.1f}s", flush=True)

    X_train, X_test, records = run_ensemble(
        args.dill, args.n, args.n_test, args.seeds, args.gens, args.steps,
        args.chunk, out_dir, args.seed0)

    sobol = compute_all_strategies(records, X_train, X_test, args.seeds, args.seed0)
    sobol["run_config"] = {"mode": args.mode, "n_train": args.n, "n_test": args.n_test,
                           "seeds": args.seeds, "gens": args.gens, "steps": args.steps,
                           "param_names": PARAM_NAMES, "bounds": BOUNDS.tolist(),
                           "obs_names": OBS_NAMES,
                           "gfp_monomer_id": GFP_MONOMER_ID}
    with open(os.path.join(out_dir, "sobol.json"), "w") as f:
        json.dump(sobol, f, indent=2)
    print(f"[uq] wrote {os.path.join(out_dir, 'sobol.json')}", flush=True)
    print(json.dumps(sobol.get("strategy_1_population", {}), indent=2), flush=True)
    print("[uq] ALL DONE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
