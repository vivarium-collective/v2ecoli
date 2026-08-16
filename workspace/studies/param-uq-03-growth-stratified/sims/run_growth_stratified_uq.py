#!/usr/bin/env python
"""param-uq-03-growth-stratified — cell-cycle-stratified (θ-binned) forward-UQ.

Adapted from pbg-uq/examples/v2ecoli_demo.py (the PHYSIOLOGY config_overrides
sweep), NOT from the new-gene driver. Uses the STANDARD baseline (no GFP) and
sweeps the three effective elongation knobs via `config_overrides`, plus one
adversarial INERT decoy knob (drawn but never applied — must return ~0 Sobol).

The new "strategy 4" (RFC006) this study needs — per-cell-cycle-stage Sobol:
  - Run ≥2 generations through ≥1 division (~2760 steps/gen) so a full birth→
    division cycle is captured (cache_dir passed to baseline() so daughters
    divide — the same division fix from param-uq-05).
  - For each GENERATION-1 timepoint compute cell-cycle progress
        θ = [log m(t) − log m_birth] / [log m_div − log m_birth]  ∈ [0,1]
    (m = cell_mass; m_birth = first gen-1 point, m_div = last gen-1 point before
    division). Bin θ into 10 equal stages.
  - Per θ-bin, fit an order-2 PCE and compute total-order Sobol of the 3 knobs
    (+ decoy) on the growth rate in that bin → does knob sensitivity vary across
    the cell cycle?

IMPORTANT — why generation-1 only: v2ecoli's Division step rebuilds each
daughter via baseline(cache_dir=...) WITHOUT propagating the parent's
`config_overrides`, so generation ≥2 cells revert to unperturbed configs. The
knob perturbation is therefore only valid in the founder generation. Generation
1 is a complete birth→division cycle with the perturbation active, so the
θ-binned analysis uses gen-1 data; the run goes through the first division only
to observe m_div and confirm the cycle completed.

Usage (from a pbg_emitters-compatible worktree, PYTHONPATH=<worktree>):
    python run_growth_stratified_uq.py --n 24 --n-test 6 --seeds 2 \
        --steps 3000 --gens 2 --chunk 60 --out results_n24
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time

os.environ.setdefault("V2ECOLI_SKIP_CACHE_VERIFY", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
import pint

from v2ecoli.types.quantity import ureg
pint.set_application_registry(ureg)

HERE = os.path.dirname(os.path.abspath(__file__))
STD_CACHE = "/Users/eranagmon/code/v2ecoli/out/cache"   # standard (no-GFP) baseline cache
N_BINS = 10

# 3 effective elongation knobs (config_overrides) + 1 inert decoy.
PARAM_NAMES = ["basal_elongation_rate", "kS", "chrom_basal_elongation_rate",
               "inert_decoy"]
# maps the first three PARAM_NAMES to their <process>.<key> config-override paths
OVERRIDE_KEYS = {
    "basal_elongation_rate": "ecoli-polypeptide-elongation.basal_elongation_rate",
    "kS": "ecoli-polypeptide-elongation.kS",
    "chrom_basal_elongation_rate": "ecoli-chromosome-replication.basal_elongation_rate",
}
BOUNDS = np.array([
    [15.0, 28.0],     # basal_elongation_rate  (aa/s)   — same as param-uq-01
    [60.0, 140.0],    # kS                              — same as param-uq-01
    [700.0, 1200.0],  # chrom basal_elongation_rate     — same as param-uq-01
    [0.0, 1.0],       # inert_decoy — adversarial null: drawn, NEVER applied
])
OBS = "instantaneous_growth_rate"

# lazy heavy singletons
_core = None
_baseline = None
_run_multigen_xarray = None
_view = None
_RunReader = None
_set_null = None
_bundle = None
_Composite = None


def _setup():
    global _core, _baseline, _run_multigen_xarray, _view, _RunReader
    global _set_null, _bundle, _Composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import (
        baseline, set_null_emitter_override, load_cache_bundle)
    from v2ecoli.library.xarray_run import view_from_emit_paths, run_multigen_xarray
    try:
        from viva_emitters.run_reader import RunReader   # current (renamed) engine
    except ImportError:
        from pbg_emitters.run_reader import RunReader
    from process_bigraph import Composite
    _baseline = baseline
    _set_null = set_null_emitter_override
    _run_multigen_xarray = run_multigen_xarray
    _RunReader = RunReader
    _Composite = Composite
    _core = build_core()
    _bundle = load_cache_bundle(STD_CACHE)
    _view = view_from_emit_paths([
        "listeners.mass.instantaneous_growth_rate",
        "listeners.mass.cell_mass",
        "listeners.mass.dry_mass",
    ])


def _lineage_series(reader):
    """{generation: (times, cell_mass, growth)} for every generation present."""
    def _ser(suffix):
        obs = [o for o in reader.observables() if o.endswith(suffix)]
        return reader.series(obs[0]) if obs else None
    m = _ser("cell_mass")
    g = _ser("instantaneous_growth_rate")
    if m is None or g is None or m.shape[0] < 3:
        return {}
    j = m.join(g, on=["generation", "time"], how="inner", suffix="_g") \
         .sort(["generation", "time"])
    out = {}
    for gen in sorted(set(j["generation"].to_list())):
        sub = j.filter(j["generation"] == gen)
        out[int(gen)] = (sub["time"].to_numpy(), sub["value"].to_numpy(),
                         sub["value_g"].to_numpy())
    return out


def _theta_bins_lineage(per_gen, n_bins=N_BINS):
    """Pool per-generation cell-cycle progress θ (each generation normalised
    birth→division separately) into θ-bins across the FULL lineage.

    Returns (bins {b: mean growth}, population mean growth, per-gen mean growth
    dict). Now that the division fix propagates config_overrides, BOTH
    generations carry the perturbation, so both contribute to the θ-profile."""
    all_theta, all_growth = [], []
    per_gen_growth = {}
    for gen, (t, m, g) in sorted(per_gen.items()):
        m = np.asarray(m, dtype=float)
        g = np.asarray(g, dtype=float)
        keep = m > 0
        m, g = m[keep], g[keep]
        if len(m) < 3:
            continue
        lm = np.log(m)
        lb, ld = lm[0], lm[-1]
        if not (ld > lb):
            continue
        theta = np.clip((lm - lb) / (ld - lb), 0.0, 1.0)
        all_theta.extend(theta.tolist())
        all_growth.extend(g.tolist())
        per_gen_growth[int(gen)] = float(np.mean(g))
    if len(all_growth) < 3:
        return None, None, None
    theta = np.asarray(all_theta)
    g = np.asarray(all_growth)
    bins = {}
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (theta >= lo) & (theta <= hi) if b == n_bins - 1 else \
               (theta >= lo) & (theta < hi)
        if mask.any():
            bins[b] = float(np.mean(g[mask]))
    return bins, float(np.mean(g)), per_gen_growth


def _run_one(x, seed, steps, gens, chunk):
    """One (sample, seed) run -> dict with θ-binned + population gen-1 growth."""
    config_overrides = {OVERRIDE_KEYS[PARAM_NAMES[i]]: float(x[i])
                        for i in range(3)}
    tmp = tempfile.mkdtemp(prefix="pq03_uq_")
    try:
        meta = {"experiment_id": "pq03_uq", "variant": 0, "lineage_seed": int(seed),
                "time_step": 1.0, "max_duration": float(steps)}
        _set_null(True)
        try:
            # cache_dir passed (param-uq-05 division fix) so daughters divide.
            # NB config_overrides are NOT propagated to daughters by the Division
            # step, so only generation 1 carries the perturbation — which is all
            # the θ-binned (birth→division) analysis uses.
            doc = _baseline(core=_core, seed=int(seed), bundle=_bundle,
                            cache_dir=STD_CACHE, config_overrides=config_overrides,
                            emitter="xarray")
        finally:
            _set_null(False)
        comp = _Composite(doc, core=_core)
        store = os.path.join(tmp, "run.zarr")
        res = _run_multigen_xarray(comp, store_path=store, view=_view,
                                   metadata_base=meta, max_steps=int(steps),
                                   max_generations=int(gens), chunk=int(chunk),
                                   single_daughters=True)
        gens_seen = res.get("generations", [])
        reader = _RunReader.open(store, kind="xarray")
        per_gen = _lineage_series(reader)
        if not per_gen:
            return None
        bins, pop_growth, per_gen_growth = _theta_bins_lineage(per_gen)
        if bins is None:
            return None
        g1 = per_gen.get(1)
        return {
            "theta_bins": {str(b): v for b, v in bins.items()},
            "population_growth": pop_growth,
            "per_gen_growth": {str(k): v for k, v in per_gen_growth.items()},
            "n_points": int(sum(len(v[0]) for v in per_gen.values())),
            "divided": bool(2 in gens_seen),
            "gens_seen": [int(gg) for gg in gens_seen],
            "m_birth": float(g1[1][0]) if g1 is not None else None,
            "m_div": float(g1[1][-1]) if g1 is not None else None,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_ensemble(n, n_test, seeds, steps, gens, chunk, out_dir, seed0):
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
                rec = _run_one(x, seed, steps, gens, chunk)
                if rec is None:
                    print(f"[uq] sample {i+1}/{n_all} seed {seed} produced no "
                          f"gen-1 series (skipped)", flush=True)
                else:
                    rec.update({"sample": i, "split": split, "seed": seed,
                                **{PARAM_NAMES[k]: float(x[k]) for k in range(4)}})
                    records.append(rec)
                    nb = len(rec["theta_bins"])
                    pgg = rec.get("per_gen_growth", {})
                    print(f"[uq] sample {i+1}/{n_all} ({split}) seed {seed} "
                          f"basal={x[0]:.1f} gens={rec['gens_seen']} "
                          f"bins={nb}/{N_BINS} per_gen_gr={pgg} "
                          f"in {time.time()-t0:.1f}s", flush=True)
            except Exception as exc:
                print(f"[uq] sample {i+1}/{n_all} seed {seed} FAILED: {exc!r}",
                      flush=True)
            with open(raw_path, "w") as f:
                json.dump(records, f, indent=2)
    print(f"[uq] ensemble done: {len(records)} records in "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)
    return X_train, X_test, records


# ---------------------------------------------------------------------------
# Aggregation -> Sobol
# ---------------------------------------------------------------------------
def _sobol_single(X, Y, X_test=None, Y_test=None):
    """order-2 PCE + analytic Sobol for a single scalar output Y (n,)."""
    from pbg_uq.uqpc import fit_pce_and_sobol
    kw = {}
    if X_test is not None and Y_test is not None and len(X_test) > 0:
        kw = {"X_test": X_test.reshape(-1, X.shape[1]),
              "Y_test": np.asarray(Y_test).reshape(-1, 1)}
    res = fit_pce_and_sobol(X, np.asarray(Y).reshape(-1, 1), BOUNDS,
                            polynomial_order=2, parameter_names=PARAM_NAMES, **kw)
    tot = np.asarray(res.sobol.total_order).ravel()
    fir = np.asarray(res.sobol.first_order).ravel()
    relerr = res.relerr_test if getattr(res, "relerr_test", None) is not None \
        and np.asarray(res.relerr_test).size else res.relerr_cv
    out = {
        "total_order": {PARAM_NAMES[p]: float(tot[p]) for p in range(4)},
        "first_order": {PARAM_NAMES[p]: float(fir[p]) for p in range(4)},
    }
    if relerr is not None and np.asarray(relerr).size:
        out["rel_test_error"] = float(np.asarray(relerr).ravel()[0])
    return out


def _mean_over_seeds(records, split, offset, X, key_fn):
    """Y[i] = mean over seeds of key_fn(record) for sample offset+i; NaN if none."""
    n = X.shape[0]
    Y = np.full(n, np.nan)
    for i in range(n):
        vals = [key_fn(r) for r in records
                if r["sample"] == offset + i and r["split"] == split
                and key_fn(r) is not None]
        if vals:
            Y[i] = float(np.mean(vals))
    return Y


def compute_strategies(records, X_train, X_test):
    n_train = X_train.shape[0]
    all_seeds = sorted({r["seed"] for r in records})
    n_div = sum(1 for r in records if r.get("divided"))
    result = {"meta": {"n_records": len(records), "seeds_present": all_seeds,
                       "n_bins": N_BINS, "n_divided": n_div,
                       "observable": OBS}}

    def _fit_scalar(key_fn):
        Ytr = _mean_over_seeds(records, "train", 0, X_train, key_fn)
        m = ~np.isnan(Ytr)
        if m.sum() <= 4:
            return {"error": f"insufficient samples ({int(m.sum())})"}
        Xte = Yte = None
        if X_test is not None and X_test.shape[0] > 0:
            Yte_full = _mean_over_seeds(records, "test", n_train, X_test, key_fn)
            mt = ~np.isnan(Yte_full)
            if mt.sum() > 0:
                Xte, Yte = X_test[mt], Yte_full[mt]
        return _sobol_single(X_train[m], Ytr[m], Xte, Yte)

    # Strategy 1: population — whole gen-1 mean growth
    result["strategy_1_population"] = _fit_scalar(
        lambda r: r.get("population_growth"))

    # Strategy 4: per θ-bin — growth within each cell-cycle stage
    s4 = {}
    for b in range(N_BINS):
        s4[f"theta_bin_{b}"] = _fit_scalar(
            lambda r, b=b: r.get("theta_bins", {}).get(str(b)))
    result["strategy_4_theta_binned"] = s4

    # Strategy 3: by-seed on population growth (robustness)
    s3 = {}
    for s in all_seeds:
        subset = [r for r in records if r["seed"] == s]
        Ytr = _mean_over_seeds(subset, "train", 0, X_train,
                               lambda r: r.get("population_growth"))
        m = ~np.isnan(Ytr)
        s3[f"seed_{s}"] = (_sobol_single(X_train[m], Ytr[m])
                           if m.sum() > 4 else {"error": "insufficient"})
    result["strategy_3_by_seed"] = s3
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results_n24"))
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    print(f"[uq] param-uq-03 θ-binned: n={args.n}+{args.n_test}test "
          f"seeds={args.seeds} gens={args.gens} steps={args.steps} "
          f"bins={N_BINS} out={out_dir}", flush=True)
    t = time.time()
    _setup()
    print(f"[uq] setup done in {time.time()-t:.1f}s", flush=True)

    X_train, X_test, records = run_ensemble(
        args.n, args.n_test, args.seeds, args.steps, args.gens, args.chunk,
        out_dir, args.seed0)

    sobol = compute_strategies(records, X_train, X_test)
    sobol["run_config"] = {"n_train": args.n, "n_test": args.n_test,
                           "seeds": args.seeds, "gens": args.gens,
                           "steps": args.steps, "n_bins": N_BINS,
                           "param_names": PARAM_NAMES, "bounds": BOUNDS.tolist(),
                           "observable": OBS, "override_keys": OVERRIDE_KEYS}
    with open(os.path.join(out_dir, "sobol.json"), "w") as f:
        json.dump(sobol, f, indent=2)
    print(f"[uq] wrote {os.path.join(out_dir, 'sobol.json')}", flush=True)
    print(json.dumps(sobol.get("strategy_1_population", {}), indent=2), flush=True)
    print("[uq] ALL DONE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
