"""param-uq-04-deep-params — forward UQ over DEEP sim_data physiological params.

Drives the deep sim_data injection adapter (pbg_v2ecoli.uq_sim_data_injection)
through pbg-uq's PCE + Sobol machinery.  Three deep params spanning both
injection mechanisms:

  rnap_elongation_rate      POST_PARCA  (transcription RNAP speed)   -> growth
  cell_dry_mass_fraction    REBUILD     (dry/wet mass partition)     -> mass
  kinetic_objective_weight  POST_PARCA  (FBA objective weight)       -> control

Observables: dry_mass, cell_mass, instantaneous_growth_rate (listeners.mass.*).
Design: order-2 PCE, n_train + n_test PCRV samples per seed, >=2 PCRV seeds.
The v2ecoli sim seed is FIXED per design (= PCRV seed) so within a design all
variance is parametric; across designs both the samples and the sim seed differ,
testing rank stability.  Per-observable total-order Sobol is computed by fitting
each observable column independently (pbg-uq's default aggregates across outputs).
"""
from __future__ import annotations
import json, os, sys, time, logging
logging.getLogger("v2ecoli").setLevel(logging.ERROR)
logging.getLogger("process_bigraph").setLevel(logging.ERROR)
os.environ.setdefault("POLARS_MAX_THREADS", "1")
import numpy as np
import pint
from v2ecoli.types.quantity import ureg
pint.set_application_registry(ureg)

from pbg_v2ecoli.uq_sim_data_injection import (
    rnap_elongation_rate, cell_dry_mass_fraction, kinetic_objective_weight,
    build_deep_param_evaluator,
)
from pbg_uq.sampling import draw_samples
from pbg_uq.uqpc import fit_pce_and_sobol

OUT = os.path.join(os.path.dirname(__file__), "..", "workspace", "studies",
                   "param-uq-04-deep-params")
OUT = os.path.abspath(OUT)
ART = os.path.join(OUT, "artifacts")
os.makedirs(ART, exist_ok=True)

OBS = ["dry_mass", "cell_mass", "instantaneous_growth_rate"]
N_STEPS = 150
N_TRAIN = int(os.environ.get("UQ_N_TRAIN", 26))
N_TEST = int(os.environ.get("UQ_N_TEST", 8))
ORDER = 2
SEEDS = [int(s) for s in os.environ.get("UQ_SEEDS", "0,1").split(",")]


def make_params():
    return [rnap_elongation_rate(), cell_dry_mass_fraction(), kinetic_objective_weight()]


def run_design(seed: int):
    params = make_params()
    pnames = [p.name for p in params]
    evaluate, inj, _core = build_deep_param_evaluator(
        params, observables=OBS, n_steps=N_STEPS, chunk=30, seed=seed)
    bounds = inj.bounds
    X, X_test = draw_samples(bounds, N_TRAIN, N_TEST, seed=seed)
    print(f"\n=== design seed={seed}: {N_TRAIN} train + {N_TEST} test ===", flush=True)
    t0 = time.perf_counter()
    Y = evaluate(X)
    print(f"  train evaluated in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    Y_test = evaluate(X_test)
    # incremental save of raw samples
    np.savez(os.path.join(ART, f"samples_seed{seed}.npz"),
             X=X, Y=Y, X_test=X_test, Y_test=Y_test,
             param_names=np.array(pnames), obs_names=np.array(OBS))
    # per-observable PCE + Sobol
    per_obs = {}
    for j, o in enumerate(OBS):
        res = fit_pce_and_sobol(
            X, Y[:, j:j+1], bounds, polynomial_order=ORDER,
            parameter_names=pnames, X_test=X_test, Y_test=Y_test[:, j:j+1], seed=seed)
        per_obs[o] = {
            "total_order": res.sobol.total_order.ravel().tolist(),
            "first_order": res.sobol.first_order.ravel().tolist(),
            "relerr_test": float(np.ravel(res.relerr_test)[0]) if res.relerr_test is not None else None,
            "Y_std": float(np.std(Y[:, j])),
            "Y_mean": float(np.mean(Y[:, j])),
        }
    return {"seed": seed, "param_names": pnames, "per_obs": per_obs}


def main():
    t_all = time.perf_counter()
    designs = [run_design(s) for s in SEEDS]
    pnames = designs[0]["param_names"]
    # aggregate total-order Sobol across seeds: mean + min/max
    agg = {}
    for o in OBS:
        mat = np.array([d["per_obs"][o]["total_order"] for d in designs])  # (n_seed, n_param)
        agg[o] = {
            "param_names": pnames,
            "total_order_mean": mat.mean(axis=0).tolist(),
            "total_order_min": mat.min(axis=0).tolist(),
            "total_order_max": mat.max(axis=0).tolist(),
            "relerr_test_mean": float(np.mean(
                [d["per_obs"][o]["relerr_test"] for d in designs
                 if d["per_obs"][o]["relerr_test"] is not None] or [np.nan])),
            "Y_std_mean": float(np.mean([d["per_obs"][o]["Y_std"] for d in designs])),
        }
    out = {
        "study": "param-uq-04-deep-params",
        "n_train": N_TRAIN, "n_test": N_TEST, "order": ORDER,
        "seeds": SEEDS, "n_steps": N_STEPS, "observables": OBS,
        "param_names": pnames,
        "designs": designs,
        "aggregate": agg,
        "wall_min": (time.perf_counter() - t_all) / 60.0,
    }
    with open(os.path.join(ART, "sobol_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 64)
    print(f"TOTAL-ORDER SOBOL (mean across seeds {SEEDS}):")
    for o in OBS:
        a = agg[o]
        print(f"\n {o}  (test relerr {a['relerr_test_mean']*100:.1f}%, Y_std {a['Y_std_mean']:.4g}):")
        order = np.argsort(a["total_order_mean"])[::-1]
        for i in order:
            print(f"   {pnames[i]:26s} S_T = {a['total_order_mean'][i]:.3f} "
                  f"[{a['total_order_min'][i]:.3f}, {a['total_order_max'][i]:.3f}]")
    print(f"\nwall time {out['wall_min']:.1f} min  ->  {ART}/sobol_results.json")


if __name__ == "__main__":
    main()
