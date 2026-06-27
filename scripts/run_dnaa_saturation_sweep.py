#!/usr/bin/env python
"""Staged parameter sweep for once-per-cell-cycle oriC-low saturation (dnaa-6/7).

Runs a list of env-knob configs through scripts/run_condition_multigen_parquet.py
(each its own subprocess, bounded concurrency), against the fresh SUCCINATE cache.
Knobs swept (per dnaa5_oric_saturation_parameter_space.md): the initiation-control
axes that govern "fills once / fires once / resets" under the mechanistic oriC-low
trigger — init threshold, SeqA eclipse, RIDA, hydrolysis — with cooperativity +
autoregulation + V held at the dnaa reference.

Each config is a dict of ENV overrides; fixed reference knobs are merged in.
Results land in out/<tag>/; score them with scripts/score_dnaa_saturation.py.

Usage:
  run_dnaa_saturation_sweep.py --configs <json> --gens 6 --seed 1 --conc 3 \
      [--resume-dill <burnin.dill>] [--cache out/cache_dnaa_succ] [--dry-run]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

REF_ENV = {  # dnaa reference operating point (held fixed across the sweep)
    "DNAA_ORIC_COOP_MODE": "hill",
    "DNAA_ORIC_COOPERATIVITY_N": "4",
    "DNAA_ORIC_HALF_SAT_nM": "30",
    "DNAA_AUTOREG_STRENGTH": "0.6",
    "DNAA_AUTOREG_FORM": "linear",
    "DNAA_INITIATION_TRIGGER": "mechanistic",
    "DNAA_INIT_TRIGGER_POOL": "low",
    "DNAA_HYDROLYSIS_RATE_PER_MIN": "0.025",
}
V_PERT = "TU00259[c]=0.0015"


def run_one(cfg, gens, seed, cache, resume_dill, max_min, dry, tag_seed=False):
    tag = cfg["tag"] + (f"_s{seed}" if tag_seed else "")
    out_dir = f"out/{tag}"
    env = dict(os.environ)
    env.update(REF_ENV)
    for k, v in cfg.get("env", {}).items():
        env[k] = str(v)
    cmd = [".venv/bin/python", "scripts/run_condition_multigen_parquet.py",
           "--cache-dir", cache, "--generations", str(gens), "--seed", str(seed),
           "--max-min", str(max_min), "--perturbation", V_PERT,
           "--experiment-id", tag, "--out-dir", out_dir,
           "--dill-dir", f"{out_dir}/gen_dills"]
    if resume_dill:
        cmd += ["--resume-dill", resume_dill]
    if dry:
        sweptknobs = " ".join(f"{k}={v}" for k, v in cfg.get("env", {}).items())
        return tag, 0, f"DRY: {sweptknobs}"
    os.makedirs(out_dir, exist_ok=True)
    log = f"{out_dir}/run.log"
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    return tag, rc, f"{round(time.time()-t0)}s rc={rc} log={log}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="JSON list of {tag, env:{...}}")
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", default=None,
                    help="comma list, e.g. 0,1,2 — runs each config x each seed (out tag _s<seed>)")
    ap.add_argument("--cache", default="out/cache_dnaa_succ")
    ap.add_argument("--resume-dill", default=None)
    ap.add_argument("--max-min", type=float, default=120.0)
    ap.add_argument("--conc", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    configs = json.load(open(args.configs))
    print(f"[{time.strftime('%H:%M:%S')}] sweep: {len(configs)} configs, gens={args.gens}, "
          f"seed={args.seed}, conc={args.conc}, cache={args.cache}, "
          f"resume={'yes' if args.resume_dill else 'cold'}")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    tag_seed = len(seeds) > 1
    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        futs = {ex.submit(run_one, c, args.gens, sd, args.cache,
                          args.resume_dill, args.max_min, args.dry_run, tag_seed):
                f'{c["tag"]}_s{sd}'
                for c in configs for sd in seeds}
        for fut in as_completed(futs):
            tag, rc, msg = fut.result()
            print(f"[{time.strftime('%H:%M:%S')}] DONE {tag}: {msg}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] sweep complete")


if __name__ == "__main__":
    main()
