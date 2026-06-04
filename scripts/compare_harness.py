#!/usr/bin/env python
"""vEcoli <-> v2ecoli comparison harness.

Runs both engines from a single vEcoli config and emits a two-column HTML
report: config/schema diff, ParCa sim_data comparison, 2-gen sim dynamics.

    .venv/bin/python scripts/compare_harness.py \
        --config /Users/eranagmon/code/vEcoli/configs/two_generations.json \
        -o out/compare/report.html
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._compare import orchestrator
from scripts._compare.config_adapter import (
    resolve_vecoli_config, schema_diff, translate_vecoli_config)
from scripts._compare.parca_section import final_sim_data_diff
from scripts._compare.report import render_report
from scripts._compare.sim_section import (
    OBSERVABLES, compare_observables, read_observables)

# sim_data diffs should be tight; dynamics looser (two engines).
PARCA_REL_TOL = 1e-6
SIM_REL_TOL = 0.05


def _config_section(vecoli_cfg, v2_cfg):
    d = schema_diff(vecoli_cfg, v2_cfg)
    rows = []
    for k in d["only_in_vecoli"]:
        rows.append({"label": k, "left": json.dumps(vecoli_cfg[k]),
                     "right": "(not used by v2ecoli)", "verdict": "drift"})
    for k in d["only_in_v2"]:
        rows.append({"label": k, "left": "(added by adapter)",
                     "right": json.dumps(v2_cfg[k]), "verdict": "drift"})
    for k, (lv, rv) in d["different"].items():
        rows.append({"label": k, "left": json.dumps(lv),
                     "right": json.dumps(rv), "verdict": "drift"})
    return {"title": "Config & schema diff", "rows": rows}


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="Path to a vEcoli JSON config (source of truth).")
    p.add_argument("-o", "--out", default="out/compare/report.html")
    p.add_argument("--workdir", default="out/compare_harness")
    p.add_argument("--mode", default="full", choices=["full", "fast"])
    p.add_argument("--fast-plumbing", action="store_true",
                   help="ParCa --mode fast for wiring iteration ONLY; the "
                        "report is stamped NOT SCIENTIFICALLY VALID.")
    args = p.parse_args(argv)
    mode = "fast" if args.fast_plumbing else args.mode

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    from ecoli.library.parquet_emitter import read_stacked_columns as v_reader  # noqa: E501
    from v2ecoli.library.parquet_emitter import read_stacked_columns as v2_reader  # noqa: E501

    # Stage 1 — config
    vecoli_cfg = resolve_vecoli_config(args.config)
    v2_cfg = translate_vecoli_config(vecoli_cfg)
    v2_cfg_path = work / "v2_config.json"
    v2_cfg_path.write_text(json.dumps(
        {k: v for k, v in v2_cfg.items() if not k.startswith("_")}))
    sections = [_config_section(vecoli_cfg, v2_cfg)]

    # Stage 2 — ParCa (both)
    v_parca = orchestrator.run_vecoli_parca(
        config_path=args.config, out_dir=work / "vecoli_parca")
    v2_parca = orchestrator.run_v2_parca(
        out_dir=work / "v2_parca", cache_dir=work / "parca_cache", mode=mode)

    # Stage 3 — ParCa / sim_data comparison
    v_sim_data = _load_pickle(v_parca / "kb" / "sim_data.cPickle")
    v2_sim_data = _load_pickle(v2_parca / "checkpoint_step_9.pkl")
    sections.append({"title": "ParCa / sim_data",
                     "rows": final_sim_data_diff(v_sim_data, v2_sim_data,
                                                 rel_tol=PARCA_REL_TOL)})

    # Stage 4 — sim (both) + dynamics
    exp_id = vecoli_cfg.get("experiment_id", "default")
    v_sim = orchestrator.run_vecoli_sim(
        config_path=args.config,
        sim_data_path=str(v_parca / "kb" / "sim_data.cPickle"),
        out_dir=work / "vecoli_sim",
        generations=int(vecoli_cfg.get("generations", 2)))
    v2_sim = orchestrator.run_v2_sim(
        config_path=str(v2_cfg_path), out_dir=work / "v2_sim")

    keys = [o["key"] for o in OBSERVABLES]
    left = read_observables(str(v_sim), exp_id, v_reader, keys)
    right = read_observables(str(v2_sim), exp_id, v2_reader, keys)
    sections.append({"title": "2-generation sim dynamics",
                     "rows": compare_observables(left, right, keys=keys,
                                                 rel_tol=SIM_REL_TOL)})

    title = "vEcoli vs v2ecoli"
    if mode == "fast":
        title += "  —  ⚠ NOT SCIENTIFICALLY VALID (fast ParCa) ⚠"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render_report(sections, title=title))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
