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


def _error_section(title: str, exc: Exception) -> dict:
    """A report section that surfaces a stage failure instead of aborting."""
    return {"title": title, "rows": [{
        "label": "stage failed", "left": "—", "right": "—",
        "verdict": "mismatch", "reason": f"{type(exc).__name__}: {exc}",
    }]}


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

    # Resolve to an absolute path: vEcoli subprocesses run with cwd=vEcoli, so
    # a relative workdir would make them write under the vEcoli tree while the
    # harness reads under v2ecoli. Absolute keeps both sides in agreement.
    work = Path(args.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    # Stage 1 — config (essential; if this raises, nothing downstream is meaningful)
    vecoli_cfg = resolve_vecoli_config(args.config)
    v2_cfg = translate_vecoli_config(vecoli_cfg)
    v2_cfg_path = work / "v2_config.json"
    v2_cfg_path.write_text(json.dumps(
        {k: v for k, v in v2_cfg.items() if not k.startswith("_")}))
    sections = [_config_section(vecoli_cfg, v2_cfg)]

    # Compute a cache token from config + mode so re-runs with different
    # settings invalidate stale outputs.
    from scripts._compare.cache import cache_key
    run_token = cache_key(vecoli_cfg, commit="", mode=mode)

    # Stages 2+3 — ParCa (both engines) + sim_data comparison.
    # These form one fault unit: Stage 3 needs Stage 2's outputs.
    parca_ok = False
    v_parca = None
    try:
        v_parca = orchestrator.run_vecoli_parca(
            config_path=args.config, out_dir=work / "vecoli_parca",
            token=run_token)
        v2_parca = orchestrator.run_v2_parca(
            out_dir=work / "v2_parca", cache_dir=work / "parca_cache",
            mode=mode, token=run_token)
        v_sim_data = _load_pickle(v_parca / "kb" / "simData.cPickle")
        v2_sim_data = _load_pickle(v2_parca / "checkpoint_step_9.pkl")
        sections.append({"title": "ParCa / sim_data",
                         "rows": final_sim_data_diff(v_sim_data, v2_sim_data,
                                                     rel_tol=PARCA_REL_TOL)})
        parca_ok = True
    except Exception as e:
        sections.append(_error_section("ParCa / sim_data", e))

    # Stage 4 — sim (both engines) + dynamics comparison.
    # Depends on ParCa outputs; if ParCa failed, record a dependency error.
    try:
        if not parca_ok:
            raise RuntimeError(
                "ParCa stage failed — vEcoli sim_data path unavailable")
        exp_id = vecoli_cfg.get("experiment_id", "default")
        vecoli_sim_out = work / "vecoli_sim"
        vecoli_sim_cfg = dict(vecoli_cfg)
        vecoli_sim_cfg["sim_data_path"] = str((v_parca / "kb" / "simData.cPickle").resolve())
        vecoli_sim_cfg["out_dir"] = str(vecoli_sim_out)
        _ea = dict(vecoli_sim_cfg.get("emitter_arg") or {})
        _ea["out_dir"] = str(vecoli_sim_out)
        vecoli_sim_cfg["emitter_arg"] = _ea
        vecoli_sim_cfg["emitter"] = "parquet"
        vecoli_sim_cfg_path = work / "vecoli_sim_config.json"
        vecoli_sim_cfg_path.write_text(json.dumps(vecoli_sim_cfg))
        v_sim = orchestrator.run_vecoli_sim(
            config_path=str(vecoli_sim_cfg_path), out_dir=vecoli_sim_out,
            token=run_token)
        v2_sim = orchestrator.run_v2_sim(
            config_path=str(v2_cfg_path), out_dir=work / "v2_sim",
            token=run_token)
        keys = [o["key"] for o in OBSERVABLES]
        # vEcoli emits at <v_sim>/<exp_id>/history; v2ecoli nests under a
        # `parquet/` subdir (<v2_sim>/parquet/<exp_id>/history).
        left = read_observables(str(v_sim), exp_id, keys)
        right = read_observables(str(v2_sim / "parquet"), exp_id, keys)
        sections.append({"title": "2-generation sim dynamics",
                         "rows": compare_observables(left, right, keys=keys,
                                                     rel_tol=SIM_REL_TOL)})
    except Exception as e:
        sections.append(_error_section("2-generation sim dynamics", e))

    title = "vEcoli vs v2ecoli"
    if mode == "fast":
        title += "  —  ⚠ NOT SCIENTIFICALLY VALID (fast ParCa) ⚠"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render_report(sections, title=title))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
