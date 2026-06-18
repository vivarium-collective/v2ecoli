#!/usr/bin/env python
"""vEcoli <-> v2ecoli comparison harness.

Runs both engines from one or more vEcoli configs (no ParCa is re-run — each
engine uses a prebuilt, cached sim_data) and emits a grouped two-column HTML
report. Each config becomes its own run-group with sections in the order:
loaded config, converted processes, ParCa/sim_data diff, 2-gen sim dynamics
(plus embedded behavior overlay + statistical report card).

    .venv/bin/python scripts/compare_harness.py \
        --config /Users/eranagmon/code/vEcoli/configs/two_generations.json \
        --vecoli-simdata /Users/eranagmon/code/vEcoli/out/kb/simData.cPickle \
        --v2-cache out/cache \
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
from scripts._compare.config_adapter import resolve_vecoli_config
from scripts._compare.parca_section import final_sim_data_diff
from scripts._compare.report import (
    config_panel_section, converted_processes_section, render_grouped_report)
from scripts._compare.sim_section import (
    OBSERVABLES, compare_observables, read_observables)

# sim_data diffs should be tight; dynamics looser (two engines).
PARCA_REL_TOL = 1e-6


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(s).lower())


def _error_section(title: str, exc: Exception) -> dict:
    """A report section that surfaces a stage failure instead of aborting."""
    return {"title": title, "rows": [{
        "label": "stage failed", "left": "—", "right": "—",
        "verdict": "mismatch", "reason": f"{type(exc).__name__}: {exc}",
    }]}


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_dill(path):
    import dill
    with open(path, "rb") as f:
        return dill.load(f)


def _resolve_exp_id(base, exp_id: str) -> str:
    """Return the actual experiment-dir name under ``base`` matching ``exp_id``.

    vEcoli timestamps its output dir when ``suffix_time`` is set (the default),
    so the emitted history lives under ``<exp_id>_<timestamp>/history`` rather
    than ``<exp_id>/history``. Prefer an exact match; otherwise pick the most
    recent ``<exp_id>*`` dir that actually contains a ``history`` subdir.
    """
    base = Path(base)
    if (base / exp_id / "history").exists():
        return exp_id
    cands = sorted(p.name for p in base.glob(f"{exp_id}*")
                   if (p / "history").exists())
    return cands[-1] if cands else exp_id


def build_injected_v2_config(vecoli_cfg: dict, *, fork_repo: str) -> dict:
    """Translate a fork's vEcoli config to a v2 config carrying an
    injected_processes block (no-op block when no add_processes)."""
    from scripts._compare.config_adapter import translate_vecoli_config
    v2 = translate_vecoli_config(vecoli_cfg)
    if vecoli_cfg.get("add_processes") or vecoli_cfg.get("swap_processes"):
        v2["injected_processes"] = {
            "fork_repo": fork_repo,
            "add_processes": vecoli_cfg.get("add_processes") or [],
            "swap_processes": vecoli_cfg.get("swap_processes") or {},
            "process_configs": vecoli_cfg.get("process_configs") or {},
            "topology": vecoli_cfg.get("topology") or {},
            "time_step": float(vecoli_cfg.get("time_step", 1.0)),
        }
    return v2


def _run_one_config(cfg_path: str, work_root: Path, args) -> dict:
    """Run both engines for a single vEcoli config and return a report group.

    Returns a group dict for ``render_grouped_report``:
    ``{"config","config_id","subtitle","sections":[...],"embedded_html":[...]}``.
    Each engine stage keeps its own try/except so one failed stage degrades to
    an ``_error_section`` rather than aborting the whole config.
    """
    # Stage 1 — config (essential; if this raises, the config is skipped above).
    vecoli_cfg = resolve_vecoli_config(cfg_path, vecoli_repo=args.vecoli_repo)
    v2_cfg = build_injected_v2_config(vecoli_cfg, fork_repo=args.vecoli_repo)

    v2_cache = Path(args.v2_cache).resolve()
    v2_cfg["cache_dir"] = str(v2_cache)

    exp_id = vecoli_cfg.get("experiment_id") or Path(cfg_path).stem
    slug = _slug(exp_id)

    work = work_root / slug
    work.mkdir(parents=True, exist_ok=True)
    v2_cfg_path = work / "v2_config.json"
    v2_cfg_path.write_text(json.dumps(
        {k: v for k, v in v2_cfg.items() if not k.startswith("_")}))

    config_sec = config_panel_section(vecoli_cfg)

    # Converted-processes specs — resolve via the inject CLI (v2 venv + fork on
    # sys.path), independent of whether the sim later succeeds.
    inj = v2_cfg.get("injected_processes")
    specs: list = []
    if inj and inj.get("add_processes"):
        import subprocess as _sp
        try:
            spec_json = _sp.check_output(
                [".venv/bin/python", "-m", "scripts._compare.inject",
                 args.vecoli_repo, str(v2_cfg_path)], text=True)
            specs = json.loads(spec_json)
        except Exception:
            specs = []

    # Cache token from config (no longer includes ParCa mode — no ParCa is run).
    from scripts._compare.cache import cache_key
    run_token = cache_key(vecoli_cfg, commit="", mode="")
    if args.force:
        import time as _time
        run_token = f"{run_token}-force-{_time.time_ns()}"

    # ParCa / sim_data diff — both sides are CACHED (no re-fit). Left = vEcoli's
    # prebuilt simData.cPickle (pickle); right = v2ecoli's sim_data_cache.dill.
    try:
        v_sim_data = _load_pickle(args.vecoli_simdata)
        v2_sim_data = _load_dill(v2_cache / "sim_data_cache.dill")
        parca_sec = {
            "title": "ParCa / sim_data (cached)",
            "desc": "Calibrated parameters (sim_data) each engine uses — "
                    "loaded from cached ParCa outputs, not re-fit.",
            "rows": final_sim_data_diff(v_sim_data, v2_sim_data,
                                        rel_tol=PARCA_REL_TOL),
        }
    except Exception as e:
        parca_sec = _error_section("ParCa / sim_data (cached)", e)

    # Sim (both engines) + dynamics comparison. Drives the behavior overlay,
    # report card, and the converted-processes "ran in both" gate.
    embedded: list[str] = []
    ran: dict = {}
    try:
        vecoli_sim_out = work / "vecoli_sim"
        vecoli_sim_cfg = dict(vecoli_cfg)
        vecoli_sim_cfg["sim_data_path"] = str(Path(args.vecoli_simdata).resolve())
        vecoli_sim_cfg["out_dir"] = str(vecoli_sim_out)
        _ea = dict(vecoli_sim_cfg.get("emitter_arg") or {})
        _ea["out_dir"] = str(vecoli_sim_out)
        vecoli_sim_cfg["emitter_arg"] = _ea
        vecoli_sim_cfg["emitter"] = "parquet"
        vecoli_sim_cfg_path = work / "vecoli_sim_config.json"
        vecoli_sim_cfg_path.write_text(json.dumps(vecoli_sim_cfg))
        v_sim = orchestrator.run_vecoli_sim(
            config_path=str(vecoli_sim_cfg_path), out_dir=vecoli_sim_out,
            token=run_token, vecoli_repo=args.vecoli_repo)
        v2_sim = orchestrator.run_v2_sim(
            config_path=str(v2_cfg_path), out_dir=work / "v2_sim",
            token=run_token)
        keys = [o["key"] for o in OBSERVABLES]
        # vEcoli emits at <v_sim>/<exp_id>/history; v2ecoli nests under a
        # `parquet/` subdir (<v2_sim>/parquet/<exp_id>/history). vEcoli may
        # suffix the experiment dir with a timestamp (suffix_time, default on),
        # so resolve the actual dir name rather than assuming the bare exp_id.
        v_exp = _resolve_exp_id(v_sim, exp_id)
        v2_exp = _resolve_exp_id(v2_sim / "parquet", exp_id)
        left = read_observables(str(v_sim), v_exp, keys)
        right = read_observables(str(v2_sim / "parquet"), v2_exp, keys)
        sim_sec = {
            "title": "2-generation sim dynamics",
            "desc": "Per-observable agreement of the simulated trajectories "
                    "(mean/min/max summary), within the relative tolerance.",
            "rows": compare_observables(left, right, keys=keys,
                                        rel_tol=args.tol_rel),
        }

        # Behavior detail — per-observable overlay plotted against the real time
        # axis (rows ordered by sim time), vEcoli vs v2ecoli on a shared scale.
        from scripts._compare.charts import multiline_svg
        from scripts._compare.sim_section import read_observable_xy
        figs = []
        for key in keys:
            lpts = read_observable_xy(str(v_sim), v_exp, key)
            rpts = read_observable_xy(str(v2_sim / "parquet"), v2_exp, key)
            if len(lpts) < 2 and len(rpts) < 2:
                continue
            svg, _rng = multiline_svg([lpts, rpts])
            figs.append(
                f"<figure style='display:inline-block;margin:6px 14px;"
                f"width:300px;vertical-align:top'>"
                f"<figcaption style='font:12px system-ui;color:#374151'>{key}"
                f" <span style='color:#9ca3af'>(vs time)</span></figcaption>"
                f"{svg}</figure>")
        if figs:
            embedded.append(
                "<section><h2>Behavior detail — vEcoli (blue) vs v2ecoli (amber)"
                "</h2><div style='display:flex;flex-wrap:wrap;gap:4px'>"
                + "".join(figs) + "</div></section>")

        # The sim completed and each process was injected -> ran in both.
        ran = {s["name"]: True for s in specs}

        # Statistical-equivalence report card. The flat per-observable arrays
        # are treated as the two value distributions for the Welch ttest axes
        # (cell_mass, growth_rate) — a behavioral equivalence proxy, not
        # bit-parity.
        from scripts._compare.report_card_section import build_report_card
        left_dist = {k: list(left.get(k, [])) for k in ("cell_mass", "growth_rate")}
        right_dist = {k: list(right.get(k, [])) for k in ("cell_mass", "growth_rate")}
        verdict_json_dict, card_html = build_report_card(
            left_dist, right_dist, reference_model="vEcoli (fork)",
            measured_model="v2ecoli", tol_rel=args.tol_rel)
        card_path = Path(args.out).with_name(f"report_card_verdict_{slug}.json")
        card_path.write_text(json.dumps(verdict_json_dict, indent=2))
        embedded.append(
            "<p style='font-size:0.85em;color:#64748b;margin:4px 0'>"
            "Note: distributions are per-timestep trajectory samples (not "
            "per-cell); relative-mean bands are meaningful, p-values indicative "
            "only.</p>" + card_html)
    except Exception as e:
        sim_sec = _error_section("2-generation sim dynamics", e)
        ran = {}

    # Converted-processes panel sits RIGHT AFTER the config, but its "ran in
    # both" status is only known once the sim has run (above).
    converted_sec = converted_processes_section(specs, ran)

    sections = [config_sec, converted_sec, parca_sec, sim_sec]
    embedded = [h for h in embedded if h]
    return {
        "config": exp_id,
        "config_id": slug,
        "subtitle": cfg_path,
        "sections": sections,
        "embedded_html": embedded,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, nargs="+",
                   help="One or more vEcoli JSON configs (source of truth). "
                        "Each becomes its own run-group in the report.")
    p.add_argument("-o", "--out", default="out/compare/report.html")
    p.add_argument("--workdir", default="out/compare_harness")
    p.add_argument("--vecoli-repo", default="/Users/eranagmon/code/vEcoli",
                   help="Path to the vEcoli fork checkout.")
    p.add_argument("--vecoli-simdata", default=None,
                   help="Prebuilt vEcoli simData.cPickle. Used as the vEcoli "
                        "sim's sim_data_path AND the left side of the ParCa "
                        "diff. Default: <vecoli-repo>/out/kb/simData.cPickle.")
    p.add_argument("--v2-cache", default="out/cache",
                   help="Prebuilt v2ecoli ParCa cache dir. Used as the v2 sim's "
                        "cache_dir AND the right side of the ParCa diff "
                        "(<v2-cache>/sim_data_cache.dill). Default out/cache.")
    p.add_argument("--tol-rel", type=float, default=0.05,
                   help="Relative tolerance shared by sim-dynamics badges "
                        "(within_tol / mismatch) and report-card bands "
                        "(mismatch band = 2 × tol-rel).  Defaults to 0.05.")
    p.add_argument("--force", action="store_true",
                   help="Bypass the run cache and re-run both engines.")
    args = p.parse_args(argv)

    if args.vecoli_simdata is None:
        args.vecoli_simdata = str(
            Path(args.vecoli_repo) / "out" / "kb" / "simData.cPickle")

    # Resolve to an absolute path: vEcoli subprocesses run with cwd=vEcoli, so
    # a relative workdir would make them write under the vEcoli tree while the
    # harness reads under v2ecoli. Absolute keeps both sides in agreement.
    work_root = Path(args.workdir).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    groups = []
    for cfg_path in args.config:
        try:
            groups.append(_run_one_config(cfg_path, work_root, args))
        except Exception as e:
            # Config resolution itself failed — surface as a degenerate group.
            slug = _slug(Path(cfg_path).stem)
            groups.append({
                "config": Path(cfg_path).stem,
                "config_id": slug,
                "subtitle": cfg_path,
                "sections": [_error_section("Loaded config", e)],
                "embedded_html": [],
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        render_grouped_report(groups, title="vEcoli vs v2ecoli"),
        encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
