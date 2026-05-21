#!/usr/bin/env python3
"""Prepare an investigation so its baseline behaviour is visible up front.

For each study in the investigation that declares ``comparative_visualizations``,
this runs the baseline + every comparison variant those overlays reference,
then renders the comparative figures. After running it, opening any prepared
study shows the baseline-vs-variant comparison without anyone manually kicking
sims.

This is the per-study companion to ``run_default_baseline.py`` (which only
produces the workspace-wide plain-baseline reference). Where default-baseline
answers "what does the cell do before any study runs", this answers "what does
each study's declared comparison look like".

Runs are driven through the live vivarium-dashboard run endpoints
(``/api/study-run-baseline`` + ``/api/study-run-variant``) so composite
resolution, the emit_paths pipeline, and the per-study declared-viz render all
go through exactly the same code path the dashboard uses interactively. The
comparative figures are then rendered directly via
``comparative_viz.render_comparative_time_series``.

Usage::

    python scripts/prepare_investigation.py                       # all studies
    python scripts/prepare_investigation.py --study dnaa-00-parameter-foundation
    python scripts/prepare_investigation.py --render-only         # skip sims
    python scripts/prepare_investigation.py --steps 120

Requires the dashboard to be running (it's the workspace run engine); the URL
is auto-detected from .pbg/dashboard/dashboard-info, override with --dashboard-url.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "workspace.yaml").is_file():
            return p
        p = p.parent
    return Path(__file__).resolve().parent.parent


REPO_ROOT = _repo_root()


def _dashboard_url(override: str | None) -> str:
    if override:
        return override.rstrip("/")
    info = REPO_ROOT / ".pbg" / "dashboard" / "dashboard-info"
    if info.is_file():
        try:
            return json.loads(info.read_text())["url"].rstrip("/")
        except Exception:
            pass
    return "http://localhost:8765"


def _post(url: str, payload: dict, timeout: float = 1800.0) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, {"raw": body[:200]}
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": f"HTTP {e.code}"}
    except Exception as e:
        return 0, {"error": str(e)}


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _new_generation_id() -> str:
    import datetime
    return "gen-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def _write_generation_manifest(inv: str, generation_id: str,
                                results: list[dict],
                                condition_caches: dict | None = None) -> Path:
    """Write the coordinated-generation manifest.

    A *generation* is one coordinated prep run: every study's baseline +
    comparison variants run, and every comparative rendered, under one id +
    one code state (git sha). The dashboard report reads the current
    manifest, shows 'results coordinated as of <generation>', and flags any
    displayed study whose latest run isn't in this generation as stale —
    so the 'results mixed across dates' problem becomes visible, not silent.
    See investigations/<inv>/feedback-2026-05-21/PLAN.md (Part 1).
    """
    import datetime
    gen_dir = REPO_ROOT / ".pbg" / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generation_id": generation_id,
        "investigation": inv,
        "created_at": datetime.datetime.now().isoformat(),
        "git_sha": _git_sha(),
        # condition_caches: {study_or_run: condition_cache_id} — pins which
        # named ParCa sim_data fit each run used (task #24). Empty until the
        # condition-cache mechanism lands.
        "condition_caches": condition_caches or {},
        "studies": [
            {
                "study": r["study"],
                "runs": [
                    {"sim_name": rr["run"], "kind": rr["kind"],
                     "run_id": rr.get("run_id")}
                    for rr in (r.get("runs") or [])
                ],
                "comparatives": [v.get("viz") for v in (r.get("rendered") or [])
                                 if "bytes" in v],
            }
            for r in results if not r.get("skipped")
        ],
    }
    (gen_dir / f"{generation_id}.json").write_text(json.dumps(manifest, indent=2))
    (gen_dir / "current.json").write_text(json.dumps(manifest, indent=2))
    return gen_dir / f"{generation_id}.json"


def _investigations() -> list[str]:
    inv_root = REPO_ROOT / "investigations"
    if not inv_root.is_dir():
        return []
    return sorted(
        d.name for d in inv_root.iterdir()
        if d.is_dir() and (d / "investigation.yaml").is_file())


def _study_slugs(inv_slug: str) -> list[str]:
    spec = yaml.safe_load((REPO_ROOT / "investigations" / inv_slug
                           / "investigation.yaml").read_text()) or {}
    studies = spec.get("studies") or []
    # entries may be plain slugs or dicts
    out = []
    for s in studies:
        out.append(s if isinstance(s, str) else (s.get("study") or s.get("name")))
    return [s for s in out if s]


def prepare_study(slug: str, dash: str, steps: int | None,
                  render_only: bool) -> dict:
    """Run baseline + comparison variants for one study, render comparatives."""
    sf = REPO_ROOT / "studies" / slug / "study.yaml"
    if not sf.is_file():
        return {"study": slug, "skipped": "no study.yaml"}
    spec = yaml.safe_load(sf.read_text()) or {}
    cvs = spec.get("comparative_visualizations") or []
    if not cvs:
        return {"study": slug, "skipped": "no comparative_visualizations"}

    # Distinct sim_names the comparatives overlay. The one whose name equals
    # the study slug is the baseline; the rest are declared variants.
    sim_names: list[str] = []
    for cv in cvs:
        for r in (cv.get("runs") or []):
            sn = r.get("sim_name")
            if sn and sn not in sim_names:
                sim_names.append(sn)

    run_results = []
    if not render_only:
        for sn in sim_names:
            if sn == slug:
                payload = {"study": slug}
                if steps is not None:
                    payload["steps"] = steps
                code, body = _post(f"{dash}/api/study-run-baseline", payload)
                kind = "baseline"
            else:
                payload = {"study": slug, "variant": sn}
                if steps is not None:
                    payload["steps"] = steps
                code, body = _post(f"{dash}/api/study-run-variant", payload)
                kind = "variant"
            run_id = (body.get("simulation_id") or body.get("run_id")
                      if isinstance(body, dict) else None)
            run_results.append({"run": sn, "kind": kind, "http": code,
                                "run_id": run_id})
            print(f"  ran {sn} ({kind}): HTTP {code} run_id={run_id}", flush=True)

    # Render comparatives directly (the run endpoints render declared viz but
    # not the comparative overlays — those only render inside run-unblocked).
    from vivarium_dashboard.lib.comparative_viz import render_comparative_time_series
    study_db = REPO_ROOT / "studies" / slug / "runs.db"
    viz_dir = REPO_ROOT / "studies" / slug / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for cv in cvs:
        runs = [{"label": r.get("label") or r.get("sim_name"),
                 "db_path": study_db, "sim_name": r.get("sim_name")}
                for r in (cv.get("runs") or [])]
        if not runs:
            continue
        out = viz_dir / f"comparative_{cv['name']}.html"
        try:
            render_comparative_time_series(
                runs=runs,
                observable_path=cv.get("observable_path", ""),
                title=cv.get("title", cv["name"]),
                y_label=cv.get("y_label", ""),
                output_path=out,
                observable_index=cv.get("observable_index"),
                target_band=cv.get("target_band"),
                target_band_label=cv.get("target_band_label"),
            )
            rendered.append({"viz": cv["name"], "bytes": out.stat().st_size})
        except Exception as e:  # noqa: BLE001
            rendered.append({"viz": cv["name"], "error": str(e)})
        print(f"  rendered {cv['name']}: {rendered[-1].get('bytes', rendered[-1].get('error'))}",
              flush=True)

    return {"study": slug, "runs": run_results, "rendered": rendered}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--investigation", default=None,
                    help="investigation slug (default: the only one present)")
    ap.add_argument("--study", default=None,
                    help="prepare only this study (default: all in the investigation)")
    ap.add_argument("--steps", type=int, default=None,
                    help="override sim length for each run (default: study params)")
    ap.add_argument("--render-only", action="store_true",
                    help="skip sims; just re-render comparatives from existing runs.db")
    ap.add_argument("--dashboard-url", default=None,
                    help="override dashboard URL (default: auto-detect)")
    args = ap.parse_args()

    inv = args.investigation
    if inv is None:
        invs = _investigations()
        if len(invs) != 1:
            print(f"error: specify --investigation (found {invs})", file=sys.stderr)
            return 2
        inv = invs[0]

    dash = _dashboard_url(args.dashboard_url)
    studies = [args.study] if args.study else _study_slugs(inv)
    generation_id = _new_generation_id()
    print(f"Preparing investigation {inv!r} via {dash}")
    print(f"  generation: {generation_id}  (git {_git_sha()})")
    print(f"  studies: {studies}")
    if not args.render_only:
        print("  (running baseline + comparison variants — this may take several "
              "minutes per study)")

    results = []
    for slug in studies:
        print(f"\n=== {slug} ===", flush=True)
        results.append(prepare_study(slug, dash, args.steps, args.render_only))

    # Record the coordinated generation. When preparing a SUBSET (--study),
    # the manifest still records what ran — but a partial generation should
    # only be promoted to current when the whole investigation was prepared.
    full_run = not args.study
    manifest_path = _write_generation_manifest(inv, generation_id, results)
    print(f"\n=== SUMMARY ===")
    print(f"generation {generation_id} → {manifest_path.relative_to(REPO_ROOT)}"
          f"{'' if full_run else '  (PARTIAL: prepared a single study)'}")
    for r in results:
        if r.get("skipped"):
            print(f"  {r['study']}: skipped ({r['skipped']})")
        else:
            nr = len(r.get("runs") or [])
            nv = sum(1 for v in (r.get("rendered") or []) if "bytes" in v)
            print(f"  {r['study']}: {nr} run(s), {nv} comparative(s) rendered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
