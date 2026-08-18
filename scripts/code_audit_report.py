"""scripts/code_audit_report.py — a REDUCED comparison report: just the code
that was transferred from the private reference repository into this one, per
config. No report cards, no trajectories, no config panels.

A "code audit" companion to comparison_report_card.py's full report — same
data (each config's v2ecoli_build_config.json sidecar under --out), same
render_report page shell, same converted_processes_section (full transferred
source, dedup'd by file, with resulting process-bigraph schema per process),
just without everything else. A config with no injected processes (e.g. the
basal control) is skipped — there is nothing to audit.

    .venv/bin/python scripts/code_audit_report.py \\
        --investigation <investigation-name> -o out/<run-dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._compare import report
from scripts.comparison_report_card import (
    _git_provenance,
    converted_processes_section,
    repositories_section,
)


def _read_v2_build(v2_dir: str) -> dict | None:
    p = Path(v2_dir) / "v2ecoli_build_config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _how_to_read_section(candidate: dict | None, reference: dict | None) -> dict:
    cand_n = (candidate or {}).get("name", "this repository")
    ref_n = (reference or {}).get("name", "the private reference repository")
    body = (
        f"<p style='margin:0 0 14px;line-height:1.55;font-size:14px'>"
        f"This is a <strong>code audit</strong>: every process transferred from "
        f"<b>{report._e(ref_n)}</b> into <b>{report._e(cand_n)}</b>, with its "
        f"<strong>full source code</strong> embedded inline, grouped by the config "
        f"that transfers it.</p>"
        f"<div style='background:var(--card,#f8fafc);border-left:4px solid #f59e0b;"
        f"padding:12px 16px;margin:0 0 16px;border-radius:4px;font-size:13.5px;line-height:1.55'>"
        f"<strong>What this report is NOT:</strong> it is <strong>not a full repository "
        f"diff</strong> and it carries <strong>no behavioral results</strong> (no report "
        f"cards, no trajectories) — see the standardized comparison report for those. "
        f"This is the code-transfer surface only.</div>"
        f"<p style='margin:0;line-height:1.55;font-size:14px'>"
        f"A config with <strong>no transferred code</strong> (e.g. the <code>basal</code> "
        f"wild-type control) is omitted below — there is nothing to audit.</p>")
    return {
        "title": "Overview — how to read this report",
        "kind": "content",
        "nav_group": "Overall",
        "desc": ("Every process transferred from the private reference repository, "
                 "full source embedded, grouped by config. No behavioral results."),
        "html": body,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--investigation", required=True,
                   help="path or name of an investigation (study-YAML-only mode).")
    p.add_argument("-o", "--out", required=True,
                   help="dir holding <out>/<config>/v2ecoli_build_config.json.")
    p.add_argument("--output-file", default=None,
                   help="output HTML path (default: <out>/code_audit_report.html).")
    args = p.parse_args(argv)

    from scripts._compare.study_spec import load_investigation
    _ctx, specs = load_investigation(args.investigation)

    cand = _git_provenance(str(Path(__file__).resolve().parents[1]))
    import os
    ref = _git_provenance(os.environ.get("V2E_VECOLI_DIR"))

    sections = [_how_to_read_section(cand, ref), repositories_section(cand, ref)]
    audited, skipped = [], []
    for spec in specs:
        v2_dir = f"{args.out}/{spec.name}"
        v2_build = _read_v2_build(v2_dir)
        sec = converted_processes_section(spec.name, v2_build)
        if sec is None:
            skipped.append(spec.name)
            continue
        sec["nav_group"] = spec.name
        sections.append(sec)
        audited.append(spec.name)

    cand_lbl = f"{cand['name']} (v2ecoli)" if cand else "v2ecoli"
    ref_lbl = f"{ref['name']} (vEcoli)" if ref else "vEcoli"
    title = f"{cand_lbl} ↔ {ref_lbl} — code audit"
    html = report.render_report(sections, title=title)

    out_path = Path(args.output_file) if args.output_file else Path(args.out) / "code_audit_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"audited configs ({len(audited)}): {', '.join(audited) or '—'}")
    print(f"skipped, no transferred code ({len(skipped)}): {', '.join(skipped) or '—'}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
