"""Populate spec.embed_visualizations[] for each pdmp-* study so the
DOWNLOADED investigation report (walkthrough.js _buildInvestigationReportHtml)
inlines viz as <iframe srcdoc="...">.

Discovery: walks reports/figures/<study>/*.html (where my generator writes
files) and adds one entry per file: {name, url, description}. The URL is
workspace-relative (/reports/figures/<study>/<file>.html); the dashboard
server's static-file fallback serves it.

The dashboard ALREADY auto-discovers studies/<name>/viz/*.html into
embed_visualizations[], but those entries aren't persisted to study.yaml,
and the downloaded-report builder fetches each URL at download time. By
writing the entries into study.yaml directly, the downloaded report has
the URL list it needs, fetches each, and inlines them.

Idempotent: existing embed_visualizations[] entries (matched by url) are
preserved.

Run from worktree root:
    python scripts/wire_embed_visualizations.py
"""
from __future__ import annotations
import os
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)


PDMP_STUDIES = {
    "pdmp-00": "pdmp-00-characterization",
    "pdmp-01": "pdmp-01-metabolism-ode",
    "pdmp-02": "pdmp-02-jump-processes",
    "pdmp-03": "pdmp-03-inference",
    "pdmp-04": "pdmp-04-compilation",
    "pdmp-05": "pdmp-05-causal-discovery",
}


def _title_from_html(path: Path) -> str | None:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:2000]
    except Exception:
        return None
    m = re.search(r"<title[^>]*>(.*?)</title>", head, re.I | re.S)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    m = re.search(r"<h1[^>]*>(.*?)</h1>", head, re.I | re.S)
    if m:
        return re.sub(r"<[^>]+>", "", re.sub(r"\s+", " ", m.group(1))).strip()
    return None


def main():
    for fig_dir_name, study_slug in PDMP_STUDIES.items():
        fig_dir = Path("reports/figures") / fig_dir_name
        yaml_path = Path("studies") / study_slug / "study.yaml"
        if not yaml_path.exists():
            print(f"  SKIP {study_slug}: yaml missing"); continue
        if not fig_dir.is_dir():
            print(f"  SKIP {study_slug}: figures dir missing"); continue

        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        existing = list(spec.get("embed_visualizations") or [])
        existing_urls = {(e or {}).get("url") for e in existing}

        added = 0
        for html in sorted(fig_dir.glob("*.html")):
            url = f"/reports/figures/{fig_dir_name}/{html.name}"
            if url in existing_urls:
                continue
            title = _title_from_html(html) or html.stem.replace("_", " ")
            entry = {
                "name": title,
                "url": url,
                "description": (
                    f"Per-study visualization rendered from reports/figures/"
                    f"{fig_dir_name}/{html.name}. Inlined into the downloaded "
                    f"report as iframe srcdoc."
                ),
            }
            existing.append(entry)
            added += 1
        if added == 0:
            print(f"  {study_slug}: no new embeds ({len(existing)} already wired)")
            continue
        spec["embed_visualizations"] = existing
        yaml_path.write_text(
            yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )
        print(f"  {study_slug}: +{added} embed_visualizations wired (total {len(existing)})")


if __name__ == "__main__":
    main()
