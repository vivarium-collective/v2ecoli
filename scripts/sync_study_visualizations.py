"""Wire every reports/figures/<study>/*.html into the corresponding
study.yaml visualizations: block.

The dashboard's walkthrough.js auto-discovers HTML files under
reports/figures/<study>/ AND honors explicit `visualizations:` entries
in study.yaml. Auto-discovery feeds embed_visualizations[]; explicit
entries control the per-study card list rendered by the Investigation
walkthrough.

This script makes both lists consistent: every HTML on disk gets a
visualizations[] entry with a name (slugified), address (file: path),
and description (best-effort from the page's <title>).

Idempotent: existing entries (matched by address) are preserved with
their existing name/description; only NEW files get appended.

Run from worktree root:
    python scripts/sync_study_visualizations.py
"""
from __future__ import annotations
import os
import re
from pathlib import Path
import sys

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


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "viz"


def main():
    for fig_dir_name, study_slug in PDMP_STUDIES.items():
        fig_dir = Path("reports/figures") / fig_dir_name
        yaml_path = Path("studies") / study_slug / "study.yaml"
        if not yaml_path.exists():
            print(f"  SKIP {study_slug}: yaml missing"); continue
        if not fig_dir.is_dir():
            print(f"  SKIP {study_slug}: figures dir {fig_dir} missing"); continue

        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        existing = list(spec.get("visualizations") or [])
        existing_addrs = {(v or {}).get("address") for v in existing}

        added = 0
        for html in sorted(fig_dir.glob("*.html")):
            address = f"file:reports/figures/{fig_dir_name}/{html.name}"
            if address in existing_addrs:
                continue  # already wired
            title = _title_from_html(html) or html.stem.replace("_", " ")
            entry = {
                "name": _slugify(html.stem.replace("_", "-")),
                "address": address,
                "description": title,
            }
            existing.append(entry)
            added += 1
        if added == 0:
            print(f"  {study_slug}: no new viz to add ({len(existing)} already wired)")
            continue
        spec["visualizations"] = existing
        yaml_path.write_text(
            yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )
        print(f"  {study_slug}: +{added} new viz wired (total now {len(existing)})")


if __name__ == "__main__":
    main()
