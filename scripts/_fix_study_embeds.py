"""Rewrite each study yaml's embed_visualizations block to point only
at figures that actually exist on disk.

Strategy:

  1. For each existing entry: keep if URL resolves; otherwise try the
     un-suffixed name (drop '_real' / '_AUTO' / '_REAL'); otherwise drop.
  2. After cleaning, add an embed entry for every figure under
     ``reports/figures/<short-slug>/`` that is not yet referenced —
     using a short auto-generated name from the filename and a stub
     description (one-liner).
  3. Sprint-3 sprint cards on pdmp-05 are already wired correctly —
     don't touch entries whose URL resolves.

The script is idempotent: rerunning produces the same yaml.
"""
from __future__ import annotations

from pathlib import Path
import re
import yaml

ROOT = Path(".")
STUDIES_DIR = ROOT / "studies"
FIGS_ROOT = ROOT / "reports" / "figures"

# Map full slug → short slug for directory lookup.
SHORT_SLUG = {
    "pdmp-00-characterization": "pdmp-00",
    "pdmp-01-metabolism-ode": "pdmp-01",
    "pdmp-02-jump-processes": "pdmp-02",
    "pdmp-03-inference": "pdmp-03",
    "pdmp-04-compilation": "pdmp-04",
    "pdmp-05-causal-discovery": "pdmp-05",
}


def humanize(stem: str) -> str:
    parts = re.split(r"[_\-]+", stem)
    return " ".join(p.capitalize() if not p.isupper() else p for p in parts)


def resolve_url(url: str) -> Path | None:
    """Try the URL as-is, then with the `_real` / `_REAL` / `_AUTO` suffix
    stripped. Return the resolved path or None.
    """
    rel = url.lstrip("/")
    p = ROOT / rel
    if p.is_file():
        return p
    name = p.name
    for suffix in ("_real.html", "_REAL.html", "_AUTO.html"):
        if name.endswith(suffix):
            alt = p.parent / (name[: -len(suffix)] + ".html")
            if alt.is_file():
                return alt
    return None


def clean_embeds(study_slug: str, existing: list[dict]) -> list[dict]:
    """Drop broken entries; rewrite '_real' to actual path where the
    un-suffixed file exists. Preserve order."""
    cleaned = []
    seen_urls = set()
    for entry in existing:
        url = entry["url"]
        resolved = resolve_url(url)
        if resolved is None:
            continue
        new_url = "/" + str(resolved.relative_to(ROOT)).replace("\\", "/")
        if new_url in seen_urls:
            continue
        seen_urls.add(new_url)
        new_entry = dict(entry)
        new_entry["url"] = new_url
        cleaned.append(new_entry)
    return cleaned


def add_missing_existing(study_slug: str,
                         current: list[dict]) -> list[dict]:
    """Append entries for every figure under reports/figures/<short>/
    that isn't already referenced by URL."""
    short = SHORT_SLUG[study_slug]
    fig_dir = FIGS_ROOT / short
    if not fig_dir.is_dir():
        return current
    existing_urls = {e["url"] for e in current}
    # Also consider the studied-named directory variant.
    extra_urls = set()
    for fig in sorted(fig_dir.glob("*.html")):
        # Skip timestamped duplicates: keep only canonical files.
        if re.search(r"_\d{8}T\d{6}_[0-9a-f]{8}", fig.stem):
            continue
        url = "/" + str(fig.relative_to(ROOT)).replace("\\", "/")
        if url in existing_urls or url in extra_urls:
            continue
        extra_urls.add(url)
        current.append({
            "name": humanize(fig.stem),
            "url": url,
            "description": (
                f"Auto-wired existing figure at {url}. "
                "Promoted from disk by the embed-wiring sweep — refine the "
                "name and description if a richer caption is wanted."
            ),
        })
    return current


def rewrite_study(study_yaml: Path):
    text = study_yaml.read_text()
    data = yaml.safe_load(text)
    slug = study_yaml.parent.name
    existing = data.get("embed_visualizations", []) or []

    cleaned = clean_embeds(slug, existing)
    final = add_missing_existing(slug, cleaned)
    data["embed_visualizations"] = final

    # Dump the embeds block as the tail of the file. Preserve
    # everything before `embed_visualizations:` and replace from there.
    lines = text.splitlines()
    cut = None
    for i, line in enumerate(lines):
        if line.startswith("embed_visualizations:"):
            cut = i
            break

    head = "\n".join(lines[:cut]) if cut is not None else text
    tail_block = yaml.dump(
        {"embed_visualizations": final},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    new_text = head.rstrip() + "\n" + tail_block
    study_yaml.write_text(new_text)
    before = len(existing)
    after = len(final)
    print(f"{slug:<30} embeds {before} → {after}  "
          f"({before - len(cleaned)} dropped, "
          f"{after - len(cleaned)} added from disk)")


def main():
    for sy in sorted(STUDIES_DIR.glob("pdmp-*/study.yaml")):
        rewrite_study(sy)


if __name__ == "__main__":
    main()
