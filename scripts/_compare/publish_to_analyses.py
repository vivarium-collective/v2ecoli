"""Publish a comparison-harness report into a workspace study's Analyses gallery.

The dashboard's **Analyses** tab discovers saved comparison report cards under
``studies/<study>/viz/report_card/<name>.html`` (+ an optional
``<name>.verdict.json`` sidecar carrying the ``overall`` verdict for the card
badge). This helper copies a self-contained harness report there so it appears
in the gallery alongside the parsimony 3D viewers and the PTools card.

Usage (CLI):
    python -m scripts._compare.publish_to_analyses \
        --report out/regen_bothfixes/report.html \
        --workspace workspace --study showcase-6-equivalence-large \
        --name vecoli-v2ecoli-comparison
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def publish_report_card(
    report_html: str | Path,
    workspace_root: str | Path,
    study: str,
    name: str = "vecoli-v2ecoli-comparison",
    overall: str | None = None,
) -> Path:
    """Copy ``report_html`` into the study's ``viz/report_card/`` so the
    Analyses gallery picks it up. Returns the written HTML path.

    ``overall`` (e.g. ``"within_tol"`` / ``"mismatch"``), when given, is written
    to the ``<name>.verdict.json`` sidecar the gallery reads for the card badge.
    """
    report_html = Path(report_html)
    if not report_html.is_file():
        raise FileNotFoundError(f"report not found: {report_html}")
    study_dir = Path(workspace_root) / "studies" / study
    if not (study_dir / "study.yaml").is_file():
        raise FileNotFoundError(
            f"{study!r} is not a study (no study.yaml under {study_dir})")
    dest_dir = study_dir / "viz" / "report_card"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{name}.html"
    shutil.copyfile(report_html, dest)
    if overall is not None:
        (dest_dir / f"{name}.verdict.json").write_text(
            json.dumps({"overall": overall}), encoding="utf-8")
    return dest


def overall_from_verdicts(verdict_jsons: list[str | Path]) -> str | None:
    """Worst-of-axes roll-up across per-condition ``report_card_verdict`` files
    (``within_tol`` < ``drift`` < ``mismatch``). Returns None if none parse."""
    rank = {"within_tol": 0, "drift": 1, "mismatch": 2}
    worst = None
    for vj in verdict_jsons:
        try:
            o = json.loads(Path(vj).read_text(encoding="utf-8")).get("overall")
        except Exception:
            continue
        if o in rank and (worst is None or rank[o] > rank[worst]):
            worst = o
    return worst


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True, help="self-contained report HTML")
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--study", required=True)
    p.add_argument("--name", default="vecoli-v2ecoli-comparison")
    p.add_argument("--overall", default=None,
                   help="within_tol | drift | mismatch (card badge)")
    p.add_argument("--verdicts", nargs="*", default=None,
                   help="per-condition verdict JSONs to roll up into --overall")
    args = p.parse_args(argv)
    overall = args.overall
    if overall is None and args.verdicts:
        overall = overall_from_verdicts(args.verdicts)
    dest = publish_report_card(args.report, args.workspace, args.study,
                               args.name, overall)
    print(f"published -> {dest}" + (f"  (overall={overall})" if overall else ""))


if __name__ == "__main__":
    main()
