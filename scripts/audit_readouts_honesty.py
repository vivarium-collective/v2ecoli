"""Audit each study's readouts[] for path honesty.

Problem: the Path column in the dashboard shows things like
'out/trajectories/lqr_sobol.csv' for readouts that are still in 'planned'
status — looks authoritative but the file doesn't exist. Reviewers can't
tell what's real.

Fix:
  1. For readouts with status != 'implemented' AND the path doesn't exist
     on disk, REPLACE path with a 'TBD — see status' marker. The dashboard
     will then either skip the line or show 'Path: TBD'.
  2. For readouts marked 'implemented', verify the path actually resolves;
     if not, demote to 'planned' with an honest note.
  3. Print a per-study audit table.

Run from worktree root:
    python scripts/audit_readouts_honesty.py
"""
from __future__ import annotations
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)


def _path_exists(path_str: str) -> bool:
    if not path_str:
        return False
    p = Path(path_str)
    if p.exists():
        return True
    # glob fallback
    if "*" in path_str:
        parts = path_str.split("/")
        for i, part in enumerate(parts):
            if "*" in part:
                base = Path("/".join(parts[:i]) or ".")
                pattern = "/".join(parts[i:])
                try:
                    return any(base.glob(pattern))
                except Exception:
                    return False
    return False


def main():
    for yaml_path in sorted(Path("workspace/studies").glob("pdmp-*/study.yaml")):
        slug = yaml_path.parent.name
        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        readouts = spec.get("readouts") or []
        if not readouts:
            print(f"  {slug}: 0 readouts"); continue

        changed = 0
        rows = []
        for r in readouts:
            status = r.get("status", "")
            path = r.get("path", "")
            exists = _path_exists(path) if path else False

            if status == "implemented" and not exists:
                r["status"] = "planned"
                r["path"] = "TBD (verification was ad-hoc; not persisted to disk yet)"
                changed += 1
                rows.append((r["name"], "DEMOTED", path, "no file at path"))
            elif status != "implemented" and not exists and path:
                # planned readout with a speculative path — clarify
                r["path"] = f"TBD — planned at {path}"
                changed += 1
                rows.append((r["name"], "CLARIFIED", path, "speculative path → TBD"))
            elif exists:
                rows.append((r["name"], "OK", path, "file resolves"))
            else:
                rows.append((r["name"], "OK", path or "(none)", ""))

        print(f"\n== {slug} ({changed} changed) ==")
        for name, action, path, note in rows:
            print(f"  [{action:9s}] {name:40s}  {note}")

        if changed:
            yaml_path.write_text(
                yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
