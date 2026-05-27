"""Audit Claude auto-memory entries for scope discipline.

Per ~/.claude/CLAUDE.md > Auto-memory: scope discipline — every memory
file should have a `scope:` block declaring applies_to / default_on /
optional required_conditions. This audit walks the memory directory and:

  - PASS  → memory has a well-formed scope block
  - WARN  → memory has type: feedback but no scope block (high
            over-application risk)
  - INFO  → memory has type other than feedback and no scope (less
            critical, but explicit scope is still recommended)

Default memory directory:
  ~/.claude/projects/-Users-eranagmon-code-v2ecoli/memory/

Run from anywhere:
    python scripts/audit_memory_scope.py [--dir <path>] [--strict]

--strict exits non-zero if any feedback memory is missing scope (for CI gates).
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from pathlib import Path

import yaml

DEFAULT_MEMORY_DIR = Path.home() / ".claude/projects/-Users-eranagmon-code-v2ecoli/memory"


def parse_frontmatter(path: Path) -> dict | None:
    """Pull the leading YAML frontmatter out of a markdown file.

    Returns the parsed dict, or None if no recognizable frontmatter is found.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return None
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None


def memory_type(fm: dict) -> str | None:
    """Pull the type field (top-level OR nested under metadata)."""
    t = fm.get("type")
    if t:
        return t
    meta = fm.get("metadata") or {}
    if isinstance(meta, dict):
        return meta.get("type")
    return None


def audit_one(path: Path) -> dict:
    fm = parse_frontmatter(path)
    if fm is None:
        return {"path": path, "level": "warn",
                "msg": "no parseable frontmatter — can't determine scope"}
    t = memory_type(fm) or "unknown"
    # scope may live at top level OR nested under metadata (both are valid)
    scope = fm.get("scope")
    if scope is None and isinstance(fm.get("metadata"), dict):
        scope = fm["metadata"].get("scope")
    if scope is None:
        if t == "feedback":
            return {"path": path, "level": "warn",
                    "msg": "type: feedback but no scope block (high over-application risk)"}
        return {"path": path, "level": "info",
                "msg": f"type: {t} has no explicit scope (recommended)"}
    if not isinstance(scope, dict):
        return {"path": path, "level": "warn",
                "msg": f"scope field present but not a mapping (got {type(scope).__name__})"}
    applies_to = scope.get("applies_to")
    if not applies_to:
        return {"path": path, "level": "warn",
                "msg": "scope present but missing 'applies_to'"}
    summary = (
        f"applies_to={applies_to}; "
        f"default={('off' if scope.get('default_off') else 'on')}"
    )
    return {"path": path, "level": "pass", "msg": summary}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=DEFAULT_MEMORY_DIR,
                   help=f"memory directory (default: {DEFAULT_MEMORY_DIR})")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if any feedback memory lacks scope")
    args = p.parse_args()

    if not args.dir.is_dir():
        sys.exit(f"memory dir not found: {args.dir}")

    files = sorted([f for f in args.dir.glob("*.md") if f.name != "MEMORY.md"])
    if not files:
        sys.exit(f"no memory files under {args.dir}")

    results = [audit_one(f) for f in files]
    by_level = {"pass": [], "warn": [], "info": []}
    for r in results:
        by_level[r["level"]].append(r)

    print(f"Auditing {len(files)} memory file(s) under {args.dir}\n")
    if by_level["pass"]:
        print(f"[PASS] ({len(by_level['pass'])})")
        for r in by_level["pass"]:
            print(f"  ✓ {r['path'].name}  {r['msg']}")
        print()
    if by_level["warn"]:
        print(f"[WARN] ({len(by_level['warn'])})")
        for r in by_level["warn"]:
            print(f"  ⚠ {r['path'].name}  {r['msg']}")
        print()
    if by_level["info"]:
        print(f"[INFO] ({len(by_level['info'])})")
        for r in by_level["info"]:
            print(f"  ℹ {r['path'].name}  {r['msg']}")
        print()

    print(
        f"Summary: {len(by_level['pass'])} pass · "
        f"{len(by_level['warn'])} warn · "
        f"{len(by_level['info'])} info"
    )

    if args.strict and by_level["warn"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
