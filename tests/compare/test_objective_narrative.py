import re
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
INV = REPO / "workspace" / "investigations" / "whole-cell-model-comparison"
BANNED = re.compile(r"\b(before\b.*\bafter|after fix|root cause|we fixed|rpoBC|exp_free|found-and-fix)\b", re.I)


def _narrative_text():
    # Scope: the whole-cell-model-comparison investigation + its OWN member
    # studies (study.yaml `investigation: whole-cell-model-comparison`).
    # workspace/studies/ is a shared top-level registry holding studies for
    # many unrelated investigations (mbp-*, pdmp-*, colonies-*, ...); those
    # are out of scope for this lint and have their own narrative/lint
    # ownership.
    docs = [INV / "investigation.yaml"]
    for p in sorted((REPO / "workspace" / "studies").glob("*/study.yaml")):
        d = yaml.safe_load(p.read_text()) or {}
        if d.get("investigation") == "whole-cell-model-comparison":
            docs.append(p)
    chunks = []
    for p in docs:
        d = yaml.safe_load(p.read_text()) or {}
        for k in ("executive", "how_to_read", "biological_story", "glossary", "narrative", "claim", "findings"):
            v = d.get(k)
            if v is not None:
                chunks.append(yaml.safe_dump(v))
    return "\n".join(chunks)


def test_no_fix_history_in_narrative():
    m = BANNED.search(_narrative_text())
    assert m is None, f"fix-history phrasing found: {m.group(0)!r}"
