"""Scaffold a whole-cell-model-comparison investigation from a reference repo
and a list of configs (condition names or reference-config paths)."""
from __future__ import annotations

from pathlib import Path
import yaml


def _entry(c: str) -> dict:
    if c.endswith(".json"):
        return {"name": Path(c).stem, "config": c}
    return {"name": c, "config": c}


def scaffold_investigation(*, name: str, reference_repo: str, configs: list,
                           out_root) -> Path:
    inv_dir = Path(out_root) / name
    inv_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": 4,
        "name": name,
        "title": "Whole-Cell Model Comparison",
        "question": "Does a candidate whole-cell model reproduce a reference "
                    "implementation across a set of configurations?",
        "comparison": {
            "candidate": "v2ecoli",
            "reference": {"repo": reference_repo, "kind": "vecoli"},
            "defaults": {"seeds": 4, "gens": 1,
                         "cards": ["summary", "parca", "statistical", "standard", "trajectory"]},
            "configs": [_entry(c) for c in configs],
        },
    }
    path = inv_dir / "investigation.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return path
