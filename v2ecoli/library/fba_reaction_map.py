"""Loader for the Millard 2017 -> v2ecoli FBA reaction map.

The YAML at ``v2ecoli/data/millard_v2ecoli_reaction_map.yaml`` maps each
Millard reaction id (SBML reaction NAME from
``millard2017_central_metabolism.xml``) onto one or more v2ecoli FBA *base*
reaction ids, with a stoichiometric ``scale`` factor. Reactions with no clean
v2ecoli counterpart live under ``millard_only`` and are excluded here.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_reaction_map(path: str) -> dict[str, list[tuple[str, float]]]:
    """millard_rxn -> [(fba_rxn_id, scale), ...]. Excludes millard_only."""
    data = yaml.safe_load(Path(path).read_text())
    out: dict[str, list[tuple[str, float]]] = {}
    for millard_rxn, targets in (data.get("pins") or {}).items():
        pins = [(t["fba"], float(t.get("scale", 1.0))) for t in targets]
        if pins:
            out[millard_rxn] = pins
    return out
