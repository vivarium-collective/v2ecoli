"""Variant grammar → declarative config-override branch specs.

Adapts vEcoli's parse_variants (runscripts/create_variants.py). Each variant
parameter declares a ``target`` path (``"<process-name>.<config-key>"``) plus
exactly one value source: ``value`` (a list) or a numpy generator such as
``linspace`` ({start, stop, num}). Multiple parameters combine via top-level
``op``: ``prod`` (cartesian), ``zip`` (elementwise), ``add`` (concatenate).
``nested`` is not supported in MVP.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BranchSpec:
    variant_index: int  # position in the full ordered branch list (baseline=0 when included)
    variant_name: str
    overrides: dict[str, Any]
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_variant_params(variant_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one variant block into a list of ``{target_path: value}`` dicts."""
    variant_config = dict(variant_config)  # don't mutate caller's dict
    operation = None
    if sum(1 for k in variant_config if k != "op") > 1:
        if "op" not in variant_config:
            raise ValueError("Variant has >1 parameter but no 'op' key.")
        operation = variant_config.pop("op")
    elif "op" in variant_config:
        raise ValueError("Single-parameter variant must not define 'op'.")

    parsed: dict[str, list[Any]] = {}
    targets: dict[str, str] = {}
    for param_name, param_conf in variant_config.items():
        param_conf = dict(param_conf)
        target = param_conf.pop("target", None)
        if target is None:
            raise ValueError(f"variant param {param_name!r} missing 'target'.")
        targets[param_name] = target
        if len(param_conf) != 1:
            raise ValueError(f"variant param {param_name!r} needs exactly one value source.")
        ptype, pvals = next(iter(param_conf.items()))
        if ptype == "value":
            if not isinstance(pvals, list):
                raise ValueError(f"{param_name!r} 'value' must be a list.")
            parsed[param_name] = pvals
        elif ptype == "nested":
            raise NotImplementedError("nested variants are deferred (MVP).")
        else:
            try:
                np_func = getattr(np, ptype)
            except AttributeError as e:
                raise ValueError(f"{param_name!r} unknown value source {ptype!r}.") from e
            parsed[param_name] = np_func(**pvals).tolist()

    names = list(parsed.keys())
    if not names:
        return []
    if operation == "prod":
        combos = itertools.product(*(parsed[k] for k in names))
        dicts = [dict(zip(names, combo)) for combo in combos]
    elif operation == "zip":
        n = len(parsed[names[0]])
        for k in names:
            if len(parsed[k]) != n:
                raise ValueError("zip requires equal-length parameters.")
        dicts = [{k: parsed[k][i] for k in names} for i in range(n)]
    elif operation == "add":
        dicts = []
        for k in names:
            dicts.extend({k: v} for v in parsed[k])
    elif operation is None:
        k = names[0]
        dicts = [{k: v} for v in parsed[k]]
    else:
        raise ValueError(f"Unknown op {operation!r}.")

    # Re-key by target path.
    return [{targets[name]: val for name, val in d.items()} for d in dicts]


def expand_branches(config: dict[str, Any]) -> list[BranchSpec]:
    """Cross the variant grid with the seed range into a flat branch list."""
    n_init_sims = int(config.get("n_init_sims", 1))
    lineage_seed = int(config.get("lineage_seed", 0))
    skip_baseline = bool(config.get("skip_baseline", False))
    different_seeds = bool(config.get("different_seeds_per_variant", False))

    variants_block = config.get("variants") or {}

    # Build the ordered list of (variant_name, overrides) entries.
    variant_entries: list[tuple[str, dict[str, Any]]] = []
    if not skip_baseline:
        variant_entries.append(("baseline", {}))
    # A variant block is either single-param shorthand (the block itself
    # carries "target", e.g. {"target": "p.k", "value": [...]}) or the
    # multi-param form (named sub-params + an "op" key). Dispatch on the
    # presence of a top-level "target".
    for vname, vconf in variants_block.items():
        for overrides in parse_variant_params({vname: vconf} if "target" in vconf
                                              else dict(vconf)):
            variant_entries.append((vname, overrides))

    branches: list[BranchSpec] = []
    for v_idx, (vname, overrides) in enumerate(variant_entries):
        if different_seeds:
            base = lineage_seed + v_idx * n_init_sims
        else:
            base = lineage_seed
        for s in range(n_init_sims):
            branches.append(BranchSpec(
                variant_index=v_idx,
                variant_name=vname,
                overrides=dict(overrides),
                seed=base + s,
                metadata={"variant_name": vname, **{f"override:{k}": v
                                                    for k, v in overrides.items()}},
            ))
    return branches
