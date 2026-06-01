#!/usr/bin/env python3
"""Cross-study consistency validator.

Walks every ``workspace/studies/*/study.yaml`` and:

  1. Validates enum-typed fields against the registered bigraph-schema
     enums in ``v2ecoli/types/biology.py`` (``BIOLOGY_TYPES``):
       target_class, narrative_confidence, failure_cause,
       perturbation_kind, verdict_result (the ``result`` field inside
       ``conclusion_verdicts`` / ``last_results``).

     Unknown enum types are skipped gracefully — if a workspace
     extends ``BIOLOGY_TYPES`` with domain-specific enums (e.g. a
     ``dnaa_pool_kind`` or similar), referencing the new type name in
     the maps below will pick it up automatically.

  2. Cross-study consistency:
       - Same ``literature_observable.name`` must map to the same
         ``biological_pool`` across studies (ERROR on divergence).
       - Same ``model_observable.state_path`` must map to the same
         ``molecular_pool`` across studies (ERROR on divergence).
       - Same ``literature_observable.name`` must cite an overlapping
         ``source_ids`` set across studies (WARNING on totally disjoint
         citations).

Exit code 0 if clean, 1 if any errors. ``--strict`` promotes warnings
to errors.

Standalone usage:
    python scripts/validate_study_biology.py
    python scripts/validate_study_biology.py --json     # machine-readable
    python scripts/validate_study_biology.py --strict   # warnings → errors
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

WS_ROOT = Path(__file__).resolve().parents[1]

# Load v2ecoli/types/biology.py directly without triggering the v2ecoli
# package __init__ (which pulls in heavy simulator deps like `multi_cell`).
# biology.py has zero runtime imports — safe to side-load.
_bio_spec = importlib.util.spec_from_file_location(
    "v2ecoli_types_biology",
    WS_ROOT / "v2ecoli" / "types" / "biology.py",
)
_bio_mod = importlib.util.module_from_spec(_bio_spec)
_bio_spec.loader.exec_module(_bio_mod)
BIOLOGY_TYPES = _bio_mod.BIOLOGY_TYPES


# Map study.yaml YAML field-names to their bigraph-schema enum type name.
# A field-path key is the LAST element of the breadcrumb (the leaf YAML key).
# Enum types not registered in BIOLOGY_TYPES are silently skipped, so
# workspaces that extend BIOLOGY_TYPES with new pool-kind / region-type
# enums can register them here without modifying the validator logic.
SCALAR_ENUM_FIELDS: dict[str, str] = {
    "target_class":         "target_class",
    "narrative_confidence": "narrative_confidence",
    # Workspace-specific pool-kind enums (skipped when not registered):
    "biological_pool":      "dnaa_pool_kind",
    "molecular_pool":       "dnaa_pool_kind",
}

# Enum fields where the parent breadcrumb matters to disambiguate. Each rule
# specifies the parent container shape:
#   shape='list_item' — leaf is a direct field on an item of a list named
#                       `parent`. Path = [..., parent, [i], leaf].
#   shape='dict_value'— leaf is a direct field on a dict value of a dict named
#                       `parent` keyed by arbitrary names.
#                       Path = [..., parent, <key>, leaf].
# Strict shape matching prevents over-matching deeper occurrences of the
# same leaf name (e.g. `measure.kind` nested two levels under a parent
# named ``perturbation_panel`` is NOT checked against perturbation_kind).
CONTEXTUAL_ENUM_FIELDS: list[tuple[str, str, str, str]] = [
    # parent_name,                leaf_name, enum_type_name,   shape
    ("conclusion_verdicts",       "result",  "verdict_result", "dict_value"),
    ("perturbation_panel",        "kind",    "perturbation_kind", "list_item"),
    # tests.last_results.<test_name> sub-blocks like {result: PASS, observed: 707}
    ("last_results",              "result",  "verdict_result", "dict_value"),
]

# Fields whose values are LISTS of enum members. Value ``None`` means
# the list is free-text and should be skipped (declared here so the
# walker doesn't try to enum-check it).
LIST_ENUM_FIELDS: dict[str, str | None] = {
    "failure_cause_candidates": "failure_cause",
    "requires_perturbations":   None,  # free-text perturbation names
    "forbidden_proxies":        None,  # free-text path strings
}


def _enum_values(type_name: str) -> set[str] | None:
    spec = BIOLOGY_TYPES.get(type_name)
    if spec is None:
        return None
    return set(spec.get("_values", []))


def _walk(node: Any, crumbs: list[str]) -> Iterable[tuple[list[str], str, Any]]:
    """Yield (breadcrumbs, leaf_name, value) for every scalar leaf.

    For list items, the breadcrumb element is f'{parent}[{i}]'.
    For dict items, the breadcrumb element is the key.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            new_crumbs = crumbs + [str(k)]
            if isinstance(v, (dict, list)):
                yield from _walk(v, new_crumbs)
            else:
                yield (new_crumbs, str(k), v)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            new_crumbs = crumbs + [f"[{i}]"]
            if isinstance(item, (dict, list)):
                yield from _walk(item, new_crumbs)
            else:
                # list of scalars — emit with parent leaf name (last non-index crumb)
                parent_name = None
                for c in reversed(crumbs):
                    if not c.startswith("["):
                        parent_name = c
                        break
                yield (new_crumbs, parent_name or "", item)


def _validate_enums(
    study_name: str,
    study_data: dict,
    errors: list[str],
    warnings: list[str],
) -> None:
    for crumbs, leaf, value in _walk(study_data, []):
        if value is None:
            continue
        path_str = ".".join(crumbs)

        # List-of-enum fields: handled via the parent crumb.
        # Detect: the second-to-last crumb is in LIST_ENUM_FIELDS and the
        # last crumb is an index marker.
        if len(crumbs) >= 2 and crumbs[-1].startswith("["):
            list_field = crumbs[-2]
            if list_field in LIST_ENUM_FIELDS:
                enum_type = LIST_ENUM_FIELDS[list_field]
                if enum_type is None:
                    continue  # explicitly skipped
                vals = _enum_values(enum_type)
                if vals is not None and str(value) not in vals:
                    errors.append(
                        f"{study_name}: {path_str} = {value!r} not in "
                        f"{enum_type} enum {sorted(vals)}"
                    )
                continue

        # Contextual enum fields. Match strict path shapes only.
        matched = False
        for parent_name, leaf_name, enum_type, shape in CONTEXTUAL_ENUM_FIELDS:
            if leaf != leaf_name:
                continue
            ok = False
            if shape == "list_item":
                # Path tail must be [..., parent_name, [i], leaf]
                if (len(crumbs) >= 3
                        and crumbs[-3] == parent_name
                        and crumbs[-2].startswith("[")):
                    ok = True
            elif shape == "dict_value":
                # Path tail must be [..., parent_name, <key>, leaf]
                # where <key> is not a list index.
                if (len(crumbs) >= 3
                        and crumbs[-3] == parent_name
                        and not crumbs[-2].startswith("[")):
                    ok = True
            if not ok:
                continue
            vals = _enum_values(enum_type)
            if vals is not None and str(value) not in vals:
                errors.append(
                    f"{study_name}: {path_str} = {value!r} not in "
                    f"{enum_type} enum {sorted(vals)}"
                )
            matched = True
            break
        if matched:
            continue
        # Scalar enum fields by leaf name (no parent disambiguation needed).
        if leaf in SCALAR_ENUM_FIELDS:
                enum_type = SCALAR_ENUM_FIELDS[leaf]
                vals = _enum_values(enum_type)
                if vals is not None and str(value) not in vals:
                    errors.append(
                        f"{study_name}: {path_str} = {value!r} not in "
                        f"{enum_type} enum {sorted(vals)}"
                    )


def _collect_observable_mappings(
    study_name: str,
    study_data: dict,
    lit_to_pool: dict[str, list[tuple[str, str, str]]],   # lit_name → [(study, pool, path)]
    state_to_pool: dict[str, list[tuple[str, str, str]]], # state_path → [(study, pool, path)]
    lit_to_sources: dict[str, list[tuple[str, set, str]]],# lit_name → [(study, source_set, path)]
) -> None:
    """Walk behavior_tests[*].measurement_mapping and harvest tuples."""
    for crumbs, leaf, value in _walk(study_data, []):
        pass  # We need a different traversal that surfaces the parent dict;
              # do that explicitly below.

    def _walk_dicts(node: Any, crumbs: list[str]) -> Iterable[tuple[list[str], dict]]:
        if isinstance(node, dict):
            yield (crumbs, node)
            for k, v in node.items():
                yield from _walk_dicts(v, crumbs + [str(k)])
        elif isinstance(node, list):
            for i, item in enumerate(node):
                yield from _walk_dicts(item, crumbs + [f"[{i}]"])

    for crumbs, d in _walk_dicts(study_data, []):
        # measurement_mapping block
        if crumbs and crumbs[-1] == "measurement_mapping":
            lit = d.get("literature_observable") or {}
            mdl = d.get("model_observable") or {}
            path = ".".join(crumbs)
            lit_name = lit.get("name")
            bio_pool = lit.get("biological_pool")
            if lit_name and bio_pool:
                lit_to_pool[str(lit_name)].append((study_name, str(bio_pool), path))
            sources = lit.get("source_ids")
            if lit_name and isinstance(sources, list) and sources:
                lit_to_sources[str(lit_name)].append((study_name, set(sources), path))
            state_path = mdl.get("state_path")
            mol_pool = mdl.get("molecular_pool")
            if state_path and mol_pool:
                state_to_pool[str(state_path)].append((study_name, str(mol_pool), path))


def _cross_study_consistency(
    lit_to_pool: dict[str, list[tuple[str, str, str]]],
    state_to_pool: dict[str, list[tuple[str, str, str]]],
    lit_to_sources: dict[str, list[tuple[str, set, str]]],
    errors: list[str],
    warnings: list[str],
) -> None:
    for lit_name, entries in lit_to_pool.items():
        pools = {pool for (_s, pool, _p) in entries}
        if len(pools) > 1:
            studies_pools = ", ".join(
                f"{s}={pool}" for (s, pool, _p) in entries
            )
            errors.append(
                f"cross-study: literature_observable '{lit_name}' mapped "
                f"to DIVERGENT biological_pool values across studies: "
                f"{studies_pools}"
            )

    for state_path, entries in state_to_pool.items():
        pools = {pool for (_s, pool, _p) in entries}
        if len(pools) > 1:
            studies_pools = ", ".join(
                f"{s}={pool}" for (s, pool, _p) in entries
            )
            errors.append(
                f"cross-study: model_observable state_path '{state_path}' "
                f"mapped to DIVERGENT molecular_pool values: {studies_pools}"
            )

    for lit_name, entries in lit_to_sources.items():
        if len(entries) < 2:
            continue
        # If two studies cite this observable with NO overlap, warn.
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                s_i, sources_i, _ = entries[i]
                s_j, sources_j, _ = entries[j]
                if not (sources_i & sources_j):
                    warnings.append(
                        f"cross-study: literature_observable '{lit_name}' "
                        f"is cited with disjoint source_ids in {s_i} "
                        f"({sorted(sources_i)}) vs {s_j} "
                        f"({sorted(sources_j)}) — verify they refer to "
                        f"the same biological quantity"
                    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="emit a machine-readable JSON summary")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as errors (non-zero exit)")
    args = ap.parse_args(argv)

    studies_dir = WS_ROOT / "workspace" / "studies"
    study_files = sorted(studies_dir.glob("*/study.yaml")) if studies_dir.exists() else []
    if not study_files:
        # No studies yet — no enums to check. Clean exit so the validator
        # is safe to invoke from lint-workspace.py on freshly-scaffolded
        # workspaces.
        if args.json:
            print(json.dumps({
                "studies_checked": 0, "errors": [], "warnings": [],
            }, indent=2))
        else:
            print("validate_study_biology: no studies/ found — nothing to check")
        return 0

    errors: list[str] = []
    warnings: list[str] = []
    lit_to_pool: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    state_to_pool: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    lit_to_sources: dict[str, list[tuple[str, set, str]]] = defaultdict(list)

    n_studies = 0
    per_study_field_counts: dict[str, int] = defaultdict(int)

    for path in study_files:
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:
            errors.append(f"{path.parent.name}: YAML parse failed — {exc}")
            continue
        study_name = data.get("name", path.parent.name)
        n_studies += 1

        _validate_enums(study_name, data, errors, warnings)
        _collect_observable_mappings(
            study_name, data, lit_to_pool, state_to_pool, lit_to_sources,
        )

        # Count enum-tagged fields for the summary.
        for crumbs, leaf, value in _walk(data, []):
            if leaf in SCALAR_ENUM_FIELDS:
                per_study_field_counts[leaf] += 1

    _cross_study_consistency(
        lit_to_pool, state_to_pool, lit_to_sources, errors, warnings,
    )

    if args.json:
        out = {
            "studies_checked": n_studies,
            "errors": errors,
            "warnings": warnings,
            "field_counts": dict(per_study_field_counts),
            "literature_observables_with_pool": len(lit_to_pool),
            "model_observables_with_pool": len(state_to_pool),
        }
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"validate_study_biology: scanned {n_studies} studies")
        print(f"  enum-tagged fields: " + ", ".join(
            f"{k}={v}" for k, v in sorted(per_study_field_counts.items())
        ) if per_study_field_counts else
            "  (no enum-tagged fields found)")
        print(f"  cross-study: {len(lit_to_pool)} literature observables "
              f"with biological_pool, {len(state_to_pool)} model observables "
              f"with molecular_pool")
        for w in warnings:
            print(f"  WARN: {w}", file=sys.stderr)
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        if not errors and not warnings:
            print("  OK")

    if errors:
        return 1
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
