"""Validate status-field consistency across investigation.yaml + study yamls.

Read-only. Checks:

  1. Investigation declares exactly one `focus_study`, and it appears in
     `studies[]`.
  2. `study_lifecycle` has an entry for every study, no extras, and every
     value is from the allowed enum (not-started, in-progress,
     awaiting-expert-review, approved, blocked-by-prerequisite).
  3. focus_study lifecycle ∈ {in-progress, awaiting-expert-review} —
     the focus_study is what we're actively working on.
  4. At most one study is `in-progress` (the focus_study).
  5. All studies positioned before focus_study in `studies[]` are
     `approved` (linear-chain assumption; the existing investigation
     yamls follow this).
  6. Per-study top-level `status:` is broadly consistent with the
     orchestration lifecycle. Soft check: warns if a `blocked-by-
     prerequisite` study has a per-study status of `passing` or similar
     terminal-good value (suggests stale: prior results recorded, but
     orchestration says "don't re-run because upstream not approved").
  7. Each study yaml passes the dashboard's strict `load_spec` parse.
     Surfaces schema-validation errors (e.g. invalid enum values in
     model_settings.gate) that the dashboard's detail endpoint would
     otherwise silently map to `status: invalid` → aggregate `failed`
     badge. Hard error — these break the dashboard render.

Exit code: 0 on clean, 1 on errors, 2 on warnings only.

Usage:
    python scripts/validate_status.py                  # all investigations
    python scripts/validate_status.py --investigation dnaa-replication
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ALLOWED_LIFECYCLE = {
    "not-started",
    "in-progress",
    "awaiting-expert-review",
    "approved",
    "blocked-by-prerequisite",
}

FOCUS_LIFECYCLE = {"in-progress", "awaiting-expert-review"}


def _load(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise SystemExit(f"yaml parse error in {p}: {e}")


def validate_investigation(inv_yaml: Path, ws_root: Path) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings) for this investigation."""
    errors: list[str] = []
    warnings: list[str] = []
    spec = _load(inv_yaml)
    inv_name = spec.get("name") or inv_yaml.parent.name

    studies = spec.get("studies") or []
    if not isinstance(studies, list):
        errors.append(f"{inv_name}: `studies` must be a list")
        return errors, warnings

    # 1. focus_study
    focus = spec.get("focus_study")
    if focus is None:
        warnings.append(
            f"{inv_name}: no `focus_study` declared — single-study cadence "
            "(haochen-one-study-at-a-time) not yet adopted")
    elif focus not in studies:
        errors.append(
            f"{inv_name}: focus_study={focus!r} not in studies list {studies}")

    # 2. study_lifecycle keys + values
    lifecycle = spec.get("study_lifecycle") or {}
    if not isinstance(lifecycle, dict):
        errors.append(f"{inv_name}: `study_lifecycle` must be a dict")
        return errors, warnings
    if focus is not None and lifecycle:
        missing = set(studies) - set(lifecycle.keys())
        extras = set(lifecycle.keys()) - set(studies)
        if missing:
            errors.append(
                f"{inv_name}: study_lifecycle missing entries for: "
                f"{sorted(missing)}")
        if extras:
            errors.append(
                f"{inv_name}: study_lifecycle has entries for unknown "
                f"studies: {sorted(extras)}")
        for slug, lc in lifecycle.items():
            if lc not in ALLOWED_LIFECYCLE:
                errors.append(
                    f"{inv_name}: study_lifecycle[{slug}]={lc!r} not in "
                    f"allowed set {sorted(ALLOWED_LIFECYCLE)}")

        # 3. focus_study lifecycle
        focus_lc = lifecycle.get(focus)
        if focus_lc and focus_lc not in FOCUS_LIFECYCLE:
            errors.append(
                f"{inv_name}: focus_study={focus} has lifecycle={focus_lc!r}; "
                f"expected one of {sorted(FOCUS_LIFECYCLE)}")

        # 4. at most one in-progress
        in_progress = [s for s, lc in lifecycle.items() if lc == "in-progress"]
        if len(in_progress) > 1:
            errors.append(
                f"{inv_name}: more than one study `in-progress`: "
                f"{in_progress} (only focus_study should be active)")

        # 5. studies before focus_study must be approved
        if focus in studies:
            focus_idx = studies.index(focus)
            unapproved_upstream = [
                s for s in studies[:focus_idx]
                if lifecycle.get(s) not in {"approved", None}
            ]
            if unapproved_upstream:
                warnings.append(
                    f"{inv_name}: studies upstream of focus_study={focus} that "
                    f"are NOT approved: {unapproved_upstream} (linear-chain "
                    "convention — verify each is intentionally not yet approved)")

    # 6. per-study status consistency (soft) + 7. strict load_spec parse (hard)
    # Try to import the dashboard's strict parser; fall back gracefully so
    # this script still runs without the dashboard install (just skips rule 7).
    try:
        import importlib
        _inv_mod = importlib.import_module("vivarium_dashboard.lib.investigations")
        _load_spec = _inv_mod.load_spec
        _InvErr = _inv_mod.InvestigationSpecError
    except Exception:  # noqa: BLE001
        _load_spec = None
        _InvErr = None

    for slug in studies:
        sp = ws_root / "workspace" / "studies" / slug / "study.yaml"
        if not sp.is_file():
            warnings.append(f"{inv_name}: study yaml missing: {sp}")
            continue
        sd = _load(sp)
        # 7. strict-parse: matches what _get_iset_detail / _build_iset_detail_for_test
        # call. If it raises, the dashboard tags status='invalid' which then
        # rolls up to a "failed" aggregate badge. Hard error.
        if _load_spec is not None:
            try:
                _load_spec(sp)
            except _InvErr as e:  # type: ignore[misc]
                errors.append(
                    f"{inv_name}/{slug}: study.yaml fails dashboard's strict "
                    f"load_spec parse — dashboard will tag this study "
                    f"status='invalid' and the investigation badge will go "
                    f"red. Error: {e}")
        per_study_status = sd.get("status") or ""
        lc = lifecycle.get(slug, "")
        # Heuristic: a study that's blocked-by-prerequisite shouldn't have
        # a per-study status that looks "done" / "passing" without a
        # superseded marker.
        if lc == "blocked-by-prerequisite":
            terminal_good = any(
                kw in per_study_status.lower()
                for kw in ("passing", "ran-", "done-", "confirmed", "in-band")
            )
            superseded = "supersed" in per_study_status.lower() or \
                         "blocked" in per_study_status.lower()
            if terminal_good and not superseded:
                warnings.append(
                    f"{inv_name}/{slug}: lifecycle=blocked-by-prerequisite "
                    f"but per-study status={per_study_status!r} suggests "
                    "prior terminal-good state without superseded marker; "
                    "consider updating status to flag the orchestration block")

    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--investigation", default=None,
        help="single investigation slug to validate (default: all)")
    args = ap.parse_args()

    ws_root = Path(__file__).resolve().parent.parent
    inv_dir = ws_root / "workspace" / "investigations"
    if not inv_dir.is_dir():
        raise SystemExit(f"no investigations dir at {inv_dir}")

    if args.investigation:
        targets = [inv_dir / args.investigation / "investigation.yaml"]
    else:
        targets = sorted(inv_dir.glob("*/investigation.yaml"))

    total_errors: list[str] = []
    total_warnings: list[str] = []
    for inv_yaml in targets:
        if not inv_yaml.is_file():
            print(f"  SKIP  {inv_yaml.parent.name}: investigation.yaml missing")
            continue
        errors, warnings = validate_investigation(inv_yaml, ws_root)
        total_errors.extend(errors)
        total_warnings.extend(warnings)
        status = "OK   " if not errors and not warnings else (
            "ERR  " if errors else "WARN "
        )
        print(f"  {status} {inv_yaml.parent.name}  "
              f"({len(errors)} error(s), {len(warnings)} warning(s))")

    if total_errors:
        print("\nErrors:")
        for e in total_errors:
            print(f"  ✗ {e}")
    if total_warnings:
        print("\nWarnings:")
        for w in total_warnings:
            print(f"  ⚠ {w}")

    if total_errors:
        return 1
    if total_warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
