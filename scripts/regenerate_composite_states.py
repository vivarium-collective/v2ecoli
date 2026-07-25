#!/usr/bin/env python3
"""Pre-resolve EVERY registered composite to a static read-only state JSON.

The published (gh-pages) dashboard serves composite state as static files at
``api/composite-state/<id>.json`` so the bigraph-loom ``?static=1`` viewer works
offline. ``vivarium_workbench.publish.build_bundle`` will use a committed
override at ``reports/composite-state/<id>.json`` verbatim (marking the composite
navigable) when present, and otherwise falls back to LIVE resolution — which
fails in CI for any composite whose generator needs the on-disk ParCa cache
(baseline and all its derivatives). The result: only the lightweight Millard
composites export, every heavy one 404s the Explore pop-out.

This script closes that gap. Run it locally / on a host WITH the ParCa cache
(``out/cache`` populated) so every composite resolves, then commit the produced
``reports/composite-state/*.json`` (force-add — ``reports/`` is gitignored).

Usage:
    .venv/bin/python scripts/regenerate_composite_states.py            # write all
    .venv/bin/python scripts/regenerate_composite_states.py --dry-run  # list only
    .venv/bin/python scripts/regenerate_composite_states.py --only baseline biological
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _study_composite_refs(ws_root: Path) -> set[str]:
    """Every distinct `composite:` ref across the workspace's study.yaml files."""
    import re
    refs: set[str] = set()
    pat = re.compile(r"composite:\s*([A-Za-z0-9_.\-]+)")
    for sy in ws_root.glob("workspace/investigations/*/studies/*/study.yaml"):
        try:
            refs.update(pat.findall(sy.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    return refs


def _live_state(build_composite_state, ws_root: Path, cid: str) -> "dict | None":
    """Build ``cid``'s document for real and return its store dict, or None.

    ``build_composite_state`` runs the ``@composite_generator`` in a subprocess
    (its own main thread) and returns the document envelope; the artifact format
    wants the bare store mapping (what ``CompositeSpec.default_state`` yields),
    so unwrap a single ``{"state": {...}}`` envelope layer.
    """
    try:
        body, status = build_composite_state(ws_root, cid, fresh=True)
    except Exception:  # noqa: BLE001 — a build failure is reported by the caller
        return None
    if status != 200:
        return None
    doc = body.get("state")
    if isinstance(doc, dict) and isinstance(doc.get("state"), dict):
        doc = doc["state"]
    return doc if isinstance(doc, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + report size, but do not write files")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these composite ids (full or short slug)")
    ap.add_argument("--no-trim", action="store_true",
                    help="keep full bulk arrays (default trims them, 5.5MB -> ~few hundred KB)")
    args = ap.parse_args()

    ws_root = Path.cwd()
    if not (ws_root / "workspace.yaml").is_file():
        print(f"ERROR: not a workspace (no workspace.yaml): {ws_root}", file=sys.stderr)
        return 1

    ws_str = str(ws_root)
    if ws_str not in sys.path:
        sys.path.insert(0, ws_str)

    from vivarium_workbench.lib.json_serialize import _json_default, _json_sanitize
    from vivarium_workbench.lib._root import set_workspace_root
    from vivarium_workbench.lib.composite_lookup import composites_data as _composites_data
    from vivarium_workbench.lib.composite_resolve import resolve_composite
    from vivarium_workbench.lib.composite_state_views import build_composite_state
    set_workspace_root(ws_root)

    # Reuse the viewers-hub trimmer: caps the long molecule-data arrays (bulk
    # counts ~16k, unique-molecule instance lists) that loom/viz never render,
    # shrinking each heavy state ~5.5MB -> a few hundred KB WITHOUT touching the
    # bigraph structure (stores/processes/wiring/schemas/describe docs).
    sys.path.insert(0, str(ws_root / "scripts"))
    from regenerate_viewers import trim_state_for_view

    composites = _composites_data(ws_root)
    comps = composites.get("composites") or []
    if not comps:
        print(f"ERROR: no composites discovered ({composites.get('error')})", file=sys.stderr)
        return 1

    registry_ids = [c.get("id") for c in comps if c.get("id")]
    # Also emit a state file under every DISTINCT composite ref that studies
    # actually pass to the loom pop-out. Discovery canonicalizes some ids (e.g.
    # `...baseline.baseline` -> `...baseline`), but the study pop-out URL is built
    # from the raw study.yaml ref, so the static file must exist under THAT exact
    # string too. Resolving the union covers both forms; refs whose package isn't
    # installed here (e.g. v2ecoli_pdmp.*, diagnostic) fail loudly and are listed.
    study_refs = _study_composite_refs(ws_root)
    ids = sorted(set(registry_ids) | study_refs)
    if args.only:
        want = set(args.only)
        ids = [cid for cid in ids
               if cid in want or cid.split(".")[-1] in want or cid in {f"*{o}" for o in want}
               or any(cid.endswith("." + o) or cid.split(".")[-1] == o for o in want)]

    out_dir = ws_root / "reports" / "composite-state"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(ids)} composites to resolve (dry-run={args.dry_run})\n")
    ok, failed = [], []
    for cid in sorted(ids):
        try:
            data = resolve_composite(ws_root, cid)
        except Exception as e:  # noqa: BLE001
            data = None
            err = repr(e)
        else:
            err = "resolve returned None"
        if data is None:
            failed.append((cid, err))
            print(f"  FAIL  {cid}  ({err})")
            continue
        # REGENERATE means rebuild — always take a fresh live build (the same
        # subprocess seam the live /api/composite-state uses) and only fall back
        # to whatever `resolve_composite` returned if that build fails.
        # `resolve_composite` reads the COMMITTED artifact for any generator
        # (none declares `default_state_ref`), so trusting its state here would
        # make this script echo the file it is supposed to be replacing: a
        # composite whose parameters changed would keep its stale wiring, and a
        # brand-new one would get a `state: null` degraded payload written out
        # under an OK line.
        built = _live_state(build_composite_state, ws_root, cid)
        if isinstance(built, dict):
            data["state"] = built
            data["wiring_status"] = "ready"
            data["notice"] = None
        elif isinstance(data.get("state"), dict):
            print(f"  note  {cid}: live build unavailable — keeping committed state")
        # NEVER write a stateless payload: the Composite Explorer's fallback
        # reads this file, so a committed `state: null` freezes "not generated
        # yet" into git and hides the real failure behind a plausible artifact.
        if not isinstance(data.get("state"), dict):
            reason = data.get("notice") or "no state (live build failed)"
            failed.append((cid, reason))
            print(f"  FAIL  {cid}  ({reason})")
            if (out_dir / f"{cid}.json").is_file():
                print(f"        note: stale artifact on disk — delete it or fix the build")
            continue
        if not args.no_trim and isinstance(data, dict) and isinstance(data.get("state"), dict):
            data["state"] = trim_state_for_view(data["state"])
        try:
            # Sanitize FIRST (inf/-inf/nan -> null), then serialize with the
            # publish.py writer (_json_default coerces ndarray/etc.). Heavy
            # v2ecoli composites carry non-finite defaults (unbounded maxima);
            # publish hides them rather than ship null-patched state, but the
            # whole point here is to make EVERY composite navigable read-only,
            # so the legitimate path (per _write_json's docstring) is to
            # sanitize. allow_nan=False then stays a hard correctness gate.
            blob = json.dumps(_json_sanitize(data), default=_json_default, allow_nan=False)
        except ValueError as e:  # any non-finite that slipped past sanitize
            failed.append((cid, f"non-finite floats: {e}"))
            print(f"  FAIL  {cid}  (non-finite floats)")
            continue
        size_mb = len(blob) / 1e6
        ok.append((cid, size_mb))
        if not args.dry_run:
            (out_dir / f"{cid}.json").write_text(blob, encoding="utf-8")
        print(f"  OK    {cid}  ({size_mb:.1f} MB)")
        # Discovery canonicalizes a generator's id (`<module>.<name>` ->
        # `<module>` when the module's stem already is the name), but the
        # workspace manifest still advertises BOTH forms and pop-out URLs are
        # built from whichever string the caller holds. Emit the alias copy so
        # the un-canonicalized id doesn't render "not generated yet".
        alias = f"{data.get('module')}.{data.get('name')}" if data.get("module") else None
        if alias and alias != cid and alias not in ids and not args.dry_run:
            (out_dir / f"{alias}.json").write_text(blob, encoding="utf-8")
            print(f"  OK    {alias}  (alias copy)")

    total = sum(s for _, s in ok)
    print(f"\nresolved {len(ok)}/{len(ids)}  ({total:.1f} MB total)  failed {len(failed)}")
    if failed:
        print("FAILED:")
        for cid, err in failed:
            print(f"  {cid}: {err}")
    if not args.dry_run:
        print(f"\nwrote to {out_dir} (force-add: git add -f reports/composite-state/*.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
