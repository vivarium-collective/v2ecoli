"""Coverage guard: every registered v2ecoli generator must ship a committed
default-state artifact at ``reports/composite-state/<id>.json``.

The Composite Explorer renders a generator's wiring from that committed artifact
(the dashboard resolver falls back to it — see vivarium-workbench
``composite_resolve._committed_default_state``). A generator with no committed
artifact shows "default state for generator '<x>' is not generated yet" and a
blank Explorer. This test fails loudly the moment a new generator lands without
its artifact, so the gap can never ship silently.

Regenerate a missing artifact with ``scripts/regenerate_composite_states.py`` on
a host with the ParCa cache (``out/cache`` populated) and commit the produced
``reports/composite-state/<id>.json`` (force-add — ``reports/`` is gitignored).
"""
import json
from pathlib import Path

import v2ecoli.composites  # noqa: F401 — import registers every @composite_generator spec
from process_bigraph.composite_spec import all_specs

REPO = Path(__file__).resolve().parent.parent
CSTATE = REPO / "reports" / "composite-state"


def _candidate_artifact_ids(spec_id: str) -> set:
    """Ids under which a generator's committed artifact may live: its own id, plus
    the clean-alias form (``<module>.<name>.<name>`` -> ``<module>.<name>``) that
    the dashboard's dedupe collapses to."""
    ids = {spec_id}
    parts = spec_id.split(".")
    if len(parts) >= 2 and parts[-1] == parts[-2]:
        ids.add(".".join(parts[:-1]))
    return ids


def _has_state(path: Path) -> bool:
    """True when the artifact carries a usable ``state`` mapping.

    Existence alone is NOT enough: ``resolve_composite`` returns a 200 payload
    with ``state: null`` + ``wiring_status: "unavailable"`` when it can't produce
    the wiring, and the regen script used to serialize that degraded payload
    verbatim — a plausible-looking file that renders as "not generated yet"
    forever (the resolver's fallback reads this same file, so the failure is
    self-sustaining). Check the content, not the filename.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — an unreadable artifact is a missing one
        return False
    return isinstance(data, dict) and isinstance(data.get("state"), dict) and bool(data["state"])


def test_every_v2ecoli_generator_has_committed_default_state():
    missing = []
    for spec in all_specs().values():
        if getattr(spec, "kind", None) != "generator":
            continue
        sid = getattr(spec, "id", "") or ""
        if not sid.startswith("v2ecoli."):
            continue
        if not any(_has_state(CSTATE / f"{i}.json")
                   for i in _candidate_artifact_ids(sid)):
            missing.append(sid)
    assert not missing, (
        "v2ecoli generators with no usable reports/composite-state/<id>.json — "
        "the Composite Explorer would show 'not generated yet' for these: "
        f"{sorted(set(missing))}. Regenerate via "
        "scripts/regenerate_composite_states.py (ParCa cache required) and commit.")


def test_no_committed_artifact_carries_a_null_state():
    """No committed artifact may be a serialized *failure*.

    Complements the coverage test above: that one asks "does every generator
    have wiring?", this one asks "is every file on disk actually wiring?" — it
    catches stateless artifacts for ids that are no longer registered here
    (other packages' composites, study-ref alias copies) too.
    """
    if not CSTATE.is_dir():
        return
    stateless = sorted(p.name for p in CSTATE.glob("*.json") if not _has_state(p))
    assert not stateless, (
        "committed composite-state artifacts with no usable `state` (these render "
        f"as 'not generated yet' in the Composite Explorer): {stateless}. "
        "Regenerate them via scripts/regenerate_composite_states.py, or delete "
        "the artifact so the miss is visible instead of frozen into git.")
