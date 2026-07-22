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


def test_every_v2ecoli_generator_has_committed_default_state():
    missing = []
    for spec in all_specs().values():
        if getattr(spec, "kind", None) != "generator":
            continue
        sid = getattr(spec, "id", "") or ""
        if not sid.startswith("v2ecoli."):
            continue
        if not any((CSTATE / f"{i}.json").is_file()
                   for i in _candidate_artifact_ids(sid)):
            missing.append(sid)
    assert not missing, (
        "v2ecoli generators with no committed reports/composite-state/<id>.json — "
        "the Composite Explorer would show 'not generated yet' for these: "
        f"{sorted(set(missing))}. Regenerate via "
        "scripts/regenerate_composite_states.py (ParCa cache required) and commit.")
