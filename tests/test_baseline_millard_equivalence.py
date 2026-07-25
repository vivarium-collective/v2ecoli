"""Structural-equivalence test for the merged baseline_millard(lqr=…) generator.

Proves — on the built composite DOCUMENT, without running a simulation — that:

  * ``baseline_millard(lqr=False)`` (default) is byte-identical in structure to
    the legacy ``baseline_millard`` composite, and
  * ``baseline_millard(lqr=True)`` reproduces the legacy ``millard-pdmp`` LQR
    composite that was merged into it.

The golden fingerprints in ``tests/fixtures/`` were captured from the pre-merge
code (commit 5d95feaf). A structural fingerprint = per-agent sorted edge keys +
each edge's address + input/output wire dicts (paths) + the doc's flow_order.
Live instances, RNG state, and config values are excluded.
"""
import json
import os
import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _wire(d):
    out = {}
    for k in sorted(d or {}):
        v = d[k]
        out[k] = list(v) if isinstance(v, (list, tuple)) else v
    return out


def _doc_fingerprint(doc):
    agents = doc["state"]["agents"]
    fp = {"flow_order": list(doc.get("flow_order", [])), "agents": {}}
    for aid in sorted(agents):
        cell = agents[aid]
        edges = {}
        for key, val in cell.items():
            if not (isinstance(val, dict) and "address" in val
                    and ("inputs" in val or "outputs" in val)):
                continue
            edges[key] = {
                "address": val.get("address"),
                "inputs": _wire(val.get("inputs")),
                "outputs": _wire(val.get("outputs")),
            }
        fp["agents"][aid] = {"edge_keys": sorted(edges), "edges": edges}
    return fp


def _build_doc(overrides):
    import v2ecoli  # noqa: F401 — forces generator registration
    from pbg_superpowers.composite_generator import _REGISTRY, build_generator
    from v2ecoli.core import build_core
    matches = [e for e in _REGISTRY.values() if e.name == "baseline_millard"]
    entry = min(matches, key=lambda e: len(e.id))
    return build_generator(entry, overrides=overrides, core=build_core())


def _requires_cache():
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; build via scripts/build_cache.py")


@pytest.mark.sim
def test_baseline_millard_lqr_false_matches_legacy_plain():
    _requires_cache()
    golden = json.loads((FIXTURES / "baseline_millard_plain.json").read_text())
    got = _doc_fingerprint(_build_doc({"seed": 0}))
    assert got == golden, "lqr=False document diverged from legacy baseline_millard"


@pytest.mark.sim
def test_baseline_millard_lqr_true_matches_legacy_pdmp():
    _requires_cache()
    golden = json.loads((FIXTURES / "baseline_millard_lqr.json").read_text())
    got = _doc_fingerprint(_build_doc({"seed": 0, "lqr": True}))
    assert got == golden, "lqr=True document diverged from legacy millard-pdmp LQR composite"
