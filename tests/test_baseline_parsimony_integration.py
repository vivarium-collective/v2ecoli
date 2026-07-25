"""Integration tests for baseline_parsimony: a real composite run crossing a
declared snapshot time and writing a pack under studies/<study>/viz/3d/.

Two tiers:

- Fast/hermetic (default-run): monkeypatches v2ecoli.structural.pack_step's
  pack_from_state with a stub that writes a small sentinel file — proves the
  composite-run -> step-fires-at-declared-time -> writes-to-out_dir pipeline
  WITHOUT the network/Rust real packer. Building + running baseline still
  needs the ParCa cache, so this is gated the same way
  test_baseline_parsimony_composite.py's @pytest.mark.sim test is.
- Slow/opt-in (@pytest.mark.slow): the real end-to-end using pbg_parsimony's
  real build_pack (network AlphaFold fetch + Rust packer), requiring
  PARSIMONY_HOME. Skipped unless explicitly requested via -m slow.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "out" / "cache"

_needs_cache = pytest.mark.skipif(
    not CACHE_DIR.is_dir() and not os.environ.get("CI"),
    reason="cache dir 'out/cache' not present; "
           "build via `python scripts/build_cache.py` (CI builds it automatically)",
)


@pytest.mark.sim
@_needs_cache
def test_initial_snapshot_pack_written(tmp_path, monkeypatch):
    """Run baseline_parsimony far enough to cross the 'initial' time and assert
    a pack file lands under the study's viz/3d, with the REAL packer stubbed
    out (no network / Rust) so this stays fast and hermetic. Proves the wiring
    end-to-end: composite run -> EcoliPackStep fires at its declared time ->
    writes <out_dir>/<name>.pack.json."""
    import v2ecoli
    from v2ecoli.core import build_core
    from v2ecoli.structural import pack_step

    calls = []

    def _stub_pack_from_state(out_dir, name, counts, volume_fl, **kwargs):
        os.makedirs(out_dir, exist_ok=True)
        doc = {"format": "parsimony.pack.v1", "name": name, "n_placed": 1}
        with open(os.path.join(out_dir, f"{name}.pack.json"), "w") as fh:
            json.dump(doc, fh)
        calls.append((name, volume_fl))
        return {"placements": [1]}

    monkeypatch.setattr(pack_step, "pack_from_state", _stub_pack_from_state)
    monkeypatch.chdir(tmp_path)

    core = build_core()
    comp = v2ecoli.build_composite(
        "baseline_parsimony", core=core, seed=0, cache_dir=str(CACHE_DIR),
        study="itest", snapshots={"initial": 2.0}, top_n=2, emitter="null")
    comp.run(4.0)  # time_step=1s -> crosses global_time=2.0

    assert [c[0] for c in calls] == ["initial"]

    pack = tmp_path / "studies" / "itest" / "viz" / "3d" / "initial.pack.json"
    assert pack.is_file()
    doc = json.loads(pack.read_text())
    assert doc["format"] == "parsimony.pack.v1"


@pytest.mark.slow
@_needs_cache
def test_initial_snapshot_real_packer(tmp_path, monkeypatch):
    """Opt-in real end-to-end: the actual pbg_parsimony.build_pack (network
    AlphaFold fetch + Rust packer) run against a live baseline_parsimony
    composite, to the 'initial' snapshot only. Requires PARSIMONY_HOME.

    NOTE: in some sandboxed shells the ambient text-decode default for
    subprocess pipes resolves to plain ASCII rather than UTF-8, and
    pbg_parsimony.engine.run_pipeline()'s subprocess.run(..., text=True) call
    (no explicit encoding=) then raises UnicodeDecodeError decoding the CLI's
    stderr (it prints a non-ASCII '→' arrow) -- AFTER the real Rust
    packer has already written a fully valid pack to disk (verified
    manually). This is an upstream pbg-parsimony subprocess-encoding issue,
    not a v2ecoli wiring bug. Workaround: run with PYTHONUTF8=1 in the
    environment if this test fails with UnicodeDecodeError."""
    if not os.environ.get("PARSIMONY_HOME"):
        pytest.skip("PARSIMONY_HOME not set; needed by the real Rust packer")
    import v2ecoli
    from v2ecoli.core import build_core
    monkeypatch.chdir(tmp_path)

    core = build_core()
    comp = v2ecoli.build_composite(
        "baseline_parsimony", core=core, seed=0, cache_dir=str(CACHE_DIR),
        study="itest", snapshots={"initial": 2.0}, top_n=2, emitter="null")
    comp.run(4.0)

    pack = tmp_path / "studies" / "itest" / "viz" / "3d" / "initial.pack.json"
    assert pack.is_file()
    doc = json.loads(pack.read_text())
    assert doc["format"] == "parsimony.pack.v1"
    assert "bounds" in doc
    assert "ingredients" in doc
    assert "placements" in doc


@pytest.mark.slow
@_needs_cache
def test_pre_division_snapshot(tmp_path, monkeypatch):
    """Full generation to division -> both packs, with the real packer. Opt-in
    (slow): runs a real baseline generation and packs ~1.3M molecules twice."""
    if not os.environ.get("PARSIMONY_HOME"):
        pytest.skip("PARSIMONY_HOME not set; needed by the real Rust packer")
    import v2ecoli
    from v2ecoli.core import build_core
    monkeypatch.chdir(tmp_path)

    core = build_core()
    comp = v2ecoli.build_composite(
        "baseline_parsimony", core=core, seed=0, cache_dir=str(CACHE_DIR),
        study="itest", emitter="null")
    comp.run(3000.0)  # to/through division

    d = tmp_path / "studies" / "itest" / "viz" / "3d"
    assert (d / "initial.pack.json").is_file()
    assert (d / "pre-division.pack.json").is_file()
