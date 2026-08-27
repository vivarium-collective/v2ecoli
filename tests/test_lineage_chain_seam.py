"""A lineage that RESUMES must keep labelling its generations honestly.

v2ecoli answers a generation-indexed induction with a **sequence of caches**, not
a mutation schedule: run the early generations against one cache, carry the
biological state across, run the later ones against another
(``v2ecoli/perturbations/design_variant.py``). That makes a resumed invocation a
first-class shape rather than an edge case — and it separates two labels that a
fresh lineage can never separate:

  * the zarr ``generation`` label, and
  * the phylogeny ``agent_id``.

On a fresh lineage both start at 1 / "0" and advance together, so nothing can
drift and nothing tests the coupling. On a resume the caller supplies both, and
``gen`` was HARDCODED to 1 — so a second stage wrote a second ``generation=1``
partition while carrying agent_id "000". A report card windowing on generation
then grades the wrong cells, and both labels look internally consistent.

⭑ The convention is ``generation == len(agent_id)`` — the wrapped-fork arm
*computes* its generation that way (``vivarium_ecoli_engine``), so the two
engines only stay comparable while this holds on both.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_multigen_last_generation import _FakeComposite  # noqa: E402

from v2ecoli.library.xarray_run import (  # noqa: E402
    run_multigen_xarray, view_from_emit_paths)


def _md(exp="chain"):
    return {"experiment_id": exp, "engine": "fake", "condition": "test",
            "variant": 0, "lineage_seed": 0, "time_step": 1.0,
            "max_duration": 900.0, "agent_id": "0"}


def _run(tmp_path, *, name, store=None, exp=None, **kw):
    """Run one stage. ``store`` lets a second stage write into the SAME store as
    the first — which a chain must do; see the module note on parent linkage."""
    from v2ecoli.core import build_core
    comp = _FakeComposite(divide_period=200, dry0=350.0)
    comp.core = build_core()
    # ⭐ NO re-keying: a resumed stage builds a FRESH composite, which names its
    # cell "0" exactly as stage 1 did. That is the point — the inner key and the
    # phylogeny key are different things, and only the latter advances.
    return run_multigen_xarray(
        comp, store_path=str(store or (tmp_path / f"{name}.zarr")),
        view=view_from_emit_paths(["listeners.mass.dry_mass"]),
        metadata_base=_md(exp or name), chunk=20, single_daughters=True, **kw)


def test_a_fresh_lineage_is_unchanged(tmp_path):
    """The default must be a no-op. `initial_generation=1` is the value at which
    every mutant below is invisible, so it is the control, not the coverage."""
    res = _run(tmp_path, name="fresh", max_steps=900, max_generations=3)
    assert res["generations"] == [1, 2, 3]


def test_a_resumed_stage_CONTINUES_the_generation_labels(tmp_path):
    """⭐ The defect, exercised as a real two-stage chain into ONE store.

    Before this, stage 2 restarted at generation 1 and the store held two
    `generation=1` partitions describing different cells."""
    store = tmp_path / "chain.zarr"
    # ⚠ SAME store AND same experiment_id — see the parent-linkage test below.
    s1 = _run(tmp_path, name="s1", store=store, exp="chain", max_steps=900,
              max_generations=3)
    assert s1["generations"] == [1, 2, 3]

    s2 = _run(tmp_path, name="s2", store=store, exp="chain", max_steps=900,
              max_generations=2, initial_generation=4, overwrite=False)
    assert s2["generations"] == [4, 5], (
        "the resumed stage restarted its generation labels — the store now has "
        "a duplicate generation and the card windows the wrong cells")


def test_a_resume_needs_its_PARENT_generation_in_the_store(tmp_path):
    """⛔ A constraint on the chain, discovered by hitting it.

    The emitter's `colony` partition strategy derives a partition from
    `agent_id` — `generation == len(agent_id)`, WITH A PARENT — so resuming at
    generation 4 looks for generation 3's time coordinate. Into a fresh store
    there is no parent and it fails.

    ⇒ **The stages of a chain must share a store AND an experiment_id** — the
    partition path is ``experiment_id=…/variant=…/lineage_seed=…/emitstep_gen=N``,
    so a stage that changes any of that prefix is looking for a parent that its
    predecessor never wrote. Writing stage 2 somewhere new does not merely lose
    the linkage, it does not run. Asserted so nobody designs a stage-per-store
    (or stage-per-experiment) layout and discovers this at integration.
    """
    with pytest.raises((KeyError, FileNotFoundError), match="emitstep_gen|gen="):
        _run(tmp_path, name="orphan", max_steps=400, max_generations=1,
             initial_generation=4, overwrite=False)


def test_max_generations_stays_a_COUNT_not_an_absolute_stop(tmp_path):
    """⚠ The alternative reading breaks the caller silently: if
    `max_generations` meant "run until generation N", then a stage resuming at 4
    and asking for 3 would run ZERO generations and report success."""
    store = tmp_path / "count.zarr"
    _run(tmp_path, name="c1", store=store, exp="count", max_steps=900,
         max_generations=2)
    res = _run(tmp_path, name="c2", store=store, exp="count", max_steps=900,
               max_generations=2, initial_generation=3, overwrite=False)
    assert res["generations"] == [3, 4], res["generations"]
    assert len(res["generations"]) == 2


@pytest.mark.parametrize("gen", [1, 2, 4])
def test_the_phylogeny_key_is_DERIVED_from_the_generation(tmp_path, gen):
    """⭐ The invariant holds BY CONSTRUCTION, so there is no mismatch to refuse.

    Taking one parameter for both the inner composite key and the phylogeny
    label is exactly how the wrapped-fork driver came to run every generation as
    the founder. Here the phylogeny key is computed from the generation, so a
    caller cannot offset one and not the other."""
    import inspect
    from v2ecoli.library import xarray_run as xr
    src = inspect.getsource(xr.run_multigen_xarray)
    assert 'partition_agent_id = "0" * gen' in src, (
        "the phylogeny key is no longer derived from the generation — a caller "
        "can now supply two labels that disagree")


def test_generation_zero_is_refused(tmp_path):
    """Labels are 1-based on both sides of the comparison; a 0 here would be the
    0-based convention leaking in from v2ecoli's other emitting path."""
    with pytest.raises(ValueError, match="1-based|>= 1"):
        _run(tmp_path, name="zero", max_steps=200, max_generations=1,
             initial_generation=0)


# ---------------------------------------------------------------------------
# The hand-off
# ---------------------------------------------------------------------------

def test_the_final_daughter_is_written_as_the_carry_state(tmp_path):
    """⭐ A chain needs the cell that the generation cap throws away.

    At the cap the runner deliberately does NOT fold the post-division daughter
    into the parent partition — that daughter is the next stage's founder. It was
    simply dropped; now it can be captured."""
    out = tmp_path / "carry.json"
    res = _run(tmp_path, name="carry", max_steps=900, max_generations=2,
               daughter_state_out_path=str(out))
    assert res["generations"] == [1, 2]
    assert out.exists(), "the chain hand-off was not written"
    import json
    payload = json.loads(out.read_text())
    assert payload, "the carry state is empty"
    assert "listeners" in payload or "bulk" in payload, sorted(payload)[:6]


def test_no_path_means_no_hand_off_and_no_crash(tmp_path):
    """The default must not write, and must not fail trying."""
    res = _run(tmp_path, name="nocarry", max_steps=900, max_generations=2)
    assert res["generations"] == [1, 2]
    assert not list(tmp_path.glob("*.json"))


def test_a_resume_that_would_DELETE_its_predecessor_is_REFUSED(tmp_path):
    """⛔⛔ The destructive default, and the reason it needs a name.

    `overwrite=True` is the default and rmtree's the store. A resumed stage run
    with default arguments therefore deletes the generations it must link back
    to, and then fails with a KeyError about a missing PARENT — which reads as a
    linkage bug, not as "stage 1 was deleted a second ago". A chain appends.
    """
    store = tmp_path / "destructive.zarr"
    _run(tmp_path, name="d1", store=store, exp="d", max_steps=900,
         max_generations=2)
    marker = list(store.glob("*"))
    assert marker, "stage 1 wrote nothing, so this test proves nothing"
    with pytest.raises(ValueError, match="overwrite=False|would DELETE"):
        _run(tmp_path, name="d2", store=store, exp="d", max_steps=900,
             max_generations=1, initial_generation=3)
    assert list(store.glob("*")), "the refusal did not happen before the rmtree"
