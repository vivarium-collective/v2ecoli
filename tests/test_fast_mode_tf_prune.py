"""Fast-mode ParCa keeps ONE transcription factor, chosen by file position.

Under ``debug=True``, step 2 reduces ``tf_to_active_inactive_conditions`` to a
single entry by taking the first key in insertion order
(``select_debug_tf_conditions``). That dict is built by iterating
``condition/tf_condition.tsv`` in row order, keeping only rows whose *active TF*
also appears in the fold-change tables (``simulation_data.py``,
``_add_condition_data``). So the TF a fast build models is decided by **which row
is first in a data file**, subject to a membership filter — not by anything that
reads as a choice.

The consequence worth knowing: anything reading regulatory behaviour out of a
fast-mode build is reading a model with **one** transcription factor. No build
fails and no gate trips, so a knockout of a TF the fast regime dropped returns a
plausible-looking null rather than an error.

**Scope, deliberately narrow.** These tests pin v2ecoli's own behaviour — that
the selection is positional, and that exactly one TF survives. They do NOT pin
*which* TF that is. Which one it happens to be is a property of
``ecoli-sources``' row ordering and fold-change membership; asserting it here
would make a v2ecoli test fail on a legitimate upstream curation change, in a
repo this one does not control. The identity of the survivor is documented in
``select_debug_tf_conditions`` and is a thing to look up, not an invariant to
enforce from here.
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import pytest

from v2ecoli.processes.parca.steps.step_02_input_adjustments import (
    select_debug_tf_conditions,
)

FIXTURE = Path(__file__).resolve().parents[1] / "models" / "parca" / "parca_state.pkl.gz"


@pytest.fixture(scope="module")
def declared_tf_conditions() -> dict:
    """``tf_to_active_inactive_conditions`` as the code sees it, from the shipped
    ParCa fixture — i.e. already past the fold-change membership filter."""
    if not FIXTURE.exists():                                    # pragma: no cover
        pytest.skip(f"ParCa fixture not present: {FIXTURE}")
    with gzip.open(FIXTURE, "rb") as handle:
        state = pickle.load(handle)
    tf_cond = state.get("tf_to_active_inactive_conditions")
    assert tf_cond, "fixture carries no tf_to_active_inactive_conditions"
    return tf_cond


# --- the code under test ------------------------------------------------------

def test_the_selection_is_positional_not_by_name():
    """Pins the SELECTION ITSELF, on the real function.

    This is the test that fails if someone replaces the positional pick with
    ``sorted(tf_cond)[0]`` or an explicit lookup. Without it, everything else
    here pins data while the behaviour it exists to protect could change freely.
    """
    a = {"first": {"active nutrients": 1}, "second": {"active nutrients": 2}}
    assert select_debug_tf_conditions(a) == {"first": {"active nutrients": 1}}

    # Same contents, different order -> different survivor. That IS the hazard.
    b = {"second": {"active nutrients": 2}, "first": {"active nutrients": 1}}
    assert select_debug_tf_conditions(b) == {"second": {"active nutrients": 2}}, (
        "the debug selection is no longer positional. If that was deliberate — e.g. "
        "choosing the TF by name — this guard is obsolete and the studies that "
        "caveat their results as 'fast-regime' should be re-read, because the regime "
        "now models a different regulator."
    )


def test_exactly_one_tf_survives(declared_tf_conditions):
    """A fast build applies ONE TF's regulation, not a reduced-but-plural set."""
    assert len(select_debug_tf_conditions(declared_tf_conditions)) == 1


def test_the_prune_actually_discards_declared_tfs(declared_tf_conditions):
    """Name the scale of what the fast regime discards.

    Also the vacuity guard: with a single declared TF the prune would be a no-op
    and the assertions above would hold for the wrong reason.
    """
    declared = len(declared_tf_conditions)
    assert declared > 1, (
        f"only {declared} TF declared; the prune would be a no-op and this "
        "module would be pinning nothing"
    )
    assert len(select_debug_tf_conditions(declared_tf_conditions)) == 1
