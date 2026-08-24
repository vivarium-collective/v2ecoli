"""Fast-mode ParCa keeps ONE transcription factor, chosen by file position.

Under ``debug=True``, step 2 reduces ``tf_to_active_inactive_conditions`` to a
single entry by taking the first key in insertion order
(``select_debug_tf_conditions``). That dict is built by iterating
``condition/tf_condition.tsv`` in row order, keeping only rows whose *active TF*
also appears in the fold-change tables (``simulation_data.py``,
``_add_condition_data``). So the TF a fast build models is decided by **which row
is first in a data file**, subject to a membership filter — not by anything that
reads as a choice.

Today that is ``trpR`` / ``CPLX-125``, and it is load-bearing rather than
incidental:

* Anything reading regulatory behaviour out of a fast-mode build is reading a
  model with **one** TF, and it happens to be the one the trpR studies are
  about. Reorder the file — or change fold-change membership — and those builds
  silently become unregulated.
* The failure is invisible by construction. No build fails, no gate trips, and a
  knockout of a now-unregulated TF produces a plausible-looking null rather than
  an error.

This module does not argue trpR *should* be first, and does not change the
selection. It makes a change **loud**.

Scope, stated rather than implied: the pin below reads the **shipped ParCa
fixture**, which is the post-filter dict the code actually prunes — not a
hand-read of the TSV, which would be blind to the fold-change filter. It is
correspondingly NOT sensitive to a ``SourceBundle`` override that supplies a
different ``tf_condition.tsv`` without the fixture being regenerated; a build
from such an override is outside what a fixture-based pin can see.
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import pytest

from v2ecoli.processes.parca.steps.step_02_input_adjustments import (
    select_debug_tf_conditions,
)

# The TF the fast regime currently applies. CPLX-125 is trpR's active form, and
# is the key form the dict uses (active TF ids, not TF names).
EXPECTED_FAST_MODE_TF = "CPLX-125"

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


# --- the data the selection lands on ------------------------------------------

def test_fast_mode_keeps_trpR(declared_tf_conditions):
    """Pin the survivor on the real, post-filter dict.

    If this fails, the fix is NOT to edit the constant until it passes: fast
    builds now apply a different TF's regulation, and any study reading
    regulation out of one needs re-checking first.
    """
    survivor = next(iter(select_debug_tf_conditions(declared_tf_conditions)))
    assert survivor == EXPECTED_FAST_MODE_TF, (
        f"fast-mode ParCa now keeps {survivor!r}, not {EXPECTED_FAST_MODE_TF!r} "
        "(trpR). The survivor is the first row of condition/tf_condition.tsv whose "
        "active TF also appears in the fold-change tables, so EITHER a row reorder "
        "OR a change in fold-change membership causes this — silently, with no build "
        "failure. Re-check any study that reads regulatory behaviour from a "
        "fast-mode build before updating this expectation."
    )


def test_the_prune_actually_discards_most_declared_tfs(declared_tf_conditions):
    """Name the scale of what the fast regime discards.

    Also the vacuity guard: with a single declared TF the prune would be a no-op
    and every assertion above would hold for the wrong reason.
    """
    declared = len(declared_tf_conditions)
    assert declared > 1, (
        f"only {declared} TF declared; the prune would be a no-op and this "
        "module would be pinning nothing"
    )
    kept = len(select_debug_tf_conditions(declared_tf_conditions))
    assert kept == 1 and declared - kept == declared - 1
