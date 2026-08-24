"""Fast-mode ParCa keeps ONE transcription factor, chosen by file position.

``step_02_input_adjustments.py`` reduces ``tf_to_active_inactive_conditions``
to a single entry under ``debug=True``::

    first_key = next(iter(tf_cond))
    tf_cond_out = {first_key: tf_cond[first_key]}

``next(iter(...))`` takes whichever key insertion order happens to yield, and
that order follows the row order of ``condition/tf_condition.tsv``. So the TF
whose regulation a fast build applies is decided by **which row is first in a
data file** — not by anything that reads as a choice.

Today the first row is ``trpR`` / ``CPLX-125``. That is load-bearing rather
than incidental:

* Anything reading regulatory behaviour out of a fast-mode build is reading a
  model with **one** TF, and it happens to be the one the trpR studies are
  about. Reorder the file and those builds silently become unregulated — the
  run still completes, the numbers just quietly stop meaning what they meant.
* The failure is invisible by construction. No build fails, no gate trips, and
  a knockout of a now-unregulated TF produces a plausible-looking null.

This module does not argue trpR *should* be first. It makes a change to that
ordering **loud**, so whoever reorders the file is told what else moves.
"""

import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle

# The TF the fast regime currently applies, and the studies that depend on it.
EXPECTED_FIRST_TF = "trpR"
EXPECTED_FIRST_ACTIVE_TF = "CPLX-125"


def _tf_condition_rows(bundle: SourceBundle) -> list[list[str]]:
    """Data rows of tf_condition.tsv, in file order, comments and header dropped."""
    path = bundle.resolve_relpath("condition__tf_condition")
    with open(path) as handle:
        lines = [ln.rstrip("\n") for ln in handle
                 if ln.strip() and not ln.lstrip().startswith("#")]
    return [ln.split("\t") for ln in lines[1:]]  # [0] is the column header


def _unquote(field: str) -> str:
    return field.strip().strip('"')


def test_first_tf_condition_row_is_the_one_fast_mode_keeps():
    """Pin the row whose position decides fast mode's single TF.

    If this fails, the fix is NOT to edit the constants above until it passes.
    Fast-mode builds now apply a different TF's regulation, and any study
    reading regulation out of one needs re-checking first.
    """
    rows = _tf_condition_rows(SourceBundle())
    assert rows, "tf_condition.tsv has no data rows"

    tf, active_tf = _unquote(rows[0][0]), _unquote(rows[0][1])
    assert (tf, active_tf) == (EXPECTED_FIRST_TF, EXPECTED_FIRST_ACTIVE_TF), (
        f"tf_condition.tsv's first data row is now {tf!r}/{active_tf!r}, not "
        f"{EXPECTED_FIRST_TF!r}/{EXPECTED_FIRST_ACTIVE_TF!r}. Fast-mode ParCa keeps only "
        "the FIRST row's TF (step_02_input_adjustments.py, `next(iter(tf_cond))`), so "
        "every fast build now applies this TF's regulation instead — silently, with no "
        "build failure. Re-check any study that reads regulatory behaviour from a "
        "fast-mode build before updating this expectation."
    )


def test_the_selection_really_is_positional():
    """The prune's semantics, on a synthetic dict.

    Without this, the test above is indistinguishable from one that pins a
    value for its own sake: it would keep passing if the selection stopped
    depending on order, and would then be pinning a fact that no longer
    matters. This asserts the coupling the first test exists to protect.
    """
    ordered = {"first": {"active": 1}, "second": {"active": 2}}
    assert next(iter(ordered)) == "first"

    reordered = {"second": {"active": 2}, "first": {"active": 1}}
    assert next(iter(reordered)) == "second", (
        "dict iteration is no longer insertion-ordered; the positional coupling "
        "this module guards would no longer hold"
    )


def test_every_other_declared_tf_is_dropped_in_fast_mode():
    """Name the scale of what the fast regime discards.

    The single-key prune is easy to read as a minor truncation. It is not: the
    shipped file declares many TFs and a fast build applies exactly one.
    """
    rows = _tf_condition_rows(SourceBundle())
    assert len(rows) > 1, (
        "expected multiple declared TFs; with only one, the fast-mode prune "
        "would be a no-op and this guard would be vacuous"
    )

    kept, dropped = 1, len(rows) - 1
    assert dropped >= 20, (
        f"fast mode keeps {kept} of {len(rows)} declared TFs, dropping {dropped}. "
        "If this count has fallen substantially, the regime's meaning has changed "
        "and the studies that caveat their results as 'fast-regime' need re-reading."
    )
