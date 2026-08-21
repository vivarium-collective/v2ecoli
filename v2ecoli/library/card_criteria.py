"""Typed grading criteria for report cards.

Moved to ``viva_superpowers.card_criteria`` (the shared home, so any workspace's
cards share one grader). This module is a thin re-export kept for back-compat —
existing ``from v2ecoli.library.card_criteria import grade_axis`` imports keep
working. Import from ``viva_superpowers`` in new code.
"""
from __future__ import annotations

from viva_superpowers.card_criteria import (  # noqa: F401
    VERDICTS,
    _SEVERITY,
    _band,
    _fit_threshold_linear,
    _r2,
    _ungraded,
    _welch,
    grade_axis,
)
