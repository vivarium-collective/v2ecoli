"""Per-metric comparison: relative error + KS, mapped to a verdict."""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy optional
    _HAVE_SCIPY = False


def compare_series(
    a,
    b,
    *,
    rel_tol: float,
    mismatch_rel: float = 0.5,
    abs_tol: float = 1e-12,
) -> dict[str, Any]:
    """Compare two numeric arrays.

    The verdict is based on the **median** relative error, not the max, so a
    handful of near-zero / outlier elements in a large distribution don't drive
    the whole metric to ``mismatch`` (which masked the fact that, e.g., RNA
    expression agrees to ~3% across two ParCa implementations). For a scalar,
    median == max, so scalars are unaffected.

    Verdicts: ``within_tol`` (median rel error <= rel_tol),
    ``mismatch`` (median rel error >= mismatch_rel), ``drift`` (between),
    ``not_compared`` (shape mismatch / empty).

    The denominator is symmetric and abs-floored:
    ``denom = max(|a|, |b|, abs_tol)``.  Element pairs where both values are
    below *abs_tol* are treated as effectively zero and contribute ~0 relative
    error, preventing spurious mismatches when one engine returns exactly 0 and
    the other returns a tiny non-zero.

    Returns ``median_rel`` (verdict driver), ``max_rel``, ``frac_within`` (the
    fraction of elements within *rel_tol*), ``max_abs`` and ``ks_p`` for
    context.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        return {"verdict": "not_compared",
                "reason": f"shape {a.shape} != {b.shape}"}
    if a.size == 0:
        return {"verdict": "not_compared", "reason": "empty"}

    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), abs_tol)
    rel = np.abs(a - b) / denom
    median_rel = float(np.median(rel))
    max_rel = float(np.max(rel))
    max_abs = float(np.max(np.abs(a - b)))
    frac_within = float(np.mean(rel <= rel_tol))

    ks_p = None
    if _HAVE_SCIPY and a.size >= 2:
        ks_p = float(_scipy_stats.ks_2samp(a, b).pvalue)

    if median_rel <= rel_tol:
        verdict = "within_tol"
    elif median_rel >= mismatch_rel:
        verdict = "mismatch"
    else:
        verdict = "drift"

    return {"verdict": verdict, "median_rel": median_rel, "max_rel": max_rel,
            "max_abs": max_abs, "frac_within": frac_within, "ks_p": ks_p}
