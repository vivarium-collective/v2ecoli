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
) -> dict[str, Any]:
    """Compare two numeric arrays.

    Verdicts: ``within_tol`` (max relative error <= rel_tol),
    ``mismatch`` (max relative error >= mismatch_rel), ``drift``
    (between the two), ``not_compared`` (shape mismatch / empty).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        return {"verdict": "not_compared",
                "reason": f"shape {a.shape} != {b.shape}"}
    if a.size == 0:
        return {"verdict": "not_compared", "reason": "empty"}

    denom = np.maximum(np.abs(a), 1e-30)
    rel = np.abs(a - b) / denom
    max_rel = float(np.max(rel))
    max_abs = float(np.max(np.abs(a - b)))

    ks_p = None
    if _HAVE_SCIPY and a.size >= 2:
        ks_p = float(_scipy_stats.ks_2samp(a, b).pvalue)

    if max_rel <= rel_tol:
        verdict = "within_tol"
    elif max_rel >= mismatch_rel:
        verdict = "mismatch"
    else:
        verdict = "drift"

    return {"verdict": verdict, "max_rel": max_rel,
            "max_abs": max_abs, "ks_p": ks_p}
