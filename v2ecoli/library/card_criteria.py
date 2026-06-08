"""Typed grading criteria for report cards.

A *criterion* says how to compare a measured axis against a reference, and what
verdict that earns. One dispatch (`grade_axis`) covers every axis type so the
organism-behavior card and the v1<->v2 equivalence card share one grader:

  - ``rel_tol``  — scalar within a relative tolerance of a reference scalar.
  - ``ttest``    — population stats: Welch's t-test of the candidate per-cell
                   values vs the reference per-cell values; a *significant*
                   difference is drift/mismatch. (Cell-level n, never timepoints.)
  - ``r2``       — vector (transcriptome/proteome): R^2 of candidate vs
                   reference ensemble-mean vector.
  - ``boolean``  — a behavioral assertion that must hold.

Verdicts use a 4-state band (adopted from the vEcoli<->v2ecoli report):
``within_tol`` (pass) / ``drift`` (amber warning) / ``mismatch`` (fail) /
``ungraded`` (no reference). Each grade returns a compact ``meter`` string and a
human-readable ``criterion`` string for the report.
"""
from __future__ import annotations

import math
from typing import Any

VERDICTS = ("within_tol", "drift", "mismatch", "ungraded")


def _band(value: float, good: float, warn: float, *, higher_is_better: bool) -> str:
    """Map a scalar to within_tol/drift/mismatch given good/warn thresholds."""
    if higher_is_better:
        if value >= good:
            return "within_tol"
        return "drift" if value >= warn else "mismatch"
    else:
        if value <= good:
            return "within_tol"
        return "drift" if value <= warn else "mismatch"


def _welch(a: list[float], b: list[float]) -> dict[str, float]:
    """Welch's t-test (unequal variance) p-value + Cohen's d, no scipy needed
    for the inputs but uses scipy for the t-distribution survival function."""
    from scipy import stats
    na, nb = len(a), len(b)
    res = stats.ttest_ind(a, b, equal_var=False)
    ma, mb = (sum(a) / na), (sum(b) / nb)
    # pooled sd for an effect-size readout
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)) or 1.0
    return {"p": float(res.pvalue), "t": float(res.statistic),
            "cohens_d": (ma - mb) / sp, "delta_rel": (ma - mb) / mb if mb else 0.0}


def _r2(cand: list[float], ref: list[float]) -> float:
    """Coefficient of determination of cand vs ref (ref as the predictor line
    y=x). Computed in log10 space on positive entries (omics are log-normal);
    falls back to linear if too few positive points."""
    pairs = [(c, r) for c, r in zip(cand, ref) if c > 0 and r > 0]
    if len(pairs) < 3:
        pairs = list(zip(cand, ref))
        xs = [r for _, r in pairs]
        ys = [c for c, _ in pairs]
    else:
        xs = [math.log10(r) for _, r in pairs]
        ys = [math.log10(c) for c, _ in pairs]
    if len(xs) < 2:
        return 0.0
    my = sum(ys) / len(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - x) ** 2 for x, y in zip(xs, ys))  # residual vs identity y=x
    return 1.0 - ss_res / ss_tot if ss_tot else 1.0


def grade_axis(measured: dict | bool | None, criterion: dict) -> dict[str, Any]:
    """Grade one axis. ``measured`` is the card node for the axis; ``criterion``
    is the reference's spec for it. Returns
    ``{verdict, value, meter, criterion, detail}``.
    """
    ctype = criterion.get("type")

    if ctype == "rel_tol":
        ref = criterion.get("reference")
        tol = criterion.get("tol_rel", 0.05)
        got = measured.get("mean") if isinstance(measured, dict) else measured
        if ref is None or got is None:
            return _ungraded(f"within {tol:.0%}")
        rel = abs(got - ref) / abs(ref) if ref else float("inf")
        verdict = _band(rel, tol, 2 * tol, higher_is_better=False)
        return {"verdict": verdict, "value": got,
                "criterion_str": f"within {tol:.0%} of {ref:.4g}",
                "meter": f"Δ = {got - ref:+.4g} ({rel:.1%} rel)", "detail": {"rel": rel}}

    if ctype == "ttest":
        # Magnitude-first: the verdict bands on the relative mean shift |Δ|, with
        # the Welch p-value as a significance GUARD (a large shift is only a
        # mismatch if it is also statistically significant; a large-but-noisy
        # shift stays a drift warning). This keeps a meaningful graded band — a
        # pure-p-value gate goes near-binary at small n (the t-test detects
        # *difference*, not *magnitude*).
        ref_vals = criterion.get("ref_values")
        within_pct = criterion.get("within_pct", 0.05)
        mismatch_pct = criterion.get("mismatch_pct", 0.10)
        p_min = criterion.get("p_min", 0.05)
        inactive_eps = criterion.get("inactive_eps", 1e-6)
        cand_vals = measured.get("values") if isinstance(measured, dict) else None
        if not ref_vals or not cand_vals or len(cand_vals) < 2 or len(ref_vals) < 2:
            return _ungraded(f"|Δ| < {within_pct:.0%}")
        ma, mb = sum(cand_vals) / len(cand_vals), sum(ref_vals) / len(ref_vals)
        # Near-zero reference (e.g. an acetate sentinel): a relative-shift t-test
        # is ill-defined (Δrel divides by ~0). Handle by activity instead — if
        # the reference is inactive (~0), the only question is whether the
        # candidate stayed inactive (pass) or became active (the sentinel
        # tripped — overflow onset = mismatch).
        if abs(mb) < inactive_eps:
            tripped = abs(ma) >= inactive_eps
            return {"verdict": "mismatch" if tripped else "within_tol", "value": ma,
                    "criterion_str": f"stays inactive (|mean| < {inactive_eps:g})",
                    "meter": (f"became active: {ma:+.3g} (ref ≈ 0)" if tripped
                              else f"both ≈ 0 (ref {mb:+.2g}, meas {ma:+.2g})"),
                    "detail": {"inactive_ref": True, "tripped": tripped}}
        w = _welch(cand_vals, ref_vals)
        eff = abs(w["delta_rel"])
        if eff < within_pct:
            verdict = "within_tol"
        elif eff < mismatch_pct:
            verdict = "drift"
        else:
            verdict = "mismatch" if w["p"] < p_min else "drift"
        return {"verdict": verdict, "value": (measured or {}).get("mean"),
                "criterion_str": (f"|Δ| < {within_pct:.0%} (drift {within_pct:.0%}–"
                                  f"{mismatch_pct:.0%}; mismatch >{mismatch_pct:.0%} & p<{p_min})"),
                "meter": f"Δ = {w['delta_rel']:+.1%} · p = {w['p']:.3f} · d = {w['cohens_d']:+.2f}",
                "detail": w}

    if ctype == "r2":
        ref_vec = criterion.get("ref_vector")
        r2_min = criterion.get("r2_min", 0.99)
        r2_drift = criterion.get("r2_drift", 0.95)
        cand_vec = measured.get("vector") if isinstance(measured, dict) else None
        if not ref_vec or not cand_vec:
            return _ungraded(f"R² ≥ {r2_min}")
        r2 = _r2(cand_vec, ref_vec)
        verdict = _band(r2, r2_min, r2_drift, higher_is_better=True)
        n = sum(1 for c, r in zip(cand_vec, ref_vec) if c > 0 and r > 0)
        return {"verdict": verdict, "value": r2,
                "criterion_str": f"log-log R² ≥ {r2_min} ({n} genes)",
                "meter": f"R² = {r2:.4f}", "detail": {"r2": r2, "n": n}}

    if ctype == "flux_scatter":
        # Whole exchange-flux vector vs the pinned reference. Two failure modes:
        # (1) QUALITATIVE — an exchange appears (ref~0 -> active) or disappears
        # (active -> ~0); any such change is a mismatch regardless of R² (the
        # cell started/stopped using a metabolite). (2) QUANTITATIVE — matched
        # (both active) fluxes drift; graded by linear R² vs identity.
        ref_vec = criterion.get("ref_vector")
        eps = criterion.get("active_eps", 1e-6)
        # Significance floor for the QUALITATIVE appeared/disappeared verdict.
        # `active_eps` (~1e-6) is the detection floor for pairing/R²; many FBA
        # exchanges jitter just across it at ~1e-6 with CV > 100% (fitting slack,
        # not metabolism). A flip whose active-side magnitude is below `qual_eps`
        # is reported but does NOT count toward mismatch — only a flux that turns
        # on/off at a metabolically meaningful level is a qualitative change.
        qual_eps = criterion.get("qual_eps", 1e-3)
        r2_min = criterion.get("r2_min", 0.99)
        r2_drift = criterion.get("r2_drift", 0.95)
        cand_vec = measured.get("vector") if isinstance(measured, dict) else None
        if not ref_vec or not cand_vec:
            return _ungraded(f"R² ≥ {r2_min}, no appeared/disappeared")
        appeared, disappeared, matched, sub_floor = [], [], [], []
        for i, (c, r) in enumerate(zip(cand_vec, ref_vec)):
            ca, ra = abs(c) > eps, abs(r) > eps
            if ca and ra:
                matched.append((c, r))
            elif ca and not ra:  # appeared: active side is the candidate
                (appeared if abs(c) >= qual_eps else sub_floor).append(i)
            elif ra and not ca:  # disappeared: active side is the reference
                (disappeared if abs(r) >= qual_eps else sub_floor).append(i)
        # linear R² vs identity on matched (signed) pairs
        if len(matched) >= 2:
            ys = [c for c, _ in matched]
            my = sum(ys) / len(ys)
            ss_tot = sum((c - my) ** 2 for c, _ in matched)
            ss_res = sum((c - r) ** 2 for c, r in matched)
            r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
        else:
            r2 = 1.0
        n_qual = len(appeared) + len(disappeared)
        if n_qual:
            verdict = "mismatch"
        else:
            verdict = _band(r2, r2_min, r2_drift, higher_is_better=True)
        sub = f" · {len(sub_floor)} sub-floor" if sub_floor else ""
        return {"verdict": verdict, "value": r2,
                "criterion_str": (f"R² ≥ {r2_min} on matched fluxes; "
                                  f"no appeared/disappeared ≥ {qual_eps:g}"),
                "meter": (f"R² = {r2:.4f} · {len(matched)} matched · "
                          f"{len(appeared)} appeared · {len(disappeared)} lost{sub}"),
                "detail": {"r2": r2, "appeared": appeared,
                           "disappeared": disappeared, "sub_floor": sub_floor,
                           "n_matched": len(matched), "qual_eps": qual_eps}}

    if ctype == "boolean":
        ok = bool(measured) if measured is not None else None
        if ok is None:
            return _ungraded("assertion holds")
        return {"verdict": "within_tol" if ok else "mismatch", "value": ok,
                "criterion_str": "assertion holds", "meter": "pass" if ok else "FAIL",
                "detail": {}}

    return _ungraded("(unknown criterion)")


def _ungraded(criterion_str: str) -> dict[str, Any]:
    return {"verdict": "ungraded", "value": None, "criterion_str": criterion_str,
            "meter": "—", "detail": {}}
