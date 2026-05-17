"""Generic evaluator for behavior-test entries in study.yaml.

Supports two schema shapes:

  Legacy (`expected_behavior:`)
    name / en / given.window / measure.{kind,...} / expect.{op,...}

  v3 (`behavior_tests:`)
    name / classification / measure.{kind,window,...} / pass_if.{op,...} /
    requires_simulation / cites / calibration_anchor

Call `evaluate(entry, history, *, monomer_ids=None)` to get
(passed, message). The parametrized test in `test_behaviors.py` wires
this up to pytest.

`monomer_ids` is the optional list-of-strings index used by the v3
`monomer_count` measure kind to resolve readouts like "PD03831[c]" to a
position in the per-step `monomer_counts.monomerCounts` array.
"""
from __future__ import annotations

import statistics


# ─── Identifiers (mirrored from conftest for standalone use) ────────────────

DNAA_MONOMER_ID = "MONOMER0-160[c]"   # the DnaA-ATP complex form (bulk)
DNAA_MONOMER_PD = "PD03831[c]"        # the v3 schema's canonical DnaA monomer id
DNAA_MRNA_ID = "EG10235_RNA"


# ─── State accessors ────────────────────────────────────────────────────────

def _agent_view(state: dict) -> dict | None:
    """Return the per-agent state subtree.

    The SQLite emitter sits inside ``agents.0`` and emits its per-step
    state directly under that node — so each row is the flat per-agent
    subtree (``listeners``, ``bulk``, ``unique``, ``global_time``).
    The synthetic-history tests wrap their snapshots in ``{agents: {0: ...}}``,
    so accept that shape too.
    """
    if not isinstance(state, dict):
        return None
    agents = state.get("agents")
    if isinstance(agents, dict) and agents:
        return next(iter(agents.values()))
    return state


def _bulk_count(state: dict, molecule_id: str) -> int | None:
    agent = _agent_view(state)
    if agent is None:
        return None
    bulk = agent.get("bulk")
    if bulk is None:
        return None
    if isinstance(bulk, dict) and "id" in bulk and "count" in bulk:
        ids = bulk["id"]
        counts = bulk["count"]
    elif isinstance(bulk, list) and bulk and isinstance(bulk[0], (list, tuple)):
        ids = [row[0] for row in bulk]
        counts = [row[1] for row in bulk]
    else:
        return None
    try:
        idx = ids.index(molecule_id)
    except ValueError:
        return None
    return counts[idx]


def _listener_value(state: dict, dotted_path: str):
    cur = _agent_view(state)
    if cur is None:
        return None
    for seg in dotted_path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return cur


# ─── Window selection ───────────────────────────────────────────────────────

def _window(history: list, name: str) -> list:
    if name == "full":
        return history
    if name == "second_half":
        n = len(history)
        return history[n // 2 :] if n >= 2 else history
    if name == "post_initiation_10min":
        # Stub for the gene-dosage test (BT-05 equivalent). Without an
        # initiation-event detector we can't slice this window correctly.
        return []
    raise ValueError(f"unknown window {name!r}")


# ─── Measure dispatch ───────────────────────────────────────────────────────

def _monomer_count(state: dict, monomer_id: str, monomer_ids: list[str] | None) -> int | None:
    """Resolve a monomer count via listeners.monomer_counts[idx],
    where idx = monomer_ids.index(monomer_id). The monomer_ids list is loaded
    from sim_data and threaded through as a kwarg on the evaluator entry
    point; fall back to a hardcoded index for the well-known DnaA case so the
    evaluator stays useful without sim_data on hand.

    Two physical shapes are tolerated:
      - ``listeners.monomer_counts`` is the flat array (current emitter)
      - ``listeners.monomer_counts.monomerCounts`` is the array (legacy
        single_cell.dill shape)
    """
    arr = _listener_value(state, "listeners.monomer_counts")
    if isinstance(arr, dict):
        arr = arr.get("monomerCounts")
    if arr is None:
        return None
    if monomer_ids is not None:
        try:
            idx = monomer_ids.index(monomer_id)
        except ValueError:
            return None
    elif monomer_id == DNAA_MONOMER_PD:
        idx = 3861   # confirmed at t=2350s in cached run (256,299 DnaA monomers)
    else:
        return None
    try:
        return int(arr[idx])
    except (IndexError, TypeError):
        return None


def _measure_series(history: list, measure: dict,
                    *, monomer_ids: list[str] | None = None) -> list[float] | None:
    kind = measure["kind"]
    if kind == "bulk_count":
        mol = measure["id"]
        series = [_bulk_count(s["state"], mol) for s in history]
        if all(v is None for v in series):
            return None
        return [v for v in series if v is not None]
    if kind == "monomer_count":
        mol = measure["id"]
        series = [_monomer_count(s["state"], mol, monomer_ids) for s in history]
        if all(v is None for v in series):
            return None
        return [v for v in series if v is not None]
    if kind == "listener_sum":
        path = measure["path"]
        series = []
        for s in history:
            v = _listener_value(s["state"], path)
            if isinstance(v, list):
                # Flatten one level for 2D matrices (e.g. n_bound_TF_per_TU
                # shape (n_TU, n_TF)). Deeper structures would need a
                # dedicated measure kind.
                if v and isinstance(v[0], list):
                    series.append(sum(sum(row) for row in v))
                else:
                    series.append(sum(v))
            else:
                series.append(v or 0)
        return series
    if kind == "listener_path":
        path = measure["path"]
        return [_listener_value(s["state"], path) for s in history]
    if kind == "tf_axis_sum":
        # Read a 2-D listener field (e.g. listeners.rna_synth_prob.n_bound_TF_per_TU
        # with shape (n_TU, n_TF)) and project onto one TF column by summing
        # across TUs. Used by autorepression tests to get the per-tick
        # binding load for a specific transcription factor.
        path = measure.get("path", "listeners.rna_synth_prob.n_bound_TF_per_TU")
        tf_idx = measure["tf_index"]
        series = []
        for s in history:
            v = _listener_value(s["state"], path)
            if isinstance(v, list) and v and isinstance(v[0], list):
                series.append(sum(row[tf_idx] for row in v if len(row) > tf_idx))
            else:
                series.append(0)
        return series
    if kind == "listener_index":
        # Read a 1-D listener field and pick out a single index. Used to slice
        # cistron-indexed or TU-indexed series to one gene's column. Example:
        #   {kind: listener_index, path: listeners.rnap_data.rna_init_event_per_cistron, index: 227}
        # → dnaA's per-step transcription init events.
        path = measure["path"]
        idx = measure["index"]
        series = []
        for s in history:
            v = _listener_value(s["state"], path)
            if isinstance(v, list) and len(v) > idx:
                series.append(v[idx])
            else:
                series.append(0)
        return series
    raise ValueError(f"unknown measure kind {kind!r}")


def _measure(history: list, measure: dict):
    """Reduce a measure to whatever shape the operator expects."""
    if measure["kind"] == "xy_correlation":
        x = _measure_series(history, measure["x"])
        y = _measure_series(history, measure["y"])
        return {"x": x, "y": y}

    reduce = measure.get("reduce", "series")
    series = _measure_series(history, measure)
    if series is None or not series:
        return None
    if reduce == "series":
        return series
    if reduce == "median":
        return statistics.median(series)
    if reduce == "mean":
        return statistics.mean(series)
    if reduce == "first_and_last":
        return {"first": series[0], "last": series[-1]}
    if reduce == "pre_post_event_ratio":
        # Used by the post-initiation-gene-dosage stub; needs an event index.
        return None
    raise ValueError(f"unknown reduce {reduce!r}")


# ─── Expect dispatch ────────────────────────────────────────────────────────

def _pearson(x: list[float], y: list[float]) -> float | None:
    if not x or not y or len(x) != len(y) or len(x) < 2:
        return None
    if statistics.stdev(x) == 0 or statistics.stdev(y) == 0:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denom = statistics.stdev(x) * statistics.stdev(y) * (len(x) - 1)
    return cov / denom if denom else None


def _check(value, expect: dict) -> tuple[bool, str]:
    op = expect["op"]
    if value is None:
        return False, f"measure returned None; cannot evaluate op={op}"

    if op == "in_range":
        lo, hi = expect["low"], expect["high"]
        return (lo <= value <= hi,
                f"value={value} expected in [{lo}, {hi}]")

    if op == "rolling_cv_below":
        series = value  # series
        w = expect.get("window_steps", 5)
        thresh = expect["threshold"]
        if len(series) < w:
            return False, f"need ≥{w} samples for rolling CV; got {len(series)}"
        cvs = []
        for i in range(len(series) - w + 1):
            block = series[i : i + w]
            m = statistics.mean(block)
            if m == 0:
                continue
            cvs.append(statistics.stdev(block) / m if len(block) > 1 else 0.0)
        max_cv = max(cvs) if cvs else 0.0
        return (max_cv < thresh,
                f"max rolling CV={max_cv:.3f} expected < {thresh}")

    if op == "ratio_at_most":
        first, last = value["first"], value["last"]
        if first == 0:
            return False, "first sample is zero; ratio undefined"
        ratio = last / first
        return (ratio <= expect["ratio"],
                f"last/first ratio={ratio:.3f} expected ≤ {expect['ratio']}")

    if op == "ratio_at_least":
        first, last = value["first"], value["last"]
        if first == 0:
            return False, "first sample is zero; ratio undefined"
        ratio = last / first
        return (ratio >= expect["ratio"],
                f"last/first ratio={ratio:.3f} expected ≥ {expect['ratio']}")

    if op == "monotonic_decreasing":
        series = value  # series
        if len(series) < 2:
            return False, f"need ≥2 samples; got {len(series)}"
        peak = series[0]
        max_rebound_pct = expect.get("allow_rebound_pct", 0) / 100.0
        for v in series[1:]:
            if peak == 0:
                continue
            if (v - peak) / peak > max_rebound_pct:
                return False, (
                    f"non-monotonic: peak={peak}, later value={v} "
                    f"(rebound > {max_rebound_pct:.1%})"
                )
            peak = min(peak, v)
        return True, f"monotonic-decreasing (first={series[0]}, last={series[-1]})"

    if op in ("pearson_below", "pearson_above"):
        # Per ADDENDUM P1b-4: distinguish FAIL from INSUFFICIENT_EVIDENCE.
        # Returns a flagged-fail with reason starting "INSUFFICIENT_EVIDENCE:"
        # which the caller (evaluate()) recognises and translates into the
        # proper outcome enum. Conditions for INSUFFICIENT_EVIDENCE:
        #   - fewer than MIN_PEARSON_SAMPLES samples (need enough variance to fit)
        #   - zero variance in either signal (constant signal can't correlate)
        MIN_PEARSON_SAMPLES = 30
        n = min(len(value["x"]), len(value["y"]))
        if n < MIN_PEARSON_SAMPLES:
            return False, (f"INSUFFICIENT_EVIDENCE: n={n} samples below "
                           f"threshold {MIN_PEARSON_SAMPLES} for reliable Pearson r")
        if len(value["x"]) > 1 and statistics.stdev(value["x"]) == 0:
            return False, "INSUFFICIENT_EVIDENCE: zero variance in x signal — sparse mRNA pattern"
        if len(value["y"]) > 1 and statistics.stdev(value["y"]) == 0:
            return False, "INSUFFICIENT_EVIDENCE: zero variance in y signal — sparse mRNA pattern"
        r = _pearson(value["x"], value["y"])
        if r is None:
            return False, "INSUFFICIENT_EVIDENCE: Pearson r undefined (denominator zero)"
        thresh = expect["threshold"]
        if op == "pearson_below":
            return r < thresh, f"r={r:.3f} expected < {thresh}"
        return r > thresh, f"r={r:.3f} expected > {thresh}"

    raise ValueError(f"unknown expect op {op!r}")


# ─── Public entry point ─────────────────────────────────────────────────────

def _normalize_pass_if(pass_if: dict) -> dict:
    """Rename v3 ops to match the existing _check() dispatcher."""
    op = pass_if.get("op")
    if op == "rolling_cv_at_most":
        return {**pass_if, "op": "rolling_cv_below"}
    if op == "pearson_at_most":
        return {**pass_if, "op": "pearson_below"}
    if op == "pearson_at_least":
        return {**pass_if, "op": "pearson_above"}
    return pass_if


def evaluate_v2(entry: dict, history: list,
                *, monomer_ids: list[str] | None = None) -> tuple[str, str]:
    """Evaluate one behavior entry; returns (outcome, message).

    outcome is one of: 'PASS' | 'FAIL' | 'INSUFFICIENT_EVIDENCE' (per
    ADDENDUM P1b-4 and the registered autorepression_test_result enum).
    INSUFFICIENT_EVIDENCE is correctly distinguished from FAIL when the
    underlying signal is too sparse or noise-free to evaluate the test
    (per the biology review: "sparse mRNA data can return insufficient
    evidence rather than false pass/fail").
    """
    passed, msg = evaluate(entry, history, monomer_ids=monomer_ids)
    if not passed and msg.startswith('INSUFFICIENT_EVIDENCE:'):
        return 'INSUFFICIENT_EVIDENCE', msg[len('INSUFFICIENT_EVIDENCE: '):]
    return ('PASS' if passed else 'FAIL'), msg


def evaluate(entry: dict, history: list,
             *, monomer_ids: list[str] | None = None) -> tuple[bool, str]:
    """Evaluate one behavior entry against a loaded history.

    Handles both schema shapes:
      - legacy: ``given.window`` + ``expect``
      - v3:     ``measure.window`` + ``pass_if``

    Returns (passed, message). Doesn't raise — caller decides whether the
    failure should be assert / xfail / skip.

    Backwards-compat wrapper around evaluate_v2: an INSUFFICIENT_EVIDENCE
    outcome is reported as `passed=False` with a message prefixed
    'INSUFFICIENT_EVIDENCE: <reason>'. New callers should use evaluate_v2
    to get the three-state outcome.
    """
    measure = entry["measure"]
    pass_if = entry.get("pass_if") or entry.get("expect")
    if pass_if is None:
        return False, "entry has neither `pass_if` nor `expect`"

    # Window resolution: v3 places window inside measure; legacy under `given`.
    window_name = measure.get("window")
    if window_name is None:
        given = entry.get("given") or {}
        window_name = given.get("window", "full")
    sub_history = _window(history, window_name)
    if not sub_history:
        return False, f"window {window_name!r} produced an empty history slice"

    # The xy_correlation kind walks sub-measures; window doesn't propagate
    # into them. _measure() handles dispatch.
    if measure["kind"] == "xy_correlation":
        # Build x and y from the windowed history.
        x = _measure_series(sub_history, measure["x"], monomer_ids=monomer_ids)
        y = _measure_series(sub_history, measure["y"], monomer_ids=monomer_ids)
        if x is None or y is None:
            return False, "xy_correlation sub-measure returned None"
        # Match lengths (n_bound_TF_per_TU might be empty at some steps).
        n = min(len(x), len(y))
        value = {"x": x[:n], "y": y[:n]}
    else:
        reduce = measure.get("reduce", "series")
        series = _measure_series(sub_history, measure, monomer_ids=monomer_ids)
        if series is None or not series:
            value = None
        elif reduce == "series":
            value = series
        elif reduce == "median":
            value = statistics.median(series)
        elif reduce == "mean":
            value = statistics.mean(series)
        elif reduce == "first_and_last":
            value = {"first": series[0], "last": series[-1]}
        elif reduce == "pre_post_event_ratio":
            value = None   # needs an event index — see req-5
        else:
            return False, f"unknown reduce {reduce!r}"

    return _check(value, _normalize_pass_if(pass_if))
